mod protocol;
mod ring;

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use self::protocol::{Request, Response, Transcript};
use self::ring::{EndpointBuffer, FRAME_SAMPLES, Utterance};
use super::audio::{AudioResampler, read_wave};
use super::{Detection, WakeWordBackend, WakeWordStream, detect_samples};
use crate::backend::Runtime;
use crate::config::Config;
use crate::paths::AppPaths;
use crate::phrase::{PhraseMatcher, normalize_tokens, record_transcript};

unsafe extern "C" {
    fn oma_whisper_open(
        library_path: *const c_char,
        verifier_path: *const c_char,
        vad_path: *const c_char,
        n_threads: c_int,
        output: *mut *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn oma_whisper_version(provider: *const c_void) -> *const c_char;
    fn oma_whisper_vad_probability(
        provider: *mut c_void,
        samples: *const f32,
        sample_count: c_int,
        probability: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn oma_whisper_vad_reset(provider: *mut c_void);
    fn oma_whisper_transcribe(
        provider: *mut c_void,
        samples: *const f32,
        sample_count: c_int,
        prompt: *const c_char,
        text: *mut c_char,
        text_capacity: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn oma_whisper_close(provider: *mut c_void);
}

pub(super) struct WhisperCppBackend {
    worker: RefCell<Worker>,
    matcher: PhraseMatcher,
    prompt: String,
    next_stream_id: Cell<u64>,
}

struct WhisperCppStream<'a> {
    backend: &'a WhisperCppBackend,
    state: RefCell<ClientStream>,
}

struct ClientStream {
    id: u64,
    started: bool,
    finished: bool,
    resampler: AudioResampler,
    pending: VecDeque<f32>,
}

impl WhisperCppBackend {
    pub(super) fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        let executable = env::current_exe().context("resolve Omawake executable")?;
        Self::load_with_executable(config, paths, executable)
    }

    fn load_with_executable(
        config: &Config,
        paths: &AppPaths,
        executable: PathBuf,
    ) -> Result<Self> {
        if config.backend.runtime != Runtime::Default {
            bail!("whispercpp currently supports only backend.runtime = default");
        }
        if config.model.sample_rate != 16_000 {
            bail!("whisper.cpp requires model.sample_rate = 16000");
        }
        if !(1..=64).contains(&config.backend.threads) {
            bail!("backend threads must be between 1 and 64");
        }
        let directory = config.model_directory(paths);
        let library = resolve_library(config, paths)?;
        let verifier = resolve_model_asset(&directory, &config.model.verifier, "verifier")?;
        let vad = resolve_model_asset(&directory, &config.model.vad, "VAD")?;
        let matcher = PhraseMatcher::compile(&config.wake_words)?;
        let prompt = config
            .wake_words
            .iter()
            .filter(|word| word.enabled)
            .flat_map(|word| {
                std::iter::once(word.phrase.as_str()).chain(word.aliases.iter().map(String::as_str))
            })
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(", ");
        let library_dirs = resolve_library_dirs(config, paths, &library)?;
        let worker = Worker::spawn_with_executable(
            executable,
            library,
            library_dirs,
            verifier,
            vad,
            i32::from(config.backend.threads),
        )?;
        Ok(Self {
            worker: RefCell::new(worker),
            matcher,
            prompt,
            next_stream_id: Cell::new(1),
        })
    }

    fn detections(&self, transcripts: Vec<Transcript>) -> Vec<Detection> {
        transcripts_to_detections(&self.matcher, transcripts)
    }
}

fn transcripts_to_detections(
    matcher: &PhraseMatcher,
    transcripts: Vec<Transcript>,
) -> Vec<Detection> {
    transcripts
        .into_iter()
        .flat_map(|transcript| {
            record_transcript(&transcript.text);
            let tokens = normalize_tokens(&transcript.text);
            matcher
                .matches(&transcript.text)
                .into_iter()
                .map(move |matched| Detection {
                    id: matched.id,
                    tokens: tokens[matched.start_token..matched.end_token].to_vec(),
                    timestamps: Vec::new(),
                    start_time: transcript.start_sample as f32 / 16_000.0,
                })
        })
        .collect()
}

impl WakeWordBackend for WhisperCppBackend {
    fn kind(&self) -> &'static str {
        "whispercpp"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        let id = self.next_stream_id.get();
        self.next_stream_id.set(id.wrapping_add(1).max(1));
        Box::new(WhisperCppStream {
            backend: self,
            state: RefCell::new(ClientStream {
                id,
                started: false,
                finished: false,
                resampler: AudioResampler::new(),
                pending: VecDeque::new(),
            }),
        })
    }

    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (sample_rate, samples) = read_wave(path)?;
        let stream = self.stream();
        detect_samples(stream.as_ref(), sample_rate, &samples)
    }
}

impl WhisperCppStream<'_> {
    fn ensure_started(&self, state: &mut ClientStream) -> Result<()> {
        if !state.started {
            self.backend
                .worker
                .try_borrow_mut()
                .map_err(|_| anyhow::anyhow!("another whisper.cpp stream is using the worker"))?
                .start(state.id, &self.backend.prompt)?;
            state.started = true;
        }
        Ok(())
    }

    fn submit_ready(&self, state: &mut ClientStream) -> Result<Vec<Detection>> {
        let mut transcripts = Vec::new();
        while state.pending.len() >= FRAME_SAMPLES {
            let frame: Vec<_> = state.pending.drain(..FRAME_SAMPLES).collect();
            transcripts.extend(
                self.backend
                    .worker
                    .try_borrow_mut()
                    .map_err(|_| anyhow::anyhow!("another whisper.cpp stream is using the worker"))?
                    .audio(state.id, &frame)?,
            );
        }
        Ok(self.backend.detections(transcripts))
    }
}

impl WakeWordStream for WhisperCppStream<'_> {
    fn accept(&self, sample_rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("audio contains a non-finite sample");
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("detector stream is already in use"))?;
        if state.finished {
            bail!("audio was supplied after the stream finished");
        }
        self.ensure_started(&mut state)?;
        let normalized = state.resampler.accept(sample_rate, samples)?;
        state.pending.extend(normalized);
        self.submit_ready(&mut state)
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("detector stream is already in use"))?;
        if state.finished {
            return Ok(Vec::new());
        }
        self.ensure_started(&mut state)?;
        let tail = state.resampler.finish()?;
        state.pending.extend(tail);
        let mut detections = self.submit_ready(&mut state)?;
        if !state.pending.is_empty() {
            state.pending.resize(FRAME_SAMPLES, 0.0);
            detections.extend(self.submit_ready(&mut state)?);
        }
        let transcripts = self
            .backend
            .worker
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("another whisper.cpp stream is using the worker"))?
            .finish(state.id)?;
        detections.extend(self.backend.detections(transcripts));
        state.finished = true;
        Ok(detections)
    }
}

impl Drop for WhisperCppStream<'_> {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        if state.started
            && !state.finished
            && let Ok(mut worker) = self.backend.worker.try_borrow_mut()
        {
            let _ = worker.cancel(state.id);
        }
    }
}

fn resolve_model_asset(directory: &Path, configured: &str, label: &str) -> Result<PathBuf> {
    if configured.trim().is_empty() {
        bail!("model.{label} must name a model file");
    }
    resolve_file_beneath(
        Path::new(configured),
        directory,
        &format!("whisper.cpp {label} model"),
    )
}

fn resolve_library(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    resolve_library_with(
        config,
        paths,
        env::var_os("OMAWAKE_WHISPER_LIBRARY").map(PathBuf::from),
        &package_library_dirs(),
    )
}

fn resolve_library_with(
    config: &Config,
    paths: &AppPaths,
    environment_library: Option<PathBuf>,
    package_dirs: &[PathBuf],
) -> Result<PathBuf> {
    if !config.backend.library.as_os_str().is_empty() {
        return resolve_file_beneath(
            &config.backend.library,
            paths.config_file.parent().unwrap_or(Path::new(".")),
            "configured whisper.cpp library",
        );
    }
    if let Some(path) = environment_library {
        if !path.is_absolute() {
            bail!("OMAWAKE_WHISPER_LIBRARY must be an absolute file path");
        }
        return resolve_file_beneath(&path, Path::new("/"), "OMAWAKE_WHISPER_LIBRARY");
    }
    for directory in package_dirs.iter().map(PathBuf::as_path).chain([
        Path::new("/usr/lib"),
        Path::new("/usr/local/lib"),
        Path::new("/usr/lib64"),
    ]) {
        for name in ["libwhisper.so.1", "libwhisper.so"] {
            let path = directory.join(name);
            if path.is_file() {
                return path
                    .canonicalize()
                    .with_context(|| format!("resolve whisper.cpp library {}", path.display()));
            }
        }
    }
    bail!("libwhisper was not found; set backend.library or OMAWAKE_WHISPER_LIBRARY")
}

fn package_library_dirs() -> Vec<PathBuf> {
    let Some(binary) = env::current_exe().ok() else {
        return Vec::new();
    };
    package_library_dirs_from(&binary)
}

fn package_library_dirs_from(binary: &Path) -> Vec<PathBuf> {
    let Some(binary_dir) = binary.parent() else {
        return Vec::new();
    };
    let mut candidates = vec![binary_dir.join("lib")];
    if matches!(
        binary_dir.file_name().and_then(|name| name.to_str()),
        Some("bin" | "sbin")
    ) && let Some(prefix) = binary_dir.parent()
    {
        candidates.push(prefix.join("lib/omawake"));
    }
    candidates
        .into_iter()
        .filter(|path| path.is_dir())
        .map(|path| path.canonicalize().unwrap_or(path))
        .collect()
}

fn resolve_library_dirs(config: &Config, paths: &AppPaths, library: &Path) -> Result<Vec<PathBuf>> {
    let base = paths.config_file.parent().unwrap_or(Path::new("."));
    let mut directories = Vec::with_capacity(config.backend.library_dirs.len() + 1);
    for configured in &config.backend.library_dirs {
        let explicit_absolute = configured.is_absolute();
        let candidate = if explicit_absolute {
            configured.clone()
        } else {
            base.join(configured)
        };
        let resolved = candidate.canonicalize().with_context(|| {
            format!("resolve backend.library_dirs entry {}", candidate.display())
        })?;
        if !resolved.is_dir() {
            bail!(
                "backend.library_dirs entry is not a directory: {}",
                resolved.display()
            );
        }
        if !explicit_absolute {
            let canonical_base = base
                .canonicalize()
                .with_context(|| format!("resolve config directory {}", base.display()))?;
            if !resolved.starts_with(&canonical_base) {
                bail!(
                    "backend.library_dirs entry escapes config directory {}",
                    canonical_base.display()
                );
            }
        }
        if !directories.contains(&resolved) {
            directories.push(resolved);
        }
    }
    let parent = library
        .parent()
        .context("configured whisper.cpp library has no parent directory")?
        .to_path_buf();
    if !directories.contains(&parent) {
        directories.push(parent);
    }
    Ok(directories)
}

fn resolve_file_beneath(path: &Path, base: &Path, label: &str) -> Result<PathBuf> {
    let explicit_absolute = path.is_absolute();
    let candidate = if explicit_absolute {
        path.to_owned()
    } else {
        base.join(path)
    };
    let resolved = candidate
        .canonicalize()
        .with_context(|| format!("resolve {label} {}", candidate.display()))?;
    if !resolved.is_file() {
        bail!("{label} is not a regular file: {}", resolved.display());
    }
    if !explicit_absolute {
        let canonical_base = base
            .canonicalize()
            .with_context(|| format!("resolve {label} base {}", base.display()))?;
        if !resolved.starts_with(&canonical_base) {
            bail!(
                "{label} escapes base directory {}",
                canonical_base.display()
            );
        }
    }
    Ok(resolved)
}

const WORKER_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const WORKER_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const WORKER_STOP_TIMEOUT: Duration = Duration::from_secs(2);

struct TimedReader {
    reader: ChildStdout,
    timeout: Duration,
}

impl io::Read for TimedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        wait_for_io(self.reader.as_raw_fd(), libc::POLLIN, self.timeout).map_err(|error| {
            io::Error::new(error.kind(), format!("wait for worker IPC read: {error}"))
        })?;
        io::Read::read(&mut self.reader, buffer)
            .map_err(|error| io::Error::new(error.kind(), format!("read worker IPC pipe: {error}")))
    }
}

struct TimedWriter {
    writer: ChildStdin,
    timeout: Duration,
}

impl io::Write for TimedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        wait_for_io(self.writer.as_raw_fd(), libc::POLLOUT, self.timeout).map_err(|error| {
            io::Error::new(error.kind(), format!("wait for worker IPC write: {error}"))
        })?;
        io::Write::write(&mut self.writer, buffer).map_err(|error| {
            io::Error::new(error.kind(), format!("write worker IPC pipe: {error}"))
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        io::Write::flush(&mut self.writer)
    }
}

fn wait_for_io(fd: libc::c_int, events: libc::c_short, timeout: Duration) -> io::Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let milliseconds = if remaining.is_zero() {
            0
        } else {
            i32::try_from(remaining.as_millis().max(1)).unwrap_or(i32::MAX)
        };
        let mut descriptor = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let status = unsafe { libc::poll(&mut descriptor, 1, milliseconds) };
        if status > 0 {
            if descriptor.revents & libc::POLLNVAL != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "whisper.cpp worker IPC descriptor is invalid",
                ));
            }
            return Ok(());
        }
        if status == 0 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "whisper.cpp worker IPC timed out",
            ));
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[derive(Clone)]
struct WorkerSpec {
    executable: PathBuf,
    library: PathBuf,
    library_dirs: Vec<PathBuf>,
    verifier: PathBuf,
    vad: PathBuf,
    threads: i32,
}

struct WorkerProcess {
    child: Child,
    input: Option<TimedWriter>,
    output: Option<TimedReader>,
}

impl WorkerProcess {
    fn launch(spec: &WorkerSpec) -> Result<Self> {
        let mut command = Command::new(&spec.executable);
        command
            .arg("__whisper-worker")
            .arg(&spec.library)
            .arg(&spec.verifier)
            .arg(&spec.vad)
            .arg(spec.threads.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        configure_library_path(&mut command, &spec.library_dirs)?;
        let mut child = command
            .spawn()
            .context("spawn isolated whisper.cpp worker")?;
        let input = child
            .stdin
            .take()
            .context("whisper.cpp worker stdin is unavailable")?;
        let output = child
            .stdout
            .take()
            .context("whisper.cpp worker stdout is unavailable")?;
        let mut process = Self {
            child,
            input: Some(TimedWriter {
                writer: input,
                timeout: WORKER_STARTUP_TIMEOUT,
            }),
            output: Some(TimedReader {
                reader: output,
                timeout: WORKER_STARTUP_TIMEOUT,
            }),
        };
        let handshake = process.read_response().context("read worker handshake");
        match handshake {
            Ok(Response::Ready { .. }) => {
                process.input.as_mut().expect("worker input exists").timeout =
                    WORKER_REQUEST_TIMEOUT;
                process
                    .output
                    .as_mut()
                    .expect("worker output exists")
                    .timeout = WORKER_REQUEST_TIMEOUT;
                Ok(process)
            }
            Ok(Response::Error { message, .. }) => {
                process.stop(false);
                bail!(message)
            }
            Ok(response) => {
                process.stop(false);
                bail!("unexpected worker handshake: {response:?}")
            }
            Err(error) => {
                process.stop(false);
                Err(error)
            }
        }
    }

    fn write_request(&mut self, request: &Request, pcm: &[f32]) -> Result<()> {
        protocol::write_request(
            self.input
                .as_mut()
                .context("whisper.cpp worker input is closed")?,
            request,
            pcm,
        )
        .context("write worker request")
    }

    fn read_response(&mut self) -> Result<Response> {
        protocol::read_response(
            self.output
                .as_mut()
                .context("whisper.cpp worker output is closed")?,
        )
        .context("read worker response")
    }

    fn stop(&mut self, graceful: bool) {
        if graceful && let Some(input) = self.input.as_mut() {
            input.timeout = Duration::from_secs(1);
            let _ = protocol::write_request(input, &Request::Shutdown, &[]);
        }
        self.input.take();
        self.output.take();
        if !wait_for_exit(&mut self.child, WORKER_STOP_TIMEOUT) {
            let _ = self.child.kill();
            let _ = wait_for_exit(&mut self.child, WORKER_STOP_TIMEOUT);
        }
    }
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            Ok(None) | Err(_) => return false,
        }
    }
}

struct Worker {
    spec: WorkerSpec,
    process: Option<WorkerProcess>,
    active_id: Option<u64>,
}

impl Worker {
    fn spawn_with_executable(
        executable: PathBuf,
        library: PathBuf,
        library_dirs: Vec<PathBuf>,
        verifier: PathBuf,
        vad: PathBuf,
        threads: i32,
    ) -> Result<Self> {
        let spec = WorkerSpec {
            executable,
            library,
            library_dirs,
            verifier,
            vad,
            threads,
        };
        let process = WorkerProcess::launch(&spec)?;
        Ok(Self {
            spec,
            process: Some(process),
            active_id: None,
        })
    }

    fn ensure_process(&mut self) -> Result<()> {
        if self.process.is_none() {
            self.process = Some(WorkerProcess::launch(&self.spec)?);
        }
        Ok(())
    }

    fn disconnect(&mut self) {
        if let Some(mut process) = self.process.take() {
            process.stop(false);
        }
        self.active_id = None;
    }

    fn exchange(&mut self, request: &Request, pcm: &[f32]) -> Result<Response> {
        self.ensure_process()?;
        let result = (|| {
            let process = self
                .process
                .as_mut()
                .context("whisper.cpp worker is unavailable")?;
            process.write_request(request, pcm)?;
            process.read_response()
        })();
        if result.is_err() {
            self.disconnect();
        }
        result
    }

    fn invalid_response<T>(&mut self, operation: &str, response: Response) -> Result<T> {
        self.disconnect();
        bail!("unexpected {operation} response: {response:?}")
    }

    fn start(&mut self, id: u64, prompt: &str) -> Result<()> {
        if let Some(active) = self.active_id {
            bail!("whisper.cpp worker is already serving stream {active}");
        }
        let request = Request::Start {
            id,
            prompt: prompt.to_owned(),
        };
        let response = match self.exchange(&request, &[]) {
            Ok(response) => response,
            Err(first_error) => self.exchange(&request, &[]).with_context(|| {
                format!("restart whisper.cpp worker after startup IPC failure: {first_error:#}")
            })?,
        };
        match response {
            Response::Ack { id: response } if response == id => {
                self.active_id = Some(id);
                Ok(())
            }
            Response::Error { message, .. } => bail!(message),
            response => self.invalid_response("start", response),
        }
    }

    fn audio(&mut self, id: u64, pcm: &[f32]) -> Result<Vec<Transcript>> {
        if self.active_id != Some(id) {
            bail!("whisper.cpp worker is not serving stream {id}");
        }
        match self.exchange(
            &Request::Audio {
                id,
                samples: pcm.len(),
            },
            pcm,
        )? {
            Response::Result {
                id: response,
                transcripts,
            } if response == id => Ok(transcripts),
            Response::Error { message, .. } => bail!(message),
            response => self.invalid_response("audio", response),
        }
    }

    fn finish(&mut self, id: u64) -> Result<Vec<Transcript>> {
        if self.active_id != Some(id) {
            bail!("whisper.cpp worker is not serving stream {id}");
        }
        let response = self.exchange(&Request::Finish { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Result {
                id: response,
                transcripts,
            } if response == id => Ok(transcripts),
            Response::Error { message, .. } => bail!(message),
            response => self.invalid_response("finish", response),
        }
    }

    fn cancel(&mut self, id: u64) -> Result<()> {
        if self.active_id != Some(id) {
            return Ok(());
        }
        let response = self.exchange(&Request::Cancel { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Ack { id: response } if response == id => Ok(()),
            Response::Error { message, .. } => bail!(message),
            response => self.invalid_response("cancel", response),
        }
    }

    fn shutdown(&mut self) {
        if let Some(mut process) = self.process.take() {
            process.stop(true);
        }
        self.active_id = None;
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn configure_library_path(command: &mut Command, configured: &[PathBuf]) -> Result<()> {
    let mut paths = configured.to_vec();
    if let Some(existing) = env::var_os("LD_LIBRARY_PATH") {
        for path in env::split_paths(&existing) {
            if !path.is_absolute() || !path.is_dir() {
                continue;
            }
            let path = path.canonicalize().unwrap_or(path);
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    let joined = env::join_paths(paths).context("encode whisper.cpp worker library path")?;
    command.env("LD_LIBRARY_PATH", joined);
    Ok(())
}

fn harden_worker_process() -> Result<()> {
    let limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let limit_status = unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) };
    let limit_error = (limit_status != 0).then(io::Error::last_os_error);
    let dump_status = unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) };
    let dump_error = (dump_status != 0).then(io::Error::last_os_error);
    if limit_status == 0 || dump_status == 0 {
        return Ok(());
    }
    bail!(
        "could not disable worker core dumps: setrlimit: {}; prctl: {}",
        limit_error.expect("failed setrlimit has an OS error"),
        dump_error.expect("failed prctl has an OS error")
    )
}

struct NativeProvider(*mut c_void);

impl NativeProvider {
    fn open(library: &Path, verifier: &Path, vad: &Path, threads: i32) -> Result<Self> {
        let library = path_to_c_string(library)?;
        let verifier = path_to_c_string(verifier)?;
        let vad = path_to_c_string(vad)?;
        let mut provider = std::ptr::null_mut();
        let mut error = vec![0 as c_char; 2048];
        let status = unsafe {
            oma_whisper_open(
                library.as_ptr(),
                verifier.as_ptr(),
                vad.as_ptr(),
                threads,
                &mut provider,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 || provider.is_null() {
            bail!(c_buffer(&error));
        }
        Ok(Self(provider))
    }

    fn version(&self) -> String {
        let version = unsafe { oma_whisper_version(self.0) };
        if version.is_null() {
            return "unknown".into();
        }
        unsafe { CStr::from_ptr(version) }
            .to_string_lossy()
            .into_owned()
    }

    fn vad_probability(&mut self, samples: &[f32]) -> Result<f32> {
        let mut probability = 0.0;
        let mut error = vec![0 as c_char; 2048];
        let status = unsafe {
            oma_whisper_vad_probability(
                self.0,
                samples.as_ptr(),
                i32::try_from(samples.len()).context("VAD frame is too large")?,
                &mut probability,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(c_buffer(&error));
        }
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            bail!("whisper VAD returned invalid probability {probability}");
        }
        Ok(probability)
    }

    fn reset_vad(&mut self) {
        unsafe { oma_whisper_vad_reset(self.0) };
    }

    fn transcribe(&mut self, samples: &[f32], prompt: &str) -> Result<String> {
        let prompt = CString::new(prompt).context("wake-word prompt contains NUL")?;
        let mut text = vec![0 as c_char; 64 * 1024];
        let mut error = vec![0 as c_char; 2048];
        let status = unsafe {
            oma_whisper_transcribe(
                self.0,
                samples.as_ptr(),
                i32::try_from(samples.len()).context("utterance is too large")?,
                prompt.as_ptr(),
                text.as_mut_ptr(),
                text.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status < 0 {
            bail!(c_buffer(&error));
        }
        Ok(unsafe { CStr::from_ptr(text.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_owned())
    }
}

impl Drop for NativeProvider {
    fn drop(&mut self) {
        unsafe { oma_whisper_close(self.0) };
    }
}

fn path_to_c_string(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_encoded_bytes()).context("native path contains NUL")
}

fn c_buffer(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

struct WorkerStream {
    id: u64,
    prompt: String,
    endpoint: EndpointBuffer,
}

impl WorkerStream {
    fn finish_utterance(
        &self,
        provider: &mut NativeProvider,
        utterance: Utterance,
    ) -> Result<Transcript> {
        let text = provider.transcribe(&utterance.samples, &self.prompt)?;
        Ok(Transcript {
            text,
            start_sample: utterance.start_sample,
            end_sample: utterance.end_sample,
        })
    }
}

pub(crate) fn worker_main(library: &Path, verifier: &Path, vad: &Path, threads: i32) -> Result<()> {
    let mut output = io::stdout().lock();
    let mut input = io::stdin().lock();
    if let Err(error) = harden_worker_process() {
        protocol::write_response(
            &mut output,
            &Response::Error {
                id: None,
                message: format!("{error:#}"),
            },
        )?;
        return Ok(());
    }
    let mut provider = match NativeProvider::open(library, verifier, vad, threads) {
        Ok(provider) => provider,
        Err(error) => {
            protocol::write_response(
                &mut output,
                &Response::Error {
                    id: None,
                    message: format!("{error:#}"),
                },
            )?;
            return Ok(());
        }
    };
    protocol::write_response(
        &mut output,
        &Response::Ready {
            version: provider.version(),
        },
    )?;
    worker_loop(&mut provider, &mut input, &mut output)
}

fn worker_loop(
    provider: &mut NativeProvider,
    input: &mut impl io::Read,
    output: &mut impl io::Write,
) -> Result<()> {
    let mut stream: Option<WorkerStream> = None;
    loop {
        let (request, pcm) = protocol::read_request(input)?;
        let response = match request {
            Request::Shutdown => return Ok(()),
            Request::Start { id, prompt } => {
                provider.reset_vad();
                stream = Some(WorkerStream {
                    id,
                    prompt,
                    endpoint: EndpointBuffer::new(),
                });
                Response::Ack { id }
            }
            Request::Audio { id, .. } => match stream.as_mut() {
                Some(active) if active.id == id && pcm.len() == FRAME_SAMPLES => {
                    let result = (|| {
                        if pcm.iter().any(|sample| !sample.is_finite()) {
                            bail!("audio contains a non-finite sample");
                        }
                        let probability = provider.vad_probability(&pcm)?;
                        let utterance = active.endpoint.push(&pcm, probability);
                        let transcript = utterance
                            .map(|utterance| active.finish_utterance(provider, utterance))
                            .transpose()?;
                        if transcript.is_some() {
                            provider.reset_vad();
                        }
                        Ok::<_, anyhow::Error>(transcript.into_iter().collect())
                    })();
                    match result {
                        Ok(transcripts) => Response::Result { id, transcripts },
                        Err(error) => Response::Error {
                            id: Some(id),
                            message: format!("{error:#}"),
                        },
                    }
                }
                Some(_) => Response::Error {
                    id: Some(id),
                    message: "worker stream id does not match".into(),
                },
                None => Response::Error {
                    id: Some(id),
                    message: "worker has no active stream".into(),
                },
            },
            Request::Finish { id } => match stream.take() {
                Some(mut active) if active.id == id => {
                    let result = active
                        .endpoint
                        .finish()
                        .map(|utterance| active.finish_utterance(provider, utterance))
                        .transpose();
                    provider.reset_vad();
                    match result {
                        Ok(transcript) => Response::Result {
                            id,
                            transcripts: transcript.into_iter().collect(),
                        },
                        Err(error) => Response::Error {
                            id: Some(id),
                            message: format!("{error:#}"),
                        },
                    }
                }
                Some(active) => {
                    stream = Some(active);
                    Response::Error {
                        id: Some(id),
                        message: "worker stream id does not match".into(),
                    }
                }
                None => Response::Error {
                    id: Some(id),
                    message: "worker has no active stream".into(),
                },
            },
            Request::Cancel { id } => match stream.as_ref() {
                Some(active) if active.id == id => {
                    stream = None;
                    provider.reset_vad();
                    Response::Ack { id }
                }
                _ => Response::Error {
                    id: Some(id),
                    message: "worker stream id does not match".into(),
                },
            },
        };
        protocol::write_response(output, &response)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WakeWord;
    use std::fs;
    use std::io::{Read, Write};
    use std::process::Command;
    use std::sync::OnceLock;

    const FAKE_WHISPER_C: &str = r#"
    #include <stdlib.h>
    #include <string.h>
    #include "whisper_abi.h"

    struct whisper_context { int segments; };
    struct whisper_vad_context { float probability; };

    const char *whisper_version(void) { return "1.9.3"; }
    void whisper_log_set(ggml_log_callback callback, void *data) { (void) callback; (void) data; }
    struct whisper_context_params whisper_context_default_params(void) {
    struct whisper_context_params value = {0}; return value;
    }
    struct whisper_context *whisper_init_from_file_with_params(
        const char *path, struct whisper_context_params params) {
    (void) params; return strstr(path, "reject") ? NULL : calloc(1, sizeof(struct whisper_context));
    }
    struct whisper_full_params whisper_full_default_params(enum whisper_sampling_strategy strategy) {
    struct whisper_full_params value = {0}; value.strategy = strategy; return value;
    }
    int whisper_full(struct whisper_context *context, struct whisper_full_params params,
                 const float *samples, int count) {
    (void) params; if (!context || !samples || count <= 0 || samples[0] == -2.0f) return -1;
    context->segments = 1; return 0;
    }
    int whisper_full_n_segments(struct whisper_context *context) { return context->segments; }
    const char *whisper_full_get_segment_text(struct whisper_context *context, int index) {
    (void) context; return index == 0 ? "  hey computer  " : NULL;
    }
    void whisper_free(struct whisper_context *context) { free(context); }
    struct whisper_vad_context_params whisper_vad_default_context_params(void) {
    struct whisper_vad_context_params value = {0}; return value;
    }
    struct whisper_vad_context *whisper_vad_init_from_file_with_params(
        const char *path, struct whisper_vad_context_params params) {
    (void) params; return strstr(path, "reject") ? NULL : calloc(1, sizeof(struct whisper_vad_context));
    }
    bool whisper_vad_detect_speech_no_reset(struct whisper_vad_context *context,
                                        const float *samples, int count) {
    if (!context || !samples || count != 512 || samples[0] == -1.0f) return false;
    context->probability = samples[0]; return true;
    }
    void whisper_vad_reset_state(struct whisper_vad_context *context) {
    if (context) context->probability = 0.0f;
    }
    int whisper_vad_n_probs(struct whisper_vad_context *context) { (void) context; return 1; }
    float *whisper_vad_probs(struct whisper_vad_context *context) { return &context->probability; }
    void whisper_vad_free(struct whisper_vad_context *context) { free(context); }
    "#;

    fn fake_whisper_library() -> &'static Path {
        static LIBRARY: OnceLock<PathBuf> = OnceLock::new();
        LIBRARY
            .get_or_init(|| {
                let root = env::temp_dir()
                    .join(format!("omawake-safe-fake-whisper-{}", std::process::id()));
                fs::create_dir_all(&root).unwrap();
                let source = root.join("fake_whisper.c");
                let library = root.join("libwhisper.so.1");
                fs::write(&source, FAKE_WHISPER_C).unwrap();
                let include = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/whispercpp-1.9.3");
                let output = Command::new("cc")
                    .args(["-shared", "-fPIC", "-O0"])
                    .arg(format!("-I{}", include.display()))
                    .arg(&source)
                    .arg("-o")
                    .arg(&library)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "compile fake whisper.cpp DSO: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                library
            })
            .as_path()
    }

    fn fake_models(name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = env::temp_dir().join(format!(
            "omawake-whisper-model-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let verifier = root.join("verifier.bin");
        let vad = root.join("vad.bin");
        fs::write(&verifier, b"safe fake verifier").unwrap();
        fs::write(&vad, b"safe fake vad").unwrap();
        (root, verifier, vad)
    }

    fn fake_worker_spec(name: &str) -> (PathBuf, WorkerSpec) {
        let (root, verifier, vad) = fake_models(name);
        (
            root,
            WorkerSpec {
                executable: crate::engine::audiocpp::tests::fake_worker().to_path_buf(),
                library: fake_whisper_library().to_path_buf(),
                library_dirs: Vec::new(),
                verifier,
                vad,
                threads: 2,
            },
        )
    }

    fn fake_worker(name: &str) -> (PathBuf, Worker) {
        let (root, spec) = fake_worker_spec(name);
        let process = WorkerProcess::launch(&spec).unwrap();
        (
            root,
            Worker {
                spec,
                process: Some(process),
                active_id: None,
            },
        )
    }

    fn fake_worker_mode(name: &str, mode: i32) -> (PathBuf, Worker) {
        let (root, mut spec) = fake_worker_spec(name);
        spec.threads = mode;
        let process = WorkerProcess::launch(&spec).unwrap();
        (
            root,
            Worker {
                spec,
                process: Some(process),
                active_id: None,
            },
        )
    }

    #[test]
    fn transcript_conversion_uses_whole_phrase_matcher() {
        let matcher = PhraseMatcher::compile(&[WakeWord {
            engine: None,
            enrollment: None,
            id: "computer".into(),
            phrase: "hey computer".into(),
            aliases: Vec::new(),
            enabled: true,
            command: vec!["true".into()],
        }])
        .unwrap();
        let detections = transcripts_to_detections(
            &matcher,
            vec![Transcript {
                text: "A computerized voice said: hey, computer!".into(),
                start_sample: 8_000,
                end_sample: 16_000,
            }],
        );
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].id, "computer");
        assert_eq!(detections[0].tokens, ["hey", "computer"]);
        assert_eq!(detections[0].start_time, 0.5);
    }

    #[test]
    fn supervised_worker_and_high_level_stream_complete_without_native_crashes() {
        let (root, mut worker) = fake_worker("client-lifecycle");
        assert!(worker.audio(1, &[0.0; FRAME_SAMPLES]).is_err());
        worker.cancel(1).unwrap();
        worker.start(1, "hello oma").unwrap();
        assert!(worker.start(2, "hello oma").is_err());
        assert!(worker.audio(1, &[0.0; FRAME_SAMPLES]).unwrap().is_empty());
        assert_eq!(worker.finish(1).unwrap()[0].text, "hello oma");
        assert!(worker.finish(1).is_err());
        worker.start(2, "hello oma").unwrap();
        worker.cancel(2).unwrap();
        worker.shutdown();

        let (backend_root, verifier, vad) = fake_models("backend-lifecycle");
        let paths = AppPaths {
            config_file: backend_root.join("config/config.toml"),
            data_dir: backend_root.join("data"),
            cache_dir: backend_root.join("cache"),
            state_dir: backend_root.join("state"),
            runtime_dir: backend_root.join("run"),
        };
        let mut config = Config::default();
        config.backend.kind = "whispercpp".into();
        config.backend.library = fake_whisper_library().to_path_buf();
        config.backend.threads = 2;
        config.model.directory = backend_root.display().to_string();
        config.model.verifier = verifier.file_name().unwrap().to_string_lossy().into_owned();
        config.model.vad = vad.file_name().unwrap().to_string_lossy().into_owned();
        config.wake_words = vec![WakeWord {
            engine: None,
            enrollment: None,
            id: "greeting".into(),
            phrase: "hello oma".into(),
            aliases: vec!["hello oh ma".into()],
            enabled: true,
            command: vec!["true".into()],
        }];
        let backend = WhisperCppBackend::load_with_executable(
            &config,
            &paths,
            crate::engine::audiocpp::tests::fake_worker().to_path_buf(),
        )
        .unwrap();
        assert_eq!(backend.kind(), "whispercpp");
        assert_eq!(backend.prompt, "hello oma, hello oh ma");
        let stream = backend.stream();
        assert!(stream.accept(16_000, &[f32::NAN]).is_err());
        assert!(
            stream
                .accept(16_000, &[0.0; FRAME_SAMPLES])
                .unwrap()
                .is_empty()
        );
        assert_eq!(stream.finish().unwrap()[0].id, "greeting");
        assert!(stream.finish().unwrap().is_empty());
        assert!(stream.accept(16_000, &[0.0]).is_err());
        drop(stream);

        let short = backend.stream();
        short.accept(16_000, &[0.0; 16]).unwrap();
        assert_eq!(short.finish().unwrap()[0].id, "greeting");
        drop(short);
        let abandoned = backend.stream();
        abandoned.accept(16_000, &[0.0; 16]).unwrap();
        drop(abandoned);
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(backend_root).unwrap();
    }

    #[test]
    fn supervised_worker_rejects_handshake_and_operation_protocol_errors_cleanly() {
        for (name, mode) in [
            ("handshake-error", 101),
            ("handshake-unexpected", 102),
            ("handshake-eof", 103),
        ] {
            let (root, mut spec) = fake_worker_spec(name);
            spec.threads = mode;
            assert!(WorkerProcess::launch(&spec).is_err());
            fs::remove_dir_all(root).unwrap();
        }

        for (name, mode) in [("start-error", 201), ("start-unexpected", 202)] {
            let (root, mut worker) = fake_worker_mode(name, mode);
            assert!(worker.start(7, "hello oma").is_err());
            fs::remove_dir_all(root).unwrap();
        }
        for (name, mode) in [("audio-error", 201), ("audio-unexpected", 202)] {
            let (root, mut worker) = fake_worker_mode(name, mode);
            worker.active_id = Some(7);
            assert!(worker.audio(7, &[0.0; FRAME_SAMPLES]).is_err());
            fs::remove_dir_all(root).unwrap();
        }
        for (name, mode) in [("finish-error", 201), ("finish-unexpected", 202)] {
            let (root, mut worker) = fake_worker_mode(name, mode);
            worker.active_id = Some(7);
            assert!(worker.finish(7).is_err());
            fs::remove_dir_all(root).unwrap();
        }
        for (name, mode) in [("cancel-error", 201), ("cancel-unexpected", 202)] {
            let (root, mut worker) = fake_worker_mode(name, mode);
            worker.active_id = Some(7);
            assert!(worker.cancel(7).is_err());
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn cpu_checkpoint_rejects_accelerator_runtime_without_spawning() {
        let mut config = Config::default();
        config.backend.kind = "whispercpp".into();
        config.backend.runtime = Runtime::Cuda;
        let error = WhisperCppBackend::load(&config, &AppPaths::discover())
            .err()
            .expect("unsupported runtime should fail");
        assert!(error.to_string().contains("only backend.runtime = default"));
    }

    #[test]
    fn native_paths_are_canonical_and_relative_paths_cannot_escape() {
        let root = env::temp_dir().join(format!("omawake-whisper-paths-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let config_dir = root.join("config");
        let library_dir = config_dir.join("lib");
        let model_dir = root.join("models");
        fs::create_dir_all(&library_dir).unwrap();
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(library_dir.join("libwhisper.so"), b"fixture").unwrap();
        fs::write(model_dir.join("verifier.bin"), b"fixture").unwrap();
        fs::write(root.join("outside.bin"), b"fixture").unwrap();
        fs::write(root.join("outside.so"), b"fixture").unwrap();

        let paths = AppPaths {
            config_file: config_dir.join("config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
            state_dir: root.join("state"),
            runtime_dir: root.join("run"),
        };
        let mut config = Config::default();
        config.backend.library = PathBuf::from("lib/libwhisper.so");
        config.backend.library_dirs = vec![PathBuf::from("lib")];
        let library = resolve_library(&config, &paths).unwrap();
        assert_eq!(
            library,
            library_dir.join("libwhisper.so").canonicalize().unwrap()
        );
        assert_eq!(
            resolve_library_dirs(&config, &paths, &library).unwrap(),
            [library_dir.canonicalize().unwrap()]
        );
        assert_eq!(
            resolve_model_asset(&model_dir, "verifier.bin", "verifier").unwrap(),
            model_dir.join("verifier.bin").canonicalize().unwrap()
        );

        config.backend.library.clear();
        let discovered = root.join("discovered");
        fs::create_dir_all(&discovered).unwrap();
        fs::write(discovered.join("libwhisper.so.1"), b"fixture").unwrap();
        assert_eq!(
            resolve_library_with(&config, &paths, None, std::slice::from_ref(&discovered)).unwrap(),
            discovered.join("libwhisper.so.1").canonicalize().unwrap()
        );
        assert_eq!(
            resolve_library_with(
                &config,
                &paths,
                Some(fake_whisper_library().to_path_buf()),
                &[]
            )
            .unwrap(),
            fake_whisper_library().canonicalize().unwrap()
        );
        assert!(
            resolve_library_with(
                &config,
                &paths,
                Some(PathBuf::from("relative/libwhisper.so")),
                &[]
            )
            .is_err()
        );
        assert!(resolve_library_with(&config, &paths, None, &[]).is_err());

        config.backend.library = PathBuf::from("../outside.so");
        assert!(resolve_library(&config, &paths).is_err());
        assert!(resolve_model_asset(&model_dir, "", "verifier").is_err());
        assert!(resolve_model_asset(&model_dir, "../outside.bin", "verifier").is_err());
        config.backend.library_dirs = vec![root.join("outside.bin")];
        assert!(resolve_library_dirs(&config, &paths, &library).is_err());
        config.backend.library_dirs = vec![PathBuf::from("../models")];
        assert!(resolve_library_dirs(&config, &paths, &library).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn package_discovery_stays_with_the_archive_or_installed_prefix() {
        let root = env::temp_dir().join(format!("omawake-whisper-package-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("archive/lib")).unwrap();
        fs::create_dir_all(root.join("prefix/bin")).unwrap();
        fs::create_dir_all(root.join("prefix/lib/omawake")).unwrap();
        assert_eq!(
            package_library_dirs_from(&root.join("archive/omawake")),
            [root.join("archive/lib").canonicalize().unwrap()]
        );
        assert_eq!(
            package_library_dirs_from(&root.join("prefix/bin/omawake")),
            [root.join("prefix/lib/omawake").canonicalize().unwrap()]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn bounded_wait_reaps_a_normal_child_exit() {
        let mut child = Command::new("true").spawn().unwrap();
        assert!(wait_for_exit(&mut child, Duration::from_secs(1)));
    }

    #[test]
    fn worker_pipes_round_trip_without_socket_permissions() {
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = TimedWriter {
            writer: child.stdin.take().unwrap(),
            timeout: Duration::from_secs(1),
        };
        let mut output = TimedReader {
            reader: child.stdout.take().unwrap(),
            timeout: Duration::from_secs(1),
        };

        input.write_all(b"worker pipe").unwrap();
        input.flush().unwrap();
        let mut echoed = [0_u8; 11];
        output.read_exact(&mut echoed).unwrap();
        assert_eq!(&echoed, b"worker pipe");

        drop(input);
        drop(output);
        assert!(wait_for_exit(&mut child, Duration::from_secs(1)));
    }

    #[test]
    fn worker_pipe_deadline_times_out_and_child_exits_normally() {
        let mut child = Command::new("sleep")
            .arg("0.05")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut output = TimedReader {
            reader: child.stdout.take().unwrap(),
            timeout: Duration::from_millis(1),
        };
        let error = output.read(&mut [0_u8; 1]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(wait_for_exit(&mut child, Duration::from_secs(1)));
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn fake_provider_loads_real_dso_and_checks_vad_and_transcription_results() {
        let (root, verifier, vad) = fake_models("provider");
        let mut provider =
            NativeProvider::open(fake_whisper_library(), &verifier, &vad, 2).unwrap();
        assert_eq!(provider.version(), "1.9.3");
        assert_eq!(
            provider
                .vad_probability(&vec![0.75; FRAME_SAMPLES])
                .unwrap(),
            0.75
        );
        provider.reset_vad();
        assert_eq!(
            provider
                .transcribe(&vec![1.0; FRAME_SAMPLES], "hey computer")
                .unwrap(),
            "hey computer"
        );
        assert!(provider.vad_probability(&[0.0; 10]).is_err());
        assert!(
            provider
                .vad_probability(&vec![-1.0; FRAME_SAMPLES])
                .is_err()
        );
        assert!(
            provider
                .vad_probability(&vec![f32::NAN; FRAME_SAMPLES])
                .is_err()
        );
        assert!(provider.transcribe(&[], "hey computer").is_err());
        assert!(
            provider
                .transcribe(&vec![1.0; FRAME_SAMPLES], "bad\0prompt")
                .is_err()
        );
        assert!(
            provider
                .transcribe(&vec![-2.0; FRAME_SAMPLES], "hey computer")
                .is_err()
        );
        drop(provider);

        let rejected = root.join("reject-verifier.bin");
        fs::write(&rejected, b"reject").unwrap();
        assert!(NativeProvider::open(fake_whisper_library(), &rejected, &vad, 2).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn in_memory_worker_protocol_covers_stream_lifecycle_without_child_crashes() {
        let (root, verifier, vad) = fake_models("worker-loop");
        let mut provider =
            NativeProvider::open(fake_whisper_library(), &verifier, &vad, 2).unwrap();
        let mut input = Vec::new();
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 1,
                samples: FRAME_SAMPLES,
            },
            &vec![0.0; FRAME_SAMPLES],
        )
        .unwrap();
        protocol::write_request(
            &mut input,
            &Request::Start {
                id: 1,
                prompt: "hey computer".into(),
            },
            &[],
        )
        .unwrap();
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 2,
                samples: FRAME_SAMPLES,
            },
            &vec![0.0; FRAME_SAMPLES],
        )
        .unwrap();
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 1,
                samples: FRAME_SAMPLES,
            },
            &vec![f32::NAN; FRAME_SAMPLES],
        )
        .unwrap();
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 1,
                samples: FRAME_SAMPLES,
            },
            &vec![1.0; FRAME_SAMPLES],
        )
        .unwrap();
        protocol::write_request(&mut input, &Request::Finish { id: 2 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Finish { id: 1 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Cancel { id: 1 }, &[]).unwrap();
        protocol::write_request(
            &mut input,
            &Request::Start {
                id: 3,
                prompt: "cancel".into(),
            },
            &[],
        )
        .unwrap();
        protocol::write_request(&mut input, &Request::Cancel { id: 3 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Shutdown, &[]).unwrap();

        let mut output = Vec::new();
        worker_loop(&mut provider, &mut input.as_slice(), &mut output).unwrap();
        let mut responses = output.as_slice();
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Error { id: Some(1), .. }
        ));
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Ack { id: 1 }
        ));
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Error { id: Some(2), .. }
        ));
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Error { id: Some(1), .. }
        ));
        assert!(
            matches!(protocol::read_response(&mut responses).unwrap(), Response::Result { id: 1, transcripts } if transcripts.is_empty())
        );
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Error { id: Some(2), .. }
        ));
        assert!(
            matches!(protocol::read_response(&mut responses).unwrap(), Response::Result { id: 1, transcripts } if transcripts[0].text == "hey computer")
        );
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Error { id: Some(1), .. }
        ));
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Ack { id: 3 }
        ));
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Ack { id: 3 }
        ));
        assert!(responses.is_empty());
        fs::remove_dir_all(root).unwrap();
    }
}
