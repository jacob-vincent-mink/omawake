mod protocol;
mod ring;

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use self::protocol::{Request, Response, Transcript};
use self::ring::{EndpointBuffer, FRAME_SAMPLES, Utterance};
use super::onnx::resample::AudioResampler;
use super::{Detection, WakeWordBackend, WakeWordStream, detect_samples};
use crate::backend::Runtime;
use crate::config::Config;
use crate::paths::AppPaths;
use crate::phrase::{PhraseMatcher, normalize_tokens};

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
            .map(|word| word.phrase.trim())
            .collect::<Vec<_>>()
            .join(", ");
        let library_dirs = resolve_library_dirs(config, paths, &library)?;
        let worker = Worker::spawn(
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
        let (sample_rate, samples) = super::onnx::read_wave(path)?;
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
    if !config.backend.library.as_os_str().is_empty() {
        return resolve_file_beneath(
            &config.backend.library,
            paths.config_file.parent().unwrap_or(Path::new(".")),
            "configured whisper.cpp library",
        );
    }
    if let Some(path) = env::var_os("OMAWAKE_WHISPER_LIBRARY").map(PathBuf::from) {
        if !path.is_absolute() {
            bail!("OMAWAKE_WHISPER_LIBRARY must be an absolute file path");
        }
        return resolve_file_beneath(&path, Path::new("/"), "OMAWAKE_WHISPER_LIBRARY");
    }
    let package_dirs = package_library_dirs();
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

#[derive(Clone)]
struct WorkerSpec {
    library: PathBuf,
    library_dirs: Vec<PathBuf>,
    verifier: PathBuf,
    vad: PathBuf,
    threads: i32,
}

struct WorkerProcess {
    child: Child,
    stream: UnixStream,
}

impl WorkerProcess {
    fn launch(spec: &WorkerSpec) -> Result<Self> {
        let (stream, child_stream) = UnixStream::pair().context("create whisper.cpp worker IPC")?;
        stream.set_read_timeout(Some(WORKER_STARTUP_TIMEOUT))?;
        stream.set_write_timeout(Some(WORKER_STARTUP_TIMEOUT))?;
        let child_input: OwnedFd = child_stream
            .try_clone()
            .context("clone whisper.cpp worker IPC")?
            .into();
        let child_output: OwnedFd = child_stream.into();
        let executable = env::current_exe().context("resolve Omawake executable")?;
        let mut command = Command::new(executable);
        command
            .arg("__whisper-worker")
            .arg(&spec.library)
            .arg(&spec.verifier)
            .arg(&spec.vad)
            .arg(spec.threads.to_string())
            .stdin(Stdio::from(child_input))
            .stdout(Stdio::from(child_output))
            .stderr(Stdio::null());
        configure_library_path(&mut command, &spec.library_dirs)?;
        let child = command
            .spawn()
            .context("spawn isolated whisper.cpp worker")?;
        let mut process = Self { child, stream };
        let handshake =
            protocol::read_response(&mut process.stream).context("read worker handshake");
        match handshake {
            Ok(Response::Ready { .. }) => {
                if let Err(error) = process
                    .stream
                    .set_read_timeout(Some(WORKER_REQUEST_TIMEOUT))
                    .and_then(|()| {
                        process
                            .stream
                            .set_write_timeout(Some(WORKER_REQUEST_TIMEOUT))
                    })
                {
                    process.stop(false);
                    return Err(error).context("configure whisper.cpp worker IPC timeout");
                }
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

    fn stop(&mut self, graceful: bool) {
        if graceful {
            let _ = self.stream.set_write_timeout(Some(Duration::from_secs(1)));
            let _ = protocol::write_request(&mut self.stream, &Request::Shutdown, &[]);
        }
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
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
    fn spawn(
        library: PathBuf,
        library_dirs: Vec<PathBuf>,
        verifier: PathBuf,
        vad: PathBuf,
        threads: i32,
    ) -> Result<Self> {
        let spec = WorkerSpec {
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
            protocol::write_request(&mut process.stream, request, pcm)
                .context("write worker request")?;
            protocol::read_response(&mut process.stream).context("read worker response")
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
        let mut error = vec![0_i8; 2048];
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
        let mut error = vec![0_i8; 2048];
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
        let mut text = vec![0_i8; 64 * 1024];
        let mut error = vec![0_i8; 2048];
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

fn c_buffer(buffer: &[i8]) -> String {
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
    let mut stream: Option<WorkerStream> = None;
    loop {
        let (request, pcm) = protocol::read_request(&mut input)?;
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
                            .map(|utterance| active.finish_utterance(&mut provider, utterance))
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
                        .map(|utterance| active.finish_utterance(&mut provider, utterance))
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
        protocol::write_response(&mut output, &response)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WakeWord;
    use std::fs;

    #[test]
    fn transcript_conversion_uses_whole_phrase_matcher() {
        let matcher = PhraseMatcher::compile(&[WakeWord {
            id: "computer".into(),
            phrase: "hey computer".into(),
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

        config.backend.library = PathBuf::from("../outside.so");
        assert!(resolve_library(&config, &paths).is_err());
        assert!(resolve_model_asset(&model_dir, "../outside.bin", "verifier").is_err());
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
}
