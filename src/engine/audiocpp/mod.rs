mod protocol;
mod ring;

use std::cell::{Cell, RefCell};
use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::io::{self, BufReader, Read};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use libloading::Library;
use sha2::{Digest, Sha256};

use self::protocol::{FRAME_SAMPLES, Request, Response, Transcript};
use self::ring::{Activity, ActivityBuffer, Utterance};
use super::onnx::resample::AudioResampler;
use super::{Detection, WakeWordBackend, WakeWordStream, detect_samples};
use crate::backend::Runtime;
use crate::config::Config;
use crate::paths::AppPaths;
use crate::phrase::{PhraseMatcher, normalize_tokens};

const AUDIOCPP_ABI_0_1_0: u32 = 1 << 8;
const WORKER_IO_TIMEOUT: Duration = Duration::from_secs(30);
// Converted GGUF: audio-cpp/audio.cpp-gguf@6d5436fc85f7a20c2e9f4e472b7f3a532f686444.
// Original MIT model: moonshine-ai/moonshine-streaming-tiny@f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a.
const MOONSHINE_TINY_BYTES: u64 = 60_407_904;
const MOONSHINE_TINY_SHA256: &str =
    "e9a342a07327f4e1745874f137f45350697e91699a91e4eb2ac60c223718f8c3";
const SILERO_VAD_BYTES: u64 = 1_239_748;
// Author-published MIT artifact: snakers4/silero-vad@7e30209a3e901f9842f81b225f3e93d8199902b1.
const SILERO_VAD_SHA256: &str = "c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1";

struct AsrProfile {
    family: &'static str,
    file_name: &'static str,
    bytes: u64,
    sha256: &'static str,
    label: &'static str,
}

const MOONSHINE_TINY: AsrProfile = AsrProfile {
    family: "moonshine_asr",
    file_name: "moonshine-streaming-tiny-q8_0.gguf",
    bytes: MOONSHINE_TINY_BYTES,
    sha256: MOONSHINE_TINY_SHA256,
    label: "Moonshine Streaming Tiny Q8_0",
};

pub(super) struct AudioCppBackend {
    worker: RefCell<Worker>,
    matcher: PhraseMatcher,
    next_stream_id: Cell<u64>,
}

struct AudioCppStream<'a> {
    backend: &'a AudioCppBackend,
    state: RefCell<ClientStream>,
}

struct ClientStream {
    id: u64,
    started: bool,
    finished: bool,
    resampler: AudioResampler,
    pending: Vec<f32>,
}

impl AudioCppBackend {
    pub(super) fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        if config.backend.runtime != Runtime::Default {
            bail!("audiocpp currently supports only backend.runtime = default");
        }
        if !matches!(
            config.backend.device.trim().to_ascii_lowercase().as_str(),
            "auto" | "cpu"
        ) {
            bail!("audiocpp CPU checkpoint requires backend.device = auto or cpu");
        }
        if config.model.sample_rate != 16_000 {
            bail!("audio.cpp Silero and the selected ASR family require model.sample_rate = 16000");
        }
        if !(1..=64).contains(&config.backend.threads) {
            bail!("backend threads must be between 1 and 64");
        }
        let directory = config.model_directory(paths);
        let configured_library_dirs = resolve_configured_library_dirs(config, paths)?;
        let library = resolve_library(config, paths, &configured_library_dirs)?;
        let library_dirs = worker_library_dirs(&library, configured_library_dirs)?;
        let profile = asr_profile(config)?;
        let verifier = resolve_model_asset(&directory, &config.model.verifier, "ASR verifier")?;
        let vad = resolve_model_asset(&directory, &config.model.vad, "Silero VAD")?;
        require_exact_asset_name(&verifier, profile.file_name, profile.label)?;
        require_exact_asset_name(&vad, "silero_vad_16k.safetensors", "Silero VAD")?;
        verify_asset(&verifier, profile.bytes, profile.sha256, profile.label)?;
        verify_asset(&vad, SILERO_VAD_BYTES, SILERO_VAD_SHA256, "Silero VAD")?;
        let matcher = PhraseMatcher::compile(&config.wake_words)?;
        let worker = Worker::spawn(
            &library,
            &verifier,
            &vad,
            i32::from(config.backend.threads),
            &paths.cache_dir.join("native-tmp"),
            profile.family,
            &library_dirs,
        )?;
        Ok(Self {
            worker: RefCell::new(worker),
            matcher,
            next_stream_id: Cell::new(1),
        })
    }

    fn detections(&self, transcripts: Vec<Transcript>) -> Vec<Detection> {
        transcripts_to_detections(&self.matcher, transcripts)
    }
}

fn asr_profile(config: &Config) -> Result<&'static AsrProfile> {
    let family = config
        .backend
        .options
        .get("audiocpp.asr_family")
        .map(String::as_str)
        .unwrap_or(MOONSHINE_TINY.family);
    match family {
        "moonshine_asr" => Ok(&MOONSHINE_TINY),
        _ => bail!(
            "audio.cpp ASR family {family:?} is not qualified; currently supported: moonshine_asr"
        ),
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

impl WakeWordBackend for AudioCppBackend {
    fn kind(&self) -> &'static str {
        "audiocpp"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        let id = self.next_stream_id.get();
        self.next_stream_id.set(id.wrapping_add(1).max(1));
        Box::new(AudioCppStream {
            backend: self,
            state: RefCell::new(ClientStream {
                id,
                started: false,
                finished: false,
                resampler: AudioResampler::new(),
                pending: Vec::new(),
            }),
        })
    }

    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (sample_rate, samples) = super::onnx::read_wave(path)?;
        let stream = self.stream();
        detect_samples(stream.as_ref(), sample_rate, &samples)
    }
}

impl AudioCppStream<'_> {
    fn ensure_started(&self, state: &mut ClientStream) -> Result<()> {
        if !state.started {
            self.backend
                .worker
                .try_borrow_mut()
                .map_err(|_| anyhow::anyhow!("another audio.cpp stream is using the worker"))?
                .start(state.id)?;
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
                    .map_err(|_| anyhow::anyhow!("another audio.cpp stream is using the worker"))?
                    .audio(state.id, &frame)?,
            );
        }
        Ok(self.backend.detections(transcripts))
    }
}

impl WakeWordStream for AudioCppStream<'_> {
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
            .map_err(|_| anyhow::anyhow!("another audio.cpp stream is using the worker"))?
            .finish(state.id)?;
        detections.extend(self.backend.detections(transcripts));
        state.finished = true;
        Ok(detections)
    }
}

impl Drop for AudioCppStream<'_> {
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
        bail!("{label} must name a model file");
    }
    let configured = Path::new(configured);
    let path = if configured.is_absolute() {
        configured.to_owned()
    } else {
        directory.join(configured)
    };
    if !path.is_file() {
        bail!("audio.cpp {label} model is missing: {}", path.display());
    }
    Ok(path)
}

fn require_exact_asset_name(path: &Path, expected: &str, label: &str) -> Result<()> {
    if path.file_name().and_then(|name| name.to_str()) != Some(expected) {
        bail!(
            "audio.cpp {label} must be the pinned {expected} asset: {}",
            path.display()
        );
    }
    Ok(())
}

fn verify_asset(path: &Path, expected_size: u64, expected_sha256: &str, label: &str) -> Result<()> {
    let actual_size = fs::metadata(path)
        .with_context(|| format!("inspect audio.cpp {label} {}", path.display()))?
        .len();
    if actual_size != expected_size {
        bail!(
            "audio.cpp {label} has {actual_size} bytes, expected pinned size {expected_size}: {}",
            path.display()
        );
    }
    let mut input = fs::File::open(path)
        .with_context(|| format!("open audio.cpp {label} {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .with_context(|| format!("hash audio.cpp {label} {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual_sha256 = format!("{:x}", hasher.finalize());
    if actual_sha256 != expected_sha256 {
        bail!(
            "audio.cpp {label} checksum mismatch: expected {expected_sha256}, found {actual_sha256}: {}",
            path.display()
        );
    }
    Ok(())
}

const AUDIOCPP_LIBRARY_NAMES: &[&str] =
    &["libaudiocpp.so.0.1.0", "libaudiocpp.so.0", "libaudiocpp.so"];

fn resolve_library(
    config: &Config,
    paths: &AppPaths,
    configured_dirs: &[PathBuf],
) -> Result<PathBuf> {
    let executable = env::current_exe().context("resolve Omawake executable")?;
    let environment = env::var_os("OMAWAKE_AUDIOCPP_LIBRARY").map(PathBuf::from);
    resolve_library_with(
        config,
        paths,
        configured_dirs,
        environment.as_deref(),
        &executable,
    )
}

fn resolve_library_with(
    config: &Config,
    paths: &AppPaths,
    configured_dirs: &[PathBuf],
    environment: Option<&Path>,
    executable: &Path,
) -> Result<PathBuf> {
    let config_directory = paths.config_file.parent().unwrap_or(Path::new("."));
    if !config.backend.library.as_os_str().is_empty() {
        return resolve_file_beneath(
            &config.backend.library,
            config_directory,
            "configured audio.cpp library",
        );
    }
    if let Some(path) = environment {
        if !path.is_absolute() {
            bail!("OMAWAKE_AUDIOCPP_LIBRARY must be an absolute file path");
        }
        return resolve_file_beneath(path, Path::new("/"), "OMAWAKE_AUDIOCPP_LIBRARY");
    }

    let package_dirs = package_library_dirs(executable);
    for directory in configured_dirs
        .iter()
        .map(PathBuf::as_path)
        .chain(package_dirs.iter().map(PathBuf::as_path))
        .chain([
            Path::new("/usr/lib"),
            Path::new("/usr/local/lib"),
            Path::new("/usr/lib64"),
        ])
    {
        if let Some(path) = library_in_directory(directory) {
            return path
                .canonicalize()
                .with_context(|| format!("resolve audio.cpp library {}", path.display()));
        }
    }
    bail!(
        "libaudiocpp ABI 0.1 was not found; set backend.library, \
         OMAWAKE_AUDIOCPP_LIBRARY, or backend.library_dirs"
    )
}

fn resolve_configured_library_dirs(config: &Config, paths: &AppPaths) -> Result<Vec<PathBuf>> {
    let base = paths.config_file.parent().unwrap_or(Path::new("."));
    let mut directories = Vec::with_capacity(config.backend.library_dirs.len());
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
    Ok(directories)
}

fn worker_library_dirs(library: &Path, mut configured: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    let parent = library
        .parent()
        .context("audio.cpp library has no parent directory")?
        .to_path_buf();
    if !configured.contains(&parent) {
        configured.insert(0, parent);
    }
    Ok(configured)
}

fn package_library_dirs(executable: &Path) -> Vec<PathBuf> {
    let Some(binary_dir) = executable.parent() else {
        return Vec::new();
    };
    let mut candidates = vec![binary_dir.join("lib")];
    if let Some(prefix) = binary_dir.parent() {
        candidates.push(prefix.join("lib/omawake"));
        candidates.push(prefix.join("lib"));
    }
    candidates.push(binary_dir.to_owned());
    candidates
}

fn library_in_directory(directory: &Path) -> Option<PathBuf> {
    AUDIOCPP_LIBRARY_NAMES
        .iter()
        .map(|name| directory.join(name))
        .find(|path| path.is_file())
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
        bail!("{label} is not a file: {}", resolved.display());
    }
    if !explicit_absolute {
        let canonical_base = base
            .canonicalize()
            .with_context(|| format!("resolve config directory {}", base.display()))?;
        if !resolved.starts_with(&canonical_base) {
            bail!(
                "{label} escapes config directory {}",
                canonical_base.display()
            );
        }
    }
    Ok(resolved)
}

struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
    active_id: Option<u64>,
    spec: WorkerSpec,
}

#[derive(Clone)]
struct WorkerSpec {
    executable: PathBuf,
    library: PathBuf,
    verifier: PathBuf,
    vad: PathBuf,
    cache: PathBuf,
    threads: i32,
    asr_family: String,
    library_dirs: Vec<PathBuf>,
}

impl Worker {
    fn spawn(
        library: &Path,
        verifier: &Path,
        vad: &Path,
        threads: i32,
        cache: &Path,
        asr_family: &str,
        library_dirs: &[PathBuf],
    ) -> Result<Self> {
        fs::create_dir_all(cache)
            .with_context(|| format!("create audio.cpp cache {}", cache.display()))?;
        fs::set_permissions(cache, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("secure audio.cpp cache {}", cache.display()))?;
        let spec = WorkerSpec {
            executable: env::current_exe().context("resolve Omawake executable")?,
            library: library.to_owned(),
            verifier: verifier.to_owned(),
            vad: vad.to_owned(),
            cache: cache.to_owned(),
            threads,
            asr_family: asr_family.to_owned(),
            library_dirs: library_dirs.to_owned(),
        };
        let (child, input, output) = launch_worker(&spec)?;
        let mut worker = Self {
            child,
            input: Some(input),
            output,
            active_id: None,
            spec,
        };
        match worker.read_response("worker handshake") {
            Ok(Response::Ready { .. }) => Ok(worker),
            Ok(Response::Error { message, .. }) => {
                let _ = worker.terminate();
                bail!(message)
            }
            Ok(response) => {
                let _ = worker.terminate();
                bail!("unexpected worker handshake: {response:?}")
            }
            Err(error) => {
                let _ = worker.terminate();
                Err(error)
            }
        }
    }

    fn read_response(&mut self, operation: &str) -> Result<Response> {
        wait_readable(self.output.get_ref(), WORKER_IO_TIMEOUT)
            .with_context(|| format!("wait for audio.cpp {operation}"))?;
        protocol::read_response(&mut self.output)
            .with_context(|| format!("read audio.cpp {operation}"))
    }

    fn exchange(&mut self, request: &Request, pcm: &[f32]) -> Result<Response> {
        let result = (|| {
            let input = self.input.as_mut().context("worker stdin is closed")?;
            protocol::write_request(input, request, pcm).context("write worker request")?;
            self.read_response("worker response")
        })();
        match result {
            Ok(response) => Ok(response),
            Err(error) => {
                self.active_id = None;
                self.terminate().with_context(|| {
                    format!("audio.cpp IPC failed ({error:#}) and the worker was not reaped")
                })?;
                Err(error)
            }
        }
    }

    fn relaunch(&mut self) -> Result<()> {
        self.terminate()?;
        let (child, input, output) = launch_worker(&self.spec)?;
        self.child = child;
        self.input = Some(input);
        self.output = output;
        self.active_id = None;
        match self.read_response("restarted worker handshake") {
            Ok(Response::Ready { .. }) => Ok(()),
            Ok(Response::Error { message, .. }) => {
                let _ = self.terminate();
                bail!(message)
            }
            Ok(response) => {
                let _ = self.terminate();
                bail!("unexpected restarted worker handshake: {response:?}")
            }
            Err(error) => {
                let _ = self.terminate();
                Err(error)
            }
        }
    }

    fn start(&mut self, id: u64) -> Result<()> {
        if let Some(active) = self.active_id {
            bail!("audio.cpp worker is already serving stream {active}");
        }
        let request = Request::Start { id };
        let response = match self.exchange(&request, &[]) {
            Ok(response) => response,
            Err(first_error) => {
                self.relaunch().with_context(|| {
                    format!("audio.cpp worker failed ({first_error:#}); its one restart failed")
                })?;
                self.exchange(&request, &[]).with_context(|| {
                    format!(
                        "audio.cpp worker failed ({first_error:#}); restarted worker also failed"
                    )
                })?
            }
        };
        match response {
            Response::Ack { id: response } if response == id => {
                self.active_id = Some(id);
                Ok(())
            }
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected start response: {response:?}"),
        }
    }

    fn audio(&mut self, id: u64, pcm: &[f32]) -> Result<Vec<Transcript>> {
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
            response => bail!("unexpected audio response: {response:?}"),
        }
    }

    fn finish(&mut self, id: u64) -> Result<Vec<Transcript>> {
        let response = self.exchange(&Request::Finish { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Result {
                id: response,
                transcripts,
            } if response == id => Ok(transcripts),
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected finish response: {response:?}"),
        }
    }

    fn cancel(&mut self, id: u64) -> Result<()> {
        let response = self.exchange(&Request::Cancel { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Ack { id: response } if response == id => Ok(()),
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected cancel response: {response:?}"),
        }
    }

    fn terminate(&mut self) -> Result<()> {
        self.input.take();
        if self.child.try_wait()?.is_some() {
            return Ok(());
        }
        self.child.kill().context("terminate audio.cpp worker")?;
        for _ in 0..100 {
            if self.child.try_wait()?.is_some() {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(10));
        }
        bail!("audio.cpp worker did not exit within one second after termination")
    }

    fn shutdown(&mut self) {
        if let Some(input) = self.input.as_mut() {
            let _ = protocol::write_request(input, &Request::Shutdown, &[]);
        }
        self.input.take();
        for _ in 0..100 {
            if self.child.try_wait().is_ok_and(|status| status.is_some()) {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let _ = self.terminate();
    }
}

fn launch_worker(spec: &WorkerSpec) -> Result<(Child, ChildStdin, BufReader<ChildStdout>)> {
    let mut command = Command::new(&spec.executable);
    command
        .arg("__audiocpp-worker")
        .arg(&spec.library)
        .arg(&spec.verifier)
        .arg(&spec.vad)
        .arg(spec.threads.to_string())
        .arg(&spec.asr_family)
        .env("TMPDIR", &spec.cache)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    prepend_library_directories(&mut command, &spec.library_dirs)?;
    let mut child = command.spawn().context("spawn isolated audio.cpp worker")?;
    let input = child.stdin.take().context("worker stdin is unavailable")?;
    let output = child
        .stdout
        .take()
        .context("worker stdout is unavailable")?;
    Ok((child, input, BufReader::new(output)))
}

fn wait_readable(fd: &impl AsRawFd, timeout: Duration) -> io::Result<()> {
    let mut descriptor = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let timeout_ms = i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX);
    loop {
        let status = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
        if status > 0 {
            return Ok(());
        }
        if status == 0 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "worker response timed out after {} seconds",
                    timeout.as_secs()
                ),
            ));
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn prepend_library_directories(command: &mut Command, library_dirs: &[PathBuf]) -> Result<()> {
    let mut paths = Vec::new();
    for directory in library_dirs {
        if !paths.contains(directory) {
            paths.push(directory.to_owned());
        }
    }
    if let Some(existing) = env::var_os("LD_LIBRARY_PATH") {
        for directory in env::split_paths(&existing) {
            if !paths.contains(&directory) {
                paths.push(directory);
            }
        }
    }
    if !paths.is_empty() {
        let joined = env::join_paths(paths).context("construct audio.cpp worker library path")?;
        command.env("LD_LIBRARY_PATH", joined);
    }
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

type Status = c_int;

#[repr(C)]
struct ModelConfig {
    family_hint: *const c_char,
    config_id: *const c_char,
    weight_id: *const c_char,
    model_spec_override: *const c_char,
}

#[repr(C)]
struct BackendConfig {
    backend: *const c_char,
    device: c_int,
    threads: c_int,
}

struct AudioCppApi {
    _library: Library,
    build_version: unsafe extern "C" fn() -> *const c_char,
    last_error: unsafe extern "C" fn() -> *const c_char,
    registry_create: unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> Status,
    registry_free: unsafe extern "C" fn(*mut c_void),
    model_load: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *const ModelConfig,
        *const c_void,
        *mut *mut c_void,
    ) -> Status,
    model_free: unsafe extern "C" fn(*mut c_void),
    session_create: unsafe extern "C" fn(
        *const c_void,
        *const c_char,
        *const c_char,
        *const BackendConfig,
        *const c_void,
        *mut *mut c_void,
    ) -> Status,
    session_free: unsafe extern "C" fn(*mut c_void),
    request_create: unsafe extern "C" fn() -> *mut c_void,
    request_free: unsafe extern "C" fn(*mut c_void),
    request_set_audio: unsafe extern "C" fn(*mut c_void, *const f32, usize, c_int, c_int) -> Status,
    session_run: unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> Status,
    result_text:
        unsafe extern "C" fn(*const c_void, *mut *const c_char, *mut *const c_char) -> Status,
    result_segment_count: unsafe extern "C" fn(*const c_void) -> usize,
    result_segment: unsafe extern "C" fn(
        *const c_void,
        usize,
        *mut i64,
        *mut i64,
        *mut f32,
        *mut *const c_char,
    ) -> Status,
    result_free: unsafe extern "C" fn(*mut c_void),
    stream_start: unsafe extern "C" fn(*mut c_void, *const c_void) -> Status,
    stream_push: unsafe extern "C" fn(
        *mut c_void,
        *const f32,
        usize,
        c_int,
        c_int,
        i64,
        *mut *mut c_void,
    ) -> Status,
    stream_reset: unsafe extern "C" fn(*mut c_void) -> Status,
    event_free: unsafe extern "C" fn(*mut c_void),
    event_as_result: unsafe extern "C" fn(*const c_void) -> *const c_void,
    event_voice_activity_count: unsafe extern "C" fn(*const c_void) -> usize,
    event_voice_activity:
        unsafe extern "C" fn(*const c_void, usize, *mut c_int, *mut i64, *mut f32) -> Status,
}

impl AudioCppApi {
    fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load audio.cpp C ABI {}", path.display()))?;
        let abi_version: unsafe extern "C" fn() -> u32 =
            unsafe { symbol(&library, b"audiocpp_abi_version\0")? };
        let found = unsafe { abi_version() };
        if found != AUDIOCPP_ABI_0_1_0 {
            bail!(
                "audio.cpp C ABI mismatch: expected 0.1.0 ({AUDIOCPP_ABI_0_1_0}), found packed version {found}"
            );
        }
        Ok(Self {
            build_version: unsafe { symbol(&library, b"audiocpp_build_version\0")? },
            last_error: unsafe { symbol(&library, b"audiocpp_last_error\0")? },
            registry_create: unsafe { symbol(&library, b"audiocpp_registry_create\0")? },
            registry_free: unsafe { symbol(&library, b"audiocpp_registry_free\0")? },
            model_load: unsafe { symbol(&library, b"audiocpp_model_load\0")? },
            model_free: unsafe { symbol(&library, b"audiocpp_model_free\0")? },
            session_create: unsafe { symbol(&library, b"audiocpp_session_create\0")? },
            session_free: unsafe { symbol(&library, b"audiocpp_session_free\0")? },
            request_create: unsafe { symbol(&library, b"audiocpp_request_create\0")? },
            request_free: unsafe { symbol(&library, b"audiocpp_request_free\0")? },
            request_set_audio: unsafe { symbol(&library, b"audiocpp_request_set_audio\0")? },
            session_run: unsafe { symbol(&library, b"audiocpp_session_run\0")? },
            result_text: unsafe { symbol(&library, b"audiocpp_result_text\0")? },
            result_segment_count: unsafe { symbol(&library, b"audiocpp_result_segment_count\0")? },
            result_segment: unsafe { symbol(&library, b"audiocpp_result_segment\0")? },
            result_free: unsafe { symbol(&library, b"audiocpp_result_free\0")? },
            stream_start: unsafe { symbol(&library, b"audiocpp_stream_start\0")? },
            stream_push: unsafe { symbol(&library, b"audiocpp_stream_push\0")? },
            stream_reset: unsafe { symbol(&library, b"audiocpp_stream_reset\0")? },
            event_free: unsafe { symbol(&library, b"audiocpp_event_free\0")? },
            event_as_result: unsafe { symbol(&library, b"audiocpp_event_as_result\0")? },
            event_voice_activity_count: unsafe {
                symbol(&library, b"audiocpp_event_voice_activity_count\0")?
            },
            event_voice_activity: unsafe { symbol(&library, b"audiocpp_event_voice_activity\0")? },
            _library: library,
        })
    }

    fn check(&self, operation: &str, status: Status) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let detail = unsafe {
            let value = (self.last_error)();
            if value.is_null() {
                "unknown audio.cpp error".into()
            } else {
                CStr::from_ptr(value).to_string_lossy().into_owned()
            }
        };
        bail!("{operation}: {detail} (status {status})")
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    Ok(*unsafe { library.get::<T>(name) }.with_context(|| {
        format!(
            "resolve native symbol {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        )
    })?)
}

struct AudioCppProvider {
    api: AudioCppApi,
    registry: *mut c_void,
    vad_model: *mut c_void,
    asr_model: *mut c_void,
    vad_session: *mut c_void,
    asr_session: *mut c_void,
    vad_cursor: i64,
}

impl AudioCppProvider {
    fn open(
        library: &Path,
        verifier: &Path,
        vad: &Path,
        threads: i32,
        asr_family: &str,
    ) -> Result<Self> {
        let api = AudioCppApi::load(library)?;
        let mut provider = Self {
            api,
            registry: std::ptr::null_mut(),
            vad_model: std::ptr::null_mut(),
            asr_model: std::ptr::null_mut(),
            vad_session: std::ptr::null_mut(),
            asr_session: std::ptr::null_mut(),
            vad_cursor: 0,
        };
        let status =
            unsafe { (provider.api.registry_create)(std::ptr::null(), &mut provider.registry) };
        provider.api.check("create audio.cpp registry", status)?;
        provider.vad_model = provider.load_model(vad, "silero_vad")?;
        provider.asr_model = provider.load_model(verifier, asr_family)?;
        provider.vad_session =
            provider.create_session(provider.vad_model, "vad", "streaming", threads)?;
        provider.asr_session =
            provider.create_session(provider.asr_model, "asr", "offline", threads)?;
        provider.start_vad()?;
        Ok(provider)
    }

    fn load_model(&self, path: &Path, family: &str) -> Result<*mut c_void> {
        let path = path_to_c_string(path)?;
        let family = CString::new(family)?;
        let config = ModelConfig {
            family_hint: family.as_ptr(),
            config_id: std::ptr::null(),
            weight_id: std::ptr::null(),
            model_spec_override: std::ptr::null(),
        };
        let mut model = std::ptr::null_mut();
        let status = unsafe {
            (self.api.model_load)(
                self.registry,
                path.as_ptr(),
                &config,
                std::ptr::null(),
                &mut model,
            )
        };
        self.api
            .check(&format!("load audio.cpp {family:?} model"), status)?;
        if model.is_null() {
            bail!("audio.cpp returned a null {family:?} model");
        }
        Ok(model)
    }

    fn create_session(
        &self,
        model: *mut c_void,
        task: &str,
        mode: &str,
        threads: i32,
    ) -> Result<*mut c_void> {
        let task = CString::new(task)?;
        let mode = CString::new(mode)?;
        let cpu = CString::new("cpu")?;
        let backend = BackendConfig {
            backend: cpu.as_ptr(),
            device: 0,
            threads,
        };
        let mut session = std::ptr::null_mut();
        let status = unsafe {
            (self.api.session_create)(
                model,
                task.as_ptr(),
                mode.as_ptr(),
                &backend,
                std::ptr::null(),
                &mut session,
            )
        };
        self.api.check("create audio.cpp CPU session", status)?;
        if session.is_null() {
            bail!("audio.cpp returned a null session");
        }
        Ok(session)
    }

    fn version(&self) -> String {
        let version = unsafe { (self.api.build_version)() };
        if version.is_null() {
            "audio.cpp ABI 0.1".into()
        } else {
            format!(
                "audio.cpp {} (ABI 0.1.0, cpu)",
                unsafe { CStr::from_ptr(version) }.to_string_lossy()
            )
        }
    }

    fn start_vad(&mut self) -> Result<()> {
        self.vad_cursor = 0;
        let status = unsafe { (self.api.stream_start)(self.vad_session, std::ptr::null()) };
        self.api.check("start persistent Silero stream", status)
    }

    fn restart_vad(&mut self) -> Result<()> {
        let status = unsafe { (self.api.stream_reset)(self.vad_session) };
        self.api.check("reset persistent Silero stream", status)?;
        self.start_vad()
    }

    fn vad_activity(&mut self, samples: &[f32]) -> Result<Activity> {
        if samples.len() != FRAME_SAMPLES {
            bail!("audio.cpp Silero requires exactly {FRAME_SAMPLES} samples per frame");
        }
        let frame_end = self
            .vad_cursor
            .checked_add(FRAME_SAMPLES as i64)
            .context("Silero sample cursor overflow")?;
        let mut event = std::ptr::null_mut();
        let status = unsafe {
            (self.api.stream_push)(
                self.vad_session,
                samples.as_ptr(),
                samples.len(),
                16_000,
                1,
                self.vad_cursor,
                &mut event,
            )
        };
        self.api.check("run audio.cpp Silero frame", status)?;
        self.vad_cursor = frame_end;
        if event.is_null() {
            return Ok(Activity::default());
        }
        let result = (|| {
            let mut activity = Activity::default();
            let count = unsafe { (self.api.event_voice_activity_count)(event) };
            for index in 0..count {
                let mut kind = -1;
                let mut sample = 0_i64;
                let mut probability = 0.0_f32;
                let status = unsafe {
                    (self.api.event_voice_activity)(
                        event,
                        index,
                        &mut kind,
                        &mut sample,
                        &mut probability,
                    )
                };
                self.api.check("read audio.cpp Silero event", status)?;
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) || sample < 0 {
                    bail!("audio.cpp Silero returned an invalid activity event");
                }
                let before_frame_end = |value: i64| -> Result<usize> {
                    if value < 0 || value > frame_end {
                        bail!(
                            "audio.cpp Silero event timestamp {value} is outside the processed stream"
                        );
                    }
                    usize::try_from(frame_end - value)
                        .context("Silero activity offset is too large")
                };
                match kind {
                    0 => activity.start_before_frame_end = Some(before_frame_end(sample)?),
                    1 => activity.end_before_frame_end = Some(before_frame_end(sample)?),
                    2 => {
                        let view = unsafe { (self.api.event_as_result)(event) };
                        if view.is_null() || unsafe { (self.api.result_segment_count)(view) } == 0 {
                            bail!("audio.cpp Silero segment event omitted its segment bounds");
                        }
                        let mut start = 0_i64;
                        let mut end = 0_i64;
                        let status = unsafe {
                            (self.api.result_segment)(
                                view,
                                0,
                                &mut start,
                                &mut end,
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                            )
                        };
                        self.api
                            .check("read audio.cpp Silero segment bounds", status)?;
                        activity.start_before_frame_end = Some(before_frame_end(start)?);
                        activity.end_before_frame_end = Some(before_frame_end(end)?);
                    }
                    _ => bail!("audio.cpp Silero returned unknown activity kind {kind}"),
                }
            }
            Ok(activity)
        })();
        unsafe { (self.api.event_free)(event) };
        result
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let request = unsafe { (self.api.request_create)() };
        if request.is_null() {
            bail!("audio.cpp could not allocate an ASR request");
        }
        let mut result_handle = std::ptr::null_mut();
        let result = (|| {
            let status = unsafe {
                (self.api.request_set_audio)(request, samples.as_ptr(), samples.len(), 16_000, 1)
            };
            self.api.check("set audio.cpp ASR audio", status)?;
            let status =
                unsafe { (self.api.session_run)(self.asr_session, request, &mut result_handle) };
            self.api
                .check("run persistent audio.cpp ASR session", status)?;
            if result_handle.is_null() {
                bail!("audio.cpp returned a null ASR result");
            }
            let mut text = std::ptr::null();
            let status =
                unsafe { (self.api.result_text)(result_handle, &mut text, std::ptr::null_mut()) };
            self.api.check("read audio.cpp ASR transcript", status)?;
            if text.is_null() {
                bail!("audio.cpp returned a null ASR transcript");
            }
            Ok(unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .trim()
                .to_owned())
        })();
        if !result_handle.is_null() {
            unsafe { (self.api.result_free)(result_handle) };
        }
        unsafe { (self.api.request_free)(request) };
        result
    }
}

impl Drop for AudioCppProvider {
    fn drop(&mut self) {
        unsafe {
            (self.api.session_free)(self.asr_session);
            (self.api.session_free)(self.vad_session);
            (self.api.model_free)(self.asr_model);
            (self.api.model_free)(self.vad_model);
            (self.api.registry_free)(self.registry);
        }
    }
}

fn path_to_c_string(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_encoded_bytes()).context("native path contains NUL")
}

struct WorkerStream {
    id: u64,
    endpoint: ActivityBuffer,
}

impl WorkerStream {
    fn finish_utterance(
        &self,
        provider: &mut AudioCppProvider,
        utterance: Utterance,
    ) -> Result<Transcript> {
        let text = provider.transcribe(&utterance.samples)?;
        Ok(Transcript {
            text,
            start_sample: utterance.start_sample,
            end_sample: utterance.end_sample,
        })
    }
}

pub(crate) fn worker_main(
    library: &Path,
    verifier: &Path,
    vad: &Path,
    threads: i32,
    asr_family: &str,
) -> Result<()> {
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
    let mut provider = match AudioCppProvider::open(library, verifier, vad, threads, asr_family) {
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
            Request::Start { id } => match provider.restart_vad() {
                Ok(()) => {
                    stream = Some(WorkerStream {
                        id,
                        endpoint: ActivityBuffer::new(),
                    });
                    Response::Ack { id }
                }
                Err(error) => Response::Error {
                    id: Some(id),
                    message: format!("{error:#}"),
                },
            },
            Request::Audio { id, .. } => match stream.as_mut() {
                Some(active) if active.id == id && pcm.len() == FRAME_SAMPLES => {
                    let result = (|| {
                        if pcm.iter().any(|sample| !sample.is_finite()) {
                            bail!("audio contains a non-finite sample");
                        }
                        let activity = provider.vad_activity(&pcm)?;
                        let utterance = active.endpoint.push(&pcm, activity);
                        let transcript = utterance
                            .map(|utterance| active.finish_utterance(&mut provider, utterance))
                            .transpose()?;
                        if transcript.is_some() {
                            provider.restart_vad()?;
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
                        .transpose()
                        .and_then(|transcript| {
                            provider.restart_vad()?;
                            Ok(transcript)
                        });
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
                    match provider.restart_vad() {
                        Ok(()) => Response::Ack { id },
                        Err(error) => Response::Error {
                            id: Some(id),
                            message: format!("{error:#}"),
                        },
                    }
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
    use std::os::unix::net::UnixStream;

    fn test_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = env::temp_dir().join(format!("omawake-audiocpp-{name}-{}", std::process::id()));
        let paths = AppPaths {
            config_file: root.join("config/omawake/config.toml"),
            data_dir: root.join("data/omawake"),
            cache_dir: root.join("cache/omawake"),
            state_dir: root.join("state/omawake"),
            runtime_dir: root.join("run/omawake"),
        };
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
        (root, paths)
    }

    #[test]
    fn transcript_conversion_uses_whole_phrase_matcher() {
        let matcher = PhraseMatcher::compile(&[WakeWord {
            id: "lights".into(),
            phrase: "light up".into(),
            enabled: true,
            command: vec!["true".into()],
        }])
        .unwrap();
        let detections = transcripts_to_detections(
            &matcher,
            vec![Transcript {
                text: "The yellow lamps would light up here.".into(),
                start_sample: 8_000,
                end_sample: 16_000,
            }],
        );
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].id, "lights");
        assert_eq!(detections[0].tokens, ["light", "up"]);
        assert_eq!(detections[0].start_time, 0.5);
    }

    #[test]
    fn cpu_checkpoint_rejects_accelerator_runtime_without_spawning() {
        let mut config = Config::default();
        config.backend.kind = "audiocpp".into();
        config.backend.runtime = Runtime::Cuda;
        let error = AudioCppBackend::load(&config, &AppPaths::discover())
            .err()
            .unwrap();
        assert!(error.to_string().contains("only backend.runtime = default"));
    }

    #[test]
    fn exact_model_names_are_enforced() {
        let wrong = Path::new("moonshine.gguf");
        assert!(
            require_exact_asset_name(wrong, "moonshine-streaming-tiny-q8_0.gguf", "verifier")
                .is_err()
        );
    }

    #[test]
    fn asr_profile_is_an_explicit_curated_extension_point() {
        let mut config = Config::default();
        assert_eq!(asr_profile(&config).unwrap().family, "moonshine_asr");
        config
            .backend
            .options
            .insert("audiocpp.asr_family".into(), "moonshine_asr".into());
        assert_eq!(asr_profile(&config).unwrap().label, MOONSHINE_TINY.label);
        config
            .backend
            .options
            .insert("audiocpp.asr_family".into(), "whisper_asr".into());
        assert!(
            asr_profile(&config)
                .err()
                .unwrap()
                .to_string()
                .contains("is not qualified")
        );
    }

    #[test]
    fn package_library_discovery_covers_binary_and_prefix_layouts() {
        let (root, paths) = test_paths("package-layouts");
        let executable = root.join("prefix/bin/omawake");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::write(&executable, []).unwrap();
        let binary_library = executable
            .parent()
            .unwrap()
            .join("lib/libaudiocpp.so.0.1.0");
        fs::create_dir_all(binary_library.parent().unwrap()).unwrap();
        fs::write(&binary_library, []).unwrap();

        let config = Config::default();
        let found = resolve_library_with(&config, &paths, &[], None, &executable).unwrap();
        assert_eq!(found, binary_library.canonicalize().unwrap());

        fs::remove_file(&binary_library).unwrap();
        let prefix_library = root.join("prefix/lib/omawake/libaudiocpp.so.0.1.0");
        fs::create_dir_all(prefix_library.parent().unwrap()).unwrap();
        fs::write(&prefix_library, []).unwrap();
        let found = resolve_library_with(&config, &paths, &[], None, &executable).unwrap();
        assert_eq!(found, prefix_library.canonicalize().unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn configured_library_dirs_resolve_provider_and_worker_dependencies() {
        let (root, paths) = test_paths("configured-library-dirs");
        let provider_dir = root.join("native");
        let dependency_dir = root.join("dependencies");
        fs::create_dir_all(&provider_dir).unwrap();
        fs::create_dir_all(&dependency_dir).unwrap();
        let provider = provider_dir.join("libaudiocpp.so.0.1.0");
        fs::write(&provider, []).unwrap();
        let mut config = Config::default();
        config.backend.library_dirs = vec![provider_dir.clone(), dependency_dir.clone()];

        let configured = resolve_configured_library_dirs(&config, &paths).unwrap();
        let executable = root.join("prefix/bin/omawake");
        let found = resolve_library_with(&config, &paths, &configured, None, &executable).unwrap();
        assert_eq!(found, provider.canonicalize().unwrap());
        let worker_dirs = worker_library_dirs(&found, configured).unwrap();
        assert_eq!(worker_dirs[0], provider_dir.canonicalize().unwrap());
        assert!(worker_dirs.contains(&dependency_dir.canonicalize().unwrap()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn worker_read_wait_is_bounded_without_crashing_a_process() {
        let (reader, _writer) = UnixStream::pair().unwrap();
        let error = wait_readable(&reader, Duration::from_millis(1)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }
}
