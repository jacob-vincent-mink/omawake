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
use super::audio::{AudioResampler, read_wave};
use super::{Detection, WakeWordBackend, WakeWordStream, detect_samples};
use crate::backend::Runtime;
use crate::config::Config;
use crate::paths::AppPaths;
use crate::phrase::{PhraseMatcher, normalize_tokens, record_transcript};

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

pub(crate) fn probe_provider(config: &Config, paths: &AppPaths) -> Result<(PathBuf, String)> {
    if config.backend.kind != "audiocpp" {
        bail!("audio.cpp probe requires backend.kind = audiocpp");
    }
    let (backend, device) = runtime_backend(config)?;
    if !(1..=64).contains(&config.backend.threads) {
        bail!("backend threads must be between 1 and 64");
    }
    let library = discover_provider(config, paths)?;
    let api = AudioCppApi::load(&library)?;
    crate::provider_families::require(&api._library, &["moonshine_asr", "silero_vad"])?;
    let version = unsafe { (api.build_version)() };
    let version = if version.is_null() {
        format!("audio.cpp ABI 0.1.0 (requested {backend}:{device})")
    } else {
        format!(
            "audio.cpp {} (ABI 0.1.0, requested {backend}:{device})",
            unsafe { CStr::from_ptr(version) }.to_string_lossy()
        )
    };
    Ok((library, version))
}

pub(crate) fn discover_provider(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    if config.backend.kind != "audiocpp" {
        bail!("audio.cpp discovery requires backend.kind = audiocpp");
    }
    runtime_backend(config)?;
    let configured_dirs = resolve_configured_library_dirs(config, paths)?;
    resolve_library(config, paths, &configured_dirs)
}

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
        let (backend, device) = runtime_backend(config)?;
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
        let worker = Worker::spawn(WorkerSpec {
            executable: env::current_exe().context("resolve Omawake executable")?,
            library,
            verifier,
            vad,
            cache: paths.cache_dir.join("native-tmp"),
            threads: i32::from(config.backend.threads),
            asr_family: profile.family.into(),
            library_dirs,
            backend: backend.into(),
            device,
        })?;
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

fn runtime_backend(config: &Config) -> Result<(&'static str, i32)> {
    config.backend.validate_shape()?;
    let backend = match config.backend.runtime {
        Runtime::Default => "cpu",
        Runtime::Cuda => "cuda",
        Runtime::Vulkan => "vulkan",
        Runtime::Hip => "hip",
        Runtime::Openvino => {
            bail!(
                "audio.cpp does not use the OpenVINO runtime; select backend.kind = openvino-genai"
            )
        }
    };
    let device = i32::try_from(config.backend.device_id)
        .context("backend.device_id exceeds audio.cpp's supported range")?;
    Ok((backend, device))
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
        let (sample_rate, samples) = read_wave(path)?;
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

pub(in crate::engine) fn resolve_bundled_library(
    paths: &AppPaths,
    configured_dirs: &[PathBuf],
) -> Result<PathBuf> {
    let executable = env::current_exe().context("resolve Omawake executable")?;
    let environment = env::var_os("OMAWAKE_AUDIOCPP_LIBRARY").map(PathBuf::from);
    let mut config = Config::default();
    config.backend.library.clear();
    resolve_library_with(
        &config,
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
    backend: String,
    device: i32,
}

impl Worker {
    fn spawn(spec: WorkerSpec) -> Result<Self> {
        fs::create_dir_all(&spec.cache)
            .with_context(|| format!("create audio.cpp cache {}", spec.cache.display()))?;
        fs::set_permissions(&spec.cache, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("secure audio.cpp cache {}", spec.cache.display()))?;
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
        .arg(&spec.backend)
        .arg(spec.device.to_string())
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
    backend: String,
    device: i32,
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
        backend: &str,
        device: i32,
    ) -> Result<Self> {
        let api = AudioCppApi::load(library)?;
        let mut provider = Self {
            api,
            backend: backend.to_owned(),
            device,
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
        provider.vad_session = provider.create_session(
            provider.vad_model,
            "vad",
            "streaming",
            threads,
            backend,
            device,
        )?;
        provider.asr_session = provider.create_session(
            provider.asr_model,
            "asr",
            "offline",
            threads,
            backend,
            device,
        )?;
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
        backend_name: &str,
        device: i32,
    ) -> Result<*mut c_void> {
        let task = CString::new(task)?;
        let mode = CString::new(mode)?;
        let backend_name = CString::new(backend_name)?;
        let backend = BackendConfig {
            backend: backend_name.as_ptr(),
            device,
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
        self.api.check("create audio.cpp session", status)?;
        if session.is_null() {
            bail!("audio.cpp returned a null session");
        }
        Ok(session)
    }

    fn version(&self) -> String {
        let version = unsafe { (self.api.build_version)() };
        if version.is_null() {
            format!("audio.cpp ABI 0.1.0 ({}:{})", self.backend, self.device)
        } else {
            format!(
                "audio.cpp {} (ABI 0.1.0, {}:{})",
                unsafe { CStr::from_ptr(version) }.to_string_lossy(),
                self.backend,
                self.device
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
    backend: &str,
    device: i32,
) -> Result<()> {
    if let Err(error) = harden_worker_process() {
        let mut output = io::stdout().lock();
        protocol::write_response(
            &mut output,
            &Response::Error {
                id: None,
                message: format!("{error:#}"),
            },
        )?;
        return Ok(());
    }
    worker_main_io(
        library,
        verifier,
        vad,
        threads,
        asr_family,
        backend,
        device,
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
    )
}

#[allow(clippy::too_many_arguments)]
fn worker_main_io(
    library: &Path,
    verifier: &Path,
    vad: &Path,
    threads: i32,
    asr_family: &str,
    backend: &str,
    device: i32,
    input: &mut impl io::Read,
    output: &mut impl io::Write,
) -> Result<()> {
    let mut provider = match AudioCppProvider::open(
        library, verifier, vad, threads, asr_family, backend, device,
    ) {
        Ok(provider) => provider,
        Err(error) => {
            protocol::write_response(
                output,
                &Response::Error {
                    id: None,
                    message: format!("{error:#}"),
                },
            )?;
            return Ok(());
        }
    };
    protocol::write_response(
        output,
        &Response::Ready {
            version: provider.version(),
        },
    )?;
    let mut stream: Option<WorkerStream> = None;
    loop {
        let (request, pcm) = protocol::read_request(input)?;
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
        protocol::write_response(output, &response)?;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::WakeWord;
    use std::os::unix::net::UnixStream;
    use std::process::Command;
    use std::sync::OnceLock;

    const FAKE_AUDIOCPP_C: &str = r#"
    #include <stdint.h>
    #include <stddef.h>
    #include <stdlib.h>

    typedef struct { int kind; int64_t sample; float probability; int64_t start; int64_t end; } Event;
    static const char *last_error = "controlled fake audio.cpp error";
    static const char transcript[] = "  hello oma  ";

    uint32_t audiocpp_abi_version(void) { return 256; }
    const char *audiocpp_build_version(void) { return "fake-provider-1"; }
    const char *audiocpp_last_error(void) { return last_error; }
    int audiocpp_registry_create(const char *json, void **out) {
    (void) json; *out = malloc(1); return *out ? 0 : 1;
    }
size_t audiocpp_registry_family_count(const void *registry) { (void)registry; return 2; }
int audiocpp_registry_family(const void *registry, size_t index, const char **out) {
    (void)registry; static const char *names[] = {"moonshine_asr", "silero_vad"};
    if (index >= 2) return 1; *out = names[index]; return 0;
}
    void audiocpp_registry_free(void *value) { free(value); }
    int audiocpp_model_load(void *registry, const char *path, const void *config,
                        const void *options, void **out) {
    (void) registry; (void) path; (void) config; (void) options;
    *out = malloc(1); return *out ? 0 : 1;
    }
    void audiocpp_model_free(void *value) { free(value); }
    int audiocpp_session_create(const void *model, const char *task, const char *mode,
                            const void *backend, const void *options, void **out) {
    (void) model; (void) task; (void) mode; (void) backend; (void) options;
    *out = malloc(1); return *out ? 0 : 1;
    }
    void audiocpp_session_free(void *value) { free(value); }
    void *audiocpp_request_create(void) { return malloc(1); }
    void audiocpp_request_free(void *value) { free(value); }
    int audiocpp_request_set_audio(void *request, const float *pcm, size_t count,
                               int rate, int channels) {
    (void) request; (void) pcm; (void) count; (void) rate; (void) channels; return 0;
    }
    int audiocpp_session_run(void *session, const void *request, void **out) {
    (void) session; (void) request; *out = malloc(1); return *out ? 0 : 1;
    }
    int audiocpp_result_text(const void *result, const char **text, const char **json) {
    (void) result; *text = transcript; if (json) *json = 0; return 0;
    }
    size_t audiocpp_result_segment_count(const void *result) { (void) result; return 1; }
    int audiocpp_result_segment(const void *result, size_t index, int64_t *start,
                            int64_t *end, float *score, const char **text) {
    const Event *event = (const Event *) result; (void) index;
    if (start) *start = event->start;
    if (end) *end = event->end;
    if (score) *score = 0.9f;
    if (text) *text = transcript;
    return 0;
    }
    void audiocpp_result_free(void *value) { free(value); }
    int audiocpp_stream_start(void *session, const void *options) {
    (void) session; (void) options; return 0;
    }
    int audiocpp_stream_push(void *session, const float *pcm, size_t count, int rate,
                         int channels, int64_t offset, void **out) {
    (void) session; (void) rate; (void) channels; *out = 0;
    if (!count || pcm[0] == 0.0f) return 0;
    Event *event = calloc(1, sizeof(Event));
    if (!event) return 1;
    event->probability = pcm[0] == 3.0f ? 2.0f : 0.9f;
    event->sample = offset + (int64_t) count;
    if (pcm[0] == -1.0f) event->kind = 1;
    else if (pcm[0] == 2.0f) {
        event->kind = 2; event->start = offset; event->end = offset + (int64_t) count;
    } else if (pcm[0] == 4.0f) event->kind = 9;
    else { event->kind = 0; if (pcm[0] == 5.0f) event->sample++; }
    *out = event; return 0;
    }
    int audiocpp_stream_reset(void *session) { (void) session; return 0; }
    void audiocpp_event_free(void *event) { free(event); }
    const void *audiocpp_event_as_result(const void *event) { return event; }
    size_t audiocpp_event_voice_activity_count(const void *event) { (void) event; return 1; }
    int audiocpp_event_voice_activity(const void *value, size_t index, int *kind,
                                  int64_t *sample, float *probability) {
    const Event *event = (const Event *) value; (void) index;
    *kind = event->kind; *sample = event->sample; *probability = event->probability; return 0;
    }
    "#;

    const FAKE_WORKER_C: &str = r#"
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    static int read_exact(void *buffer, size_t size) {
    return fread(buffer, 1, size, stdin) == size;
    }
    static void write_response(const char *json) {
    uint32_t size = (uint32_t) strlen(json);
    fwrite(&size, sizeof(size), 1, stdout);
    fwrite(json, 1, size, stdout);
    fflush(stdout);
    }
    static unsigned long long request_id(const char *json) {
    const char *id = strstr(json, "\"id\":");
    return id ? strtoull(id + 5, 0, 10) : 0;
    }
    static size_t request_samples(const char *json) {
    const char *samples = strstr(json, "\"samples\":");
    return samples ? (size_t) strtoull(samples + 10, 0, 10) : 0;
    }
    int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "__openvino-runtime-worker") == 0) {
        fputs("{\"runtime_build\":\"fake\",\"runtime_description\":\"safe fake runtime\",\"requested_device\":\"CPU\",\"available_device\":\"CPU\",\"full_device_name\":\"Fake CPU\",\"device_architecture\":\"fake\",\"driver_version\":\"fake\",\"genai_library\":\"fake\",\"core_library\":\"fake\",\"audiocpp_library\":\"fake\"}", stdout);
        return 0;
    }
    int openvino = argc > 1 && strcmp(argv[1], "__openvino-genai-worker") == 0;
    const char *mode = openvino ? (argc > 8 ? argv[8] : "CPU") : (argc > 7 ? argv[7] : "cpu");
    if (!openvino && argc == 6 && strcmp(argv[1], "__whisper-worker") == 0) {
        if (strcmp(argv[5], "101") == 0) mode = "HANDSHAKE_ERROR";
        else if (strcmp(argv[5], "102") == 0) mode = "HANDSHAKE_UNEXPECTED";
        else if (strcmp(argv[5], "103") == 0) mode = "HANDSHAKE_EOF";
        else if (strcmp(argv[5], "201") == 0) mode = "RESPONSE_ERROR";
        else if (strcmp(argv[5], "202") == 0) mode = "RESPONSE_UNEXPECTED";
    }
    if (strcmp(mode, "HANDSHAKE_EOF") == 0) return 0;
    if (strcmp(mode, "HANDSHAKE_ERROR") == 0) {
        write_response("{\"type\":\"error\",\"id\":null,\"message\":\"controlled handshake error\"}");
        return 0;
    }
    if (strcmp(mode, "HANDSHAKE_UNEXPECTED") == 0) {
        write_response("{\"type\":\"ack\",\"id\":0}");
        return 0;
    }
    if (openvino) {
        write_response("{\"type\":\"ready\",\"evidence\":{\"profile_id\":\"whisper-base.en-int8-ov\",\"languages\":[\"en\"],\"multilingual\":false,\"runtime_build\":\"fake\",\"runtime_description\":\"fake\",\"requested_device\":\"CPU\",\"available_device\":\"CPU\",\"full_device_name\":\"Fake CPU\",\"device_architecture\":\"fake\",\"driver_version\":\"fake\",\"static_pipeline\":false,\"pipeline_load_milliseconds\":1.0,\"cache_directory\":\"/tmp\",\"cache_files\":0,\"cache_bytes\":0,\"genai_library\":\"fake\",\"core_library\":\"fake\"}}");
    } else {
        write_response("{\"type\":\"ready\",\"version\":\"safe fake worker\"}");
    }
    for (;;) {
        uint32_t size = 0;
        if (!read_exact(&size, sizeof(size))) return 0;
        if (size > 65536) return 2;
        char *json = calloc((size_t) size + 1, 1);
        if (!json || !read_exact(json, size)) return 3;
        size_t samples = request_samples(json);
        float discard[512];
        if (samples && (!read_exact(discard, samples * sizeof(float)))) return 4;
        unsigned long long id = request_id(json);
        char response[512];
        if (strstr(json, "\"type\":\"shutdown\"")) { free(json); return 0; }
        if (strcmp(mode, "RESPONSE_ERROR") == 0) {
            snprintf(response, sizeof(response), "{\"type\":\"error\",\"id\":%llu,\"message\":\"controlled request error\"}", id);
        } else if (strcmp(mode, "RESPONSE_UNEXPECTED") == 0) {
            snprintf(response, sizeof(response), "{\"type\":\"ack\",\"id\":0}");
        } else if (strstr(json, "\"type\":\"start\"") || strstr(json, "\"type\":\"cancel\"")) {
            snprintf(response, sizeof(response), "{\"type\":\"ack\",\"id\":%llu}", id);
        } else if (strstr(json, "\"type\":\"finish\"")) {
            snprintf(response, sizeof(response), "{\"type\":\"result\",\"id\":%llu,\"transcripts\":[{\"text\":\"hello oma\",\"start_sample\":0,\"end_sample\":512}]}", id);
        } else {
            snprintf(response, sizeof(response), "{\"type\":\"result\",\"id\":%llu,\"transcripts\":[]}", id);
        }
        free(json);
        write_response(response);
    }
    }
    "#;

    pub(crate) fn fake_library() -> &'static Path {
        static LIBRARY: OnceLock<PathBuf> = OnceLock::new();
        LIBRARY
            .get_or_init(|| {
                let root = env::temp_dir()
                    .join(format!("omawake-safe-fake-audiocpp-{}", std::process::id()));
                fs::create_dir_all(&root).unwrap();
                let source = root.join("fake_audiocpp.c");
                let library = root.join("libaudiocpp.so.0.1.0");
                fs::write(&source, FAKE_AUDIOCPP_C).unwrap();
                let output = Command::new("cc")
                    .args(["-shared", "-fPIC", "-O0"])
                    .arg(&source)
                    .arg("-o")
                    .arg(&library)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "compile fake audio.cpp DSO: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                library
            })
            .as_path()
    }

    pub(crate) fn fake_worker() -> &'static Path {
        static WORKER: OnceLock<PathBuf> = OnceLock::new();
        WORKER
            .get_or_init(|| {
                let root = env::temp_dir()
                    .join(format!("omawake-safe-fake-worker-{}", std::process::id()));
                fs::create_dir_all(&root).unwrap();
                let source = root.join("fake_worker.c");
                let worker = root.join("fake-worker");
                fs::write(&source, FAKE_WORKER_C).unwrap();
                let output = Command::new("cc")
                    .args(["-O0"])
                    .arg(&source)
                    .arg("-o")
                    .arg(&worker)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "compile safe fake worker: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                worker
            })
            .as_path()
    }

    fn fake_models(name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = env::temp_dir().join(format!(
            "omawake-safe-fake-models-{name}-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let verifier = root.join("verifier.gguf");
        let vad = root.join("vad.safetensors");
        fs::write(&verifier, b"fake verifier").unwrap();
        fs::write(&vad, b"fake vad").unwrap();
        (root, verifier, vad)
    }

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
            aliases: Vec::new(),
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
    fn audiocpp_rejects_openvino_runtime_without_spawning() {
        let mut config = Config::default();
        config.backend.kind = "audiocpp".into();
        config.backend.runtime = Runtime::Openvino;
        let error = AudioCppBackend::load(&config, &AppPaths::discover())
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("select backend.kind = openvino-genai")
        );
    }

    #[test]
    fn backend_load_validates_configuration_and_pinned_assets_before_spawning() {
        let (root, paths) = test_paths("backend-load-validation");
        let mut config = Config::default();
        config.backend.kind = "audiocpp".into();
        config.backend.library = fake_library().to_path_buf();
        config.model.sample_rate = 8_000;
        assert!(AudioCppBackend::load(&config, &paths).is_err());
        config.model.sample_rate = 16_000;
        config.backend.threads = 0;
        assert!(AudioCppBackend::load(&config, &paths).is_err());
        config.backend.threads = 2;
        let models = config.model_directory(&paths);
        fs::create_dir_all(&models).unwrap();
        fs::write(models.join(&config.model.verifier), b"wrong verifier").unwrap();
        assert!(AudioCppBackend::load(&config, &paths).is_err());
        fs::write(models.join(&config.model.vad), b"wrong vad").unwrap();
        assert!(
            AudioCppBackend::load(&config, &paths)
                .err()
                .unwrap()
                .to_string()
                .contains("expected pinned size")
        );
        fs::remove_dir_all(root).unwrap();
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

    #[test]
    fn runtime_mapping_covers_every_audio_cpp_backend() {
        let mut config = Config::default();
        config.backend.kind = "audiocpp".into();
        for (runtime, backend) in [
            (Runtime::Default, "cpu"),
            (Runtime::Cuda, "cuda"),
            (Runtime::Vulkan, "vulkan"),
            (Runtime::Hip, "hip"),
        ] {
            config.backend.runtime = runtime;
            config.backend.device = if runtime == Runtime::Default {
                "cpu".into()
            } else {
                "gpu".into()
            };
            config.backend.device_id = if runtime == Runtime::Default { 0 } else { 3 };
            assert_eq!(
                runtime_backend(&config).unwrap(),
                (backend, config.backend.device_id as i32)
            );
        }
        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "cpu".into();
        config.backend.device_id = 0;
        assert!(
            runtime_backend(&config)
                .unwrap_err()
                .to_string()
                .contains("openvino-genai")
        );
    }

    #[test]
    fn fake_provider_loads_real_dso_and_runs_vad_and_asr() {
        let (_root, verifier, vad) = fake_models("provider");
        let mut provider = AudioCppProvider::open(
            fake_library(),
            &verifier,
            &vad,
            2,
            "moonshine_asr",
            "cuda",
            3,
        )
        .unwrap();
        assert_eq!(
            provider.version(),
            "audio.cpp fake-provider-1 (ABI 0.1.0, cuda:3)"
        );
        assert_eq!(provider.transcribe(&[0.25; 32]).unwrap(), "hello oma");
        assert_eq!(
            provider
                .vad_activity(&vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .start_before_frame_end,
            None
        );

        let mut frame = vec![0.0; FRAME_SAMPLES];
        frame[0] = 1.0;
        assert_eq!(
            provider
                .vad_activity(&frame)
                .unwrap()
                .start_before_frame_end,
            Some(0)
        );
        frame[0] = -1.0;
        assert_eq!(
            provider.vad_activity(&frame).unwrap().end_before_frame_end,
            Some(0)
        );
        provider.restart_vad().unwrap();
        frame[0] = 2.0;
        let segment = provider.vad_activity(&frame).unwrap();
        assert_eq!(segment.start_before_frame_end, Some(FRAME_SAMPLES));
        assert_eq!(segment.end_before_frame_end, Some(0));

        frame[0] = 3.0;
        assert!(
            provider
                .vad_activity(&frame)
                .unwrap_err()
                .to_string()
                .contains("invalid")
        );
        frame[0] = 4.0;
        assert!(
            provider
                .vad_activity(&frame)
                .unwrap_err()
                .to_string()
                .contains("unknown")
        );
        frame[0] = 5.0;
        assert!(
            provider
                .vad_activity(&frame)
                .unwrap_err()
                .to_string()
                .contains("outside")
        );
        assert!(provider.vad_activity(&frame[..10]).is_err());
        provider.vad_cursor = i64::MAX;
        assert!(provider.vad_activity(&vec![0.0; FRAME_SAMPLES]).is_err());
        assert!(
            provider
                .api
                .check("controlled", 7)
                .unwrap_err()
                .to_string()
                .contains("controlled fake")
        );
    }

    #[test]
    fn provider_probe_uses_configured_real_dso() {
        let (root, paths) = test_paths("probe-real-dso");
        let mut config = Config::default();
        config.backend.kind = "audiocpp".into();
        config.backend.library = fake_library().to_path_buf();
        let (path, version) = probe_provider(&config, &paths).unwrap();
        assert_eq!(path, fake_library().canonicalize().unwrap());
        assert!(version.contains("fake-provider-1"));

        config.backend.threads = 0;
        assert!(
            probe_provider(&config, &paths)
                .unwrap_err()
                .to_string()
                .contains("threads")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn in_memory_worker_protocol_covers_stream_lifecycle() {
        let (_root, verifier, vad) = fake_models("worker-io");
        let mut input = Vec::new();
        let frame = vec![0.0; FRAME_SAMPLES];
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 1,
                samples: FRAME_SAMPLES,
            },
            &frame,
        )
        .unwrap();
        protocol::write_request(&mut input, &Request::Start { id: 1 }, &[]).unwrap();
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 2,
                samples: FRAME_SAMPLES,
            },
            &frame,
        )
        .unwrap();
        let mut non_finite = frame.clone();
        non_finite[0] = f32::NAN;
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 1,
                samples: FRAME_SAMPLES,
            },
            &non_finite,
        )
        .unwrap();
        let mut speech = frame.clone();
        speech[0] = 1.0;
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 1,
                samples: FRAME_SAMPLES,
            },
            &speech,
        )
        .unwrap();
        protocol::write_request(&mut input, &Request::Finish { id: 2 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Finish { id: 1 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Finish { id: 1 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Start { id: 3 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Cancel { id: 3 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Cancel { id: 3 }, &[]).unwrap();
        protocol::write_request(&mut input, &Request::Shutdown, &[]).unwrap();

        let mut output = Vec::new();
        worker_main_io(
            fake_library(),
            &verifier,
            &vad,
            2,
            "moonshine_asr",
            "cpu",
            0,
            &mut input.as_slice(),
            &mut output,
        )
        .unwrap();

        let mut responses = output.as_slice();
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Ready { .. }
        ));
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
            matches!(protocol::read_response(&mut responses).unwrap(), Response::Result { id: 1, transcripts } if transcripts[0].text == "hello oma")
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
        assert!(matches!(
            protocol::read_response(&mut responses).unwrap(),
            Response::Error { id: Some(3), .. }
        ));
        assert!(responses.is_empty());
    }

    #[test]
    fn worker_reports_native_open_errors_over_protocol() {
        let (_root, verifier, vad) = fake_models("worker-open-error");
        let mut output = Vec::new();
        worker_main_io(
            Path::new("/definitely/missing/libaudiocpp.so"),
            &verifier,
            &vad,
            2,
            "moonshine_asr",
            "cpu",
            0,
            &mut [].as_slice(),
            &mut output,
        )
        .unwrap();
        assert!(matches!(
            protocol::read_response(&mut output.as_slice()).unwrap(),
            Response::Error { id: None, message } if message.contains("load audio.cpp")
        ));
    }

    fn fake_worker_spec(name: &str) -> (PathBuf, WorkerSpec) {
        let (root, verifier, vad) = fake_models(name);
        let cache = root.join("cache");
        (
            root,
            WorkerSpec {
                executable: fake_worker().to_path_buf(),
                library: fake_library().to_path_buf(),
                verifier,
                vad,
                cache,
                threads: 2,
                asr_family: "moonshine_asr".into(),
                library_dirs: Vec::new(),
                backend: "cpu".into(),
                device: 0,
            },
        )
    }

    #[test]
    fn supervised_worker_restarts_and_reaps_a_normal_process() {
        let (root, spec) = fake_worker_spec("supervised-worker");
        let mut worker = Worker::spawn(spec).unwrap();
        assert!(worker.start(7).is_ok());
        assert!(
            worker
                .start(8)
                .unwrap_err()
                .to_string()
                .contains("already serving")
        );
        assert!(
            worker
                .audio(7, &vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .is_empty()
        );
        assert_eq!(worker.finish(7).unwrap()[0].text, "hello oma");
        worker.start(8).unwrap();
        worker.cancel(8).unwrap();
        worker.relaunch().unwrap();
        worker.terminate().unwrap();
        // The first exchange observes the normal EOF, then the one supervised
        // restart succeeds without ever invoking a malformed native library.
        worker.start(9).unwrap();
        worker.shutdown();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervised_worker_reports_controlled_handshake_and_response_errors() {
        for mode in ["HANDSHAKE_ERROR", "HANDSHAKE_UNEXPECTED", "HANDSHAKE_EOF"] {
            let (root, mut spec) = fake_worker_spec(&format!("handshake-{mode}"));
            spec.backend = mode.into();
            assert!(Worker::spawn(spec).err().is_some());
            fs::remove_dir_all(root).unwrap();
        }

        for mode in ["RESPONSE_ERROR", "RESPONSE_UNEXPECTED"] {
            let (root, mut spec) = fake_worker_spec(&format!("response-{mode}"));
            spec.backend = mode.into();
            let mut worker = Worker::spawn(spec).unwrap();
            assert!(worker.start(21).is_err());
            assert!(worker.audio(21, &vec![0.0; FRAME_SAMPLES]).is_err());
            assert!(worker.finish(21).is_err());
            assert!(worker.cancel(21).is_err());
            worker.shutdown();
            fs::remove_dir_all(root).unwrap();
        }

        for mode in ["HANDSHAKE_ERROR", "HANDSHAKE_UNEXPECTED", "HANDSHAKE_EOF"] {
            let (root, spec) = fake_worker_spec(&format!("relaunch-{mode}"));
            let mut worker = Worker::spawn(spec).unwrap();
            worker.spec.backend = mode.into();
            assert!(worker.relaunch().is_err());
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn high_level_stream_uses_supervised_worker_and_phrase_matcher() {
        let (root, spec) = fake_worker_spec("backend-stream");
        let backend = AudioCppBackend {
            worker: RefCell::new(Worker::spawn(spec).unwrap()),
            matcher: PhraseMatcher::compile(&[WakeWord {
                id: "greeting".into(),
                phrase: "hello oma".into(),
                aliases: Vec::new(),
                enabled: true,
                command: vec!["true".into()],
            }])
            .unwrap(),
            next_stream_id: Cell::new(u64::MAX),
        };
        assert_eq!(backend.kind(), "audiocpp");
        let stream = backend.stream();
        assert!(stream.accept(16_000, &[f32::NAN]).is_err());
        assert!(
            stream
                .accept(16_000, &vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .is_empty()
        );
        let detections = stream.finish().unwrap();
        assert_eq!(detections[0].id, "greeting");
        assert!(stream.finish().unwrap().is_empty());
        assert!(stream.accept(16_000, &[0.0]).is_err());
        drop(stream);

        let short = backend.stream();
        short.accept(16_000, &[0.0; 16]).unwrap();
        assert_eq!(short.finish().unwrap()[0].id, "greeting");
        drop(short);

        let wav = root.join("silence.wav");
        let mut writer = hound::WavWriter::create(
            &wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..FRAME_SAMPLES {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
        assert_eq!(backend.detect_file(&wav).unwrap()[0].id, "greeting");

        let abandoned = backend.stream();
        abandoned.accept(16_000, &[0.0; 16]).unwrap();
        drop(abandoned);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_and_library_path_validation_covers_success_and_tampering() {
        let (root, paths) = test_paths("paths-and-hashes");
        let model_dir = root.join("models");
        fs::create_dir_all(&model_dir).unwrap();
        let asset = model_dir.join("asset.bin");
        fs::write(&asset, b"abc").unwrap();
        assert_eq!(
            resolve_model_asset(&model_dir, "asset.bin", "test").unwrap(),
            asset
        );
        assert!(resolve_model_asset(&model_dir, "", "test").is_err());
        assert!(resolve_model_asset(&model_dir, "missing", "test").is_err());
        assert_eq!(
            resolve_model_asset(&model_dir, asset.to_str().unwrap(), "test").unwrap(),
            asset
        );
        verify_asset(
            &asset,
            3,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "test",
        )
        .unwrap();
        assert!(verify_asset(&asset, 2, "unused", "test").is_err());
        assert!(verify_asset(&asset, 3, "bad", "test").is_err());

        let configured = paths.config_file.parent().unwrap().join("provider.so");
        fs::write(&configured, b"provider").unwrap();
        let mut config = Config::default();
        config.backend.library = PathBuf::from("provider.so");
        assert_eq!(
            resolve_library_with(&config, &paths, &[], None, fake_worker()).unwrap(),
            configured.canonicalize().unwrap()
        );
        config.backend.library.clear();
        assert_eq!(
            resolve_library_with(&config, &paths, &[], Some(&configured), fake_worker()).unwrap(),
            configured.canonicalize().unwrap()
        );
        assert!(
            resolve_library_with(
                &config,
                &paths,
                &[],
                Some(Path::new("relative.so")),
                fake_worker()
            )
            .is_err()
        );

        let relative_dir = paths.config_file.parent().unwrap().join("relative-native");
        fs::create_dir_all(&relative_dir).unwrap();
        config.backend.library_dirs = vec![PathBuf::from("relative-native"), relative_dir.clone()];
        assert_eq!(
            resolve_configured_library_dirs(&config, &paths)
                .unwrap()
                .len(),
            1
        );
        config.backend.library_dirs = vec![PathBuf::from("missing-native")];
        assert!(resolve_configured_library_dirs(&config, &paths).is_err());
        config.backend.library_dirs = vec![configured.clone()];
        assert!(resolve_configured_library_dirs(&config, &paths).is_err());
        assert!(package_library_dirs(Path::new("/")).is_empty());
        assert!(worker_library_dirs(Path::new("/"), vec![]).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
