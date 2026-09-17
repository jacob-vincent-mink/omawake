//! OpenVINO GenAI Whisper verifier loaded through Intel's public C API.
//!
//! The native libraries live only in a supervised Omawake worker.  The main
//! process exchanges bounded PCM frames and structured responses with that
//! worker, so a vendor-runtime failure cannot take down the daemon.

mod protocol;
mod ring;

use std::cell::{Cell, RefCell};
use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use libloading::Library;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(crate) use self::protocol::PlacementEvidence;
use self::protocol::{FRAME_SAMPLES, Request, Response, Transcript};
use self::ring::{Activity, ActivityBuffer, Utterance};
use super::audio::{AudioResampler, read_wave};
use super::{Detection, WakeWordBackend, WakeWordStream, detect_samples};
use crate::backend::Runtime;
use crate::config::{Config, WakeWord};
use crate::paths::AppPaths;
use crate::phrase::{PhraseMatcher, normalize_tokens, record_transcript};

const WORKER_STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const WORKER_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const WORKER_STOP_TIMEOUT: Duration = Duration::from_secs(2);
const AUDIOCPP_ABI_0_1_0: u32 = 1 << 8;
#[derive(Clone, Copy, Debug)]
pub(crate) struct VerifierProfile {
    pub catalog_id: &'static str,
    pub id: &'static str,
    pub languages: &'static [&'static str],
    /// Language tokens the pinned model itself accepts for model.language.
    pub model_languages: &'static [&'static str],
    pub multilingual: bool,
}

pub(crate) const WHISPER_BASE_EN_PROFILE: VerifierProfile = VerifierProfile {
    catalog_id: crate::catalog::OPENVINO_MODEL_ID,
    id: "whisper-base.en-int8-ov",
    languages: &["en"],
    model_languages: &[],
    multilingual: false,
};

/// Whisper Base (multilingual) INT8 OpenVINO profile. `languages` stays the
/// curated qualification claim (W09: Spanish only); `model_languages` is the
/// full language set the pinned model itself supports and accepts for
/// `model.language` without asserting any additional qualification.
pub(crate) const WHISPER_BASE_MULTI_PROFILE: VerifierProfile = VerifierProfile {
    catalog_id: crate::catalog::OPENVINO_MULTILINGUAL_MODEL_ID,
    id: "whisper-base-int8-ov",
    languages: &["es"],
    model_languages: WHISPER_MODEL_LANGUAGES,
    multilingual: true,
};

/// Language tokens supported by the pinned multilingual Whisper model, taken
/// from its pinned `generation_config.json` (`lang_to_id`). Listing them here
/// validates configuration input only; it is not a qualification claim.
pub(crate) const WHISPER_MODEL_LANGUAGES: &[&str] = &[
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da",
    "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "haw", "ha", "he",
    "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la",
    "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl",
    "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so",
    "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz",
    "vi", "yi", "yo", "zh",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RuntimeEvidence {
    pub runtime_build: String,
    pub runtime_description: String,
    pub requested_device: String,
    pub available_device: String,
    pub full_device_name: String,
    pub device_architecture: String,
    pub driver_version: String,
    pub genai_library: String,
    pub core_library: String,
    pub audiocpp_library: String,
}

/// Exact external runtime and model selected by setup.
#[derive(Clone, Debug)]
pub(crate) struct ProviderSpec {
    pub profile: VerifierProfile,
    pub language: String,
    pub genai_library: PathBuf,
    pub core_library: PathBuf,
    pub audiocpp_library: PathBuf,
    pub library_dirs: Vec<PathBuf>,
    pub model_directory: PathBuf,
    pub vad_model: PathBuf,
    pub cache_directory: PathBuf,
    pub placement_log: PathBuf,
    pub device: String,
    pub vad_threads: u16,
}

impl ProviderSpec {
    pub(crate) fn from_config(config: &Config, paths: &AppPaths) -> Result<Self> {
        if config.backend.kind != "openvino-genai" || config.backend.runtime != Runtime::Openvino {
            bail!("OpenVINO GenAI requires backend.kind = openvino-genai and runtime = openvino");
        }
        if config.model.sample_rate != 16_000 {
            bail!("OpenVINO GenAI Whisper requires model.sample_rate = 16000");
        }
        let profile = verifier_profile(config);
        let language = validate_language(config, &profile)?;
        let device = canonical_device(&config.backend.device)?.to_owned();
        let base = paths.config_file.parent().unwrap_or(Path::new("."));
        let mut library_dirs = Vec::new();
        for configured in &config.backend.library_dirs {
            let candidate = if configured.is_absolute() {
                configured.clone()
            } else {
                base.join(configured)
            };
            let directory = canonical_directory(&candidate, "backend.library_dirs entry")?;
            if !library_dirs.contains(&directory) {
                library_dirs.push(directory);
            }
        }
        if let Some(loader_path) = env::var_os("LD_LIBRARY_PATH") {
            for directory in env::split_paths(&loader_path) {
                if let Ok(directory) = canonical_directory(&directory, "loader library directory")
                    && !library_dirs.contains(&directory)
                {
                    library_dirs.push(directory);
                }
            }
        }
        for directory in ["/usr/lib", "/usr/local/lib", "/usr/lib64"] {
            let directory = PathBuf::from(directory);
            if directory.is_dir() && !library_dirs.contains(&directory) {
                library_dirs.push(directory);
            }
        }
        let genai_library = if config.backend.library.as_os_str().is_empty() {
            env::var_os("OMAWAKE_OPENVINO_GENAI_LIBRARY")
                .map(PathBuf::from)
                .map(|path| canonical_file(&path, "OMAWAKE_OPENVINO_GENAI_LIBRARY"))
                .transpose()?
                .or_else(|| find_library(&library_dirs, &["libopenvino_genai_c.so"]))
                .context(
                    "libopenvino_genai_c was not found in the selected OpenVINO installation",
                )?
        } else {
            let candidate = if config.backend.library.is_absolute() {
                config.backend.library.clone()
            } else {
                base.join(&config.backend.library)
            };
            canonical_file(&candidate, "configured OpenVINO GenAI C library")?
        };
        if let Some(parent) = genai_library.parent().map(Path::to_path_buf)
            && !library_dirs.contains(&parent)
        {
            library_dirs.insert(0, parent);
        }
        let core_library = env::var_os("OMAWAKE_OPENVINO_LIBRARY")
            .map(PathBuf::from)
            .map(|path| canonical_file(&path, "OMAWAKE_OPENVINO_LIBRARY"))
            .transpose()?
            .or_else(|| find_library(&library_dirs, &["libopenvino_c.so"]))
            .context("libopenvino_c was not found in the selected OpenVINO installation")?;
        let plugin = match device.as_str() {
            "CPU" => "libopenvino_intel_cpu_plugin.so",
            "GPU" => "libopenvino_intel_gpu_plugin.so",
            "NPU" => "libopenvino_intel_npu_plugin.so",
            _ => unreachable!(),
        };
        find_library(&library_dirs, &[plugin]).with_context(|| {
            format!("selected OpenVINO installation has no {device} device plugin ({plugin})")
        })?;
        if device == "NPU" {
            for required in [
                "libopenvino_intel_npu_compiler_loader.so",
                "libopenvino_intel_npu_compiler.so",
            ] {
                find_library(&library_dirs, &[required]).with_context(|| {
                    format!("selected OpenVINO NPU installation is incomplete: missing {required}")
                })?;
            }
        }
        let audiocpp_library = super::audiocpp::resolve_bundled_library(paths, &library_dirs)
            .context(
                "OpenVINO GenAI uses Omawake's packaged audio.cpp CPU provider for Silero VAD; install the complete Omawake release layout or set OMAWAKE_AUDIOCPP_LIBRARY to that provider",
            )?;
        let model_directory = config.model_directory(paths);
        let vad_model = if Path::new(&config.model.vad).is_absolute() {
            PathBuf::from(&config.model.vad)
        } else {
            model_directory.join(&config.model.vad)
        };
        let cache_directory = super::openvino_cache_directory(config, paths)?;
        Ok(Self {
            profile,
            language,
            genai_library,
            core_library,
            audiocpp_library,
            library_dirs,
            model_directory,
            vad_model,
            placement_log: cache_directory.join("placement.log"),
            cache_directory,
            device,
            vad_threads: config.backend.threads,
        })
    }

    pub(crate) fn validate(self) -> Result<Self> {
        let profile = self.profile;
        self.validate_with(
            |path| verify_vad_with(profile, path),
            |directory| verify_model_manifest(profile.catalog_id, directory),
        )
    }

    fn validate_with(
        mut self,
        validate_vad: impl FnOnce(&Path) -> Result<()>,
        validate_model: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<Self> {
        self.genai_library = canonical_file(&self.genai_library, "OpenVINO GenAI C library")?;
        self.core_library = canonical_file(&self.core_library, "OpenVINO Runtime C library")?;
        self.audiocpp_library = canonical_file(&self.audiocpp_library, "audio.cpp C library")?;
        self.model_directory =
            canonical_directory(&self.model_directory, "OpenVINO Whisper model")?;
        self.vad_model = canonical_file(&self.vad_model, "Silero VAD model")?;
        validate_vad(&self.vad_model)?;
        validate_model(&self.model_directory)?;
        if !(1..=64).contains(&self.vad_threads) {
            bail!("VAD threads must be between 1 and 64");
        }
        self.device = canonical_device(&self.device)?.into();
        let mut directories = Vec::new();
        for directory in &self.library_dirs {
            let directory = canonical_directory(directory, "runtime library directory")?;
            if !directories.contains(&directory) {
                directories.push(directory);
            }
        }
        for library in [
            &self.genai_library,
            &self.core_library,
            &self.audiocpp_library,
        ] {
            let parent = library
                .parent()
                .context("native library has no parent directory")?;
            let parent = parent.to_path_buf();
            if !directories.contains(&parent) {
                directories.insert(0, parent);
            }
        }
        self.library_dirs = directories;
        secure_directory(&self.cache_directory, "OpenVINO cache")?;
        if let Some(parent) = self.placement_log.parent() {
            secure_directory(parent, "OpenVINO placement log directory")?;
        }
        Ok(self)
    }

    fn static_pipeline(&self) -> bool {
        self.device == "NPU"
    }
}

fn find_library(directories: &[PathBuf], names: &[&str]) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for directory in directories {
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if names
                .iter()
                .any(|name| file_name == *name || file_name.starts_with(&format!("{name}.")))
                && entry
                    .file_type()
                    .is_ok_and(|kind| kind.is_file() || kind.is_symlink())
            {
                candidates.push(entry.path());
            }
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .and_then(|path| path.canonicalize().ok())
}

fn canonical_device(device: &str) -> Result<&'static str> {
    match device.trim().to_ascii_uppercase().as_str() {
        "CPU" => Ok("CPU"),
        "GPU" => Ok("GPU"),
        "NPU" => Ok("NPU"),
        _ => bail!("OpenVINO GenAI device must be CPU, GPU, or NPU"),
    }
}

fn canonical_file(path: &Path, label: &str) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("resolve {label} {}", path.display()))?;
    if !path.is_file() {
        bail!("{label} is not a regular file: {}", path.display());
    }
    Ok(path)
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("resolve {label} {}", path.display()))?;
    if !path.is_dir() {
        bail!("{label} is not a directory: {}", path.display());
    }
    Ok(path)
}

fn secure_directory(path: &Path, label: &str) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("create {label} {}", path.display()))?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("secure {label} {}", path.display()))
}

fn verify_model_manifest(catalog_id: &str, directory: &Path) -> Result<()> {
    let spec = crate::catalog::model(catalog_id).with_context(|| {
        format!("OpenVINO Whisper profile {catalog_id} is missing from the Omawake catalog")
    })?;
    for expected in spec.assets {
        verify_file(
            &directory.join(expected.path),
            expected.size,
            expected.sha256,
            &format!("OpenVINO verifier artifact {}", expected.path),
        )?;
    }
    Ok(())
}

fn verify_vad_with(profile: VerifierProfile, path: &Path) -> Result<()> {
    let asset = crate::catalog::model(profile.catalog_id)
        .and_then(|spec| {
            spec.assets
                .iter()
                .find(|asset| asset.path == "silero_vad_16k.safetensors")
        })
        .context("Silero VAD profile is missing from the Omawake catalog")?;
    if path.file_name().and_then(|name| name.to_str()) != Some(asset.path) {
        bail!("Silero VAD must use the pinned {} artifact", asset.path);
    }
    verify_file(path, asset.size, asset.sha256, "Silero VAD")
}

/// Select the curated verifier profile from the configured catalog model.
/// Unknown names fall back to the English profile; the per-file pinned
/// manifest verification then rejects any model directory that does not
/// carry the English artifacts, so the fallback cannot activate another
/// model.
fn verifier_profile(config: &Config) -> VerifierProfile {
    verifier_profile_for_name(&config.model.name)
}

/// Profile for a catalog model id; unknown ids fall back to the English
/// profile and the pinned per-file manifest verification then rejects any
/// directory that does not carry the English artifacts.
pub(crate) fn verifier_profile_for_name(model_name: &str) -> VerifierProfile {
    if model_name == WHISPER_BASE_MULTI_PROFILE.catalog_id {
        WHISPER_BASE_MULTI_PROFILE
    } else {
        WHISPER_BASE_EN_PROFILE
    }
}

/// Validate `model.language` against the selected verifier profile.
#[cfg(test)]
pub(crate) fn validate_language_for_test(
    config: &Config,
    profile: &VerifierProfile,
) -> Result<String> {
    validate_language(config, profile)
}

fn validate_language(config: &Config, profile: &VerifierProfile) -> Result<String> {
    let language = config.model.language.trim().to_owned();
    if language.is_empty() {
        return Ok(String::new());
    }
    if !profile.multilingual {
        bail!(
            "model.language {:?} requires the multilingual verifier {}",
            language,
            WHISPER_BASE_MULTI_PROFILE.catalog_id
        );
    }
    if !profile.model_languages.contains(&language.as_str()) {
        bail!(
            "model.language {:?} is not a language token of the pinned multilingual Whisper model",
            language
        );
    }
    Ok(language)
}

fn verify_file(path: &Path, expected_bytes: u64, expected_sha256: &str, label: &str) -> Result<()> {
    let actual_bytes = fs::metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?
        .len();
    if actual_bytes != expected_bytes {
        bail!(
            "{label} has {actual_bytes} bytes, expected {expected_bytes}: {}",
            path.display()
        );
    }
    let mut file = File::open(path).with_context(|| format!("open {label} {}", path.display()))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher).with_context(|| format!("hash {label} {}", path.display()))?;
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected_sha256 {
        bail!(
            "{label} checksum mismatch: expected {expected_sha256}, found {actual}: {}",
            path.display()
        );
    }
    Ok(())
}

pub(crate) struct OpenVinoGenAiBackend {
    worker: RefCell<Worker>,
    matcher: PhraseMatcher,
    next_stream_id: Cell<u64>,
}

struct OpenVinoGenAiStream<'a> {
    backend: &'a OpenVinoGenAiBackend,
    state: RefCell<ClientStream>,
}

struct ClientStream {
    id: u64,
    started: bool,
    finished: bool,
    resampler: AudioResampler,
    pending: Vec<f32>,
}

impl OpenVinoGenAiBackend {
    pub(crate) fn open(spec: ProviderSpec, wake_words: &[WakeWord]) -> Result<Self> {
        let spec = spec.validate()?;
        let matcher = PhraseMatcher::compile(wake_words)?;
        let worker = Worker::spawn(spec)?;
        Ok(Self {
            worker: RefCell::new(worker),
            matcher,
            next_stream_id: Cell::new(1),
        })
    }

    fn detections(&self, transcripts: Vec<Transcript>) -> Vec<Detection> {
        transcripts
            .into_iter()
            .flat_map(|transcript| {
                record_transcript(&transcript.text);
                let tokens = normalize_tokens(&transcript.text);
                self.matcher
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
}

/// Construct the real pipeline and return device/cache evidence without saving config.
pub(crate) fn prepare(spec: ProviderSpec) -> Result<PlacementEvidence> {
    let mut worker = Worker::spawn(spec.validate()?)?;
    let evidence = worker.evidence.clone();
    worker.shutdown();
    Ok(evidence)
}

pub(crate) fn probe_runtime(config: &Config, paths: &AppPaths) -> Result<RuntimeEvidence> {
    let executable = env::current_exe().context("resolve Omawake executable")?;
    probe_runtime_with(config, paths, &executable)
}

fn probe_runtime_with(
    config: &Config,
    paths: &AppPaths,
    executable: &Path,
) -> Result<RuntimeEvidence> {
    let spec = ProviderSpec::from_config(config, paths)?;
    let mut command = Command::new(executable);
    command
        .arg("__openvino-runtime-worker")
        .arg(&spec.genai_library)
        .arg(&spec.core_library)
        .arg(&spec.audiocpp_library)
        .arg(&spec.device)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    prepend_library_directories(&mut command, &spec.library_dirs)?;
    let mut child = command
        .spawn()
        .context("spawn isolated OpenVINO runtime probe")?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("OpenVINO runtime probe timed out after 30 seconds");
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!("OpenVINO runtime probe failed: {}", output.status);
    }
    serde_json::from_slice(&output.stdout).context("read OpenVINO runtime probe evidence")
}

impl WakeWordBackend for OpenVinoGenAiBackend {
    fn kind(&self) -> &'static str {
        "openvino-genai"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        let id = self.next_stream_id.get();
        self.next_stream_id.set(id.wrapping_add(1).max(1));
        Box::new(OpenVinoGenAiStream {
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

impl OpenVinoGenAiStream<'_> {
    fn ensure_started(&self, state: &mut ClientStream) -> Result<()> {
        if !state.started {
            self.backend
                .worker
                .try_borrow_mut()
                .map_err(|_| anyhow::anyhow!("another OpenVINO stream is using the worker"))?
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
                    .map_err(|_| anyhow::anyhow!("another OpenVINO stream is using the worker"))?
                    .audio(state.id, &frame)?,
            );
        }
        Ok(self.backend.detections(transcripts))
    }
}

impl WakeWordStream for OpenVinoGenAiStream<'_> {
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
            .map_err(|_| anyhow::anyhow!("another OpenVINO stream is using the worker"))?
            .finish(state.id)?;
        detections.extend(self.backend.detections(transcripts));
        state.finished = true;
        Ok(detections)
    }
}

impl Drop for OpenVinoGenAiStream<'_> {
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

struct WorkerProcess {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl WorkerProcess {
    fn launch_with(spec: &ProviderSpec, executable: &Path) -> Result<(Self, PlacementEvidence)> {
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&spec.placement_log)
            .with_context(|| format!("open placement log {}", spec.placement_log.display()))?;
        let mut command = Command::new(executable);
        command
            .arg("__openvino-genai-worker")
            .arg(&spec.genai_library)
            .arg(&spec.core_library)
            .arg(&spec.audiocpp_library)
            .arg(&spec.model_directory)
            .arg(&spec.vad_model)
            .arg(&spec.cache_directory)
            .arg(&spec.device)
            .arg(spec.vad_threads.to_string())
            .arg(&spec.language)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log));
        prepend_library_directories(&mut command, &spec.library_dirs)?;
        let mut child = command
            .spawn()
            .context("spawn isolated OpenVINO GenAI worker")?;
        let input = child
            .stdin
            .take()
            .context("OpenVINO worker stdin is unavailable")?;
        let output = child
            .stdout
            .take()
            .context("OpenVINO worker stdout is unavailable")?;
        let mut process = Self {
            child,
            input: Some(input),
            output: BufReader::new(output),
        };
        let handshake = wait_for_io(
            process.output.get_ref().as_raw_fd(),
            libc::POLLIN,
            WORKER_STARTUP_TIMEOUT,
        )
        .context("wait for OpenVINO worker handshake")
        .and_then(|()| {
            protocol::read_response(&mut process.output).context("read OpenVINO worker handshake")
        });
        match handshake {
            Ok(Response::Ready { evidence }) => Ok((process, *evidence)),
            Ok(Response::Error { message, .. }) => {
                process.stop(false);
                bail!(message)
            }
            Ok(response) => {
                process.stop(false);
                bail!("unexpected OpenVINO worker handshake: {response:?}")
            }
            Err(error) => {
                process.stop(false);
                Err(error)
            }
        }
    }

    fn stop(&mut self, graceful: bool) {
        if graceful
            && let Some(input) = self.input.as_mut()
            && wait_for_io(input.as_raw_fd(), libc::POLLOUT, Duration::from_secs(1)).is_ok()
        {
            let _ = protocol::write_request(input, &Request::Shutdown, &[]);
        }
        self.input.take();
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

fn wait_for_io(fd: c_int, events: i16, timeout: Duration) -> io::Result<()> {
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
            if descriptor.revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "OpenVINO worker IPC failed",
                ));
            }
            return Ok(());
        }
        if status == 0 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "OpenVINO worker IPC timed out",
            ));
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

struct Worker {
    spec: ProviderSpec,
    executable: PathBuf,
    process: Option<WorkerProcess>,
    active_id: Option<u64>,
    evidence: PlacementEvidence,
}

impl Worker {
    fn spawn(spec: ProviderSpec) -> Result<Self> {
        let executable = env::current_exe().context("resolve Omawake executable")?;
        Self::spawn_with(spec, executable)
    }

    fn spawn_with(spec: ProviderSpec, executable: PathBuf) -> Result<Self> {
        let (process, evidence) = WorkerProcess::launch_with(&spec, &executable)?;
        Ok(Self {
            spec,
            executable,
            process: Some(process),
            active_id: None,
            evidence,
        })
    }

    fn ensure_process(&mut self) -> Result<()> {
        if self.process.is_none() {
            let (process, evidence) = WorkerProcess::launch_with(&self.spec, &self.executable)?;
            self.process = Some(process);
            self.evidence = evidence;
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
                .context("OpenVINO worker is unavailable")?;
            let input = process
                .input
                .as_mut()
                .context("OpenVINO worker stdin is closed")?;
            wait_for_io(input.as_raw_fd(), libc::POLLOUT, WORKER_REQUEST_TIMEOUT)
                .context("wait to write OpenVINO worker request")?;
            protocol::write_request(input, request, pcm)
                .context("write OpenVINO worker request")?;
            wait_for_io(
                process.output.get_ref().as_raw_fd(),
                libc::POLLIN,
                WORKER_REQUEST_TIMEOUT,
            )
            .context("wait for OpenVINO worker response")?;
            protocol::read_response(&mut process.output).context("read OpenVINO worker response")
        })();
        if result.is_err() {
            self.disconnect();
        }
        result
    }

    fn start(&mut self, id: u64) -> Result<()> {
        if let Some(active) = self.active_id {
            bail!("OpenVINO worker is already serving stream {active}");
        }
        let request = Request::Start { id };
        let response = match self.exchange(&request, &[]) {
            Ok(response) => response,
            Err(first_error) => self.exchange(&request, &[]).with_context(|| {
                format!("restart OpenVINO worker after startup IPC failure: {first_error:#}")
            })?,
        };
        match response {
            Response::Ack { id: response } if response == id => {
                self.active_id = Some(id);
                Ok(())
            }
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected OpenVINO start response: {response:?}"),
        }
    }

    fn audio(&mut self, id: u64, pcm: &[f32]) -> Result<Vec<Transcript>> {
        if self.active_id != Some(id) {
            bail!("OpenVINO worker is not serving stream {id}");
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
            response => bail!("unexpected OpenVINO audio response: {response:?}"),
        }
    }

    fn finish(&mut self, id: u64) -> Result<Vec<Transcript>> {
        if self.active_id != Some(id) {
            bail!("OpenVINO worker is not serving stream {id}");
        }
        let response = self.exchange(&Request::Finish { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Result {
                id: response,
                transcripts,
            } if response == id => Ok(transcripts),
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected OpenVINO finish response: {response:?}"),
        }
    }

    fn cancel(&mut self, id: u64) -> Result<()> {
        let response = self.exchange(&Request::Cancel { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Ack { id: response } if response == id => Ok(()),
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected OpenVINO cancel response: {response:?}"),
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

fn prepend_library_directories(command: &mut Command, directories: &[PathBuf]) -> Result<()> {
    let mut paths = directories.to_vec();
    if let Some(existing) = env::var_os("LD_LIBRARY_PATH") {
        for directory in env::split_paths(&existing) {
            if !paths.contains(&directory) {
                paths.push(directory);
            }
        }
    }
    let joined = env::join_paths(paths).context("construct OpenVINO worker library path")?;
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

type Status = c_int;

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    Ok(*unsafe { library.get::<T>(name) }.with_context(|| {
        format!(
            "resolve native symbol {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        )
    })?)
}

#[repr(C)]
struct OvVersion {
    build_number: *const c_char,
    description: *const c_char,
}

#[repr(C)]
struct OvAvailableDevices {
    devices: *mut *mut c_char,
    size: usize,
}

struct CoreApi {
    _library: Library,
    get_version: unsafe extern "C" fn(*mut OvVersion) -> Status,
    version_free: unsafe extern "C" fn(*mut OvVersion),
    core_create: unsafe extern "C" fn(*mut *mut c_void) -> Status,
    core_free: unsafe extern "C" fn(*mut c_void),
    available_devices: unsafe extern "C" fn(*const c_void, *mut OvAvailableDevices) -> Status,
    available_devices_free: unsafe extern "C" fn(*mut OvAvailableDevices),
    get_property: unsafe extern "C" fn(
        *const c_void,
        *const c_char,
        *const c_char,
        *mut *mut c_char,
    ) -> Status,
    free_string: unsafe extern "C" fn(*const c_char),
    error_info: unsafe extern "C" fn(Status) -> *const c_char,
    last_error: unsafe extern "C" fn() -> *const c_char,
}

impl CoreApi {
    fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load OpenVINO Runtime C library {}", path.display()))?;
        Ok(Self {
            get_version: unsafe { symbol(&library, b"ov_get_openvino_version\0")? },
            version_free: unsafe { symbol(&library, b"ov_version_free\0")? },
            core_create: unsafe { symbol(&library, b"ov_core_create\0")? },
            core_free: unsafe { symbol(&library, b"ov_core_free\0")? },
            available_devices: unsafe { symbol(&library, b"ov_core_get_available_devices\0")? },
            available_devices_free: unsafe { symbol(&library, b"ov_available_devices_free\0")? },
            get_property: unsafe { symbol(&library, b"ov_core_get_property\0")? },
            free_string: unsafe { symbol(&library, b"ov_free\0")? },
            error_info: unsafe { symbol(&library, b"ov_get_error_info\0")? },
            last_error: unsafe { symbol(&library, b"ov_get_last_err_msg\0")? },
            _library: library,
        })
    }

    fn check(&self, operation: &str, status: Status) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let name = unsafe { optional_c_string((self.error_info)(status)) };
        let detail = unsafe { optional_c_string((self.last_error)()) };
        bail!("{operation}: {name}: {detail} (status {status})")
    }

    fn inspect(&self, requested: &str) -> Result<DeviceEvidence> {
        let mut version = OvVersion {
            build_number: std::ptr::null(),
            description: std::ptr::null(),
        };
        self.check("query OpenVINO version", unsafe {
            (self.get_version)(&mut version)
        })?;
        let runtime_build = unsafe { optional_c_string(version.build_number) };
        let runtime_description = unsafe { optional_c_string(version.description) };
        unsafe { (self.version_free)(&mut version) };

        let mut core = std::ptr::null_mut();
        self.check("create OpenVINO Core", unsafe {
            (self.core_create)(&mut core)
        })?;
        let result = (|| {
            let mut devices = OvAvailableDevices {
                devices: std::ptr::null_mut(),
                size: 0,
            };
            self.check("query OpenVINO devices", unsafe {
                (self.available_devices)(core, &mut devices)
            })?;
            let names = if devices.devices.is_null() {
                Vec::new()
            } else {
                (0..devices.size)
                    .map(|index| unsafe { optional_c_string(*devices.devices.add(index)) })
                    .collect::<Vec<_>>()
            };
            unsafe { (self.available_devices_free)(&mut devices) };
            let available_device = names
                .iter()
                .find(|name| name.as_str() == requested)
                .or_else(|| {
                    names
                        .iter()
                        .find(|name| name.starts_with(&format!("{requested}.")))
                })
                .cloned()
                .with_context(|| {
                    format!(
                        "OpenVINO device {requested} is unavailable; found {}",
                        names.join(", ")
                    )
                })?;
            Ok(DeviceEvidence {
                runtime_build,
                runtime_description,
                full_device_name: self.property(core, &available_device, "FULL_DEVICE_NAME")?,
                device_architecture: self.property(
                    core,
                    &available_device,
                    "DEVICE_ARCHITECTURE",
                )?,
                driver_version: self.property_optional(core, &available_device, "DRIVER_VERSION"),
                available_device,
            })
        })();
        unsafe { (self.core_free)(core) };
        result
    }

    fn property(&self, core: *const c_void, device: &str, key: &str) -> Result<String> {
        let device = CString::new(device)?;
        let key = CString::new(key)?;
        let mut value = std::ptr::null_mut();
        self.check(
            &format!("query OpenVINO property {}", key.to_string_lossy()),
            unsafe { (self.get_property)(core, device.as_ptr(), key.as_ptr(), &mut value) },
        )?;
        if value.is_null() {
            bail!("OpenVINO returned a null property value");
        }
        let output = unsafe { optional_c_string(value) };
        unsafe { (self.free_string)(value) };
        Ok(output)
    }

    fn property_optional(&self, core: *const c_void, device: &str, key: &str) -> String {
        self.property(core, device, key).unwrap_or_default()
    }
}

struct DeviceEvidence {
    runtime_build: String,
    runtime_description: String,
    available_device: String,
    full_device_name: String,
    device_architecture: String,
    driver_version: String,
}

unsafe fn optional_c_string(value: *const c_char) -> String {
    if value.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned()
    }
}

type PipelineCreate =
    unsafe extern "C" fn(*const c_char, *const c_char, usize, *mut *mut c_void, ...) -> Status;

struct GenAiApi {
    _library: Library,
    pipeline_create: PipelineCreate,
    pipeline_free: unsafe extern "C" fn(*mut c_void),
    generate: unsafe extern "C" fn(
        *mut c_void,
        *const f32,
        usize,
        *const c_void,
        *mut *mut c_void,
    ) -> Status,
    result_string: unsafe extern "C" fn(*const c_void, *mut c_char, *mut usize) -> Status,
    result_free: unsafe extern "C" fn(*mut c_void),
    whisper_config_create_from_json:
        unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> Status,
    whisper_config_set_language: unsafe extern "C" fn(*mut c_void, *const c_char) -> Status,
    whisper_config_validate: unsafe extern "C" fn(*const c_void) -> Status,
    whisper_config_free: unsafe extern "C" fn(*mut c_void),
}

impl GenAiApi {
    fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load OpenVINO GenAI C library {}", path.display()))?;
        Ok(Self {
            pipeline_create: unsafe { symbol(&library, b"ov_genai_whisper_pipeline_create\0")? },
            pipeline_free: unsafe { symbol(&library, b"ov_genai_whisper_pipeline_free\0")? },
            generate: unsafe { symbol(&library, b"ov_genai_whisper_pipeline_generate\0")? },
            result_string: unsafe {
                symbol(&library, b"ov_genai_whisper_decoded_results_get_string\0")?
            },
            result_free: unsafe { symbol(&library, b"ov_genai_whisper_decoded_results_free\0")? },
            whisper_config_create_from_json: unsafe {
                symbol(
                    &library,
                    b"ov_genai_whisper_generation_config_create_from_json\0",
                )?
            },
            whisper_config_set_language: unsafe {
                symbol(
                    &library,
                    b"ov_genai_whisper_generation_config_set_language\0",
                )?
            },
            whisper_config_validate: unsafe {
                symbol(&library, b"ov_genai_whisper_generation_config_validate\0")?
            },
            whisper_config_free: unsafe {
                symbol(&library, b"ov_genai_whisper_generation_config_free\0")?
            },
            _library: library,
        })
    }
}

#[repr(C)]
struct AudioCppModelConfig {
    family_hint: *const c_char,
    config_id: *const c_char,
    weight_id: *const c_char,
    model_spec_override: *const c_char,
}

#[repr(C)]
struct AudioCppBackendConfig {
    backend: *const c_char,
    device: c_int,
    threads: c_int,
}

struct VadApi {
    _library: Library,
    last_error: unsafe extern "C" fn() -> *const c_char,
    registry_create: unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> Status,
    registry_free: unsafe extern "C" fn(*mut c_void),
    model_load: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *const AudioCppModelConfig,
        *const c_void,
        *mut *mut c_void,
    ) -> Status,
    model_free: unsafe extern "C" fn(*mut c_void),
    session_create: unsafe extern "C" fn(
        *const c_void,
        *const c_char,
        *const c_char,
        *const AudioCppBackendConfig,
        *const c_void,
        *mut *mut c_void,
    ) -> Status,
    session_free: unsafe extern "C" fn(*mut c_void),
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
    result_segment_count: unsafe extern "C" fn(*const c_void) -> usize,
    result_segment: unsafe extern "C" fn(
        *const c_void,
        usize,
        *mut i64,
        *mut i64,
        *mut f32,
        *mut *const c_char,
    ) -> Status,
}

impl VadApi {
    fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load audio.cpp C library {}", path.display()))?;
        let abi_version: unsafe extern "C" fn() -> u32 =
            unsafe { symbol(&library, b"audiocpp_abi_version\0")? };
        let found = unsafe { abi_version() };
        if found != AUDIOCPP_ABI_0_1_0 {
            bail!("audio.cpp C ABI mismatch: expected 0.1.0 ({AUDIOCPP_ABI_0_1_0}), found {found}");
        }
        Ok(Self {
            last_error: unsafe { symbol(&library, b"audiocpp_last_error\0")? },
            registry_create: unsafe { symbol(&library, b"audiocpp_registry_create\0")? },
            registry_free: unsafe { symbol(&library, b"audiocpp_registry_free\0")? },
            model_load: unsafe { symbol(&library, b"audiocpp_model_load\0")? },
            model_free: unsafe { symbol(&library, b"audiocpp_model_free\0")? },
            session_create: unsafe { symbol(&library, b"audiocpp_session_create\0")? },
            session_free: unsafe { symbol(&library, b"audiocpp_session_free\0")? },
            stream_start: unsafe { symbol(&library, b"audiocpp_stream_start\0")? },
            stream_push: unsafe { symbol(&library, b"audiocpp_stream_push\0")? },
            stream_reset: unsafe { symbol(&library, b"audiocpp_stream_reset\0")? },
            event_free: unsafe { symbol(&library, b"audiocpp_event_free\0")? },
            event_as_result: unsafe { symbol(&library, b"audiocpp_event_as_result\0")? },
            event_voice_activity_count: unsafe {
                symbol(&library, b"audiocpp_event_voice_activity_count\0")?
            },
            event_voice_activity: unsafe { symbol(&library, b"audiocpp_event_voice_activity\0")? },
            result_segment_count: unsafe { symbol(&library, b"audiocpp_result_segment_count\0")? },
            result_segment: unsafe { symbol(&library, b"audiocpp_result_segment\0")? },
            _library: library,
        })
    }

    fn check(&self, operation: &str, status: Status) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let detail = unsafe { optional_c_string((self.last_error)()) };
        bail!("{operation}: {detail} (status {status})")
    }
}

struct AudioCppVad {
    api: VadApi,
    registry: *mut c_void,
    model: *mut c_void,
    session: *mut c_void,
    cursor: i64,
}

impl AudioCppVad {
    fn open(library: &Path, model: &Path, threads: i32) -> Result<Self> {
        let api = VadApi::load(library)?;
        let mut vad = Self {
            api,
            registry: std::ptr::null_mut(),
            model: std::ptr::null_mut(),
            session: std::ptr::null_mut(),
            cursor: 0,
        };
        vad.api.check("create audio.cpp registry", unsafe {
            (vad.api.registry_create)(std::ptr::null(), &mut vad.registry)
        })?;
        let path = path_to_c_string(model)?;
        let family = CString::new("silero_vad")?;
        let config = AudioCppModelConfig {
            family_hint: family.as_ptr(),
            config_id: std::ptr::null(),
            weight_id: std::ptr::null(),
            model_spec_override: std::ptr::null(),
        };
        vad.api.check("load Silero VAD", unsafe {
            (vad.api.model_load)(
                vad.registry,
                path.as_ptr(),
                &config,
                std::ptr::null(),
                &mut vad.model,
            )
        })?;
        let task = CString::new("vad")?;
        let mode = CString::new("streaming")?;
        let backend_name = CString::new("cpu")?;
        let backend = AudioCppBackendConfig {
            backend: backend_name.as_ptr(),
            device: 0,
            threads,
        };
        vad.api.check("create Silero VAD session", unsafe {
            (vad.api.session_create)(
                vad.model,
                task.as_ptr(),
                mode.as_ptr(),
                &backend,
                std::ptr::null(),
                &mut vad.session,
            )
        })?;
        vad.api.check("start Silero VAD stream", unsafe {
            (vad.api.stream_start)(vad.session, std::ptr::null())
        })?;
        Ok(vad)
    }

    fn restart(&mut self) -> Result<()> {
        self.cursor = 0;
        self.api.check("reset Silero VAD stream", unsafe {
            (self.api.stream_reset)(self.session)
        })?;
        self.api.check("start Silero VAD stream", unsafe {
            (self.api.stream_start)(self.session, std::ptr::null())
        })
    }

    fn activity(&mut self, samples: &[f32]) -> Result<Activity> {
        if samples.len() != FRAME_SAMPLES {
            bail!("Silero VAD requires exactly {FRAME_SAMPLES} samples per frame");
        }
        let frame_end = self
            .cursor
            .checked_add(FRAME_SAMPLES as i64)
            .context("VAD sample cursor overflow")?;
        let mut event = std::ptr::null_mut();
        self.api.check("run Silero VAD frame", unsafe {
            (self.api.stream_push)(
                self.session,
                samples.as_ptr(),
                samples.len(),
                16_000,
                1,
                self.cursor,
                &mut event,
            )
        })?;
        self.cursor = frame_end;
        if event.is_null() {
            return Ok(Activity::default());
        }
        let result = (|| {
            let mut activity = Activity::default();
            let count = unsafe { (self.api.event_voice_activity_count)(event) };
            for index in 0..count {
                let mut kind = -1;
                let mut sample = 0_i64;
                let mut probability = 0.0;
                self.api.check("read Silero VAD event", unsafe {
                    (self.api.event_voice_activity)(
                        event,
                        index,
                        &mut kind,
                        &mut sample,
                        &mut probability,
                    )
                })?;
                if !probability.is_finite()
                    || !(0.0..=1.0).contains(&probability)
                    || sample < 0
                    || sample > frame_end
                {
                    bail!("Silero VAD returned an invalid activity event");
                }
                let before_end =
                    usize::try_from(frame_end - sample).context("VAD timestamp is too large")?;
                match kind {
                    0 => activity.start_before_frame_end = Some(before_end),
                    1 => activity.end_before_frame_end = Some(before_end),
                    2 => {
                        let view = unsafe { (self.api.event_as_result)(event) };
                        if view.is_null() || unsafe { (self.api.result_segment_count)(view) } == 0 {
                            bail!("Silero segment event omitted its bounds");
                        }
                        let mut start = 0_i64;
                        let mut end = 0_i64;
                        self.api.check("read Silero segment bounds", unsafe {
                            (self.api.result_segment)(
                                view,
                                0,
                                &mut start,
                                &mut end,
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                            )
                        })?;
                        if start < 0 || end < start || end > frame_end {
                            bail!("Silero segment bounds are invalid");
                        }
                        activity.start_before_frame_end = Some(usize::try_from(frame_end - start)?);
                        activity.end_before_frame_end = Some(usize::try_from(frame_end - end)?);
                    }
                    _ => bail!("Silero VAD returned unknown activity kind {kind}"),
                }
            }
            Ok(activity)
        })();
        unsafe { (self.api.event_free)(event) };
        result
    }
}

impl Drop for AudioCppVad {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                (self.api.session_free)(self.session);
            }
            if !self.model.is_null() {
                (self.api.model_free)(self.model);
            }
            if !self.registry.is_null() {
                (self.api.registry_free)(self.registry);
            }
        }
    }
}

struct OpenVinoProvider {
    core: CoreApi,
    genai: GenAiApi,
    pipeline: *mut c_void,
    vad: AudioCppVad,
    evidence: PlacementEvidence,
    model_directory: PathBuf,
}

impl OpenVinoProvider {
    /// Build the optional generation config that pins the configured
    /// language for every generate call; `None` follows the model default
    /// (auto-detection).
    fn language_config(&self) -> Result<Option<GenAiConfigGuard<'_>>> {
        if self.evidence.language.is_empty() {
            return Ok(None);
        }
        Ok(Some(GenAiConfigGuard {
            api: &self.genai,
            config: create_language_config(
                &self.core,
                &self.genai,
                &self.model_directory,
                &self.evidence.language,
            )?,
        }))
    }
}

/// Build a validated generation config carrying the explicit language.
///
/// The pinned model's `generation_config.json` provides `lang_to_id` and
/// `is_multilingual` (the JSON constructor ignores `language`, so the value
/// cannot be delivered through a file); `set_language` then receives the
/// wrapped Whisper token (`"<|es|>"`) because the pinned runtime's
/// `validate()` looks the language up verbatim in `lang_to_id` — plain
/// two-letter codes are only normalized in runtime builds that include
/// openvinotoolkit/openvino.genai#4258.
fn create_language_config(
    core: &CoreApi,
    api: &GenAiApi,
    model_directory: &Path,
    language: &str,
) -> Result<*mut c_void> {
    let pinned_config = model_directory.join("generation_config.json");
    let path = CString::new(pinned_config.as_os_str().as_encoded_bytes())
        .context("pinned generation config path contains a NUL byte")?;
    let mut config: *mut c_void = std::ptr::null_mut();
    core.check("create OpenVINO Whisper generation config", unsafe {
        (api.whisper_config_create_from_json)(path.as_ptr(), &mut config)
    })?;
    if config.is_null() {
        bail!("OpenVINO GenAI returned a null Whisper generation config");
    }
    let token = format!("<|{language}|>");
    let token_c = CString::new(token.as_str()).context("language token contains a NUL byte")?;
    core.check("set OpenVINO Whisper generation language", unsafe {
        (api.whisper_config_set_language)(config, token_c.as_ptr())
    })?;
    core.check("validate OpenVINO Whisper generation config", unsafe {
        (api.whisper_config_validate)(config)
    })?;
    Ok(config)
}

struct GenAiConfigGuard<'a> {
    api: &'a GenAiApi,
    config: *mut c_void,
}

impl Drop for GenAiConfigGuard<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.whisper_config_free)(self.config) };
    }
}

impl OpenVinoProvider {
    fn open(spec: &ProviderSpec) -> Result<Self> {
        let core = CoreApi::load(&spec.core_library)?;
        let device = core.inspect(&spec.device)?;
        let genai = GenAiApi::load(&spec.genai_library)?;
        let model = path_to_c_string(&spec.model_directory)?;
        let requested = CString::new(spec.device.as_str())?;
        let cache_key = CString::new("CACHE_DIR")?;
        let cache_value = path_to_c_string(&spec.cache_directory)?;
        let static_key = CString::new("STATIC_PIPELINE")?;
        let true_value = CString::new("true")?;
        let mut pipeline = std::ptr::null_mut();
        let pipeline_started = Instant::now();
        let status = unsafe {
            if spec.static_pipeline() {
                (genai.pipeline_create)(
                    model.as_ptr(),
                    requested.as_ptr(),
                    4,
                    &mut pipeline,
                    cache_key.as_ptr(),
                    cache_value.as_ptr(),
                    static_key.as_ptr(),
                    true_value.as_ptr(),
                )
            } else {
                (genai.pipeline_create)(
                    model.as_ptr(),
                    requested.as_ptr(),
                    2,
                    &mut pipeline,
                    cache_key.as_ptr(),
                    cache_value.as_ptr(),
                )
            }
        };
        core.check("create OpenVINO GenAI Whisper pipeline", status)?;
        if pipeline.is_null() {
            bail!("OpenVINO GenAI returned a null Whisper pipeline");
        }
        let pipeline_load_milliseconds = pipeline_started.elapsed().as_secs_f64() * 1_000.0;
        let (cache_files, cache_bytes) = cache_artifacts(&spec.cache_directory)?;
        if spec.device != "CPU" && cache_files == 0 {
            unsafe { (genai.pipeline_free)(pipeline) };
            bail!(
                "OpenVINO {} pipeline compiled without creating cache artifacts",
                spec.device
            );
        }
        let vad = match AudioCppVad::open(
            &spec.audiocpp_library,
            &spec.vad_model,
            i32::from(spec.vad_threads),
        ) {
            Ok(vad) => vad,
            Err(error) => {
                unsafe { (genai.pipeline_free)(pipeline) };
                return Err(error);
            }
        };
        let evidence = PlacementEvidence {
            profile_id: spec.profile.id.into(),
            languages: spec
                .profile
                .languages
                .iter()
                .map(|value| (*value).into())
                .collect(),
            language: spec.language.clone(),
            multilingual: WHISPER_BASE_EN_PROFILE.multilingual,
            runtime_build: device.runtime_build,
            runtime_description: device.runtime_description,
            requested_device: spec.device.clone(),
            available_device: device.available_device,
            full_device_name: device.full_device_name,
            device_architecture: device.device_architecture,
            driver_version: device.driver_version,
            static_pipeline: spec.static_pipeline(),
            pipeline_load_milliseconds,
            cache_directory: spec.cache_directory.display().to_string(),
            cache_files,
            cache_bytes,
            genai_library: spec.genai_library.display().to_string(),
            core_library: spec.core_library.display().to_string(),
        };
        Ok(Self {
            core,
            genai,
            pipeline,
            vad,
            evidence,
            model_directory: spec.model_directory.clone(),
        })
    }

    fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() || samples.len() > 16_000 * 30 {
            bail!("OpenVINO Whisper utterance must contain 1 to 480000 samples");
        }
        let mut results = std::ptr::null_mut();
        let config = self.language_config()?;
        let config_pointer = config
            .as_ref()
            .map_or(std::ptr::null(), |guard| guard.config.cast_const());
        self.core.check("run OpenVINO GenAI Whisper", unsafe {
            (self.genai.generate)(
                self.pipeline,
                samples.as_ptr(),
                samples.len(),
                config_pointer,
                &mut results,
            )
        })?;
        if results.is_null() {
            bail!("OpenVINO GenAI returned null Whisper results");
        }
        let result = (|| {
            let mut size = 0;
            self.core.check("size OpenVINO transcript", unsafe {
                (self.genai.result_string)(results, std::ptr::null_mut(), &mut size)
            })?;
            if size == 0 || size > 4 * 1024 * 1024 {
                bail!("OpenVINO returned an invalid transcript size {size}");
            }
            let mut text = vec![0_u8; size];
            self.core.check("read OpenVINO transcript", unsafe {
                (self.genai.result_string)(results, text.as_mut_ptr().cast(), &mut size)
            })?;
            let text = CStr::from_bytes_until_nul(&text)
                .context("OpenVINO transcript is not NUL terminated")?;
            Ok(text.to_string_lossy().trim().to_owned())
        })();
        unsafe { (self.genai.result_free)(results) };
        result
    }

    fn verify_utterance(&self, samples: &[f32]) -> Result<Option<String>> {
        let transcript = self.transcribe(samples)?;
        // Deliberate extension point: an enrollment-based speaker verifier belongs
        // here, after VAD/ASR and before the transcript reaches phrase acceptance.
        Ok(Some(transcript))
    }
}

impl Drop for OpenVinoProvider {
    fn drop(&mut self) {
        if !self.pipeline.is_null() {
            unsafe { (self.genai.pipeline_free)(self.pipeline) };
            self.pipeline = std::ptr::null_mut();
        }
    }
}

fn cache_artifacts(directory: &Path) -> Result<(usize, u64)> {
    fn walk(directory: &Path, count: &mut usize, bytes: &mut u64) -> Result<()> {
        for entry in fs::read_dir(directory)
            .with_context(|| format!("read cache {}", directory.display()))?
        {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                walk(&entry.path(), count, bytes)?;
            } else if metadata.is_file()
                && metadata.len() > 0
                && entry.file_name() != "placement.log"
            {
                *count += 1;
                *bytes = bytes.saturating_add(metadata.len());
            }
        }
        Ok(())
    }
    let mut count = 0;
    let mut bytes = 0;
    walk(directory, &mut count, &mut bytes)?;
    Ok((count, bytes))
}

fn path_to_c_string(path: &Path) -> Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes()).context("native path contains NUL")
}

struct WorkerStream {
    id: u64,
    endpoint: ActivityBuffer,
}

impl WorkerStream {
    fn finish_utterance(
        &self,
        provider: &OpenVinoProvider,
        utterance: Utterance,
    ) -> Result<Transcript> {
        Ok(Transcript {
            text: provider
                .verify_utterance(&utterance.samples)?
                .unwrap_or_default(),
            start_sample: utterance.start_sample,
            end_sample: utterance.end_sample,
        })
    }
}

pub(crate) fn runtime_worker_main(
    genai_library: &Path,
    core_library: &Path,
    audiocpp_library: &Path,
    requested_device: &str,
) -> Result<()> {
    harden_worker_process()?;
    runtime_worker_main_io(
        genai_library,
        core_library,
        audiocpp_library,
        requested_device,
        &mut std::io::stdout().lock(),
    )
}

fn runtime_worker_main_io(
    genai_library: &Path,
    core_library: &Path,
    audiocpp_library: &Path,
    requested_device: &str,
    output: &mut impl io::Write,
) -> Result<()> {
    let requested_device = canonical_device(requested_device)?;
    let core = CoreApi::load(core_library)?;
    let device = core.inspect(requested_device)?;
    let _genai = GenAiApi::load(genai_library)?;
    let _audiocpp = VadApi::load(audiocpp_library)?;
    serde_json::to_writer(
        output,
        &RuntimeEvidence {
            runtime_build: device.runtime_build,
            runtime_description: device.runtime_description,
            requested_device: requested_device.into(),
            available_device: device.available_device,
            full_device_name: device.full_device_name,
            device_architecture: device.device_architecture,
            driver_version: device.driver_version,
            genai_library: genai_library.display().to_string(),
            core_library: core_library.display().to_string(),
            audiocpp_library: audiocpp_library.display().to_string(),
        },
    )?;
    Ok(())
}

pub(crate) fn worker_main(spec: ProviderSpec) -> Result<()> {
    worker_main_io(
        spec,
        &mut std::io::stdin().lock(),
        &mut std::io::stdout().lock(),
    )
}

fn worker_main_io(
    spec: ProviderSpec,
    input: &mut impl io::Read,
    output: &mut impl io::Write,
) -> Result<()> {
    if let Err(error) = harden_worker_process() {
        protocol::write_response(
            output,
            &Response::Error {
                id: None,
                message: format!("{error:#}"),
            },
        )?;
        return Ok(());
    }
    let spec = match spec.validate() {
        Ok(spec) => spec,
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
    let mut provider = match OpenVinoProvider::open(&spec) {
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
    if let Ok(mut evidence_log) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&spec.placement_log)
    {
        let _ = serde_json::to_writer(&mut evidence_log, &provider.evidence);
        use std::io::Write;
        let _ = evidence_log.write_all(b"\n");
    }
    worker_loop(&mut provider, input, output)
}

fn worker_loop(
    provider: &mut OpenVinoProvider,
    input: &mut impl io::Read,
    output: &mut impl io::Write,
) -> Result<()> {
    protocol::write_response(
        output,
        &Response::Ready {
            evidence: Box::new(provider.evidence.clone()),
        },
    )?;
    let mut active: Option<WorkerStream> = None;
    loop {
        let (request, pcm) = protocol::read_request(input)?;
        let response = match request {
            Request::Shutdown => return Ok(()),
            Request::Start { id } => match provider.vad.restart() {
                Ok(()) => {
                    active = Some(WorkerStream {
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
            Request::Audio { id, .. } => match active.as_mut() {
                Some(stream) if stream.id == id && pcm.len() == FRAME_SAMPLES => {
                    let result = (|| {
                        if pcm.iter().any(|sample| !sample.is_finite()) {
                            bail!("audio contains a non-finite sample");
                        }
                        let activity = provider.vad.activity(&pcm)?;
                        let transcript = stream
                            .endpoint
                            .push(&pcm, activity)
                            .map(|utterance| stream.finish_utterance(provider, utterance))
                            .transpose()?;
                        if transcript.is_some() {
                            provider.vad.restart()?;
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
            Request::Finish { id } => match active.take() {
                Some(mut stream) if stream.id == id => {
                    let result = stream
                        .endpoint
                        .finish()
                        .map(|utterance| stream.finish_utterance(provider, utterance))
                        .transpose()
                        .and_then(|transcript| {
                            provider.vad.restart()?;
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
                Some(stream) => {
                    active = Some(stream);
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
            Request::Cancel { id } => match active.as_ref() {
                Some(stream) if stream.id == id => {
                    active = None;
                    match provider.vad.restart() {
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
mod tests {
    use super::*;
    use std::process::Command;
    use std::sync::OnceLock;

    const FAKE_OPENVINO_C: &str = r#"
    #include <stdarg.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>
    #ifndef FAKE_MODE
    #define FAKE_MODE 0
    #endif

    typedef struct { const char *build_number; const char *description; } OvVersion;
    typedef struct { char **devices; size_t size; } OvAvailableDevices;
    static char cpu[] = "CPU";
    static char gpu[] = "GPU.0";
    static char npu[] = "NPU";
    static char *device_names[] = { cpu, gpu, npu };
    static const char transcript[] = "  hello oma  ";

    int ov_get_openvino_version(OvVersion *out) {
    if (FAKE_MODE == 1) return 7;
    out->build_number = "2026.3.fake"; out->description = "safe fake OpenVINO"; return 0;
    }
    void ov_version_free(OvVersion *version) { (void) version; }
    int ov_core_create(void **out) { if (FAKE_MODE == 2) return 7; *out = malloc(1); return *out ? 0 : 1; }
    void ov_core_free(void *core) { free(core); }
    int ov_core_get_available_devices(const void *core, OvAvailableDevices *out) {
    if (FAKE_MODE == 3) return 7;
    (void) core; out->devices = device_names; out->size = 3; return 0;
    }
    void ov_available_devices_free(OvAvailableDevices *devices) { (void) devices; }
    int ov_core_get_property(const void *core, const char *device, const char *key, char **out) {
    (void) core; (void) device;
    if (FAKE_MODE == 4) return 7;
    if (FAKE_MODE == 5) { *out = 0; return 0; }
    const char *value = strcmp(key, "FULL_DEVICE_NAME") == 0 ? "Safe Fake Device" :
                        strcmp(key, "DEVICE_ARCHITECTURE") == 0 ? "fake-arch" :
                        strcmp(key, "DRIVER_VERSION") == 0 ? "fake-driver" : "fake-value";
    *out = strdup(value); return *out ? 0 : 1;
    }
    void ov_free(const char *value) { free((void *) value); }
    const char *ov_get_error_info(int status) { (void) status; return "FAKE_STATUS"; }
    const char *ov_get_last_err_msg(void) { return "controlled fake OpenVINO error"; }

    int ov_genai_whisper_pipeline_create(const char *model, const char *device,
                                     size_t properties, void **out, ...) {
    (void) model; (void) device; (void) properties;
    if (FAKE_MODE == 6) return 7;
    if (FAKE_MODE == 7) { *out = 0; return 0; }
    *out = malloc(1); return *out ? 0 : 1;
    }
    void ov_genai_whisper_pipeline_free(void *pipeline) { free(pipeline); }
    int ov_genai_whisper_pipeline_generate(void *pipeline, const float *pcm, size_t count,
                                       const void *config, void **out) {
    (void) pipeline; (void) pcm; (void) count; (void) config;
    if (FAKE_MODE == 8) return 7;
    if (FAKE_MODE == 9) { *out = 0; return 0; }
    *out = malloc(1); return *out ? 0 : 1;
    }
    int ov_genai_whisper_decoded_results_get_string(const void *results, char *out, size_t *size) {
    (void) results;
    if (!out) {
        if (FAKE_MODE == 10) { *size = 0; return 0; }
        if (FAKE_MODE == 11) { *size = 5 * 1024 * 1024; return 0; }
        *size = sizeof(transcript); return 0;
    }
    if (FAKE_MODE == 12) return 7;
    if (FAKE_MODE == 13) { memset(out, 'x', *size); return 0; }
    memcpy(out, transcript, sizeof(transcript)); *size = sizeof(transcript); return 0;
    }
    void ov_genai_whisper_decoded_results_free(void *results) { free(results); }
    int ov_genai_whisper_generation_config_create_from_json(const char *path, void **config) {
        (void)path; *config = malloc(16); return FAKE_MODE == 14 ? 5 : 0;
    }
    int ov_genai_whisper_generation_config_set_language(void *config, const char *language) {
        (void)config; (void)language; return 0;
    }
    int ov_genai_whisper_generation_config_validate(void *config) { (void)config; return 0; }
    void ov_genai_whisper_generation_config_free(void *config) { free(config); }
    "#;

    fn fake_openvino_library() -> &'static Path {
        static LIBRARY: OnceLock<PathBuf> = OnceLock::new();
        LIBRARY
            .get_or_init(|| {
                let root = env::temp_dir()
                    .join(format!("omawake-safe-fake-openvino-{}", std::process::id()));
                fs::create_dir_all(&root).unwrap();
                let source = root.join("fake_openvino.c");
                let library = root.join("libopenvino_fake.so");
                fs::write(&source, FAKE_OPENVINO_C).unwrap();
                let output = Command::new("cc")
                    .args(["-shared", "-fPIC", "-O0"])
                    .arg(&source)
                    .arg("-o")
                    .arg(&library)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "compile fake OpenVINO DSO: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                library
            })
            .as_path()
    }

    fn fake_openvino_variant(mode: u8) -> PathBuf {
        let root = env::temp_dir().join(format!(
            "omawake-safe-fake-openvino-variant-{}-{mode}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("fake_openvino.c");
        let library = root.join("libopenvino_fake.so");
        fs::write(&source, FAKE_OPENVINO_C).unwrap();
        let output = Command::new("cc")
            .args(["-shared", "-fPIC", "-O0"])
            .arg(format!("-DFAKE_MODE={mode}"))
            .arg(&source)
            .arg("-o")
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "compile fake OpenVINO variant: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        library
    }

    fn fake_spec(name: &str, device: &str) -> (PathBuf, ProviderSpec) {
        let root = temporary(name);
        let _ = fs::remove_dir_all(&root);
        let model_directory = root.join("model");
        let cache_directory = root.join("cache");
        fs::create_dir_all(&model_directory).unwrap();
        fs::create_dir_all(&cache_directory).unwrap();
        let vad_model = root.join("vad.safetensors");
        fs::write(&vad_model, b"safe fake vad").unwrap();
        if device != "CPU" {
            fs::write(cache_directory.join("compiled.blob"), b"fake cache").unwrap();
        }
        (
            root.clone(),
            ProviderSpec {
                profile: WHISPER_BASE_EN_PROFILE,
                language: String::new(),
                genai_library: fake_openvino_library().to_path_buf(),
                core_library: fake_openvino_library().to_path_buf(),
                audiocpp_library: super::super::audiocpp::tests::fake_library().to_path_buf(),
                library_dirs: vec![fake_openvino_library().parent().unwrap().to_path_buf()],
                model_directory,
                vad_model,
                cache_directory: cache_directory.clone(),
                placement_log: cache_directory.join("placement.log"),
                device: device.into(),
                vad_threads: 2,
            },
        )
    }

    fn fake_runtime_config(name: &str) -> (PathBuf, AppPaths, Config) {
        let root = temporary(name);
        let _ = fs::remove_dir_all(&root);
        let runtime = root.join("runtime");
        fs::create_dir_all(&runtime).unwrap();
        let genai = runtime.join("libopenvino_genai_c.so");
        fs::copy(fake_openvino_library(), &genai).unwrap();
        fs::copy(fake_openvino_library(), runtime.join("libopenvino_c.so")).unwrap();
        fs::copy(
            super::super::audiocpp::tests::fake_library(),
            runtime.join("libaudiocpp.so.0.1.0"),
        )
        .unwrap();
        for plugin in [
            "libopenvino_intel_cpu_plugin.so",
            "libopenvino_intel_gpu_plugin.so",
            "libopenvino_intel_npu_plugin.so",
            "libopenvino_intel_npu_compiler_loader.so",
            "libopenvino_intel_npu_compiler.so",
        ] {
            fs::write(runtime.join(plugin), b"present").unwrap();
        }
        let paths = AppPaths {
            config_file: root.join("config/config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
            state_dir: root.join("state"),
            runtime_dir: root.join("run"),
        };
        fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
        let mut config = Config::default();
        config.backend.kind = "openvino-genai".into();
        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "cpu".into();
        config.backend.library = genai;
        config.backend.library_dirs = vec![runtime];
        (root, paths, config)
    }

    fn temporary(name: &str) -> PathBuf {
        env::temp_dir().join(format!("omawake-openvino-{name}-{}", std::process::id()))
    }

    #[test]
    fn only_explicit_physical_devices_are_accepted() {
        assert_eq!(canonical_device(" cpu ").unwrap(), "CPU");
        assert_eq!(canonical_device("gpu").unwrap(), "GPU");
        assert_eq!(canonical_device("NPU").unwrap(), "NPU");
        assert!(canonical_device("AUTO").is_err());
        assert!(canonical_device("cuda").is_err());
    }

    #[test]
    fn cache_inventory_counts_nested_regular_files() {
        let root = temporary("cache");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("one.blob"), [1, 2, 3]).unwrap();
        fs::write(root.join("nested/two.bin"), [4, 5]).unwrap();
        assert_eq!(cache_artifacts(&root).unwrap(), (2, 5));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_rejects_missing_files_before_native_loading() {
        let root = temporary("manifest");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let error = verify_model_manifest(WHISPER_BASE_EN_PROFILE.catalog_id, &root).unwrap_err();
        assert!(error.to_string().contains("config.json"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn phrase_matching_uses_provider_independent_matcher() {
        let matcher = PhraseMatcher::compile(&[WakeWord {
            id: "lights".into(),
            phrase: "light up".into(),
            aliases: Vec::new(),
            enabled: true,
            command: vec!["true".into()],
        }])
        .unwrap();
        let transcripts = vec![Transcript {
            text: "Please light up the room.".into(),
            start_sample: 8_000,
            end_sample: 16_000,
        }];
        let detections: Vec<_> = transcripts
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
            .collect();
        assert_eq!(detections[0].id, "lights");
        assert_eq!(detections[0].tokens, ["light", "up"]);
        assert_eq!(detections[0].start_time, 0.5);
    }

    #[test]
    fn fake_core_dso_reports_devices_and_properties() {
        let core = CoreApi::load(fake_openvino_library()).unwrap();
        for (requested, available) in [("CPU", "CPU"), ("GPU", "GPU.0"), ("NPU", "NPU")] {
            let evidence = core.inspect(requested).unwrap();
            assert_eq!(evidence.available_device, available);
            assert_eq!(evidence.runtime_build, "2026.3.fake");
            assert_eq!(evidence.runtime_description, "safe fake OpenVINO");
            assert_eq!(evidence.full_device_name, "Safe Fake Device");
            assert_eq!(evidence.device_architecture, "fake-arch");
            assert_eq!(evidence.driver_version, "fake-driver");
        }
        assert!(
            core.inspect("GNA")
                .err()
                .unwrap()
                .to_string()
                .contains("unavailable")
        );
        assert!(
            core.check("controlled", 9)
                .unwrap_err()
                .to_string()
                .contains("FAKE_STATUS")
        );
        assert_eq!(unsafe { optional_c_string(std::ptr::null()) }, "");
    }

    #[test]
    fn fake_core_variants_cover_safe_runtime_error_cleanup() {
        for mode in 1..=5 {
            let library = fake_openvino_variant(mode);
            let core = CoreApi::load(&library).unwrap();
            let error = core.inspect("CPU").err().unwrap();
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn runtime_probe_and_worker_evidence_use_isolated_valid_processes() {
        let mut output = Vec::new();
        runtime_worker_main_io(
            fake_openvino_library(),
            fake_openvino_library(),
            super::super::audiocpp::tests::fake_library(),
            "gpu",
            &mut output,
        )
        .unwrap();
        let evidence: RuntimeEvidence = serde_json::from_slice(&output).unwrap();
        assert_eq!(evidence.requested_device, "GPU");
        assert_eq!(evidence.available_device, "GPU.0");
        assert!(
            runtime_worker_main_io(
                fake_openvino_library(),
                fake_openvino_library(),
                super::super::audiocpp::tests::fake_library(),
                "AUTO",
                &mut Vec::new(),
            )
            .is_err()
        );

        let (root, paths, config) = fake_runtime_config("runtime-probe");
        let evidence = probe_runtime_with(
            &config,
            &paths,
            super::super::audiocpp::tests::fake_worker(),
        )
        .unwrap();
        assert_eq!(evidence.runtime_description, "safe fake runtime");
        assert!(probe_runtime_with(&config, &paths, Path::new("/bin/false")).is_err());
        assert!(
            probe_runtime_with(
                &config,
                &paths,
                Path::new("/definitely/missing/omawake-worker")
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fake_openvino_provider_runs_genai_and_vad_on_cpu_and_npu() {
        for device in ["CPU", "NPU"] {
            let (root, spec) = fake_spec(&format!("provider-{device}"), device);
            let mut provider = OpenVinoProvider::open(&spec).unwrap();
            assert_eq!(provider.evidence.requested_device, device);
            assert_eq!(provider.evidence.static_pipeline, device == "NPU");
            assert_eq!(provider.transcribe(&[0.25; 32]).unwrap(), "hello oma");
            assert_eq!(
                provider.verify_utterance(&[0.25; 32]).unwrap().as_deref(),
                Some("hello oma")
            );
            assert!(provider.transcribe(&[]).is_err());
            assert!(provider.transcribe(&vec![0.0; 16_000 * 30 + 1]).is_err());

            let mut frame = vec![0.0; FRAME_SAMPLES];
            assert_eq!(
                provider
                    .vad
                    .activity(&frame)
                    .unwrap()
                    .start_before_frame_end,
                None
            );
            frame[0] = 1.0;
            assert_eq!(
                provider
                    .vad
                    .activity(&frame)
                    .unwrap()
                    .start_before_frame_end,
                Some(0)
            );
            frame[0] = -1.0;
            assert_eq!(
                provider.vad.activity(&frame).unwrap().end_before_frame_end,
                Some(0)
            );
            provider.vad.restart().unwrap();
            frame[0] = 2.0;
            let segment = provider.vad.activity(&frame).unwrap();
            assert_eq!(segment.start_before_frame_end, Some(FRAME_SAMPLES));
            assert_eq!(segment.end_before_frame_end, Some(0));
            frame[0] = 3.0;
            assert!(
                provider
                    .vad
                    .activity(&frame)
                    .unwrap_err()
                    .to_string()
                    .contains("invalid")
            );
            frame[0] = 4.0;
            assert!(
                provider
                    .vad
                    .activity(&frame)
                    .unwrap_err()
                    .to_string()
                    .contains("unknown")
            );
            assert!(provider.vad.activity(&frame[..10]).is_err());
            provider.vad.cursor = i64::MAX;
            assert!(provider.vad.activity(&vec![0.0; FRAME_SAMPLES]).is_err());
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn language_config_requires_the_pinned_generation_metadata_and_validates() {
        let (root, mut spec) = fake_spec("language-missing-metadata", "CPU");
        spec.language = "es".into();
        let provider = OpenVinoProvider::open(&spec).unwrap();
        assert_eq!(provider.evidence.language, "es");
        // A failing create_from_json surfaces as a controlled error.
        spec.genai_library = fake_openvino_variant(14).to_path_buf();
        let failing = OpenVinoProvider::open(&spec).unwrap();
        let error = failing
            .transcribe(&vec![0.0_f32; 1600])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("create OpenVINO Whisper generation config"),
            "{error}"
        );
        // With the pinned metadata present, the config is created from the
        // model's generation_config.json, the wrapped language token is set,
        // the config validates, and no file is ever written.
        fs::write(
            spec.model_directory.join("generation_config.json"),
            r#"{"task": "transcribe", "lang_to_id": {"<|es|>": 17451, "<|en|>": 50259}}"#,
        )
        .unwrap();
        spec.genai_library = fake_openvino_library().to_path_buf();
        let provider = OpenVinoProvider::open(&spec).unwrap();
        assert_eq!(provider.evidence.language, "es");
        let transcript = provider.transcribe(&vec![0.0_f32; 1600]).unwrap();
        assert_eq!(transcript.trim(), "hello oma");
        assert!(
            !spec
                .cache_directory
                .join("generation_config.es.json")
                .exists(),
            "no derived config file should be written"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fake_genai_variants_cover_safe_pipeline_and_transcript_errors() {
        for mode in 6..=7 {
            let (root, mut spec) = fake_spec(&format!("pipeline-error-{mode}"), "CPU");
            spec.genai_library = fake_openvino_variant(mode);
            assert!(OpenVinoProvider::open(&spec).err().is_some());
            fs::remove_dir_all(root).unwrap();
        }

        let (root, spec) = fake_spec("accelerator-cache-required", "NPU");
        fs::remove_file(spec.cache_directory.join("compiled.blob")).unwrap();
        assert!(
            OpenVinoProvider::open(&spec)
                .err()
                .unwrap()
                .to_string()
                .contains("cache artifacts")
        );
        fs::remove_dir_all(root).unwrap();

        let (root, mut spec) = fake_spec("vad-loader-error", "CPU");
        spec.audiocpp_library = fake_openvino_library().to_path_buf();
        assert!(OpenVinoProvider::open(&spec).err().is_some());
        fs::remove_dir_all(root).unwrap();

        for mode in 8..=13 {
            let (root, mut spec) = fake_spec(&format!("transcript-error-{mode}"), "CPU");
            spec.genai_library = fake_openvino_variant(mode);
            let provider = OpenVinoProvider::open(&spec).unwrap();
            assert!(provider.transcribe(&[0.25; 32]).is_err());
            drop(provider);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn in_memory_openvino_worker_covers_stream_lifecycle() {
        let (root, spec) = fake_spec("worker-loop", "CPU");
        let mut provider = OpenVinoProvider::open(&spec).unwrap();
        let frame = vec![0.0; FRAME_SAMPLES];
        let mut input = Vec::new();
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
        let mut invalid = frame.clone();
        invalid[0] = f32::INFINITY;
        protocol::write_request(
            &mut input,
            &Request::Audio {
                id: 1,
                samples: FRAME_SAMPLES,
            },
            &invalid,
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
        worker_loop(&mut provider, &mut input.as_slice(), &mut output).unwrap();
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
        drop(provider);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_spec_validation_and_worker_errors_are_protocol_safe() {
        let (root, mut spec) = fake_spec("spec-errors", "CPU");
        assert!(
            spec.clone()
                .validate()
                .unwrap_err()
                .to_string()
                .contains("Silero VAD")
        );
        spec.device = "AUTO".into();
        assert!(spec.clone().validate().is_err());

        let mut output = Vec::new();
        let mut input = [].as_slice();
        worker_main_io(spec, &mut input, &mut output).unwrap();
        assert!(matches!(
            protocol::read_response(&mut output.as_slice()).unwrap(),
            Response::Error { id: None, .. }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn io_waits_paths_and_cache_permissions_are_bounded() {
        use std::os::unix::ffi::OsStringExt;
        use std::os::unix::net::UnixStream;

        let (reader, writer) = UnixStream::pair().unwrap();
        assert_eq!(
            wait_for_io(reader.as_raw_fd(), libc::POLLIN, Duration::from_millis(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::TimedOut
        );
        wait_for_io(writer.as_raw_fd(), libc::POLLOUT, Duration::from_millis(10)).unwrap();

        let mut child = Command::new("true").spawn().unwrap();
        assert!(wait_for_exit(&mut child, Duration::from_secs(1)));

        let root = temporary("secure-directory");
        let _ = fs::remove_dir_all(&root);
        secure_directory(&root, "test cache").unwrap();
        assert_eq!(
            fs::metadata(&root).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(cache_artifacts(&root).unwrap(), (0, 0));
        assert!(canonical_file(&root, "directory").is_err());
        assert!(canonical_directory(&root.join("missing"), "missing").is_err());
        let nul = std::ffi::OsString::from_vec(b"bad\0path".to_vec());
        assert!(path_to_c_string(Path::new(&nul)).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervised_openvino_worker_reconnects_and_reaps_normally() {
        let (root, spec) = fake_spec("supervised-process", "CPU");
        let executable = super::super::audiocpp::tests::fake_worker().to_path_buf();
        let mut worker = Worker::spawn_with(spec, executable).unwrap();
        assert_eq!(worker.evidence.available_device, "CPU");
        worker.start(7).unwrap();
        assert!(
            worker
                .start(8)
                .unwrap_err()
                .to_string()
                .contains("already serving")
        );
        assert!(
            worker
                .audio(8, &[])
                .unwrap_err()
                .to_string()
                .contains("not serving")
        );
        assert!(
            worker
                .audio(7, &vec![0.0; FRAME_SAMPLES])
                .unwrap()
                .is_empty()
        );
        assert_eq!(worker.finish(7).unwrap()[0].text, "hello oma");
        assert!(worker.finish(7).is_err());
        worker.start(8).unwrap();
        worker.cancel(8).unwrap();
        worker.disconnect();
        worker.start(9).unwrap();
        worker.shutdown();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervised_openvino_worker_reports_controlled_protocol_errors() {
        let executable = super::super::audiocpp::tests::fake_worker().to_path_buf();
        for mode in ["HANDSHAKE_ERROR", "HANDSHAKE_UNEXPECTED", "HANDSHAKE_EOF"] {
            let (root, mut spec) = fake_spec(&format!("handshake-{mode}"), "CPU");
            spec.device = mode.into();
            assert!(Worker::spawn_with(spec, executable.clone()).err().is_some());
            fs::remove_dir_all(root).unwrap();
        }

        for mode in ["RESPONSE_ERROR", "RESPONSE_UNEXPECTED"] {
            let (root, mut spec) = fake_spec(&format!("response-{mode}"), "CPU");
            spec.device = mode.into();
            let mut worker = Worker::spawn_with(spec, executable.clone()).unwrap();
            assert!(worker.start(21).is_err());
            worker.active_id = Some(21);
            assert!(worker.audio(21, &vec![0.0; FRAME_SAMPLES]).is_err());
            worker.active_id = Some(21);
            assert!(worker.finish(21).is_err());
            assert!(worker.cancel(21).is_err());
            worker.shutdown();
            fs::remove_dir_all(root).unwrap();
        }

        let (root, spec) = fake_spec("closed-input-reconnect", "CPU");
        let mut worker = Worker::spawn_with(spec, executable).unwrap();
        worker.process.as_mut().unwrap().input.take();
        worker.start(22).unwrap();
        worker.shutdown();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn high_level_openvino_stream_uses_supervised_worker() {
        let (root, spec) = fake_spec("backend-stream", "CPU");
        let executable = super::super::audiocpp::tests::fake_worker().to_path_buf();
        let backend = OpenVinoGenAiBackend {
            worker: RefCell::new(Worker::spawn_with(spec, executable).unwrap()),
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
        assert_eq!(backend.kind(), "openvino-genai");
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
    fn provider_spec_discovery_is_explicit_and_never_installs_a_runtime() {
        let root = temporary("provider-discovery");
        let _ = fs::remove_dir_all(&root);
        let runtime = root.join("runtime");
        fs::create_dir_all(&runtime).unwrap();
        let genai = runtime.join("libopenvino_genai_c.so");
        let core = runtime.join("libopenvino_c.so");
        let audiocpp = runtime.join("libaudiocpp.so.0.1.0");
        fs::copy(fake_openvino_library(), &genai).unwrap();
        fs::copy(fake_openvino_library(), &core).unwrap();
        fs::copy(super::super::audiocpp::tests::fake_library(), &audiocpp).unwrap();
        for plugin in [
            "libopenvino_intel_cpu_plugin.so",
            "libopenvino_intel_gpu_plugin.so",
            "libopenvino_intel_npu_plugin.so",
            "libopenvino_intel_npu_compiler_loader.so",
            "libopenvino_intel_npu_compiler.so",
        ] {
            fs::write(runtime.join(plugin), b"present").unwrap();
        }
        let paths = AppPaths {
            config_file: root.join("config/config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
            state_dir: root.join("state"),
            runtime_dir: root.join("run"),
        };
        fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
        let mut config = Config::default();
        config.backend.kind = "openvino-genai".into();
        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "cpu".into();
        config.backend.library = genai.clone();
        config.backend.library_dirs = vec![runtime.clone()];
        let cpu = ProviderSpec::from_config(&config, &paths).unwrap();
        assert_eq!(cpu.device, "CPU");
        assert_eq!(cpu.core_library, core.canonicalize().unwrap());
        assert_eq!(cpu.audiocpp_library, audiocpp.canonicalize().unwrap());
        assert!(!cpu.static_pipeline());

        config.backend.library.clear();
        config.backend.library_dirs = vec![PathBuf::from("../runtime")];
        let discovered = ProviderSpec::from_config(&config, &paths).unwrap();
        assert_eq!(discovered.genai_library, genai.canonicalize().unwrap());
        config.backend.library = PathBuf::from("../runtime/libopenvino_genai_c.so");
        assert_eq!(
            ProviderSpec::from_config(&config, &paths)
                .unwrap()
                .genai_library,
            genai.canonicalize().unwrap()
        );
        config.backend.library = genai.clone();
        config.backend.library_dirs = vec![runtime.clone()];
        config.backend.device = "gpu".into();
        assert_eq!(
            ProviderSpec::from_config(&config, &paths).unwrap().device,
            "GPU"
        );

        config.backend.device = "npu".into();
        let npu = ProviderSpec::from_config(&config, &paths).unwrap();
        assert!(npu.static_pipeline());
        config.model.vad = root.join("absolute-vad.safetensors").display().to_string();
        assert_eq!(
            ProviderSpec::from_config(&config, &paths)
                .unwrap()
                .vad_model,
            root.join("absolute-vad.safetensors")
        );
        config.backend.kind = "audiocpp".into();
        assert!(ProviderSpec::from_config(&config, &paths).is_err());
        config.backend.kind = "openvino-genai".into();
        config.model.sample_rate = 8_000;
        assert!(ProviderSpec::from_config(&config, &paths).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
