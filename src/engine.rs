use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::backend::{Fallback, Runtime};
use crate::config::{Config, WakeWord};
use crate::paths::AppPaths;

pub(crate) mod audio;
pub(crate) mod embedding;
pub(crate) mod embedding_worker;
pub(crate) mod routing;
pub(crate) mod trained;
pub(crate) mod whisper_features;
pub use routing::GroupStatus;
pub(crate) mod audiocpp;
pub(crate) mod openvino_genai;
pub(crate) mod whisper;
use self::audio::read_wave;
use self::audiocpp::AudioCppBackend;
use self::openvino_genai::{OpenVinoGenAiBackend, ProviderSpec as OpenVinoProviderSpec};
use self::whisper::WhisperCppBackend;

pub trait WakeWordBackend {
    fn engine_statuses(&self) -> Vec<GroupStatus> {
        Vec::new()
    }
    fn kind(&self) -> &'static str;
    fn stream(&self) -> Box<dyn WakeWordStream + '_>;
    fn live_stream(&self) -> Box<dyn WakeWordStream + '_> {
        self.stream()
    }
    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>>;
}

pub trait WakeWordStream {
    fn accept(&self, sample_rate: i32, samples: &[f32]) -> Result<Vec<Detection>>;
    fn finish(&self) -> Result<Vec<Detection>>;
}

pub struct Detector {
    backend: Box<dyn WakeWordBackend>,
    actions: HashMap<String, WakeWord>,
    pub backend_kind: &'static str,
    pub keywords_buffer: String,
    pub load_time: Duration,
    pub effective_runtime: Runtime,
    pub fallback_used: bool,
}

pub struct DetectionSession<'a> {
    detector: &'a Detector,
    stream: Box<dyn WakeWordStream + 'a>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub id: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f32>,
    pub start_time: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ActionResult {
    pub id: String,
    pub program: String,
    pub arguments: Vec<String>,
    pub state: &'static str,
    pub pid: u32,
}

struct SpawnedAction {
    id: String,
    program: String,
    child: Child,
}

static ACTION_REAPER: OnceLock<std::result::Result<Sender<SpawnedAction>, String>> =
    OnceLock::new();

pub fn wav_duration(path: &Path) -> Result<Duration> {
    let (sample_rate, samples) =
        read_wave(path).with_context(|| format!("read WAV metadata {}", path.display()))?;
    if sample_rate <= 0 {
        bail!("WAV sample rate must be positive: {}", path.display());
    }
    Ok(Duration::from_secs_f64(
        samples.len() as f64 / sample_rate as f64,
    ))
}

impl Detector {
    pub fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        let groups = routing::plans(config)?;
        if groups.len() > 1
            || groups
                .first()
                .is_some_and(|(name, _)| !name.starts_with("default:"))
        {
            let started = Instant::now();
            let backend = routing::RoutedBackend::load(groups, paths)?;
            let statuses = backend.statuses();
            let fallback = statuses.iter().any(|s| s.fallback_used);
            return Self::from_backend(
                config,
                Box::new(backend),
                String::new(),
                config.backend.runtime,
                fallback,
                started.elapsed(),
            );
        }
        if let Some((_, single)) = groups.into_iter().next() {
            return Self::load_single(&single, paths);
        }
        Self::load_single(config, paths)
    }
    pub(crate) fn load_single(config: &Config, paths: &AppPaths) -> Result<Self> {
        if config
            .wake_words
            .iter()
            .any(|w| w.enabled && w.uses_trained_head())
        {
            let started = Instant::now();
            let backend = trained::TrainedBackend::load(config, paths)?;
            return Self::from_backend(
                config,
                Box::new(backend),
                String::new(),
                Runtime::Openvino,
                false,
                started.elapsed(),
            );
        }
        Self::load_with(
            config,
            paths,
            |candidate, paths| Ok(Box::new(AudioCppBackend::load(candidate, paths)?)),
            |candidate, paths| Ok(Box::new(WhisperCppBackend::load(candidate, paths)?)),
            |candidate, paths| {
                let spec = OpenVinoProviderSpec::from_config(candidate, paths)?;
                Ok(Box::new(OpenVinoGenAiBackend::open(
                    spec,
                    &candidate.wake_words,
                )?))
            },
        )
    }

    fn load_with<FA, FW, FO>(
        config: &Config,
        paths: &AppPaths,
        mut load_audiocpp: FA,
        mut load_whisper: FW,
        mut load_openvino: FO,
    ) -> Result<Self>
    where
        FA: FnMut(&Config, &AppPaths) -> Result<Box<dyn WakeWordBackend>>,
        FW: FnMut(&Config, &AppPaths) -> Result<Box<dyn WakeWordBackend>>,
        FO: FnMut(&Config, &AppPaths) -> Result<Box<dyn WakeWordBackend>>,
    {
        if config.backend.kind == "audiocpp" {
            config.backend.validate_shape()?;
            let keywords_buffer = config
                .wake_words
                .iter()
                .filter(|entry| entry.enabled)
                .map(|entry| entry.phrase.trim())
                .collect::<Vec<_>>()
                .join(", ");
            let started = Instant::now();
            let mut load = |candidate: &Config| load_audiocpp(candidate, paths);
            let (backend, effective_runtime, fallback_used) = match load(config) {
                Ok(backend) => (backend, config.backend.runtime, false),
                Err(accelerator_error)
                    if config.backend.runtime != Runtime::Default
                        && config.backend.fallback == Fallback::Cpu =>
                {
                    eprintln!(
                        "omawake: warning: audio.cpp accelerator initialization failed: {accelerator_error:#}; falling back to audio.cpp CPU"
                    );
                    let mut cpu = config.clone();
                    cpu.backend.runtime = Runtime::Default;
                    cpu.backend.device = "cpu".into();
                    cpu.backend.device_id = 0;
                    (
                        load(&cpu).with_context(|| {
                            format!(
                                "audio.cpp accelerator initialization failed ({accelerator_error:#}); audio.cpp CPU fallback also failed"
                            )
                        })?,
                        Runtime::Default,
                        true,
                    )
                }
                Err(error) => return Err(error),
            };
            return Self::from_backend(
                config,
                backend,
                keywords_buffer,
                effective_runtime,
                fallback_used,
                started.elapsed(),
            );
        }
        if config.backend.kind == "whispercpp" {
            config.backend.validate_shape()?;
            let keywords_buffer = config
                .wake_words
                .iter()
                .filter(|entry| entry.enabled)
                .map(|entry| entry.phrase.trim())
                .collect::<Vec<_>>()
                .join(", ");
            let started = Instant::now();
            let backend = load_whisper(config, paths)?;
            return Self::from_backend(
                config,
                backend,
                keywords_buffer,
                Runtime::Default,
                false,
                started.elapsed(),
            );
        }
        if config.backend.kind == "openvino-genai" {
            config.backend.validate_shape()?;
            let keywords_buffer = config
                .wake_words
                .iter()
                .filter(|entry| entry.enabled)
                .map(|entry| entry.phrase.trim())
                .collect::<Vec<_>>()
                .join(", ");
            let started = Instant::now();
            let mut load = |candidate: &Config| load_openvino(candidate, paths);
            let (backend, fallback_used) = match load(config) {
                Ok(backend) => (backend, false),
                Err(accelerator_error)
                    if config.backend.fallback == Fallback::Cpu
                        && !config.backend.device.eq_ignore_ascii_case("cpu") =>
                {
                    eprintln!(
                        "omawake: warning: OpenVINO accelerator initialization failed: {accelerator_error:#}; falling back to OpenVINO CPU"
                    );
                    let mut cpu = config.clone();
                    cpu.backend.device = "cpu".into();
                    (
                        load(&cpu).with_context(|| {
                            format!(
                                "OpenVINO accelerator initialization failed ({accelerator_error:#}); OpenVINO CPU fallback also failed"
                            )
                        })?,
                        true,
                    )
                }
                Err(error) => return Err(error),
            };
            return Self::from_backend(
                config,
                backend,
                keywords_buffer,
                Runtime::Openvino,
                fallback_used,
                started.elapsed(),
            );
        }
        bail!(
            "unsupported wake-word backend {}; run `omawake setup runtime`",
            config.backend.kind
        )
    }

    pub fn from_backend(
        config: &Config,
        backend: Box<dyn WakeWordBackend>,
        keywords_buffer: String,
        effective_runtime: Runtime,
        fallback_used: bool,
        load_time: Duration,
    ) -> Result<Self> {
        let backend_kind = backend.kind();
        let actions = config
            .wake_words
            .iter()
            .filter(|entry| entry.enabled)
            .cloned()
            .map(|entry| (entry.id.clone(), entry))
            .collect();
        Ok(Self {
            backend,
            actions,
            backend_kind,
            keywords_buffer,
            load_time,
            effective_runtime,
            fallback_used,
        })
    }

    pub fn engine_statuses(&self) -> Vec<GroupStatus> {
        self.backend.engine_statuses()
    }

    pub fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let detections = self.backend.detect_file(path)?;
        self.validate_detections(&detections)?;
        Ok(detections)
    }

    pub fn session(&self) -> DetectionSession<'_> {
        DetectionSession {
            detector: self,
            stream: self.backend.stream(),
        }
    }

    pub fn live_session(&self) -> DetectionSession<'_> {
        DetectionSession {
            detector: self,
            stream: self.backend.live_stream(),
        }
    }

    fn validate_detections(&self, detections: &[Detection]) -> Result<()> {
        for detection in detections {
            if !self.actions.contains_key(&detection.id) {
                bail!("detector returned unknown wake-word id {}", detection.id);
            }
        }
        Ok(())
    }

    pub fn run_action(&self, id: &str) -> Result<ActionResult> {
        let action = self
            .actions
            .get(id)
            .with_context(|| format!("no action for wake-word id {id}"))?;
        let (program, arguments) = action
            .command
            .split_first()
            .context("empty action command")?;
        let reaper = action_reaper()?;
        let mut child = Command::new(program)
            .args(arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("start wake-word action {program}"))?;
        let pid = child.id();
        if let Err(error) = reaper.send(SpawnedAction {
            id: id.into(),
            program: program.clone(),
            child,
        }) {
            child = error.0.child;
            let _ = child.kill();
            let _ = child.wait();
            bail!("action reaper stopped before it accepted {program}");
        }
        Ok(ActionResult {
            id: id.into(),
            program: program.clone(),
            arguments: arguments.to_vec(),
            state: "started",
            pid,
        })
    }
}

fn action_reaper() -> Result<&'static Sender<SpawnedAction>> {
    match ACTION_REAPER.get_or_init(start_action_reaper) {
        Ok(sender) => Ok(sender),
        Err(error) => bail!("start action reaper: {error}"),
    }
}

fn start_action_reaper() -> std::result::Result<Sender<SpawnedAction>, String> {
    let (sender, receiver) = channel();
    thread::Builder::new()
        .name("omawake-action-reaper".into())
        .spawn(move || reap_actions(receiver))
        .map_err(|error| error.to_string())?;
    Ok(sender)
}

fn reap_actions(receiver: Receiver<SpawnedAction>) {
    let mut actions = Vec::new();
    loop {
        match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(action) => actions.push(action),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        actions.extend(receiver.try_iter());

        let mut index = 0;
        while index < actions.len() {
            match actions[index].child.try_wait() {
                Ok(Some(status)) => {
                    let action = actions.swap_remove(index);
                    if !status.success() {
                        eprintln!(
                            "action for {} ({}) exited with {status}",
                            action.id, action.program
                        );
                    }
                }
                Ok(None) => index += 1,
                Err(error) => {
                    let mut action = actions.swap_remove(index);
                    eprintln!(
                        "action for {} ({}) could not be reaped: {error}",
                        action.id, action.program
                    );
                    let _ = action.child.kill();
                    let _ = action.child.wait();
                }
            }
        }
    }
}

impl DetectionSession<'_> {
    pub fn accept(&self, sample_rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        let detections = self.stream.accept(sample_rate, samples)?;
        self.detector.validate_detections(&detections)?;
        Ok(detections)
    }

    pub fn finish(&self) -> Result<Vec<Detection>> {
        let detections = self.stream.finish()?;
        self.detector.validate_detections(&detections)?;
        Ok(detections)
    }
}

pub(crate) fn openvino_cache_directory(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    Ok(openvino_device_directory(config, paths)?.join("compiled"))
}

fn openvino_device_directory(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    let canonical = config.backend.canonical_device()?.to_ascii_uppercase();
    Ok(paths
        .cache_dir
        .join("openvino")
        .join(device_path_component(&canonical)))
}

fn device_path_component(device: &str) -> String {
    device
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn detect_samples(
    stream: &dyn WakeWordStream,
    sample_rate: i32,
    samples: &[f32],
) -> Result<Vec<Detection>> {
    let chunk_sizes = [317usize, 1600, 89, 2711, 503, 997];
    let mut offset = 0usize;
    let mut chunk_index = 0usize;
    let mut detections = Vec::new();
    while offset < samples.len() {
        let end = (offset + chunk_sizes[chunk_index % chunk_sizes.len()]).min(samples.len());
        detections.extend(stream.accept(sample_rate, &samples[offset..end])?);
        offset = end;
        chunk_index += 1;
    }
    let padding = vec![0.0; sample_rate.max(1) as usize / 2];
    detections.extend(stream.accept(sample_rate, &padding)?);
    detections.extend(stream.finish()?);
    Ok(detections)
}

#[cfg(test)]
#[path = "../tests/unit/engine.rs"]
mod tests;
