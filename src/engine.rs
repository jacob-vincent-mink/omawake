use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::backend::{Fallback, Runtime};
use crate::config::{Config, WakeWord};
use crate::keyword::KeywordCompiler;
use crate::paths::AppPaths;

pub(crate) mod onnx;
pub(crate) mod whisper;
use self::onnx::{OmaOnnxBackend, read_wave};
use self::whisper::WhisperCppBackend;

pub trait WakeWordBackend {
    fn kind(&self) -> &'static str;
    fn stream(&self) -> Box<dyn WakeWordStream + '_>;
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
    pub status: i32,
}

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
            let backend = Box::new(WhisperCppBackend::load(config, paths)?);
            return Self::from_backend(
                config,
                backend,
                keywords_buffer,
                Runtime::Default,
                false,
                started.elapsed(),
            );
        }
        if config.backend.kind == "omawake-onnx" {
            return Self::load_with(config, paths, |config, directory, runtime, keywords| {
                Ok(Box::new(OmaOnnxBackend::load(
                    config, paths, directory, runtime, keywords,
                )?))
            });
        }
        bail!(
            "unsupported wake-word backend {}; run `omawake setup runtime`",
            config.backend.kind
        )
    }

    pub fn load_with<F>(config: &Config, paths: &AppPaths, mut load_backend: F) -> Result<Self>
    where
        F: FnMut(&Config, &Path, Runtime, &str) -> Result<Box<dyn WakeWordBackend>>,
    {
        config.backend.validate_shape()?;
        let mut effective_runtime = config.backend.runtime;
        let mut fallback_used = false;
        let directory = config.model_directory(paths);
        let compiler = KeywordCompiler::open(&directory.join(&config.model.bpe_model))?;
        let keywords_buffer = compiler.compile(&config.wake_words)?;

        let started = Instant::now();
        let backend = match load_backend(config, &directory, effective_runtime, &keywords_buffer) {
            Ok(backend) => backend,
            Err(accelerator_error)
                if effective_runtime != Runtime::Default
                    && config.backend.fallback == Fallback::Cpu =>
            {
                eprintln!(
                    "omawake: warning: accelerated backend initialization failed: {accelerator_error:#}; falling back to cpu"
                );
                effective_runtime = Runtime::Default;
                fallback_used = true;
                load_backend(config, &directory, Runtime::Default, &keywords_buffer).with_context(
                    || {
                        format!(
                            "accelerated backend initialization failed ({accelerator_error:#}); CPU fallback also failed"
                        )
                    },
                )?
            }
            Err(error) => return Err(error),
        };
        Self::from_backend(
            config,
            backend,
            keywords_buffer,
            effective_runtime,
            fallback_used,
            started.elapsed(),
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
        let status = Command::new(program)
            .args(arguments)
            .status()
            .with_context(|| format!("run wake-word action {program}"))?;
        Ok(ActionResult {
            id: id.into(),
            program: program.clone(),
            arguments: arguments.to_vec(),
            status: action_status_code(&status),
        })
    }
}

fn action_status_code(status: &std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
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
