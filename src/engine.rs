use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use sherpa_onnx::KeywordSpotterConfig;

use crate::backend::{Fallback, Runtime, compiled_capabilities};
use crate::config::{Config, WakeWord};
use crate::keyword::KeywordCompiler;
use crate::paths::AppPaths;

mod sherpa;
use self::sherpa::SherpaOnnxBackend;

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

impl Detector {
    pub fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        if config.backend.kind != "sherpa-onnx" {
            bail!(
                "wake-word backend {} is not available in this build; run `omawake setup runtime`",
                config.backend.kind
            );
        }
        Self::load_with(config, paths, |config, directory, runtime, keywords| {
            Ok(Box::new(SherpaOnnxBackend::load(
                config, directory, runtime, keywords,
            )?))
        })
    }

    pub fn load_with<F>(config: &Config, paths: &AppPaths, load_backend: F) -> Result<Self>
    where
        F: FnOnce(&Config, &Path, Runtime, &str) -> Result<Box<dyn WakeWordBackend>>,
    {
        config.backend.validate_shape()?;
        let (effective_runtime, fallback_used) = match config
            .backend
            .validate_capabilities(compiled_capabilities())
        {
            Ok(()) => (config.backend.runtime, false),
            Err(error) if config.backend.fallback == Fallback::Cpu => {
                eprintln!("omawake: warning: {error}; falling back to cpu");
                (Runtime::Default, true)
            }
            Err(error) => return Err(error.into()),
        };
        let directory = config.model_directory(paths);
        let compiler = KeywordCompiler::open(&directory.join(&config.model.bpe_model))?;
        let keywords_buffer = compiler.compile(&config.wake_words)?;

        let started = Instant::now();
        let backend = load_backend(config, &directory, effective_runtime, &keywords_buffer)?;
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
            status: status.code().unwrap_or(-1),
        })
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

fn drain_ready(mut next: impl FnMut() -> Option<Option<Detection>>) -> Vec<Detection> {
    let mut detections = Vec::new();
    while let Some(result) = next() {
        if let Some(detection) = result {
            detections.push(detection);
        }
    }
    detections
}

fn detection_from_parts(
    id: String,
    tokens: Vec<String>,
    timestamps: Vec<f32>,
    start_time: f32,
) -> Option<Detection> {
    (!id.is_empty()).then_some(Detection {
        id,
        tokens,
        timestamps,
        start_time,
    })
}

fn build_sherpa_config(
    config: &Config,
    directory: &Path,
    runtime: Runtime,
    keywords_buffer: &str,
) -> Result<KeywordSpotterConfig> {
    let required = |name: &str| -> Result<String> {
        let path = directory.join(name);
        if !path.exists() {
            bail!("required model asset is missing: {}", path.display());
        }
        Ok(path.to_string_lossy().into_owned())
    };
    let mut sherpa_config = KeywordSpotterConfig::default();
    sherpa_config.model_config.transducer.encoder = Some(required(&config.model.encoder)?);
    sherpa_config.model_config.transducer.decoder = Some(required(&config.model.decoder)?);
    sherpa_config.model_config.transducer.joiner = Some(required(&config.model.joiner)?);
    sherpa_config.model_config.tokens = Some(required(&config.model.tokens)?);
    sherpa_config.model_config.num_threads = config.backend.threads.into();
    sherpa_config.model_config.provider = Some(match runtime {
        Runtime::Default => "cpu".into(),
        Runtime::Cuda => "cuda".into(),
        Runtime::Openvino => {
            bail!("OpenVINO provider file generation is unavailable in this build")
        }
    });
    sherpa_config.feat_config.sample_rate = config.model.sample_rate;
    sherpa_config.max_active_paths = config.model.max_active_paths;
    sherpa_config.num_trailing_blanks = config.model.num_trailing_blanks;
    sherpa_config.keywords_score = config.model.keywords_score;
    sherpa_config.keywords_threshold = config.model.keywords_threshold;
    sherpa_config.keywords_buf = Some(keywords_buffer.into());
    Ok(sherpa_config)
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
