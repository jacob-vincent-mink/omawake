use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig, Wave};

use crate::backend::{Fallback, Runtime, compiled_capabilities};
use crate::config::{Config, WakeWord};
use crate::keyword::KeywordCompiler;
use crate::paths::AppPaths;

pub struct Detector {
    spotter: KeywordSpotter,
    actions: HashMap<String, WakeWord>,
    pub keywords_buffer: String,
    pub load_time: Duration,
    pub effective_runtime: Runtime,
    pub fallback_used: bool,
}

pub struct DetectionSession<'a> {
    detector: &'a Detector,
    stream: sherpa_onnx::OnlineStream,
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
        let required = |name: &str| -> Result<String> {
            let path = directory.join(name);
            if !path.exists() {
                bail!("required model asset is missing: {}", path.display());
            }
            Ok(path.to_string_lossy().into_owned())
        };

        let compiler = KeywordCompiler::open(&directory.join(&config.model.bpe_model))?;
        let keywords_buffer = compiler.compile(&config.wake_words)?;
        let mut sherpa_config = KeywordSpotterConfig::default();
        sherpa_config.model_config.transducer.encoder = Some(required(&config.model.encoder)?);
        sherpa_config.model_config.transducer.decoder = Some(required(&config.model.decoder)?);
        sherpa_config.model_config.transducer.joiner = Some(required(&config.model.joiner)?);
        sherpa_config.model_config.tokens = Some(required(&config.model.tokens)?);
        sherpa_config.model_config.num_threads = config.backend.threads.into();
        sherpa_config.model_config.provider = Some(match effective_runtime {
            Runtime::Default => "cpu".into(),
            Runtime::Cuda => "cuda".into(),
            Runtime::Openvino => {
                bail!("OpenVINO provider file generation is unavailable in the CPU build")
            }
        });
        sherpa_config.feat_config.sample_rate = config.model.sample_rate;
        sherpa_config.max_active_paths = config.model.max_active_paths;
        sherpa_config.num_trailing_blanks = config.model.num_trailing_blanks;
        sherpa_config.keywords_score = config.model.keywords_score;
        sherpa_config.keywords_threshold = config.model.keywords_threshold;
        sherpa_config.keywords_buf = Some(keywords_buffer.clone());

        let started = Instant::now();
        let spotter = KeywordSpotter::create(&sherpa_config)
            .context("sherpa-onnx could not create the keyword spotter")?;
        let load_time = started.elapsed();
        let actions = config
            .wake_words
            .iter()
            .filter(|entry| entry.enabled)
            .cloned()
            .map(|entry| (entry.id.clone(), entry))
            .collect();
        Ok(Self {
            spotter,
            actions,
            keywords_buffer,
            load_time,
            effective_runtime,
            fallback_used,
        })
    }

    pub fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let wave = Wave::read(path.to_string_lossy().as_ref())
            .with_context(|| format!("read audio fixture {}", path.display()))?;
        let stream = self.spotter.create_stream();
        let chunk_sizes = [317usize, 1600, 89, 2711, 503, 997];
        let mut offset = 0usize;
        let mut chunk_index = 0usize;
        let samples = wave.samples();
        let mut detections = Vec::new();
        while offset < samples.len() {
            let end = (offset + chunk_sizes[chunk_index % chunk_sizes.len()]).min(samples.len());
            stream.accept_waveform(wave.sample_rate(), &samples[offset..end]);
            self.decode_ready(&stream, &mut detections)?;
            offset = end;
            chunk_index += 1;
        }
        let padding = vec![0.0; wave.sample_rate().max(1) as usize / 2];
        stream.accept_waveform(wave.sample_rate(), &padding);
        stream.input_finished();
        self.decode_ready(&stream, &mut detections)?;
        Ok(detections)
    }

    pub fn session(&self) -> DetectionSession<'_> {
        DetectionSession {
            detector: self,
            stream: self.spotter.create_stream(),
        }
    }

    fn decode_ready(
        &self,
        stream: &sherpa_onnx::OnlineStream,
        detections: &mut Vec<Detection>,
    ) -> Result<()> {
        while self.spotter.is_ready(stream) {
            self.spotter.decode(stream);
            if let Some(result) = self.spotter.get_result(stream) {
                if result.keyword.is_empty() {
                    continue;
                }
                if !self.actions.contains_key(&result.keyword) {
                    bail!("detector returned unknown wake-word id {}", result.keyword);
                }
                detections.push(Detection {
                    id: result.keyword,
                    tokens: result.tokens_arr,
                    timestamps: result.timestamps,
                    start_time: result.start_time,
                });
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
        self.stream.accept_waveform(sample_rate, samples);
        let mut detections = Vec::new();
        self.detector.decode_ready(&self.stream, &mut detections)?;
        Ok(detections)
    }

    pub fn finish(&self) -> Result<Vec<Detection>> {
        self.stream.input_finished();
        let mut detections = Vec::new();
        self.detector.decode_ready(&self.stream, &mut detections)?;
        Ok(detections)
    }
}
