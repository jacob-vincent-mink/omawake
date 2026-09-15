use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::backend::BackendConfig;
use crate::paths::AppPaths;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub backend: BackendConfig,
    pub model: ModelConfig,
    pub audio: AudioConfig,
    pub daemon: DaemonConfig,
    pub wake_words: Vec<WakeWord>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let input =
            fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
        toml::from_str(&input).with_context(|| format!("parse config {}", path.display()))
    }

    pub fn model_directory(&self, paths: &AppPaths) -> PathBuf {
        if self.model.directory.trim().is_empty() {
            paths.data_dir.join("models").join(&self.model.name)
        } else {
            PathBuf::from(&self.model.directory)
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("toml.tmp");
        fs::write(&temporary, toml::to_string_pretty(self)?)?;
        fs::rename(&temporary, path)
            .with_context(|| format!("install config {}", path.display()))?;
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: BackendConfig::default(),
            model: ModelConfig::default(),
            audio: AudioConfig::default(),
            daemon: DaemonConfig::default(),
            wake_words: vec![WakeWord {
                id: "computer".into(),
                phrase: "Computer".into(),
                enabled: true,
                command: vec!["notify-send".into(), "Wake word heard".into()],
            }],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub directory: String,
    pub verifier: String,
    pub vad: String,
    pub encoder: String,
    pub decoder: String,
    pub joiner: String,
    pub tokens: String,
    pub bpe_model: String,
    pub sample_rate: i32,
    pub keywords_score: f32,
    pub keywords_threshold: f32,
    pub max_active_paths: i32,
    pub num_trailing_blanks: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01".into(),
            directory: String::new(),
            verifier: "ggml-tiny.en.bin".into(),
            vad: "ggml-silero-v6.2.0.bin".into(),
            encoder: "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx".into(),
            decoder: "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx".into(),
            joiner: "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx".into(),
            tokens: "tokens.txt".into(),
            bpe_model: "bpe.model".into(),
            sample_rate: 16_000,
            keywords_score: 1.5,
            keywords_threshold: 0.25,
            max_active_paths: 4,
            num_trailing_blanks: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    pub device: String,
    pub channels: String,
    pub buffer_milliseconds: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: "default".into(),
            channels: "mono".into(),
            buffer_milliseconds: 200,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    pub cooldown_milliseconds: u64,
    pub queue_capacity: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            cooldown_milliseconds: 1_500,
            queue_capacity: 8,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WakeWord {
    pub id: String,
    pub phrase: String,
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    pub command: Vec<String>,
}

fn enabled_by_default() -> bool {
    true
}

#[cfg(test)]
#[path = "../tests/unit/config.rs"]
mod tests;
