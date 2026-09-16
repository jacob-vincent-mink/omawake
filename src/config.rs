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
                aliases: Vec::new(),
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
    pub sample_rate: i32,
    /// Language code passed to multilingual verifiers; empty uses the model
    /// default. Only the curated Spanish profile is qualified (W09).
    pub language: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: crate::catalog::DEFAULT_MODEL_ID.into(),
            directory: String::new(),
            verifier: "moonshine-streaming-tiny-q8_0.gguf".into(),
            vad: "silero_vad_16k.safetensors".into(),
            sample_rate: 16_000,
            language: String::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    pub device: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: "default".into(),
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
    /// Exact alternate ASR transcripts accepted for this wake word.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
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
