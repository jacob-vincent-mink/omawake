use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::backend::BackendConfig;
use crate::paths::AppPaths;

/// The resolved engine for one detection group: which backend/model runs it
/// (by value, so owner threads can move it) and whether it is the top-level
/// default or a named profile.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolvedEngine {
    /// `None` for the top-level default backend/model.
    pub name: Option<String>,
    pub backend: BackendConfig,
    pub model: ModelConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub backend: BackendConfig,
    pub model: ModelConfig,
    /// Named engine profiles. Each profile is an independent (backend, model)
    /// pair that wake words may reference by `wake_words[].engine`. The
    /// top-level `backend`/`model` are the default profile, used by words
    /// whose `engine` is `None`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub engines: BTreeMap<String, EngineProfile>,
    pub audio: AudioConfig,
    pub daemon: DaemonConfig,
    pub wake_words: Vec<WakeWord>,
}

/// One wake-word engine: an independent backend + model pair.
///
/// A profile is intentionally *not* tied to any particular wake word — it is
/// selected by wake words via `WakeWord.engine`, and it is never rewritten when
/// the default (top-level) backend/model changes.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EngineProfile {
    pub backend: BackendConfig,
    pub model: ModelConfig,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let input =
            fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
        let config: Self =
            toml::from_str(&input).with_context(|| format!("parse config {}", path.display()))?;
        config.validate_engine_references()?;
        Ok(config)
    }

    /// Validate the `engines` map before any native load: reject empty or
    /// whitespace profile IDs, the reserved `default` profile, and wake words
    /// that reference a profile that is not defined.
    pub fn validate_engine_references(&self) -> Result<()> {
        for id in self.engines.keys() {
            if id.trim().is_empty() {
                anyhow::bail!("engines must not contain an empty profile id");
            }
            anyhow::ensure!(
                id.len() <= 96
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
                "engine profile identifiers must use letters, numbers, hyphens or underscores"
            );
            if id.eq_ignore_ascii_case("default") {
                anyhow::bail!(
                    "engine profile {id:?} is reserved for the top-level default backend/model"
                );
            }
        }
        for word in &self.wake_words {
            if let Some(engine) = word.engine.as_deref() {
                if engine.trim().is_empty() {
                    anyhow::bail!("wake word {} references an empty engine id", word.id);
                }
                if !self.engines.contains_key(engine) {
                    anyhow::bail!(
                        "wake word {} references undefined engine {}; define that profile or remove the word engine field to use the default",
                        word.id,
                        engine
                    );
                }
            }
        }
        Ok(())
    }

    /// Resolve `engine` to the profile that detects its words. `None` selects
    /// the top-level default backend/model; `Some(name)` must name a profile in
    /// `engines`. Whitespace-only or `default` (any case) never resolves to a
    /// named profile — `default` is the reserved word for the top-level
    /// backend/model, so a config cannot shadow it with a named profile.
    pub fn engine_profile(&self, engine: Option<&str>) -> Result<ResolvedEngine> {
        let name = engine.map(str::trim).filter(|name| !name.is_empty());
        match name {
            None => Ok(self.default_profile()),
            Some(name) if name.eq_ignore_ascii_case("default") => Ok(self.default_profile()),
            Some(name) => {
                let profile = self.engines.get(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "undefined engine {name}; available: {}",
                        self.engines.keys().cloned().collect::<Vec<_>>().join(", ")
                    )
                })?;
                Ok(ResolvedEngine {
                    name: Some(name.to_owned()),
                    backend: profile.backend.clone(),
                    model: profile.model.clone(),
                })
            }
        }
    }

    /// The top-level default profile (the un-named engine for words with
    /// `engine = null`). Always present and never shadowed by `engines`.
    pub fn default_profile(&self) -> ResolvedEngine {
        ResolvedEngine {
            name: None,
            backend: self.backend.clone(),
            model: self.model.clone(),
        }
    }

    /// All engines that detect at least one *enabled* wake word, in stable
    /// order (default first, then named profiles by id).
    ///
    /// This is the grouping contract for the detector and for future parent
    /// extensions (e.g. separating ASR from trained/enrolled words): each
    /// returned `ResolvedEngine` owns its backend/model by value, so one model
    /// load and one detection pass happens per group.
    pub fn active_engines(&self) -> Result<Vec<ResolvedEngine>> {
        self.validate_engine_references()?;
        let mut out: Vec<ResolvedEngine> = Vec::new();
        let default_used = self
            .wake_words
            .iter()
            .any(|word| word.enabled && word.engine.as_deref().is_none());
        if default_used {
            out.push(self.default_profile());
        }
        for (name, profile) in &self.engines {
            let used = self.wake_words.iter().any(|word| {
                word.enabled
                    && word
                        .engine
                        .as_deref()
                        .map(str::trim)
                        .filter(|n| !n.is_empty())
                        == Some(name.as_str())
            });
            if used {
                out.push(ResolvedEngine {
                    name: Some(name.clone()),
                    backend: profile.backend.clone(),
                    model: profile.model.clone(),
                });
            }
        }
        Ok(out)
    }

    /// Words routed to `engine` (None = default, un-named words). Used with
    /// `for_engine`/`active_engines` to build one group's detector config.
    pub fn words_for_engine(&self, engine: Option<&str>) -> Vec<&WakeWord> {
        let target = engine.map(str::trim).filter(|name| !name.is_empty());
        self.wake_words
            .iter()
            .filter(|word| {
                word.engine
                    .as_deref()
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    == target
            })
            .collect()
    }

    /// Build a Config that runs exactly one engine profile: the backend + model
    /// selected by `engine` (the default profile when `None`), and only the
    /// wake words assigned to that engine. Word definitions (id, phrase,
    /// aliases, enabled, command) are preserved verbatim — only the backend/
    /// model fields are replaced by the profile's. The result carries no
    /// `engines` map, so it cannot re-route (no recursive routing).
    ///
    /// Onboarding uses this to materialize a profile for a single word: load
    /// the word's `for_engine` config and load the detector from it.
    pub fn for_engine(&self, engine: Option<&str>) -> Result<Config> {
        self.validate_engine_references()?;
        let resolved = self.engine_profile(engine)?;
        let target = resolved.name.as_deref();
        let words = self
            .wake_words
            .iter()
            .filter(|word| {
                word.engine
                    .as_deref()
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    == target
            })
            .cloned()
            .map(|mut word| {
                // Clear the routing label so the materialized config cannot
                // re-route this word into a named profile (no recursive
                // routing): it now belongs to the top-level default backend.
                word.engine = None;
                word
            })
            .collect();
        Ok(Config {
            backend: resolved.backend,
            model: resolved.model,
            engines: BTreeMap::new(),
            audio: self.audio.clone(),
            daemon: self.daemon.clone(),
            wake_words: words,
        })
    }

    pub fn model_directory(&self, paths: &AppPaths) -> PathBuf {
        if self.model.directory.trim().is_empty() {
            paths.data_dir.join("models").join(&self.model.name)
        } else {
            PathBuf::from(&self.model.directory)
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate_engine_references()?;
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
            engines: BTreeMap::new(),
            audio: AudioConfig::default(),
            daemon: DaemonConfig::default(),
            wake_words: vec![WakeWord {
                engine: None,
                enrollment: None,
                id: "computer".into(),
                phrase: "Computer".into(),
                aliases: Vec::new(),
                enabled: true,
                command: vec!["notify-send".into(), "Wake word heard".into()],
            }],
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub directory: String,
    pub verifier: String,
    pub vad: String,
    pub sample_rate: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: crate::catalog::DEFAULT_MODEL_ID.into(),
            directory: String::new(),
            verifier: "moonshine-streaming-tiny-q8_0.gguf".into(),
            vad: "silero_vad_16k.safetensors".into(),
            sample_rate: 16_000,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enrollment: Option<crate::enrollment::artifact::EnrollmentBinding>,
    pub id: String,
    pub phrase: String,
    /// Exact alternate ASR transcripts accepted for this wake word.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    pub command: Vec<String>,
    /// Named engine profile this word is detected on. `None` (the default) runs
    /// on the top-level default backend/model. The engine is an assignment
    /// only — it is independent of the word's definition and is never rewritten
    /// when a profile's backend/model changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
}

fn enabled_by_default() -> bool {
    true
}

#[cfg(test)]
#[path = "../tests/unit/config.rs"]
mod tests;

impl WakeWord {
    pub fn uses_trained_head(&self) -> bool {
        self.enrollment.as_ref().is_some_and(|e| e.active)
    }
}
