use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
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

pub fn wav_duration(path: &Path) -> Result<Duration> {
    let wave = sherpa_onnx::Wave::read(path.to_string_lossy().as_ref())
        .with_context(|| format!("read WAV metadata {}", path.display()))?;
    let sample_rate = wave.sample_rate();
    if sample_rate <= 0 {
        bail!("WAV sample rate must be positive: {}", path.display());
    }
    Ok(Duration::from_secs_f64(
        wave.num_samples().max(0) as f64 / sample_rate as f64,
    ))
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
                config, paths, directory, runtime, keywords,
            )?))
        })
    }

    pub fn load_with<F>(config: &Config, paths: &AppPaths, load_backend: F) -> Result<Self>
    where
        F: FnMut(&Config, &Path, Runtime, &str) -> Result<Box<dyn WakeWordBackend>>,
    {
        Self::load_with_capabilities(config, paths, compiled_capabilities(), load_backend)
    }

    fn load_with_capabilities<F>(
        config: &Config,
        paths: &AppPaths,
        capabilities: &[&str],
        mut load_backend: F,
    ) -> Result<Self>
    where
        F: FnMut(&Config, &Path, Runtime, &str) -> Result<Box<dyn WakeWordBackend>>,
    {
        config.backend.validate_shape()?;
        let (mut effective_runtime, mut fallback_used) =
            match config.backend.validate_capabilities(capabilities) {
                Ok(()) => (config.backend.runtime, false),
                Err(error) if config.backend.fallback == Fallback::Cpu => {
                    eprintln!("omawake: warning: {error}; falling back to cpu");
                    (Runtime::Default, true)
                }
                Err(error) => return Err(error.into()),
            };
        if effective_runtime == Runtime::Openvino
            && let Some(spec) = crate::catalog::model(&config.model.name)
            && spec.uses_openvino_accelerator(config)
            && config.model.encoder == spec.encoder
        {
            bail!(
                "model {} uses an encoder that loses detections on OpenVINO accelerators; run `omawake setup model --set {}` to select {}",
                spec.id,
                spec.id,
                spec.openvino_accelerator_encoder
            );
        }
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
    paths: &AppPaths,
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
        Runtime::Cuda => cuda_provider(config, paths)?,
        Runtime::Openvino => openvino_provider(config, paths)?,
    });
    sherpa_config.feat_config.sample_rate = config.model.sample_rate;
    sherpa_config.max_active_paths = config.model.max_active_paths;
    sherpa_config.num_trailing_blanks = config.model.num_trailing_blanks;
    sherpa_config.keywords_score = config.model.keywords_score;
    sherpa_config.keywords_threshold = config.model.keywords_threshold;
    sherpa_config.keywords_buf = Some(keywords_buffer.into());
    Ok(sherpa_config)
}

fn cuda_provider(config: &Config, paths: &AppPaths) -> Result<String> {
    let supplied = config.backend.provider_config.trim();
    if !supplied.is_empty() {
        return provider_config_path("CUDA", "cuda", supplied, paths);
    }

    for (key, value) in &config.backend.options {
        validate_provider_option(key, value)?;
        if key == "device_id" {
            bail!("backend.options.device_id is managed by backend.device_id");
        }
    }
    let device_directory = paths
        .state_dir
        .join("cache/cuda")
        .join(format!("device-{}", config.backend.device_id));
    fs::create_dir_all(&device_directory).with_context(|| {
        format!(
            "create CUDA provider directory {}",
            device_directory.display()
        )
    })?;
    make_directory_private(&device_directory)?;
    let device_directory = fs::canonicalize(&device_directory).with_context(|| {
        format!(
            "resolve CUDA provider directory {}",
            device_directory.display()
        )
    })?;

    let mut properties = config.backend.options.clone();
    properties.insert("device_id".to_owned(), config.backend.device_id.to_string());
    properties
        .entry("cudnn_conv_algo_search".to_owned())
        .or_insert_with(|| "HEURISTIC".to_owned());
    let mut contents = String::new();
    for (key, value) in properties {
        contents.push_str(&key);
        contents.push('=');
        contents.push_str(&value);
        contents.push('\n');
    }
    let config_path = device_directory.join("provider.config");
    atomic_write_private(&config_path, contents.as_bytes())?;
    Ok(format!("cuda:{}", utf8_path(&config_path)?))
}

fn openvino_provider(config: &Config, paths: &AppPaths) -> Result<String> {
    let supplied = config.backend.provider_config.trim();
    if !supplied.is_empty() {
        return provider_config_path("OpenVINO", "openvino", supplied, paths);
    }

    let canonical = config.backend.canonical_device()?.to_ascii_uppercase();
    for (key, value) in &config.backend.options {
        validate_provider_option(key, value)?;
        match key.as_str() {
            "cache_dir" => {
                bail!("backend.options.cache_dir is managed by Omawake for each OpenVINO device")
            }
            "device_type" if value != &canonical => {
                bail!("backend.options.device_type must match canonical device {canonical}")
            }
            _ => {}
        }
    }
    let device_directory = paths
        .state_dir
        .join("cache/openvino")
        .join(device_path_component(&canonical));
    fs::create_dir_all(&device_directory).with_context(|| {
        format!(
            "create OpenVINO provider directory {}",
            device_directory.display()
        )
    })?;
    make_directory_private(&device_directory)?;
    let device_directory = fs::canonicalize(&device_directory).with_context(|| {
        format!(
            "resolve OpenVINO provider directory {}",
            device_directory.display()
        )
    })?;
    let cache_directory = device_directory.join("compiled");
    fs::create_dir_all(&cache_directory).with_context(|| {
        format!(
            "create OpenVINO cache directory {}",
            cache_directory.display()
        )
    })?;
    make_directory_private(&cache_directory)?;

    let cache_value = utf8_path(&cache_directory)?.to_owned();
    let mut properties = std::collections::BTreeMap::from([
        ("cache_dir".to_owned(), cache_value.clone()),
        ("device_type".to_owned(), canonical.clone()),
    ]);
    if canonical == "NPU" {
        properties.insert("disable_dynamic_shapes".into(), "True".into());
        properties.insert("enable_qdq_optimizer".into(), "True".into());
    }
    for (key, value) in &config.backend.options {
        properties.insert(key.clone(), value.clone());
    }

    let mut contents = String::new();
    for (key, value) in properties {
        contents.push_str(&key);
        contents.push('=');
        contents.push_str(&value);
        contents.push('\n');
    }
    let config_path = device_directory.join("provider.config");
    atomic_write_private(&config_path, contents.as_bytes())?;
    Ok(format!("openvino:{}", utf8_path(&config_path)?))
}

fn provider_config_path(
    runtime: &str,
    provider: &str,
    supplied: &str,
    paths: &AppPaths,
) -> Result<String> {
    let supplied = Path::new(supplied);
    let path = if supplied.is_absolute() {
        supplied.to_owned()
    } else {
        paths
            .config_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(supplied)
    };
    let path = fs::canonicalize(&path)
        .with_context(|| format!("resolve {runtime} provider config {}", path.display()))?;
    if !path.is_file() {
        bail!(
            "{runtime} provider config is not a regular file: {}",
            path.display()
        );
    }
    Ok(format!("{provider}:{}", utf8_path(&path)?))
}

fn validate_provider_option(key: &str, value: &str) -> Result<()> {
    let mut characters = key.chars();
    let valid_first = characters
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric());
    let valid_rest = characters
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-'));
    if !valid_first || !valid_rest {
        bail!("backend option key {key:?} is invalid; use ASCII letters, digits, '.', '_' or '-'");
    }
    if value.contains(['\0', '\r', '\n']) {
        bail!("backend option {key} must not contain NUL or line breaks");
    }
    Ok(())
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

fn utf8_path(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn atomic_write_private(path: &Path, contents: &[u8]) -> Result<()> {
    static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .with_context(|| format!("provider config has no parent: {}", path.display()))?;
    let temporary = parent.join(format!(
        ".provider.config.{}.{}.tmp",
        std::process::id(),
        TEMPORARY_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary)
        .with_context(|| format!("create temporary provider config {}", temporary.display()))?;
    let install = (|| -> Result<()> {
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
            .with_context(|| format!("install provider config {}", path.display()))?;
        Ok(())
    })();
    if install.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    install
}

fn make_directory_private(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("make directory private {}", path.display()))?;
    }
    Ok(())
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
