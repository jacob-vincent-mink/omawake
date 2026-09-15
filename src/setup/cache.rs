use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::backend::{Fallback, Runtime};
use crate::config::Config;
use crate::engine::Detector;
use crate::paths::AppPaths;
use crate::setup::model::ProgressFormat;

const PREPARE_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_SIGNAL_ATTEMPTS: usize = 5;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CacheReport {
    pub required: bool,
    pub prepared: bool,
    pub directory: Option<PathBuf>,
    pub artifacts: usize,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_milliseconds: Option<f64>,
}

pub fn required(config: &Config) -> bool {
    config.backend.runtime == Runtime::Openvino
        && config
            .backend
            .canonical_device()
            .is_ok_and(|device| matches!(device.as_str(), "gpu" | "npu"))
}

pub fn status(config: &Config, paths: &AppPaths) -> Result<CacheReport> {
    if !required(config) {
        return Ok(CacheReport {
            required: false,
            prepared: true,
            directory: None,
            artifacts: 0,
            bytes: 0,
            elapsed_milliseconds: None,
        });
    }
    if !config.backend.provider_config.trim().is_empty() {
        bail!(
            "setup cannot verify a persistent OpenVINO accelerator cache when backend.provider_config is supplied"
        );
    }
    let directory = crate::engine::openvino_cache_directory(config, paths)?;
    let (artifacts, bytes) = cache_artifacts(&directory)?;
    Ok(CacheReport {
        required: true,
        prepared: artifacts > 0,
        directory: Some(directory),
        artifacts,
        bytes,
        elapsed_milliseconds: None,
    })
}

pub fn prepare(
    config: &Config,
    config_path: &Path,
    paths: &AppPaths,
    progress: ProgressFormat,
) -> Result<Option<CacheReport>> {
    prepare_with(config, config_path, paths, progress, isolated)
}

pub fn prepare_for_runtime(
    config: &Config,
    config_path: &Path,
    paths: &AppPaths,
    progress: ProgressFormat,
) -> Result<Option<CacheReport>> {
    prepare_for_runtime_with(config, config_path, paths, progress, isolated)
}

fn prepare_for_runtime_with(
    config: &Config,
    config_path: &Path,
    paths: &AppPaths,
    progress: ProgressFormat,
    run: impl FnOnce(&Config, &Path, &AppPaths) -> Result<CacheReport>,
) -> Result<Option<CacheReport>> {
    if !required(config) {
        return Ok(None);
    }
    if !config.backend.provider_config.trim().is_empty() {
        bail!(
            "setup-time OpenVINO accelerator cache preparation requires the managed provider config; unset backend.provider_config"
        );
    }
    let Some(probe_audio) = catalog_probe_audio(config, paths) else {
        emit_deferred(progress, config)?;
        return Ok(None);
    };
    if !probe_audio.is_file() {
        emit_deferred(progress, config)?;
        return Ok(None);
    }
    prepare_with(config, config_path, paths, progress, run)
}

fn prepare_with(
    config: &Config,
    config_path: &Path,
    paths: &AppPaths,
    progress: ProgressFormat,
    run: impl FnOnce(&Config, &Path, &AppPaths) -> Result<CacheReport>,
) -> Result<Option<CacheReport>> {
    if !required(config) {
        return Ok(None);
    }
    emit(progress, "model-cache-prepare-start", config, None)?;
    let report = run(config, config_path, paths)?;
    if !report.prepared || report.artifacts == 0 {
        bail!(
            "OpenVINO {} inference completed without a persistent compiled cache artifact",
            cache_device(config)?.to_ascii_uppercase()
        );
    }
    emit(progress, "model-cache-prepared", config, Some(&report))?;
    Ok(Some(report))
}

fn isolated(config: &Config, config_path: &Path, _paths: &AppPaths) -> Result<CacheReport> {
    if !config.backend.provider_config.trim().is_empty() {
        bail!(
            "setup-time OpenVINO accelerator cache preparation requires the managed provider config; unset backend.provider_config"
        );
    }
    let mut candidate = config.clone();
    candidate.backend.fallback = Fallback::Error;
    candidate.backend = crate::runtime_inventory::resolve(&candidate.backend, config_path);
    let library_path = std::env::join_paths(&candidate.backend.library_dirs)?;
    let device = cache_device(&candidate)?.to_ascii_uppercase();
    retry_signaled(&device, || {
        isolated_attempt(&candidate, config_path, &library_path, &device)
    })
}

enum AttemptOutcome {
    Complete(CacheReport),
    Failed {
        status: String,
        signal: Option<i32>,
        stderr: String,
    },
}

fn retry_signaled(
    device: &str,
    mut run: impl FnMut() -> Result<AttemptOutcome>,
) -> Result<CacheReport> {
    for attempt in 1..=MAX_SIGNAL_ATTEMPTS {
        match run()? {
            AttemptOutcome::Complete(report) => return Ok(report),
            AttemptOutcome::Failed {
                signal: Some(signal),
                ..
            } if attempt < MAX_SIGNAL_ATTEMPTS => {
                eprintln!(
                    "OpenVINO {device} model-cache child attempt {attempt}/{MAX_SIGNAL_ATTEMPTS} exited on signal {signal}; retrying attempt {}/{} against the partial cache",
                    attempt + 1,
                    MAX_SIGNAL_ATTEMPTS,
                );
            }
            AttemptOutcome::Failed {
                status,
                signal,
                stderr,
            } => {
                let signal =
                    signal.map_or_else(String::new, |signal| format!(" (signal {signal})"));
                let detail = stderr.trim();
                bail!(
                    "OpenVINO {device} model-cache preparation failed on attempt {attempt}/{MAX_SIGNAL_ATTEMPTS}: {status}{signal}{}",
                    if detail.is_empty() {
                        String::new()
                    } else {
                        format!(": {detail}")
                    }
                );
            }
        }
    }
    unreachable!("all cache preparation attempts are handled above")
}

fn isolated_attempt(
    candidate: &Config,
    config_path: &Path,
    library_path: &std::ffi::OsStr,
    device: &str,
) -> Result<AttemptOutcome> {
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("--config")
        .arg(config_path)
        .arg("__model-cache-prepare")
        .arg(serde_json::to_string(&candidate)?)
        .env("LD_LIBRARY_PATH", library_path)
        .env("ORT_DISABLE_TELEMETRY", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("start isolated OpenVINO {device} model-cache preparation"))?;
    let mut child_stderr = child
        .stderr
        .take()
        .with_context(|| format!("capture isolated OpenVINO {device} model-cache stderr"))?;
    let mut child_stdout = child
        .stdout
        .take()
        .with_context(|| format!("capture isolated OpenVINO {device} model-cache stdout"))?;
    let stderr_reader = thread::spawn(move || {
        let mut stderr = Vec::new();
        child_stderr.read_to_end(&mut stderr).map(|_| stderr)
    });
    let stdout_reader = thread::spawn(move || {
        let mut stdout = Vec::new();
        child_stdout.read_to_end(&mut stdout).map(|_| stdout)
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() > PREPARE_TIMEOUT {
            child.kill()?;
            child.wait()?;
            let _ = stderr_reader.join();
            let _ = stdout_reader.join();
            bail!(
                "OpenVINO {device} model-cache preparation timed out after {} seconds",
                PREPARE_TIMEOUT.as_secs()
            );
        }
        thread::sleep(Duration::from_millis(25));
    };
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("read isolated OpenVINO {device} model-cache stderr"))??;
    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("read isolated OpenVINO {device} model-cache stdout"))??;
    if !status.success() {
        return Ok(AttemptOutcome::Failed {
            status: status.to_string(),
            signal: exit_signal(&status),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        });
    }
    let report = serde_json::from_slice(&stdout)
        .with_context(|| format!("read isolated OpenVINO {device} model-cache report"))?;
    Ok(AttemptOutcome::Complete(report))
}

#[cfg(unix)]
fn exit_signal(status: &ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn exit_signal(_status: &ExitStatus) -> Option<i32> {
    None
}

pub fn child(config: &Config, paths: &AppPaths) -> Result<CacheReport> {
    child_with(config, paths, |candidate, paths, audio| {
        let detector = Detector::load(candidate, paths)?;
        if detector.effective_runtime != Runtime::Openvino || detector.fallback_used {
            bail!("OpenVINO accelerator cache preparation fell back to CPU");
        }
        detector.detect_file(audio)?;
        Ok(())
    })
}

fn child_with(
    config: &Config,
    paths: &AppPaths,
    infer: impl FnOnce(&Config, &AppPaths, &Path) -> Result<()>,
) -> Result<CacheReport> {
    if !required(config) {
        bail!("model-cache preparation is only valid for an explicit OpenVINO GPU or NPU runtime");
    }
    if config.backend.fallback != Fallback::Error {
        bail!("model-cache preparation requires fallback = error");
    }
    if !config.backend.provider_config.trim().is_empty() {
        bail!("model-cache preparation requires the managed OpenVINO provider config");
    }
    let spec = crate::catalog::model(&config.model.name)
        .with_context(|| format!("model {} has no catalog probe audio", config.model.name))?;
    let audio = config.model_directory(paths).join(spec.probe_audio);
    if !audio.is_file() {
        bail!("model cache probe audio is missing: {}", audio.display());
    }
    let started = Instant::now();
    infer(config, paths, &audio)?;
    let mut report = status(config, paths)?;
    report.elapsed_milliseconds = Some(started.elapsed().as_secs_f64() * 1000.0);
    if !report.prepared {
        bail!(
            "OpenVINO {} inference produced no compiled cache artifact in {}",
            cache_device(config)?.to_ascii_uppercase(),
            report
                .directory
                .as_deref()
                .unwrap_or_else(|| Path::new("<unknown>"))
                .display()
        );
    }
    Ok(report)
}

fn cache_artifacts(directory: &Path) -> Result<(usize, u64)> {
    if !directory.is_dir() {
        return Ok((0, 0));
    }
    let mut pending = vec![directory.to_owned()];
    let mut artifacts = 0;
    let mut bytes = 0_u64;
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)
            .with_context(|| format!("inspect OpenVINO cache {}", current.display()))?
        {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file()
                && entry.path().extension().and_then(|value| value.to_str()) == Some("blob")
            {
                let metadata = entry.metadata()?;
                if metadata.len() == 0 {
                    continue;
                }
                artifacts += 1;
                bytes = bytes.saturating_add(metadata.len());
            }
        }
    }
    Ok((artifacts, bytes))
}

fn cache_device(config: &Config) -> Result<String> {
    let device = config.backend.canonical_device()?;
    if config.backend.runtime != Runtime::Openvino || !matches!(device.as_str(), "gpu" | "npu") {
        bail!("an explicit OpenVINO GPU or NPU device is required");
    }
    Ok(device)
}

fn catalog_probe_audio(config: &Config, paths: &AppPaths) -> Option<PathBuf> {
    crate::catalog::model(&config.model.name)
        .map(|spec| config.model_directory(paths).join(spec.probe_audio))
}

fn emit_deferred(format: ProgressFormat, config: &Config) -> Result<()> {
    match format {
        ProgressFormat::Human => eprintln!(
            "OpenVINO {} model cache preparation deferred until the catalog model and probe WAV are installed",
            cache_device(config)?.to_ascii_uppercase()
        ),
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "event": "model-cache-deferred",
                "runtime": "openvino",
                "device": config.backend.device,
                "reason": "catalog model probe audio is not installed",
            }))?
        ),
    }
    Ok(())
}

fn emit(
    format: ProgressFormat,
    event: &'static str,
    config: &Config,
    report: Option<&CacheReport>,
) -> Result<()> {
    match format {
        ProgressFormat::Human if report.is_none() => eprintln!(
            "preparing OpenVINO {} model cache; first-time compilation may take a while",
            cache_device(config)?.to_ascii_uppercase()
        ),
        ProgressFormat::Human => {
            let report = report.expect("report checked above");
            eprintln!(
                "prepared OpenVINO {} model cache: {} artifact(s), {} bytes in {}",
                cache_device(config)?.to_ascii_uppercase(),
                report.artifacts,
                report.bytes,
                report
                    .directory
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .display()
            );
        }
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "event": event,
                "runtime": "openvino",
                "device": config.backend.device,
                "cache": report,
            }))?
        ),
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_cache.rs"]
mod tests;
