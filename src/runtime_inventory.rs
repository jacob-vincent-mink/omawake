//! Read-only discovery and isolated native validation. No model or service setup.
use crate::{
    backend::{BackendConfig, Runtime},
    config::Config,
    runtime_paths,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    env,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Evidence {
    pub versions: Vec<String>,
    pub provider_registration: bool,
    pub available_devices: Vec<String>,
    pub selected_device: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Probe {
    pub loadable: bool,
    pub device_accessible: bool,
    pub ready: bool,
    pub evidence: Evidence,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct State {
    pub runtime: &'static str,
    pub device: String,
    pub supported: bool,
    pub discovered: bool,
    pub source: &'static str,
    pub configured: bool,
    #[serde(flatten)]
    pub probe: Probe,
    pub paths: BackendConfig,
    pub remediation: Vec<String>,
}

pub fn name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
    }
}

pub fn inventory(config: &BackendConfig, path: &Path) -> Vec<State> {
    [ (Runtime::Default, &["auto", "cpu"][..]),
      (Runtime::Openvino, &["auto", "cpu", "gpu", "npu"][..]),
      (Runtime::Cuda, &["auto", "gpu"][..]) ]
    .into_iter().flat_map(|(runtime, devices)| devices.iter().map(move |device| (runtime, *device)))
    .map(|(runtime, device)| {
        let mut candidate = config.clone();
        if runtime != config.runtime { candidate.provider_library.clear(); candidate.device_id = 0; }
        candidate.runtime = runtime;
        candidate.device = device.into();
        let locations = runtime_paths::discover(&candidate, path);
        let exact = resolve(&candidate, path);
        let anchor = if runtime == Runtime::Default { &exact.onnxruntime_library } else { &exact.provider_library };
        let configured = path.is_file() && config.runtime == runtime && config.device.eq_ignore_ascii_case(device);
        let source = if locations.configured_library_dirs.iter().any(|dir| anchor.starts_with(dir)) { "configured" }
            else if locations.environment_library_dirs.iter().any(|dir| anchor.starts_with(dir)) { "environment" }
            else if locations.package_library_dirs.iter().any(|dir| anchor.starts_with(dir)) { "package" }
            else if env::var_os("LD_LIBRARY_PATH").is_some_and(|value| env::split_paths(&value).any(|dir| anchor.starts_with(dir))) { "environment" }
            else if anchor.is_file() { "system" } else { "candidate" };
        State { runtime: name(runtime), device: device.into(), supported: true,
            discovered: required(&exact).iter().all(|p| p.is_file()), source, configured,
            probe: probe(&candidate, path), paths: exact,
            remediation: vec![format!("Supply external {} libraries and dependencies, then run `omawake setup runtime --runtime {} --device {device} --dir /absolute/runtime --apply`. Required: ONNX Runtime 1.29.0, sherpa with the extended API 1.13.8{}.", name(runtime), name(runtime), match runtime {Runtime::Default => "", Runtime::Openvino => ", libonnxruntime_providers_openvino.so and Intel OpenVINO", Runtime::Cuda => ", libonnxruntime_providers_cuda.so and NVIDIA CUDA"})],
        }
    }).collect()
}

pub fn resolve(config: &BackendConfig, path: &Path) -> BackendConfig {
    let locations = runtime_paths::discover(config, path);
    resolve_with_locations(config, locations)
}

fn resolve_with_locations(
    config: &BackendConfig,
    locations: runtime_paths::RuntimeLibraryReport,
) -> BackendConfig {
    let mut candidate = config.clone();
    candidate.onnxruntime_library = locations.onnxruntime_library.unwrap_or_default();
    candidate.sherpa_library = locations.sherpa_library.unwrap_or_default();
    candidate.provider_library = locations.provider_library.unwrap_or_default();
    candidate.library_dirs = locations.configured_library_dirs;
    // The isolated child replaces LD_LIBRARY_PATH. Retain directories supplied
    // through OMAWAKE_LIBRARY_PATH so split provider/vendor stacks keep their
    // dependencies, while deliberately excluding the ambient loader path.
    for directory in locations.environment_library_dirs {
        if !candidate.library_dirs.contains(&directory) {
            candidate.library_dirs.push(directory);
        }
    }
    for library in [
        &candidate.onnxruntime_library,
        &candidate.sherpa_library,
        &candidate.provider_library,
    ] {
        if let Some(parent) = library.parent().filter(|p| !p.as_os_str().is_empty())
            && !candidate.library_dirs.iter().any(|dir| dir == parent)
        {
            candidate.library_dirs.push(parent.to_owned());
        }
    }
    candidate
}

fn required(config: &BackendConfig) -> Vec<&Path> {
    let mut paths = vec![
        config.onnxruntime_library.as_path(),
        config.sherpa_library.as_path(),
    ];
    if config.runtime != Runtime::Default {
        paths.push(&config.provider_library);
    }
    paths
}

pub fn probe(config: &BackendConfig, path: &Path) -> Probe {
    let exact = resolve(config, path);
    let result = (|| -> Result<Probe> {
        exact.validate_shape()?;
        if exact.runtime == Runtime::Cuda && exact.device_id != 0 {
            bail!(
                "the sherpa device probe cannot verify CUDA device_id {}; select device_id 0",
                exact.device_id
            );
        }
        for (library, required_name) in required(&exact).into_iter().zip([
            "libonnxruntime.so (ONNX Runtime 1.29.0)",
            "libsherpa-onnx-c-api.so (patched sherpa 1.13.8)",
            if exact.runtime == Runtime::Openvino {
                "libonnxruntime_providers_openvino.so"
            } else {
                "libonnxruntime_providers_cuda.so"
            },
        ]) {
            if !library.is_file() {
                bail!(
                    "required native library missing: {required_name}; selected path: {}",
                    library.display()
                );
            }
        }
        for directory in &exact.library_dirs {
            if !directory.is_absolute() || !directory.is_dir() {
                bail!("invalid dependency directory: {}", directory.display());
            }
        }
        isolated(&exact)
    })();
    result.unwrap_or_else(|error| Probe {
        errors: vec![format!("{error:#}")],
        ..Probe::default()
    })
}

fn isolated(config: &BackendConfig) -> Result<Probe> {
    let mut command = Command::new(env::current_exe()?);
    command
        .arg("__inventory-probe")
        .arg(serde_json::to_string(config)?)
        .env("LD_LIBRARY_PATH", env::join_paths(&config.library_dirs)?)
        // ORT 1.29 POSIX initializes telemetry storage unless explicitly disabled.
        .env("ORT_DISABLE_TELEMETRY", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = command.spawn().context("start isolated native probe")?;
    let start = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if start.elapsed() > Duration::from_secs(20) {
            child.kill()?;
            child.wait()?;
            bail!("native probe timed out after 20 seconds");
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!("native probe terminated: {}", output.status);
    }
    serde_json::from_slice(&output.stdout).context("read isolated native probe evidence")
}

/// Only called by the hidden child-process entry point.
pub fn child(config: &BackendConfig) -> Probe {
    let mut result = Probe::default();
    let attempt = (|| -> Result<()> {
        crate::engine::sherpa::validate_runtime_libraries(
            &config.onnxruntime_library,
            &config.sherpa_library,
        )?;
        result.evidence.versions = vec![
            "ONNX Runtime 1.29.0".into(),
            "sherpa-onnx 1.13.8; extended sherpa API v1".into(),
        ];
        if config.runtime != Runtime::Default {
            let device = config.canonical_device()?;
            let (provider, target) = match (config.runtime, device.as_str()) {
                (Runtime::Openvino, "auto") => ("OpenVINOExecutionProvider.AUTO", ""),
                (Runtime::Openvino, _) => ("OpenVINOExecutionProvider", device.as_str()),
                (Runtime::Cuda, _) => ("CUDAExecutionProvider", "gpu"),
                _ => unreachable!(),
            };
            crate::engine::sherpa::validate_runtime_provider_with(
                &config.onnxruntime_library,
                &config.sherpa_library,
                &config.provider_library,
                "omawake-setup-probe",
                provider,
                target,
                || {
                    result.loadable = true;
                    result.evidence.provider_registration = true;
                },
            )?;
            result.evidence.provider_registration = true;
        }
        result.loadable = true;
        result.device_accessible = true;
        result.ready = true;
        let device = config.canonical_device()?;
        result.evidence.selected_device = if config.runtime == Runtime::Default {
            Some("cpu".into())
        } else if device != "auto" {
            Some(device)
        } else {
            None
        };
        // This API proves the requested device only; it does not enumerate all devices.
        Ok(())
    })();
    if let Err(error) = attempt {
        result.errors.push(format!("{error:#}"));
    }
    result
}

pub fn apply_with(
    config: &Config,
    path: &Path,
    apply: bool,
    probe: impl FnOnce(&BackendConfig, &Path) -> Probe,
) -> Result<Probe> {
    let result = probe(&config.backend, path);
    if !result.ready {
        bail!(
            "runtime candidate rejected; config unchanged: {}",
            result.errors.join("; ")
        );
    }
    if apply {
        config.save(path)?;
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../tests/unit/runtime_inventory.rs"]
mod tests;
