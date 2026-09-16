pub mod cache;
pub mod menu;
pub mod model;
pub mod systemd;
pub mod wizard;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::catalog;
use crate::config::Config;
use crate::paths::AppPaths;

#[derive(Serialize)]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
    pub remediation: Option<String>,
}

/// Load configuration for setup without making obsolete pre-release files a
/// dead end. Invalid contents are represented by current defaults until an
/// explicit setup action succeeds and atomically saves the current schema.
/// Read-only setup paths therefore leave the original bytes untouched.
pub fn load_config(path: &Path) -> Result<Config> {
    setup_config(path).map(|(config, _)| config)
}

/// Report that setup is recovering from an obsolete configuration. I/O
/// failures remain fatal so an unreadable file is never mistaken for one that
/// setup may safely replace.
pub fn config_recovery(path: &Path) -> Result<Option<String>> {
    setup_config(path).map(|(_, error)| error)
}

fn setup_config(path: &Path) -> Result<(Config, Option<String>)> {
    let input = match fs::read_to_string(path) {
        Ok(input) => input,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Config::default(), None));
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read config {}", path.display()));
        }
    };
    match toml::from_str(&input) {
        Ok(config) => Ok((config, None)),
        Err(error) => Ok((
            Config::default(),
            Some(format!("parse config {}: {error}", path.display())),
        )),
    }
}

pub fn ensure_config(path: &Path) -> Result<Config> {
    let exists = path.exists();
    let config = load_config(path)?;
    if !exists {
        config.save(path)?;
    }
    Ok(config)
}

pub fn checks(path: &Path, paths: &AppPaths) -> Vec<Check> {
    let mut result = checks_with(
        path,
        paths,
        &|config, paths| {
            let detector = crate::engine::Detector::load(config, paths)?;
            let integration = match detector.backend_kind {
                "openvino-genai" => {
                    "official OpenVINO GenAI Whisper C API plus audio.cpp Silero VAD C API"
                }
                _ => "audio.cpp public C ABI",
            };
            Ok(format!(
                "{} initialized in {:.1} ms through {integration}",
                detector.backend_kind,
                detector.load_time.as_secs_f64() * 1000.0
            ))
        },
        command_exists("systemctl"),
        &systemd::is_active,
    );
    if let Ok(config) = Config::load(path) {
        let inventory = crate::audio::device_inventory(&config.audio.device);
        if crate::audio_devices::is_default(&config.audio.device) {
            result.push(ok(
                "audio_device",
                "System default (resolved when audio opens)",
            ));
        } else if inventory["devices"].as_array().is_some_and(|devices| {
            devices
                .iter()
                .any(|d| d["selector"] == config.audio.device && d["available"] == true)
        }) {
            result.push(ok("audio_device", config.audio.device));
        } else {
            result.push(fail(
                "audio_device",
                format!("{} is unavailable", config.audio.device),
                "connect the device or run `omawake setup audio`",
            ));
        }
    }
    result
}

fn checks_with(
    path: &Path,
    paths: &AppPaths,
    check_engine: &dyn Fn(&Config, &AppPaths) -> Result<String>,
    systemctl_available: bool,
    service_is_active: &dyn Fn() -> bool,
) -> Vec<Check> {
    let mut result = Vec::new();
    let config = match Config::load(path) {
        Ok(config) => {
            result.push(ok("config", path.display().to_string()));
            config
        }
        Err(error) => {
            result.push(fail(
                "config",
                format!("{error:#}"),
                "run `omawake setup all`",
            ));
            return result;
        }
    };

    match catalog::backends()
        .iter()
        .find(|backend| backend.kind == config.backend.kind)
    {
        Some(backend) if backend.built => {
            result.push(ok("backend", format!("{} is supported", backend.kind)))
        }
        Some(backend) => result.push(fail(
            "backend",
            format!("{} is registered but unavailable", backend.kind),
            "install a build containing that backend",
        )),
        None => result.push(fail(
            "backend",
            format!("{} is not registered", config.backend.kind),
            "choose a backend shown by `omawake setup runtime`",
        )),
    }

    match catalog::model(&config.model.name) {
        Some(spec) => match model::verify(paths, spec) {
            Ok(()) => result.push(ok(
                "model",
                model::model_directory(paths, spec).display().to_string(),
            )),
            Err(error) => result.push(fail(
                "model",
                format!("{error:#}"),
                format!("run `omawake setup model --download {}`", spec.id),
            )),
        },
        None => {
            let directory = config.model_directory(paths);
            if directory.exists() {
                result.push(ok(
                    "model",
                    format!("custom model at {}", directory.display()),
                ));
            } else {
                result.push(fail(
                    "model",
                    format!("custom model directory is missing: {}", directory.display()),
                    "set model.directory or install a catalog model",
                ));
            }
        }
    }
    let cache_device = config
        .backend
        .canonical_device()
        .unwrap_or_else(|_| config.backend.device.clone());
    match cache::status(&config, paths) {
        Ok(report) if !report.required => result.push(ok(
            "model-cache",
            "setup-time compiled cache is not required for this runtime/device",
        )),
        Ok(report) if report.prepared => result.push(ok(
            "model-cache",
            format!(
                "{} compiled artifact(s), {} bytes in {}",
                report.artifacts,
                report.bytes,
                report
                    .directory
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .display()
            ),
        )),
        Ok(report) => result.push(fail(
            "model-cache",
            format!(
                "OpenVINO {} model cache is not prepared: {}",
                cache_device.to_ascii_uppercase(),
                report
                    .directory
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<unknown>"))
                    .display()
            ),
            format!(
                "rerun OpenVINO {} runtime setup with `omawake setup runtime --runtime openvino --device {} --apply` after installing the model",
                cache_device.to_ascii_uppercase(),
                cache_device
            ),
        )),
        Err(error) => result.push(fail(
            "model-cache",
            format!("{error:#}"),
            "use Omawake's managed OpenVINO provider config and rerun GPU/NPU setup",
        )),
    }
    match check_engine(&config, paths) {
        Ok(detail) => result.push(ok("engine", detail)),
        Err(error) => result.push(fail(
            "engine",
            format!("{error:#}"),
            "fix the backend/runtime/model/wake-word settings shown above",
        )),
    }
    result.push(if !config.wake_words.is_empty() {
        ok(
            "wake-words",
            format!("{} mapping(s) configured", config.wake_words.len()),
        )
    } else {
        fail(
            "wake-words",
            "no wake-word mappings are configured",
            "run `omawake wake-word add --id NAME --phrase PHRASE -- COMMAND ...`",
        )
    });
    let launcher = menu::launcher_path(paths);
    result.push(if launcher.is_file() {
        ok("launcher", launcher.display().to_string())
    } else {
        ok(
            "launcher",
            format!(
                "not installed (optional); run `omawake setup menu` to add {}",
                launcher.display()
            ),
        )
    });
    let service = systemd::service_path(paths);
    let service_active = systemctl_available && service_is_active();
    result.push(if !systemctl_available {
        ok(
            "systemd",
            "not available (optional); run the daemon directly with `omawake daemon`",
        )
    } else if service.is_file() {
        ok(
            "systemd",
            format!(
                "optional service installed and {}: {}",
                if service_active { "active" } else { "inactive" },
                service.display()
            ),
        )
    } else if service_active {
        ok(
            "systemd",
            format!(
                "optional service is active from an external unit; no app-installed unit at {}",
                service.display()
            ),
        )
    } else {
        ok(
            "systemd",
            "not installed (optional); run `omawake setup systemd` to install it",
        )
    });
    result
}

pub fn print_checks(path: &Path, paths: &AppPaths, json: bool) -> Result<()> {
    let checks = checks(path, paths);
    if json {
        println!("{}", serde_json::to_string_pretty(&checks)?);
    } else {
        for check in &checks {
            println!(
                "{} {:<9} {}",
                if check.ok { "ok" } else { "error" },
                check.name,
                check.detail
            );
            if let Some(remediation) = &check.remediation {
                println!("  fix: {remediation}");
            }
        }
    }
    if checks.iter().any(|check| !check.ok) {
        bail!("setup checks failed")
    }
    Ok(())
}

pub fn print_checks_event(path: &Path, paths: &AppPaths) -> Result<()> {
    let checks = checks(path, paths);
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "event": "checks",
            "checks": checks,
        }))?
    );
    if checks.iter().any(|check| !check.ok) {
        bail!("setup checks failed")
    }
    Ok(())
}

pub fn print_runtime(config: &Config, config_path: &Path, json: bool) -> Result<()> {
    let configured = crate::runtime_inventory::probe(config, config_path);
    let hardware = crate::hardware::detect();
    let providers = crate::app::setup_provider_availability(config, config_path);
    let recommendation = crate::hardware::recommend(&hardware, providers);
    let mut candidate = config.clone();
    candidate.backend.kind = "audiocpp".into();
    candidate.backend.runtime = crate::backend::Runtime::Default;
    candidate.backend.device = "cpu".into();
    candidate.backend.device_id = 0;
    candidate.backend.library.clear();
    candidate.backend.library_dirs.clear();
    candidate.backend.options.clear();
    let mut probe_paths = AppPaths::discover();
    probe_paths.config_file = config_path.to_owned();
    let provider = crate::engine::audiocpp::probe_provider(&candidate, &probe_paths);
    let (loadable, library, version, error) = match provider {
        Ok((library, version)) => (true, Some(library), Some(version), None),
        Err(error) => (false, None, None, Some(format!("{error:#}"))),
    };
    let value = serde_json::json!({
        "backends": catalog::backends(),
        "integration": "Omawake dynamically loads audio.cpp's public C ABI; it never invokes the audio.cpp CLI. The hidden worker is an Omawake re-exec for isolation and warm sessions.",
        "runtime_device_matrix": {
            "default": ["cpu"],
            "openvino": ["cpu", "gpu", "npu"],
            "cuda": ["gpu"],
            "vulkan": ["gpu"],
            "hip": ["gpu"]
        },
        "models": catalog::models(),
        "provider": {
            "kind": "audiocpp",
            "loadable": loadable,
            "ready": false,
            "library": library,
            "version": version,
            "error": error,
        },
        "configured_provider": {
            "runtime": crate::runtime_inventory::name(config.backend.runtime),
            "device": config.backend.device,
            "probe": configured,
        },
        "hardware": hardware,
        "provider_availability": providers,
        "recommendation": recommendation,
        "runtime_installation": "Omawake discovers complete provider directories supplied by the user; setup never installs vendor runtimes.",
        "loader_environment": std::env::var_os("LD_LIBRARY_PATH")
            .map(|value| value.to_string_lossy().into_owned()),
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        println!("Integration: direct audio.cpp public C ABI library; no audio.cpp CLI process");
        println!("Recommendation: {}", recommendation.detail);
        if hardware.devices.is_empty() {
            println!("Detected accelerator hardware: none");
        } else {
            for device in &hardware.devices {
                println!(
                    "Detected hardware: {} vendor={} class={} driver={} capabilities={}",
                    device.address,
                    device.vendor,
                    device.class,
                    device.driver.as_deref().unwrap_or("unbound"),
                    device.capabilities.join(",")
                );
            }
        }
        println!(
            "Readiness: hardware discovery and provider availability are advisory; model proof runs only at Apply."
        );
        println!("Backends:");
        for backend in catalog::backends() {
            println!(
                "  {}\t{}\t{}",
                backend.kind,
                if backend.built { "built" } else { "not built" },
                backend.description
            );
        }
        println!(
            "Provider: {}",
            if loadable {
                "found"
            } else {
                "not found or incompatible"
            }
        );
        if let Some(path) = library {
            println!("  library: {}", path.display());
        }
        if let Some(version) = version {
            println!("  version: {version}");
        }
        if let Some(error) = error {
            println!("  error: {error}");
            println!(
                "  fix: install the Omawake release package or choose a complete libaudiocpp build directory"
            );
        }
        println!(
            "Selected: {} / {} ({})",
            crate::runtime_inventory::name(config.backend.runtime),
            config.backend.device,
            if configured.loadable {
                "provider found"
            } else {
                "needs setup"
            }
        );
        for error in &configured.errors {
            println!("  selected provider: {error}");
        }
        println!("Runtime/device choices:");
        println!("  default   cpu                       integrated audio.cpp provider");
        println!("  openvino  Intel CPU, GPU, NPU       external complete OpenVINO GenAI install");
        println!("  cuda      NVIDIA GPU                external complete audio.cpp provider");
        println!("  vulkan    Vulkan GPU                external complete audio.cpp provider");
        println!("  hip       AMD GPU                   external complete audio.cpp provider");
        println!(
            "Setup discovers or accepts complete provider builds; it never installs a system runtime."
        );
        println!("Browse catalog models and license status with `omawake setup model --list`.");
    }
    Ok(())
}

fn command_exists(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| directory.join(name).is_file())
    })
}

fn ok(name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        name: name.into(),
        ok: true,
        detail: detail.into(),
        remediation: None,
    }
}

fn fail(
    name: impl Into<String>,
    detail: impl Into<String>,
    remediation: impl Into<String>,
) -> Check {
    Check {
        name: name.into(),
        ok: false,
        detail: detail.into(),
        remediation: Some(remediation.into()),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/setup_mod.rs"]
mod tests;

pub mod audio;
