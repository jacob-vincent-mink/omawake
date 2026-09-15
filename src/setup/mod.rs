pub mod cache;
pub mod menu;
pub mod model;
pub mod systemd;
pub mod wizard;

use std::path::Path;

use anyhow::{Result, bail};
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

pub fn ensure_config(path: &Path) -> Result<Config> {
    if path.exists() {
        Config::load(path)
    } else {
        let config = Config::default();
        config.save(path)?;
        Ok(config)
    }
}

pub fn checks(path: &Path, paths: &AppPaths) -> Vec<Check> {
    checks_with(
        path,
        paths,
        &|config, paths| {
            let probe = crate::runtime_inventory::probe(&config.backend, &paths.config_file);
            if !probe.ready {
                bail!("{}", probe.errors.join("; "));
            }
            Ok("runtime/device probe passed; model inference is not verified by this check".into())
        },
        command_exists("systemctl"),
        &systemd::is_active,
    )
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
        fail(
            "launcher",
            format!("desktop launcher is missing: {}", launcher.display()),
            "run `omawake setup menu`",
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
    let libraries = crate::runtime_paths::report(&config.backend, config_path);
    let inventory = crate::runtime_inventory::inventory(&config.backend, config_path);
    let value = serde_json::json!({
        "backends": catalog::backends(),
        "supported_capabilities": crate::backend::supported_capabilities(),
        "runtime_device_matrix": {
            "default": ["auto", "cpu"],
            "cuda": ["auto", "gpu"],
            "openvino": ["auto", "cpu", "gpu", "npu"]
        },
        "models": catalog::models(),
        "libraries": libraries,
        "inventory": inventory,
        "loader_environment": std::env::var_os("LD_LIBRARY_PATH")
            .map(|value| value.to_string_lossy().into_owned()),
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        for state in &inventory {
            println!(
                "{} / {}: supported={} discovered={} configured={} loadable={} device_accessible={} ready={} source={}",
                state.runtime,
                state.device,
                state.supported,
                state.discovered,
                state.configured,
                state.probe.loadable,
                state.probe.device_accessible,
                state.probe.ready,
                state.source
            );
            for error in &state.probe.errors {
                println!("  error: {error}");
            }
            if !state.probe.ready {
                for action in &state.remediation {
                    println!("  fix: {action}");
                }
            }
        }
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
            "Runtime loader capabilities: {}",
            crate::backend::supported_capabilities().join(", ")
        );
        println!("Runtime library discovery:");
        println!(
            "  configured: {}",
            format_paths(&libraries.configured_library_dirs)
        );
        println!(
            "  environment ({}): {}",
            crate::runtime_paths::LIBRARY_PATH_ENV,
            format_paths(&libraries.environment_library_dirs)
        );
        println!(
            "  package: {}",
            format_paths(&libraries.package_library_dirs)
        );
        println!(
            "  effective: {}",
            format_paths(&libraries.effective_library_dirs)
        );
        println!(
            "  missing: {}",
            format_paths(&libraries.missing_library_dirs)
        );
        println!(
            "  ONNX Runtime: {}",
            libraries
                .onnxruntime_library
                .as_deref()
                .map_or_else(|| "(not found)".into(), |path| path.display().to_string())
        );
        println!(
            "  sherpa: {}",
            libraries
                .sherpa_library
                .as_deref()
                .map_or_else(|| "(not found)".into(), |path| path.display().to_string())
        );
        println!(
            "  provider: {}",
            libraries.provider_library.as_deref().map_or_else(
                || "(not selected)".into(),
                |path| path.display().to_string()
            )
        );
        println!("Runtime/device choices:");
        println!(
            "  default   auto, cpu                 {}",
            if libraries.runtime_loadable["default"] {
                "runtime ready"
            } else {
                "runtime not found"
            }
        );
        println!(
            "  openvino  auto, cpu, gpu, npu       {}",
            if libraries.runtime_loadable["openvino"] {
                "external runtime ready"
            } else {
                "external runtime not found"
            }
        );
        println!(
            "  cuda      auto, gpu                 {}",
            if libraries.runtime_loadable["cuda"] {
                "external runtime ready"
            } else {
                "external runtime not found"
            }
        );
        println!(
            "Configure backend.runtime and backend.device independently; unsupported combinations fail unless fallback = \"cpu\"."
        );
        println!("Browse catalog models and license status with `omawake setup model --list`.");
        for remediation in &libraries.remediation {
            println!("  fix: {remediation}");
        }
    }
    Ok(())
}

fn format_paths(paths: &[std::path::PathBuf]) -> String {
    if paths.is_empty() {
        "(none)".into()
    } else {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(":")
    }
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
