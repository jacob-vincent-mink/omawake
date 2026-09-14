pub mod menu;
pub mod model;
pub mod systemd;

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
            result.push(ok("backend", format!("{} is compiled", backend.kind)))
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
    match crate::engine::Detector::load(&config, paths) {
        Ok(detector) => result.push(ok(
            "engine",
            format!(
                "{} initialized {} wake-word mapping(s)",
                detector.backend_kind,
                config.wake_words.iter().filter(|word| word.enabled).count()
            ),
        )),
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
    result.push(if command_exists("systemctl") && service.is_file() {
        ok("systemd", service.display().to_string())
    } else {
        fail(
            "systemd",
            format!(
                "systemctl or user service is missing: {}",
                service.display()
            ),
            "run `omawake setup systemd`",
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

pub fn print_runtime(json: bool) -> Result<()> {
    let value = serde_json::json!({
        "backends": catalog::backends(),
        "compiled_capabilities": crate::backend::compiled_capabilities(),
        "runtime_device_matrix": {
            "default": ["auto", "cpu"],
            "cuda": ["auto", "gpu"],
            "openvino": ["auto", "cpu", "gpu", "npu", "AUTO:<devices>", "HETERO:<devices>", "MULTI:<devices>"]
        },
        "models": catalog::models(),
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
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
            "Compiled runtime capabilities: {}",
            crate::backend::compiled_capabilities().join(", ")
        );
        println!(
            "Configure backend.runtime and backend.device independently; unsupported combinations fail unless fallback = \"cpu\"."
        );
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
