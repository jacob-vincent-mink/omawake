use std::{
    collections::BTreeMap,
    env,
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::backend::{BackendConfig, Runtime};

pub const LIBRARY_PATH_ENV: &str = "OMAWAKE_LIBRARY_PATH";
pub const LIBRARY_PATH_READY_ENV: &str = "OMAWAKE_LIBRARY_PATH_READY";
pub const ONNXRUNTIME_LIBRARY_ENV: &str = "OMAWAKE_ONNXRUNTIME_LIBRARY";
pub const PROVIDER_LIBRARY_ENV: &str = "OMAWAKE_PROVIDER_LIBRARY";

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLibraryReport {
    pub onnxruntime_library: Option<PathBuf>,
    pub provider_library: Option<PathBuf>,
    pub configured_library_dirs: Vec<PathBuf>,
    pub environment_library_dirs: Vec<PathBuf>,
    pub package_library_dirs: Vec<PathBuf>,
    pub effective_library_dirs: Vec<PathBuf>,
    pub missing_library_dirs: Vec<PathBuf>,
    pub runtime_loadable: BTreeMap<&'static str, bool>,
    pub remediation: Vec<String>,
}

impl RuntimeLibraryReport {
    pub fn tui_context(&self) -> String {
        format!(
            "Configured: {}\r\nEffective: {}\r\nONNX Runtime: {}\r\nProvider: {}\r\nRemediation: {}",
            display_or_none(&self.configured_library_dirs),
            display_or_none(&self.effective_library_dirs),
            self.onnxruntime_library
                .as_deref()
                .map_or_else(|| "not found".into(), |path| path.display().to_string()),
            self.provider_library
                .as_deref()
                .map_or_else(|| "not selected".into(), |path| path.display().to_string()),
            if self.remediation.is_empty() {
                "none".into()
            } else {
                self.remediation.join("; ")
            },
        )
    }
}

pub fn report(config: &BackendConfig, config_path: &Path) -> RuntimeLibraryReport {
    let mut report = discover(config, config_path);
    report.remediation.clear();
    for runtime in [Runtime::Default, Runtime::Openvino, Runtime::Cuda] {
        let mut candidate = config.clone();
        candidate.runtime = runtime;
        candidate.device = "auto".into();
        candidate.device_id = 0;
        if runtime != config.runtime {
            candidate.provider_library.clear();
        }
        let probe = crate::runtime_inventory::probe(&candidate, config_path);
        report
            .runtime_loadable
            .insert(crate::runtime_inventory::name(runtime), probe.ready);
        if !probe.ready {
            let action = if runtime == Runtime::Default {
                "reinstall Omawake to restore its ONNX Runtime 1.30.0 core"
            } else {
                "install a compatible official provider plugin and select its directory with `omawake setup runtime`"
            };
            report.remediation.push(format!(
                "{}: {}; {action}",
                crate::runtime_inventory::name(runtime),
                probe.errors.join("; ")
            ));
        }
    }
    report
}

/// Resolve app-owned core and optional provider libraries without loading code.
pub fn discover(config: &BackendConfig, config_path: &Path) -> RuntimeLibraryReport {
    let executable = env::current_exe().ok();
    let absolute_config = if config_path.is_absolute() {
        config_path.to_owned()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(config_path)
    };
    let config_directory = absolute_config
        .parent()
        .map(Path::to_owned)
        .unwrap_or_else(|| PathBuf::from("."));
    let configured_library_dirs = deduplicate(
        config
            .library_dirs
            .iter()
            .map(|path| absolute_from(path, &config_directory))
            .chain(
                [&config.onnxruntime_library, &config.provider_library]
                    .into_iter()
                    .filter(|path| !path.as_os_str().is_empty())
                    .filter_map(|path| {
                        absolute_from(path, &config_directory)
                            .parent()
                            .map(Path::to_owned)
                    }),
            ),
    );
    let mut environment_library_dirs = split_paths(env::var_os(LIBRARY_PATH_ENV).as_deref());
    environment_library_dirs.extend(
        [
            env::var_os(ONNXRUNTIME_LIBRARY_ENV),
            env::var_os(PROVIDER_LIBRARY_ENV),
        ]
        .into_iter()
        .flatten()
        .filter_map(|path| PathBuf::from(path).parent().map(Path::to_owned)),
    );
    environment_library_dirs = deduplicate(environment_library_dirs);
    let package_library_dirs = executable
        .as_deref()
        .map_or_else(Vec::new, package_library_dirs);
    let effective_library_dirs = deduplicate(
        configured_library_dirs
            .iter()
            .chain(&environment_library_dirs)
            .chain(&package_library_dirs)
            .cloned(),
    );
    let missing_library_dirs: Vec<PathBuf> = effective_library_dirs
        .iter()
        .filter(|path| !path.is_absolute() || !path.is_dir())
        .cloned()
        .collect();
    let loader_dirs = split_paths(env::var_os("LD_LIBRARY_PATH").as_deref());
    let search_dirs = deduplicate(effective_library_dirs.iter().chain(&loader_dirs).cloned());
    let ldconfig = Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    let onnxruntime_library = exact_or_discover(
        &config.onnxruntime_library,
        env::var_os(ONNXRUNTIME_LIBRARY_ENV)
            .as_deref()
            .map(Path::new),
        "libonnxruntime.so",
        &config_directory,
        &search_dirs,
        ldconfig.as_deref(),
    );
    let provider_names: &[&str] = match config.runtime {
        Runtime::Default => &[],
        Runtime::Openvino => &[
            "libonnxruntime_providers_openvino_plugin.so",
            "libonnxruntime_providers_openvino.so",
        ],
        Runtime::Cuda => &["libonnxruntime_providers_cuda.so"],
    };
    let provider_library = if provider_names.is_empty() {
        None
    } else if !config.provider_library.as_os_str().is_empty() {
        Some(absolute_from(&config.provider_library, &config_directory))
    } else if let Some(path) = env::var_os(PROVIDER_LIBRARY_ENV).filter(|path| !path.is_empty()) {
        Some(PathBuf::from(path))
    } else {
        provider_names
            .iter()
            .find_map(|name| runtime_library(name, &search_dirs, ldconfig.as_deref()))
    };
    let core_ready = missing_library_dirs.is_empty()
        && onnxruntime_library.as_deref().is_some_and(Path::is_file);
    let mut remediation = Vec::new();
    if !core_ready {
        remediation
            .push("the app-owned ONNX Runtime 1.30.0 core is missing; reinstall Omawake".into());
    } else if config.runtime != Runtime::Default
        && !provider_library.as_deref().is_some_and(Path::is_file)
    {
        remediation.push(
            "the selected provider plugin is missing; install a compatible official plugin package and select its directory with `omawake setup runtime`".into(),
        );
    }
    RuntimeLibraryReport {
        onnxruntime_library,
        provider_library,
        configured_library_dirs,
        environment_library_dirs,
        package_library_dirs,
        effective_library_dirs,
        missing_library_dirs,
        runtime_loadable: BTreeMap::from([
            ("default", core_ready),
            (
                "openvino",
                core_ready
                    && [
                        "libonnxruntime_providers_openvino_plugin.so",
                        "libonnxruntime_providers_openvino.so",
                    ]
                    .iter()
                    .any(|name| runtime_library(name, &search_dirs, ldconfig.as_deref()).is_some()),
            ),
            (
                "cuda",
                core_ready
                    && runtime_library(
                        "libonnxruntime_providers_cuda.so",
                        &search_dirs,
                        ldconfig.as_deref(),
                    )
                    .is_some(),
            ),
        ]),
        remediation,
    }
}

pub fn effective_library_path(
    config: &BackendConfig,
    config_path: &Path,
) -> Result<Option<OsString>> {
    let report = discover(config, config_path);
    validate(&report)?;
    (!report.effective_library_dirs.is_empty())
        .then(|| env::join_paths(&report.effective_library_dirs))
        .transpose()
        .context("backend library paths cannot be represented in the loader environment")
}

#[cfg(target_os = "linux")]
pub fn ensure_engine_library_path(config: &BackendConfig, config_path: &Path) -> Result<()> {
    use std::os::unix::process::CommandExt;
    let report = discover(config, config_path);
    validate(&report)?;
    if report.effective_library_dirs.is_empty() {
        return Ok(());
    }
    let current = deduplicate(split_paths(env::var_os("LD_LIBRARY_PATH").as_deref()));
    if report
        .effective_library_dirs
        .iter()
        .all(|path| equivalent_in(&current, path))
    {
        return Ok(());
    }
    if env::var_os(LIBRARY_PATH_READY_ENV).is_some() {
        bail!("app-owned native libraries remain unavailable after loader re-exec");
    }
    let combined = deduplicate(
        report
            .effective_library_dirs
            .iter()
            .chain(&current)
            .cloned(),
    );
    let error = Command::new(env::current_exe()?)
        .args(env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", env::join_paths(combined)?)
        .env(LIBRARY_PATH_READY_ENV, "1")
        .exec();
    Err(error).context("re-exec Omawake with its app-owned libraries")
}

#[cfg(not(target_os = "linux"))]
pub fn ensure_engine_library_path(config: &BackendConfig, config_path: &Path) -> Result<()> {
    validate(&discover(config, config_path))
}

fn validate(report: &RuntimeLibraryReport) -> Result<()> {
    if !report.missing_library_dirs.is_empty() {
        bail!(
            "native library directories must be absolute existing directories: {}",
            display(&report.missing_library_dirs)
        );
    }
    Ok(())
}

fn package_library_dirs(executable: &Path) -> Vec<PathBuf> {
    let Some(binary) = executable.parent() else {
        return Vec::new();
    };
    deduplicate(
        [
            binary.join("lib"),
            binary.to_owned(),
            binary.join("../lib/omawake"),
        ]
        .into_iter()
        .filter(|directory| {
            library_in_directory("libonnxruntime.so", directory)
                || library_in_directory("libonnxruntime_providers_openvino_plugin.so", directory)
                || library_in_directory("libonnxruntime_providers_openvino.so", directory)
                || library_in_directory("libonnxruntime_providers_cuda.so", directory)
        }),
    )
}

fn exact_or_discover(
    configured: &Path,
    environment: Option<&Path>,
    name: &str,
    base: &Path,
    dirs: &[PathBuf],
    ldconfig: Option<&str>,
) -> Option<PathBuf> {
    if !configured.as_os_str().is_empty() {
        return Some(absolute_from(configured, base));
    }
    environment
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_owned)
        .or_else(|| runtime_library(name, dirs, ldconfig))
}

fn runtime_library(name: &str, dirs: &[PathBuf], ldconfig: Option<&str>) -> Option<PathBuf> {
    dirs.iter()
        .find_map(|directory| library_path_in_directory(name, directory))
        .or_else(|| {
            ldconfig?.lines().find_map(|line| {
                let (description, path) = line.split_once("=>")?;
                description
                    .contains(name)
                    .then(|| PathBuf::from(path.trim()))
            })
        })
}

fn library_in_directory(name: &str, directory: &Path) -> bool {
    library_path_in_directory(name, directory).is_some()
}

fn library_path_in_directory(name: &str, directory: &Path) -> Option<PathBuf> {
    let mut matches = fs::read_dir(directory)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            (entry
                .file_type()
                .is_ok_and(|kind| kind.is_file() || kind.is_symlink())
                && (file_name == name || file_name.starts_with(&format!("{name}."))))
            .then(|| entry.path())
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches.into_iter().next()
}

fn absolute_from(path: &Path, base: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}
fn split_paths(value: Option<&OsStr>) -> Vec<PathBuf> {
    value
        .filter(|value| !value.is_empty())
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .collect()
}
fn deduplicate(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for path in paths {
        if !equivalent_in(&result, &path) {
            result.push(path);
        }
    }
    result
}
fn equivalent_in(paths: &[PathBuf], candidate: &Path) -> bool {
    let canonical = candidate
        .canonicalize()
        .unwrap_or_else(|_| candidate.to_owned());
    paths.iter().any(|path| {
        path == candidate || path.canonicalize().unwrap_or_else(|_| path.clone()) == canonical
    })
}
fn display(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
fn display_or_none(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        "none".into()
    } else {
        display(paths)
    }
}
