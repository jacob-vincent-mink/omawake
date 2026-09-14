use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::backend::{BackendConfig, Runtime};

pub const LIBRARY_PATH_ENV: &str = "OMAWAKE_LIBRARY_PATH";
pub const LIBRARY_PATH_READY_ENV: &str = "OMAWAKE_LIBRARY_PATH_READY";
pub const ONNXRUNTIME_LIBRARY_ENV: &str = "OMAWAKE_ONNXRUNTIME_LIBRARY";
pub const SHERPA_LIBRARY_ENV: &str = "OMAWAKE_SHERPA_LIBRARY";
pub const PROVIDER_LIBRARY_ENV: &str = "OMAWAKE_PROVIDER_LIBRARY";

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeLibraryReport {
    pub onnxruntime_library: Option<PathBuf>,
    pub sherpa_library: Option<PathBuf>,
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
            "Configured: {}\r\nEffective: {}\r\nONNX Runtime: {}\r\nSherpa: {}\r\nProvider: {}\r\nRemediation: {}",
            display_paths_or_none(&self.configured_library_dirs),
            display_paths_or_none(&self.effective_library_dirs),
            self.onnxruntime_library
                .as_deref()
                .map_or_else(|| "not found".into(), |path| path.display().to_string()),
            self.sherpa_library
                .as_deref()
                .map_or_else(|| "not found".into(), |path| path.display().to_string()),
            self.provider_library
                .as_deref()
                .map_or_else(|| "not selected".into(), |path| path.display().to_string()),
            if self.remediation.is_empty() {
                "none".to_owned()
            } else {
                self.remediation.join("; ")
            }
        )
    }
}

pub fn report(config: &BackendConfig, config_path: &Path) -> RuntimeLibraryReport {
    let executable = env::current_exe().ok();
    let mut app_environment_dirs = split_paths(env::var_os(LIBRARY_PATH_ENV).as_deref());
    app_environment_dirs.extend(
        [
            env::var_os(ONNXRUNTIME_LIBRARY_ENV),
            env::var_os(SHERPA_LIBRARY_ENV),
            env::var_os(PROVIDER_LIBRARY_ENV),
        ]
        .into_iter()
        .flatten()
        .filter(|path| !path.is_empty())
        .filter_map(|path| PathBuf::from(path).parent().map(Path::to_owned)),
    );
    let app_environment = env::join_paths(app_environment_dirs).ok();
    let loader_environment = env::var_os("LD_LIBRARY_PATH");
    let ldconfig = Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
    report_with(
        config,
        config_path,
        executable.as_deref(),
        app_environment.as_deref(),
        loader_environment.as_deref(),
        ldconfig.as_deref(),
        [
            env::var_os(ONNXRUNTIME_LIBRARY_ENV).map(PathBuf::from),
            env::var_os(SHERPA_LIBRARY_ENV).map(PathBuf::from),
            env::var_os(PROVIDER_LIBRARY_ENV).map(PathBuf::from),
        ],
        provider_dependencies_resolve,
        |ort, sherpa| crate::engine::sherpa::validate_runtime_libraries(ort, sherpa).is_ok(),
        provider_runtime_loadable,
    )
}

#[allow(clippy::too_many_arguments)]
fn report_with(
    config: &BackendConfig,
    config_path: &Path,
    executable: Option<&Path>,
    app_environment: Option<&OsStr>,
    loader_environment: Option<&OsStr>,
    ldconfig: Option<&str>,
    environment_libraries: [Option<PathBuf>; 3],
    provider_loadable: impl Fn(&Path, &[PathBuf]) -> bool,
    stack_loadable: impl Fn(&Path, &Path) -> bool,
    provider_runtime_loadable: impl Fn(&Path, &Path, &Path, Runtime, &str, &[PathBuf]) -> bool,
) -> RuntimeLibraryReport {
    let absolute_config_path = if config_path.is_absolute() {
        config_path.to_owned()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(config_path)
    };
    let config_directory = absolute_config_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let configured_library_dirs = deduplicate_paths(
        config
            .library_dirs
            .iter()
            .map(|path| {
                if path.is_absolute() {
                    path.clone()
                } else {
                    config_directory.join(path)
                }
            })
            .chain(
                [
                    &config.onnxruntime_library,
                    &config.sherpa_library,
                    &config.provider_library,
                ]
                .into_iter()
                .filter(|path| !path.as_os_str().is_empty())
                .filter_map(|path| {
                    let path = if path.is_absolute() {
                        path.clone()
                    } else {
                        config_directory.join(path)
                    };
                    path.parent().map(Path::to_owned)
                }),
            ),
    );
    let environment_library_dirs = split_paths(app_environment);
    let package_library_dirs = executable.map_or_else(Vec::new, package_library_dirs);
    let effective_library_dirs = deduplicate_paths(
        configured_library_dirs
            .iter()
            .chain(&environment_library_dirs)
            .chain(&package_library_dirs)
            .cloned(),
    );
    let missing_library_dirs = effective_library_dirs
        .iter()
        .filter(|path| !path.is_absolute() || !path.is_dir())
        .cloned()
        .collect::<Vec<_>>();
    let search_dirs = deduplicate_paths(
        effective_library_dirs
            .iter()
            .chain(split_paths(loader_environment).iter())
            .cloned(),
    );
    let paths_valid = missing_library_dirs.is_empty();
    let onnxruntime_library = effective_library(
        &config.onnxruntime_library,
        environment_libraries[0].as_deref(),
        "libonnxruntime.so",
        config_directory,
        &search_dirs,
        ldconfig,
    );
    let sherpa_library = effective_library(
        &config.sherpa_library,
        environment_libraries[1].as_deref(),
        "libsherpa-onnx-c-api.so",
        config_directory,
        &search_dirs,
        ldconfig,
    );
    let provider_prefix = match config.runtime {
        crate::backend::Runtime::Openvino => "libonnxruntime_providers_openvino.so",
        crate::backend::Runtime::Cuda => "libonnxruntime_providers_cuda.so",
        crate::backend::Runtime::Default => "",
    };
    let provider_library = (!provider_prefix.is_empty())
        .then(|| {
            effective_library(
                &config.provider_library,
                environment_libraries[2].as_deref(),
                provider_prefix,
                config_directory,
                &search_dirs,
                ldconfig,
            )
        })
        .flatten();
    let base_loadable = paths_valid
        && onnxruntime_library.as_deref().is_some_and(Path::is_file)
        && sherpa_library
            .as_deref()
            .is_some_and(|library| library.is_file() && provider_loadable(library, &search_dirs))
        && onnxruntime_library
            .as_deref()
            .zip(sherpa_library.as_deref())
            .is_some_and(|(ort, sherpa)| stack_loadable(ort, sherpa));
    let openvino_library = if config.runtime == crate::backend::Runtime::Openvino {
        provider_library.clone()
    } else {
        runtime_library(
            "libonnxruntime_providers_openvino.so",
            &search_dirs,
            ldconfig,
        )
    };
    let cuda_library = if config.runtime == crate::backend::Runtime::Cuda {
        provider_library.clone()
    } else {
        runtime_library("libonnxruntime_providers_cuda.so", &search_dirs, ldconfig)
    };
    let openvino_device = if config.runtime == Runtime::Openvino {
        config.canonical_device().ok()
    } else {
        Some("auto".to_owned())
    };
    let cuda_device = if config.runtime == Runtime::Cuda {
        config.canonical_device().ok()
    } else {
        Some("auto".to_owned())
    };
    let provider_ready = |provider: Option<&Path>, runtime, device: Option<&str>| {
        base_loadable
            && provider.is_some_and(|provider| {
                provider.is_file()
                    && provider_loadable(provider, &search_dirs)
                    && onnxruntime_library
                        .as_deref()
                        .zip(sherpa_library.as_deref())
                        .zip(device)
                        .is_some_and(|((ort, sherpa), device)| {
                            provider_runtime_loadable(
                                ort,
                                sherpa,
                                provider,
                                runtime,
                                device,
                                &search_dirs,
                            )
                        })
            })
    };
    let runtime_loadable = BTreeMap::from([
        ("default", base_loadable),
        (
            "openvino",
            provider_ready(
                openvino_library.as_deref(),
                Runtime::Openvino,
                openvino_device.as_deref(),
            ),
        ),
        (
            "cuda",
            provider_ready(
                cuda_library.as_deref(),
                Runtime::Cuda,
                cuda_device.as_deref(),
            ),
        ),
    ]);
    let mut remediation = Vec::new();
    if !missing_library_dirs.is_empty() {
        remediation.push(format!(
            "remove or correct missing/non-absolute directories in backend.library_dirs or {LIBRARY_PATH_ENV}"
        ));
    }
    if !base_loadable {
        remediation.push(format!(
            "provide matching ONNX Runtime 1.29.0 and patched sherpa-onnx 1.13.8 libraries, then set their exact paths or add their directories to backend.library_dirs or {LIBRARY_PATH_ENV}"
        ));
    }
    if base_loadable {
        for (runtime, provider, device) in [
            (
                "openvino",
                openvino_library.as_deref(),
                openvino_device.as_deref().unwrap_or("invalid"),
            ),
            (
                "cuda",
                cuda_library.as_deref(),
                cuda_device.as_deref().unwrap_or("invalid"),
            ),
        ] {
            if !runtime_loadable[runtime] {
                remediation.push(provider.map_or_else(
                    || {
                        format!(
                            "set backend.library_dirs or {LIBRARY_PATH_ENV} to the directory containing the {runtime} ONNX Runtime provider"
                        )
                    },
                    |provider| {
                        format!(
                            "the {runtime} provider {} failed dependency resolution, registration, or its {device} device probe; supply a provider matching the selected ONNX Runtime and vendor stack",
                            provider.display()
                        )
                    },
                ));
            }
        }
    }
    RuntimeLibraryReport {
        onnxruntime_library,
        sherpa_library,
        provider_library,
        configured_library_dirs,
        environment_library_dirs,
        package_library_dirs,
        effective_library_dirs,
        missing_library_dirs,
        runtime_loadable,
        remediation,
    }
}

fn effective_library(
    configured: &Path,
    environment: Option<&Path>,
    prefix: &str,
    config_directory: &Path,
    search_dirs: &[PathBuf],
    ldconfig: Option<&str>,
) -> Option<PathBuf> {
    if !configured.as_os_str().is_empty() {
        return Some(if configured.is_absolute() {
            configured.to_owned()
        } else {
            config_directory.join(configured)
        });
    }
    if let Some(environment) = environment.filter(|path| !path.as_os_str().is_empty()) {
        return Some(environment.to_owned());
    }
    runtime_library(prefix, search_dirs, ldconfig)
}

pub fn effective_library_path(
    config: &BackendConfig,
    config_path: &Path,
) -> Result<Option<OsString>> {
    let report = report(config, config_path);
    validate_report(&report)?;
    if report.effective_library_dirs.is_empty() {
        Ok(None)
    } else {
        Ok(Some(
            env::join_paths(&report.effective_library_dirs)
                .context("backend library directory cannot be represented in a loader path")?,
        ))
    }
}

#[cfg(target_os = "linux")]
pub fn ensure_engine_library_path(config: &BackendConfig, config_path: &Path) -> Result<()> {
    use std::os::unix::process::CommandExt;

    let report = report(config, config_path);
    let current = env::var_os("LD_LIBRARY_PATH");
    let sentinel = env::var_os(LIBRARY_PATH_READY_ENV).is_some();
    let Some(library_path) = reexec_library_path(&report, current.as_deref(), sentinel)? else {
        return Ok(());
    };
    let executable = env::current_exe().context("locate the Omawake executable for re-exec")?;
    let error = Command::new(executable)
        .args(env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", library_path)
        .env(LIBRARY_PATH_READY_ENV, "1")
        .exec();
    Err(error).context("re-exec Omawake with its configured native library path")
}

#[cfg(not(target_os = "linux"))]
pub fn ensure_engine_library_path(config: &BackendConfig, config_path: &Path) -> Result<()> {
    validate_report(&report(config, config_path))
}

fn reexec_library_path(
    report: &RuntimeLibraryReport,
    current: Option<&OsStr>,
    sentinel: bool,
) -> Result<Option<OsString>> {
    validate_report(report)?;
    if report.effective_library_dirs.is_empty() {
        return Ok(None);
    }
    let current_paths = deduplicate_paths(split_paths(current));
    let missing_from_loader = report
        .effective_library_dirs
        .iter()
        .filter(|required| !contains_equivalent_path(&current_paths, required))
        .cloned()
        .collect::<Vec<_>>();
    if missing_from_loader.is_empty() {
        return Ok(None);
    }
    if sentinel {
        bail!(
            "configured native library directories are still absent after re-exec: {}",
            display_paths(&missing_from_loader)
        );
    }
    let augmented = deduplicate_paths(
        report
            .effective_library_dirs
            .iter()
            .chain(&current_paths)
            .cloned(),
    );
    Ok(Some(env::join_paths(augmented).context(
        "native library directories cannot be represented in LD_LIBRARY_PATH",
    )?))
}

fn validate_report(report: &RuntimeLibraryReport) -> Result<()> {
    if !report.missing_library_dirs.is_empty() {
        bail!(
            "native library directories must be absolute existing directories: {}; run `omawake setup runtime` for remediation",
            display_paths(&report.missing_library_dirs)
        );
    }
    Ok(())
}

fn split_paths(value: Option<&OsStr>) -> Vec<PathBuf> {
    value
        .filter(|value| !value.is_empty())
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .collect()
}

fn package_library_dirs(executable: &Path) -> Vec<PathBuf> {
    let Some(binary_dir) = executable.parent() else {
        return Vec::new();
    };
    let candidates = [
        binary_dir.join("lib"),
        binary_dir.to_owned(),
        binary_dir.join("../lib/omawake"),
    ];
    deduplicate_paths(
        candidates
            .into_iter()
            .filter(|directory| contains_runtime_anchor(directory)),
    )
}

fn contains_runtime_anchor(directory: &Path) -> bool {
    library_in_directory("libonnxruntime.so", directory)
        || library_in_directory("libsherpa-onnx-c-api.so", directory)
        || library_in_directory("libonnxruntime_providers_openvino.so", directory)
        || library_in_directory("libonnxruntime_providers_cuda.so", directory)
}

fn runtime_library(name: &str, search_dirs: &[PathBuf], ldconfig: Option<&str>) -> Option<PathBuf> {
    search_dirs
        .iter()
        .find_map(|directory| library_path_in_directory(name, directory))
        .or_else(|| {
            ldconfig.and_then(|output| {
                output.lines().find_map(|line| {
                    let (description, path) = line.split_once("=>")?;
                    description
                        .contains(name)
                        .then(|| PathBuf::from(path.trim()))
                })
            })
        })
}

fn library_in_directory(name: &str, directory: &Path) -> bool {
    library_path_in_directory(name, directory).is_some()
}

fn library_path_in_directory(name: &str, directory: &Path) -> Option<PathBuf> {
    fs::read_dir(directory).ok().and_then(|entries| {
        let mut matches = entries
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
    })
}

fn provider_dependencies_resolve(provider: &Path, search_dirs: &[PathBuf]) -> bool {
    let library_path = env::join_paths(search_dirs).ok();
    let mut command = Command::new("ldd");
    command.arg(provider);
    if let Some(library_path) = library_path {
        command.env("LD_LIBRARY_PATH", library_path);
    }
    command.output().is_ok_and(|output| {
        output.status.success()
            && !String::from_utf8_lossy(&output.stdout).contains("not found")
            && !String::from_utf8_lossy(&output.stderr).contains("not found")
    })
}

fn provider_runtime_loadable(
    ort: &Path,
    sherpa: &Path,
    provider: &Path,
    runtime: Runtime,
    device: &str,
    search_dirs: &[PathBuf],
) -> bool {
    let (registration, ep_name, device) = match (runtime, device) {
        (Runtime::Openvino, "auto") => (
            "omawake-readiness-openvino",
            "OpenVINOExecutionProvider.AUTO",
            "",
        ),
        (Runtime::Openvino, "cpu" | "gpu" | "npu") => (
            "omawake-readiness-openvino",
            "OpenVINOExecutionProvider",
            device,
        ),
        (Runtime::Cuda, "auto" | "gpu") => {
            ("omawake-readiness-cuda", "CUDAExecutionProvider", "gpu")
        }
        _ => return false,
    };
    let executable = match env::current_exe() {
        Ok(executable) => executable,
        Err(_) => return false,
    };
    let loader_dirs = deduplicate_paths(
        search_dirs
            .iter()
            .chain(split_paths(env::var_os("LD_LIBRARY_PATH").as_deref()).iter())
            .cloned(),
    );
    let loader_path = match env::join_paths(loader_dirs) {
        Ok(loader_path) => loader_path,
        Err(_) => return false,
    };
    Command::new(executable)
        .arg("__runtime-probe")
        .arg("--onnxruntime")
        .arg(ort)
        .arg("--sherpa")
        .arg(sherpa)
        .arg("--provider")
        .arg(provider)
        .arg("--registration")
        .arg(registration)
        .arg("--ep-name")
        .arg(ep_name)
        .arg("--device")
        .arg(device)
        .env("LD_LIBRARY_PATH", loader_path)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn deduplicate_paths(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::new();
    for path in paths {
        if !contains_equivalent_path(&unique, &path) {
            unique.push(path);
        }
    }
    unique
}

fn contains_equivalent_path(paths: &[PathBuf], candidate: &Path) -> bool {
    let canonical_candidate = candidate
        .canonicalize()
        .unwrap_or_else(|_| candidate.to_owned());
    paths.iter().any(|existing| {
        existing == candidate
            || existing.canonicalize().unwrap_or_else(|_| existing.clone()) == canonical_candidate
    })
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn display_paths_or_none(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        "none".to_owned()
    } else {
        display_paths(paths)
    }
}

#[cfg(test)]
#[path = "../tests/unit/runtime_paths.rs"]
mod tests;
