use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::paths::AppPaths;

const UNIT: &str = "omawake.service";

pub fn service_path(paths: &AppPaths) -> PathBuf {
    paths
        .config_file
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join("systemd/user/omawake.service")
}

pub fn generate(binary: &Path, config: &Path) -> String {
    generate_with_library_path(binary, config, None)
}

fn generate_with_library_path(
    binary: &Path,
    config: &Path,
    library_path: Option<&OsStr>,
) -> String {
    let library_environment = library_path
        .filter(|value| !value.is_empty())
        .map(|value| {
            format!(
                "Environment=\"LD_LIBRARY_PATH={}\"\n",
                escape_environment_value(value)
            )
        })
        .unwrap_or_default();
    format!(
        "[Unit]\nDescription=Omawake local wake-word daemon\nPartOf=graphical-session.target\nAfter=graphical-session.target pipewire.service\n\n[Service]\nType=simple\nExecStart={} --config {} daemon\nRestart=on-failure\nRestartSec=1\nEnvironment=XDG_RUNTIME_DIR=%t\n{}\n[Install]\nWantedBy=graphical-session.target\n",
        quote(binary),
        quote(config),
        library_environment
    )
}

pub fn install(paths: &AppPaths, config: &Path, start: bool) -> Result<PathBuf> {
    let path = service_path(paths);
    let binary = std::env::current_exe()?.canonicalize()?;
    let unit = generate_with_library_path(
        &binary,
        config,
        std::env::var_os("LD_LIBRARY_PATH").as_deref(),
    );
    write_atomic(&path, unit.as_bytes())?;
    systemctl(["daemon-reload"])?;
    if start {
        systemctl(["enable", UNIT])?;
        restart()?;
    } else {
        systemctl(["enable", UNIT])?;
    }
    Ok(path)
}

pub fn uninstall(paths: &AppPaths) -> Result<()> {
    let _ = systemctl(["disable", "--now", UNIT]);
    let path = service_path(paths);
    if path.exists() {
        fs::remove_file(&path)?;
    }
    systemctl(["daemon-reload"])
}

pub fn status(paths: &AppPaths) -> Result<()> {
    let path = service_path(paths);
    println!("unit: {}", path.display());
    if !path.exists() {
        bail!("omawake systemd user service is not installed");
    }
    let status = Command::new("systemctl")
        .args(["--user", "status", UNIT, "--no-pager"])
        .status()
        .context("run systemctl --user status")?;
    if !status.success() {
        bail!("omawake.service is not running");
    }
    Ok(())
}

pub fn is_active() -> bool {
    Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", UNIT])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn restart() -> Result<()> {
    systemctl(["restart", UNIT])?;
    if !is_active() {
        bail!("{UNIT} did not remain active after restart");
    }
    Ok(())
}

pub fn reload_if_was_active(was_active: bool) -> Result<bool> {
    reload_if_was_active_with(was_active, || systemctl(["try-restart", UNIT]), is_active)
}

fn reload_if_was_active_with(
    was_active: bool,
    try_restart: impl FnOnce() -> Result<()>,
    active_after: impl FnOnce() -> bool,
) -> Result<bool> {
    if !was_active {
        return Ok(false);
    }
    try_restart()?;
    if !active_after() {
        bail!("{UNIT} stopped while applying setup changes");
    }
    Ok(true)
}

fn systemctl<const N: usize>(args: [&str; N]) -> Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
        .context("run systemctl --user")?;
    if !status.success() {
        bail!("systemctl --user command failed");
    }
    Ok(())
}

fn quote(path: &Path) -> String {
    format!("\"{}\"", path.display().to_string().replace('"', "\\\""))
}

fn escape_environment_value(value: &OsStr) -> String {
    value
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('%', "%%")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_systemd.rs"]
mod tests;
