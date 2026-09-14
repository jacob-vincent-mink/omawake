use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths::AppPaths;

pub fn service_path(paths: &AppPaths) -> PathBuf {
    paths
        .config_file
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join("systemd/user/omawake.service")
}

pub fn generate(binary: &Path, config: &Path) -> String {
    format!(
        "[Unit]\nDescription=Omawake local wake-word daemon\nPartOf=graphical-session.target\nAfter=graphical-session.target pipewire.service\n\n[Service]\nType=simple\nExecStart={} --config {} daemon\nRestart=on-failure\nRestartSec=1\nEnvironment=XDG_RUNTIME_DIR=%t\n\n[Install]\nWantedBy=graphical-session.target\n",
        quote(binary),
        quote(config)
    )
}

pub fn install(paths: &AppPaths, config: &Path, start: bool) -> Result<PathBuf> {
    let path = service_path(paths);
    let binary = std::env::current_exe()?.canonicalize()?;
    write_atomic(&path, generate(&binary, config).as_bytes())?;
    systemctl(["daemon-reload"])?;
    if start {
        systemctl(["enable", "--now", "omawake.service"])?;
    } else {
        systemctl(["enable", "omawake.service"])?;
    }
    Ok(path)
}

pub fn uninstall(paths: &AppPaths) -> Result<()> {
    let _ = systemctl(["disable", "--now", "omawake.service"]);
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
        .args(["--user", "status", "omawake.service", "--no-pager"])
        .status()
        .context("run systemctl --user status")?;
    if !status.success() {
        bail!("omawake.service is not running");
    }
    Ok(())
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
mod tests {
    use super::*;

    #[test]
    fn unit_uses_absolute_binary_and_config() {
        let unit = generate(Path::new("/opt/oma speak"), Path::new("/tmp/config.toml"));
        assert!(unit.contains("ExecStart=\"/opt/oma speak\" --config \"/tmp/config.toml\" daemon"));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("XDG_RUNTIME_DIR=%t"));
    }
}
