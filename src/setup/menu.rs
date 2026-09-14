use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use super::systemd::write_atomic;
use crate::paths::AppPaths;

pub fn launcher_path(paths: &AppPaths) -> PathBuf {
    paths
        .data_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("applications/omawake-settings.desktop")
}

pub fn install(paths: &AppPaths) -> Result<PathBuf> {
    let binary = std::env::current_exe()?.canonicalize()?;
    let contents = format!(
        "[Desktop Entry]\nType=Application\nName=Omawake Setup\nComment=Install and configure a local wake-word model\nExec=\"{}\" setup model\nTerminal=true\nCategories=Settings;\nKeywords=voice;wake word;speech;\n",
        binary.display().to_string().replace('"', "\\\"")
    );
    let path = launcher_path(paths);
    write_atomic(&path, contents.as_bytes())?;
    Ok(path)
}

pub fn uninstall(paths: &AppPaths) -> Result<()> {
    let path = launcher_path(paths);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub fn status(paths: &AppPaths) -> Result<()> {
    let path = launcher_path(paths);
    if path.exists() {
        println!("installed: {}", path.display());
        Ok(())
    } else {
        bail!("Omawake setup launcher is not installed")
    }
}
