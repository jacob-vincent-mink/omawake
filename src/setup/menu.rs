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
    let config = if paths.config_file.is_absolute() {
        paths.config_file.clone()
    } else {
        std::env::current_dir()?.join(&paths.config_file)
    };
    let contents = format!(
        "[Desktop Entry]\nType=Application\nName=Omawake Setup\nComment=Configure wake words and the background service\nExec=\"{}\" --config \"{}\" setup\nTerminal=true\nCategories=Settings;\nKeywords=voice;wake word;speech;\n",
        desktop_exec_path(&binary),
        desktop_exec_path(&config)
    );
    let path = launcher_path(paths);
    write_atomic(&path, contents.as_bytes())?;
    Ok(path)
}

fn desktop_exec_path(path: &Path) -> String {
    let mut escaped = String::new();
    for character in path.display().to_string().chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '$' => escaped.push_str("\\$"),
            '`' => escaped.push_str("\\`"),
            '%' => escaped.push_str("%%"),
            character => escaped.push(character),
        }
    }
    escaped
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

#[cfg(test)]
#[path = "../../tests/unit/setup_menu.rs"]
mod tests;
