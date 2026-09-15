use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppPaths {
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub state_dir: PathBuf,
    pub runtime_dir: PathBuf,
}

impl AppPaths {
    pub fn discover() -> Self {
        Self::discover_with(|key| std::env::var_os(key), &std::env::temp_dir())
    }

    fn discover_with(mut variable: impl FnMut(&str) -> Option<OsString>, temp: &Path) -> Self {
        let home = variable("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let config = variable("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"));
        let data = variable("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/share"));
        let cache = variable("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".cache"));
        let state = variable("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/state"));
        let runtime = variable("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let user = variable("USER").unwrap_or_else(|| "unknown".into());
                temp.join(format!("omavoice-{}", user.to_string_lossy()))
            });
        Self {
            config_file: config.join("omawake/config.toml"),
            data_dir: data.join("omawake"),
            cache_dir: cache.join("omawake"),
            state_dir: state.join("omawake"),
            runtime_dir: runtime.join("omawake"),
        }
    }

    pub fn socket(&self) -> PathBuf {
        self.runtime_dir.join("control.sock")
    }

    pub fn status_file(&self) -> PathBuf {
        self.state_dir.join("status.json")
    }
}

#[cfg(test)]
#[path = "../tests/unit/paths.rs"]
mod tests;
