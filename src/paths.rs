use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppPaths {
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub state_dir: PathBuf,
    pub runtime_dir: PathBuf,
}

impl AppPaths {
    pub fn discover() -> Self {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let config = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"));
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/share"));
        let state = std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/state"));
        let runtime = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let user = std::env::var("USER").unwrap_or_else(|_| "unknown".into());
                std::env::temp_dir().join(format!("omavoice-{user}"))
            });
        Self {
            config_file: config.join("omawake/config.toml"),
            data_dir: data.join("omawake"),
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
