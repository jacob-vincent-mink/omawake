use super::*;

#[test]
fn derives_runtime_files() {
    let paths = AppPaths {
        config_file: "/cfg/omawake/config.toml".into(),
        data_dir: "/data/omawake".into(),
        state_dir: "/state/omawake".into(),
        runtime_dir: "/run/omawake".into(),
    };
    assert_eq!(paths.socket(), PathBuf::from("/run/omawake/control.sock"));
    assert_eq!(
        paths.status_file(),
        PathBuf::from("/state/omawake/status.json")
    );
}

#[test]
fn discovery_honors_xdg_and_has_stable_fallbacks() {
    let custom = AppPaths::discover_with(
        |key| match key {
            "HOME" => Some("/home/test".into()),
            "XDG_CONFIG_HOME" => Some("/cfg".into()),
            "XDG_DATA_HOME" => Some("/data".into()),
            "XDG_STATE_HOME" => Some("/state".into()),
            "XDG_RUNTIME_DIR" => Some("/run".into()),
            _ => None,
        },
        Path::new("/tmp"),
    );
    assert_eq!(
        custom.config_file,
        PathBuf::from("/cfg/omawake/config.toml")
    );
    assert_eq!(custom.data_dir, PathBuf::from("/data/omawake"));
    assert_eq!(custom.state_dir, PathBuf::from("/state/omawake"));
    assert_eq!(custom.runtime_dir, PathBuf::from("/run/omawake"));

    let home_defaults = AppPaths::discover_with(
        |key| (key == "HOME").then(|| OsString::from("/home/test")),
        Path::new("/tmp"),
    );
    assert_eq!(
        home_defaults.config_file,
        PathBuf::from("/home/test/.config/omawake/config.toml")
    );
    assert_eq!(
        home_defaults.data_dir,
        PathBuf::from("/home/test/.local/share/omawake")
    );
    assert_eq!(
        home_defaults.state_dir,
        PathBuf::from("/home/test/.local/state/omawake")
    );
    assert_eq!(
        home_defaults.runtime_dir,
        PathBuf::from("/tmp/omavoice-unknown/omawake")
    );

    let no_home = AppPaths::discover_with(|_| None, Path::new("/tmp"));
    assert_eq!(
        no_home.config_file,
        PathBuf::from("./.config/omawake/config.toml")
    );
}
