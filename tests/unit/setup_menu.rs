use super::*;

#[test]
fn launcher_path_falls_back_and_uninstall_is_idempotent() {
    let paths = AppPaths {
        config_file: "config.toml".into(),
        data_dir: "/".into(),
        cache_dir: "cache".into(),
        state_dir: "state".into(),
        runtime_dir: "run".into(),
    };
    assert_eq!(
        launcher_path(&paths),
        PathBuf::from("./applications/omawake-settings.desktop")
    );
    let mut isolated = paths;
    isolated.data_dir = std::env::temp_dir().join("omawake-menu-missing/data");
    uninstall(&isolated).unwrap();
    let installed = install(&isolated).unwrap();
    assert!(installed.is_file());
    status(&isolated).unwrap();
    uninstall(&isolated).unwrap();
    assert!(!installed.exists());
    assert!(status(&isolated).is_err());
}
