use super::*;

#[test]
fn launcher_path_falls_back_and_uninstall_is_idempotent() {
    let paths = AppPaths {
        config_file: "config.toml".into(),
        config_home: ".".into(),
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
    let contents = fs::read_to_string(&installed).unwrap();
    assert!(
        contents
            .lines()
            .any(|line| line.starts_with("Exec=") && line.ends_with(" setup"))
    );
    status(&isolated).unwrap();
    uninstall(&isolated).unwrap();
    assert!(!installed.exists());
    assert!(status(&isolated).is_err());
}

#[test]
fn desktop_exec_path_escapes_reserved_characters() {
    assert_eq!(
        desktop_exec_path(Path::new("/opt/oma%wake/quote\"back\\dollar$`")),
        r#"/opt/oma%%wake/quote\"back\\dollar\$\`"#
    );
}
