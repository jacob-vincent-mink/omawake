use super::*;

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("omawake-config-{}-{name}", std::process::id()))
}

#[test]
fn parses_multiple_wake_words() {
    let config: Config = toml::from_str(
        r#"
            [[wake_words]]
            id = "one"
            phrase = "Lovely Child"
            command = ["touch", "/tmp/one"]
            [[wake_words]]
            id = "two"
            phrase = "Forever"
            enabled = false
            command = ["touch", "/tmp/two"]
        "#,
    )
    .unwrap();
    assert_eq!(config.wake_words.len(), 2);
    assert!(config.wake_words[0].enabled);
    assert!(!config.wake_words[1].enabled);
}

#[test]
fn missing_save_load_and_model_paths_round_trip() {
    let root = temp("round-trip");
    let path = root.join("nested/config.toml");
    let _ = fs::remove_dir_all(&root);
    let defaults = Config::load(&path).unwrap();
    assert_eq!(defaults.audio.device, "default");
    assert_eq!(defaults.audio.channels, "mono");
    assert_eq!(defaults.daemon.queue_capacity, 8);
    defaults.save(&path).unwrap();
    assert_eq!(Config::load(&path).unwrap().wake_words.len(), 1);

    let paths = AppPaths {
        config_file: path.clone(),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    assert!(
        defaults
            .model_directory(&paths)
            .starts_with(root.join("data/models"))
    );
    let mut custom = defaults;
    custom.model.directory = root.join("custom").display().to_string();
    assert_eq!(custom.model_directory(&paths), root.join("custom"));
}

#[test]
fn malformed_and_unknown_config_is_rejected() {
    let root = temp("invalid");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    fs::write(&path, "not = [valid").unwrap();
    assert!(Config::load(&path).is_err());
    fs::write(&path, "unknown = true").unwrap();
    assert!(Config::load(&path).is_err());

    let target = root.join("directory-as-config");
    fs::create_dir_all(&target).unwrap();
    assert!(Config::default().save(&target).is_err());
}
