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
            aliases = ["Lovely Childe", "Love Lee Child"]
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
    assert_eq!(config.wake_words[0].aliases.len(), 2);
    assert!(!config.wake_words[1].enabled);
}

#[test]
fn missing_save_load_and_model_paths_round_trip() {
    let root = temp("round-trip");
    let path = root.join("nested/config.toml");
    let _ = fs::remove_dir_all(&root);
    let defaults = Config::load(&path).unwrap();
    assert_eq!(defaults.audio.device, "default");
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

#[test]
fn language_field_defaults_to_empty_and_round_trips() {
    let mut config = Config::default();
    assert_eq!(config.model.language, "");
    config.model.language = "es".into();
    let path = std::env::temp_dir().join(format!("omawake-language-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    let file = path.join("config.toml");
    config.save(&file).unwrap();
    let loaded = Config::load(&file).unwrap();
    assert_eq!(loaded.model.language, "es");
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn multilingual_language_validation_rejects_non_supported_models_and_unknown_codes() {
    let mut config = Config::default();
    config.backend.kind = "openvino-genai".into();
    config.backend.runtime = crate::backend::Runtime::Openvino;
    config.model.name = crate::catalog::OPENVINO_MULTILINGUAL_MODEL_ID.into();
    config.model.language = "es".into();
    let profile = crate::engine::openvino_genai::verifier_profile_for_name(&config.model.name);
    assert!(profile.multilingual);
    let language = crate::engine::openvino_genai::validate_language_for_test(&config, &profile).unwrap();
    assert_eq!(language, "es");
    config.model.language = "xx".into();
    assert!(crate::engine::openvino_genai::validate_language_for_test(&config, &profile).is_err());
    config.model.name = crate::catalog::OPENVINO_MODEL_ID.into();
    config.model.language = "es".into();
    let english = crate::engine::openvino_genai::verifier_profile_for_name(&config.model.name);
    assert!(!english.multilingual);
    let error = crate::engine::openvino_genai::validate_language_for_test(&config, &english).unwrap_err();
    assert!(error.to_string().contains("multilingual verifier"), "{error}");
}
