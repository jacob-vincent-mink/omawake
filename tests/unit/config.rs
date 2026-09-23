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
        config_home: root.to_path_buf(),
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
    let language =
        crate::engine::openvino_genai::validate_language_for_test(&config, &profile).unwrap();
    assert_eq!(language, "es");
    config.model.language = "xx".into();
    assert!(crate::engine::openvino_genai::validate_language_for_test(&config, &profile).is_err());
    config.model.name = crate::catalog::OPENVINO_MODEL_ID.into();
    config.model.language = "es".into();
    let english = crate::engine::openvino_genai::verifier_profile_for_name(&config.model.name);
    assert!(!english.multilingual);
    let error =
        crate::engine::openvino_genai::validate_language_for_test(&config, &english).unwrap_err();
    assert!(
        error.to_string().contains("multilingual verifier"),
        "{error}"
    );
}
fn profile(backend_kind: &str, model_name: &str) -> EngineProfile {
    let mut profile = EngineProfile {
        backend: BackendConfig::default(),
        model: ModelConfig::default(),
    };
    profile.backend.kind = backend_kind.into();
    profile.model.name = model_name.into();
    profile
}

#[test]
fn for_engine_selects_named_profile_and_preserves_word_definitions() {
    let mut config = Config::default();
    config
        .engines
        .insert("gpu".into(), profile("openvino-genai", "other-model"));
    config.wake_words.push(WakeWord {
        enrollment: None,
        id: "assistant".into(),
        phrase: "Hey Assistant".into(),
        aliases: vec!["hey asst".into()],
        enabled: true,
        command: vec!["touch".into(), "/tmp/assistant".into()],
        engine: Some("gpu".into()),
    });

    let materialized = config.for_engine(Some("gpu")).unwrap();
    assert_eq!(materialized.backend.kind, "openvino-genai");
    assert_eq!(materialized.model.name, "other-model");
    assert!(
        materialized.engines.is_empty(),
        "materialized config must not re-route"
    );
    assert_eq!(materialized.wake_words.len(), 1);
    let word = &materialized.wake_words[0];
    assert_eq!(word.id, "assistant");
    assert_eq!(word.phrase, "Hey Assistant");
    assert_eq!(word.aliases, vec!["hey asst".to_string()]);
    assert!(word.enabled);
    assert_eq!(word.command, vec!["touch", "/tmp/assistant"]);
    assert!(
        word.engine.is_none(),
        "materialized words must clear their engine label"
    );

    // Default words stay on the default backend; named profiles are untouched.
    let default = config.for_engine(None).unwrap();
    assert_eq!(default.backend.kind, "audiocpp");
    assert_eq!(default.wake_words.len(), 1);
    assert_eq!(default.wake_words[0].id, "computer");
    assert_eq!(
        config.engines.get("gpu").unwrap().backend.kind,
        "openvino-genai"
    );
}

#[test]
fn for_engine_rejects_unknown_and_reserved_profiles() {
    let mut config = Config::default();
    config.wake_words[0].engine = Some("missing".into());
    let error = config.validate_engine_references().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("references undefined engine missing")
    );
    let error = config.for_engine(None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("references undefined engine missing")
    );

    let mut reserved = Config::default();
    reserved
        .engines
        .insert("default".into(), profile("whispercpp", "x"));
    assert!(reserved.validate_engine_references().is_err());
    reserved
        .engines
        .insert("Default".into(), profile("whispercpp", "x"));
    assert!(reserved.validate_engine_references().is_err());
    reserved
        .engines
        .insert("".into(), profile("whispercpp", "x"));
    assert!(reserved.validate_engine_references().is_err());
    // A word pointing at the reserved name cannot shadow the top-level default.
    reserved.wake_words[0].engine = Some("default".into());
    assert!(reserved.validate_engine_references().is_err());
}

#[test]
fn changing_default_runtime_does_not_rewrite_named_profiles() {
    let mut config = Config::default();
    config
        .engines
        .insert("fast".into(), profile("openvino-genai", "fast-model"));
    let snapshot = config.engines.clone();
    config.backend.runtime = crate::backend::Runtime::Vulkan;
    config.backend.device = "gpu".into();
    config.model.name = "default-model".into();
    assert_eq!(
        config.engines, snapshot,
        "profiles must be independent of the default runtime"
    );
    assert_eq!(config.engines["fast"].model.name, "fast-model");
    // Grouping only returns engines that actually have enabled words.
    let active = config.active_engines().unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].name, None);
    assert_eq!(active[0].backend.kind, "audiocpp");
}

#[test]
fn active_engines_lists_default_then_profiles_with_enabled_words() {
    let mut config = Config::default();
    config
        .engines
        .insert("gpu".into(), profile("openvino-genai", "m"));
    config.wake_words.push(WakeWord {
        enrollment: None,
        id: "assistant".into(),
        phrase: "Hey Assistant".into(),
        aliases: Vec::new(),
        enabled: true,
        command: vec!["true".into()],
        engine: Some("gpu".into()),
    });
    // Disabled group members must not open a group.
    config.wake_words.push(WakeWord {
        enrollment: None,
        id: "off".into(),
        phrase: "Off".into(),
        aliases: Vec::new(),
        enabled: false,
        command: vec!["true".into()],
        engine: Some("gpu".into()),
    });
    let active = config.active_engines().unwrap();
    assert_eq!(active.len(), 2);
    assert_eq!(active[0].name, None);
    assert_eq!(active[1].name.as_deref(), Some("gpu"));
    assert_eq!(active[1].backend.kind, "openvino-genai");

    let words = config.words_for_engine(Some("gpu"));
    assert!(words.iter().any(|word| word.id == "assistant"));
    assert!(!words.iter().any(|word| word.id == "computer"));
}

#[test]
fn engines_serialize_and_parse_with_defaults() {
    let mut config = Config::default();
    config
        .engines
        .insert("gpu".into(), profile("whispercpp", "m"));
    let toml = toml::to_string(&config).unwrap();
    let parsed: Config = toml::from_str(&toml).unwrap();
    assert_eq!(parsed.engines["gpu"].backend.kind, "whispercpp");
    // Existing files without [engines] keep working: optional field, empty default.
    let legacy: Config = toml::from_str(
        "[[wake_words]]\nid = \"computer\"\nphrase = \"Computer\"\ncommand = [\"true\"]\n",
    )
    .unwrap();
    assert!(legacy.engines.is_empty());
    assert_eq!(legacy.wake_words[0].engine, None);
}

#[test]
fn invalid_engine_edit_does_not_replace_a_readable_configuration() {
    let root = crate::test_support::unique_directory("config-engine", "transaction");
    let path = root.join("config.toml");
    let mut config = Config::default();
    config.save(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    config.wake_words[0].engine = Some("not-configured".into());
    assert!(config.save(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(Config::load(&path).is_ok());
    std::fs::remove_dir_all(root).unwrap();
}
