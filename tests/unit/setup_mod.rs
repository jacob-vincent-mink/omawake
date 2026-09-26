use super::*;
use std::fs;

fn fixture(name: &str) -> (std::path::PathBuf, AppPaths) {
    let root =
        std::env::temp_dir().join(format!("omawake-setup-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let paths = AppPaths {
        config_file: root.join("config/omawake/config.toml"),
        config_home: root.join("config"),
        data_dir: root.join("data/omawake"),
        cache_dir: root.join("cache/omawake"),
        state_dir: root.join("state/omawake"),
        runtime_dir: root.join("run/omawake"),
    };
    (root, paths)
}

#[test]
fn ensure_config_creates_and_reloads_defaults() {
    let (_, paths) = fixture("ensure");
    let created = ensure_config(&paths.config_file).unwrap();
    assert_eq!(created.wake_words.len(), 1);
    let loaded = ensure_config(&paths.config_file).unwrap();
    assert_eq!(loaded.backend.kind, "audiocpp");
}

#[test]
fn setup_loader_uses_defaults_for_invalid_config_without_changing_the_file() {
    let (_, paths) = fixture("invalid-recovery");
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    let invalid = b"[backend]\nremoved_option = \"\"\n";
    fs::write(&paths.config_file, invalid).unwrap();

    let issue = config_recovery(&paths.config_file).unwrap().unwrap();
    assert!(issue.contains("unknown field `removed_option`"));
    let config = ensure_config(&paths.config_file).unwrap();
    assert_eq!(config.backend.kind, "audiocpp");
    assert_eq!(fs::read(&paths.config_file).unwrap(), invalid);

    let unreadable = paths.config_file.with_file_name("config-directory");
    fs::create_dir_all(&unreadable).unwrap();
    assert!(config_recovery(&unreadable).is_err());
}

#[test]
fn checks_report_malformed_missing_custom_and_empty_states() {
    let (root, paths) = fixture("checks");
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    fs::write(&paths.config_file, "bad = [toml").unwrap();
    let malformed = checks(&paths.config_file, &paths);
    assert_eq!(malformed.len(), 1);
    assert!(!malformed[0].ok);

    let mut config = Config::default();
    config.backend.kind = "future".into();
    config.model.name = "custom".into();
    config.model.directory = root.join("custom-model").display().to_string();
    config.wake_words.clear();
    config.save(&paths.config_file).unwrap();
    let missing = checks(&paths.config_file, &paths);
    assert!(
        missing
            .iter()
            .any(|check| check.name == "backend" && !check.ok)
    );
    assert!(
        missing
            .iter()
            .any(|check| check.name == "model" && !check.ok)
    );
    assert!(
        missing
            .iter()
            .any(|check| check.name == "wake-words" && !check.ok)
    );
    let optional_service = missing
        .iter()
        .find(|check| check.name == "systemd")
        .unwrap();
    assert!(optional_service.ok);
    assert!(optional_service.detail.contains("optional"));
    assert!(optional_service.remediation.is_none());
    let optional_launcher = missing
        .iter()
        .find(|check| check.name == "launcher")
        .unwrap();
    assert!(optional_launcher.ok);
    assert!(optional_launcher.detail.contains("optional"));
    fs::create_dir_all(&config.model.directory).unwrap();
    let launcher = menu::launcher_path(&paths);
    fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    fs::write(&launcher, "launcher").unwrap();
    let service = systemd::service_path(&paths);
    fs::create_dir_all(service.parent().unwrap()).unwrap();
    fs::write(&service, "service").unwrap();
    let present = checks(&paths.config_file, &paths);
    assert!(
        present
            .iter()
            .any(|check| check.name == "model" && check.ok)
    );
    assert!(
        present
            .iter()
            .any(|check| check.name == "engine" && !check.ok)
    );
    assert!(
        present
            .iter()
            .any(|check| check.name == "launcher" && check.ok)
    );
    if command_exists("systemctl") {
        assert!(
            present
                .iter()
                .any(|check| check.name == "systemd" && check.ok)
        );
    }
}

#[test]
fn check_printers_fail_when_remediation_is_required() {
    let (_, paths) = fixture("print");
    ensure_config(&paths.config_file).unwrap();
    assert!(print_checks(&paths.config_file, &paths, false).is_err());
    assert!(print_checks(&paths.config_file, &paths, true).is_err());
    assert!(print_checks_event(&paths.config_file, &paths).is_err());
    let config = Config::default();
    print_runtime(&config, &paths.config_file, false).unwrap();
    print_runtime(&config, &paths.config_file, true).unwrap();
}

#[test]
fn checks_report_successful_engine_and_optional_service_states() {
    let (root, paths) = fixture("injected-checks");
    let mut config = Config::default();
    config.model.name = "custom".into();
    config.model.directory = root.join("custom-model").display().to_string();
    fs::create_dir_all(&config.model.directory).unwrap();
    config.save(&paths.config_file).unwrap();
    let launcher = menu::launcher_path(&paths);
    fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    fs::write(launcher, "launcher").unwrap();

    let unavailable = checks_with(
        &paths.config_file,
        &paths,
        &|_, _| Ok("fake initialized one mapping".into()),
        false,
        &|| panic!("inactive system manager must not be queried"),
    );
    assert!(unavailable.iter().all(|check| check.ok));
    assert!(
        unavailable
            .iter()
            .any(|check| check.name == "engine" && check.detail.contains("fake initialized"))
    );
    assert!(
        unavailable
            .iter()
            .any(|check| check.name == "systemd" && check.detail.contains("not available"))
    );

    let active_external = checks_with(
        &paths.config_file,
        &paths,
        &|_, _| Ok("fake initialized one mapping".into()),
        true,
        &|| true,
    );
    assert!(active_external.iter().all(|check| check.ok));
    assert!(active_external.iter().any(|check| {
        check.name == "systemd" && check.detail.contains("active from an external unit")
    }));
}

#[test]
fn checks_require_a_prepared_cache_for_openvino_accelerators() {
    for device in ["gpu", "npu"] {
        let (root, paths) = fixture(&format!("{device}-cache-check"));
        let mut config = Config::default();
        config.backend.runtime = crate::backend::Runtime::Openvino;
        config.backend.device = device.into();
        config.model.name = "custom".into();
        config.model.directory = root.join("custom-model").display().to_string();
        fs::create_dir_all(&config.model.directory).unwrap();
        config.save(&paths.config_file).unwrap();
        let launcher = menu::launcher_path(&paths);
        fs::create_dir_all(launcher.parent().unwrap()).unwrap();
        fs::write(launcher, "launcher").unwrap();

        let missing = checks_with(
            &paths.config_file,
            &paths,
            &|_, _| Ok("runtime ready".into()),
            false,
            &|| unreachable!(),
        );
        let cache = missing
            .iter()
            .find(|check| check.name == "model-cache")
            .unwrap();
        assert!(!cache.ok);
        assert!(cache.detail.contains("not prepared"));
        assert!(cache.detail.contains(&device.to_ascii_uppercase()));

        let directory = crate::engine::openvino_cache_directory(&config, &paths).unwrap();
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("compiled.blob"), b"ready").unwrap();
        let ready = checks_with(
            &paths.config_file,
            &paths,
            &|_, _| Ok("runtime ready".into()),
            false,
            &|| unreachable!(),
        );
        let cache = ready
            .iter()
            .find(|check| check.name == "model-cache")
            .unwrap();
        assert!(cache.ok);
        assert!(cache.detail.contains("1 compiled artifact"));
    }
}
