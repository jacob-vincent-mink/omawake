use super::*;
use std::fs;

#[test]
fn failed_and_cancelled_candidates_preserve_exact_config_bytes() {
    let root = env::temp_dir().join(format!("omawake-apply-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    let original = b"# Preserve this comment\n[backend]\ndevice = 'cpu'\n";
    fs::write(&path, original).unwrap();
    let mut candidate = Config::load(&path).unwrap();
    candidate.backend.device = "auto".into();
    for error in [
        "missing files",
        "wrong version",
        "provider registration failed",
        "device unavailable",
        "probe crashed",
    ] {
        assert!(
            apply_with(&candidate, &path, true, |_, _| Probe {
                errors: vec![error.into()],
                ..Default::default()
            })
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), original);
    }
    let success = |_: &BackendConfig, _: &Path| Probe {
        ready: true,
        loadable: true,
        device_accessible: true,
        ..Default::default()
    };
    apply_with(&candidate, &path, false, success).unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    apply_with(&candidate, &path, true, success).unwrap();
    assert_eq!(Config::load(&path).unwrap().backend.device, "auto");
    assert!(!path.with_extension("toml.tmp").exists());
    assert!(!probe(&BackendConfig::default(), &root.join("absent/config.toml")).ready);
    assert!(!root.join("absent").exists());
}

#[test]
fn absent_optional_runtimes_keep_stable_state_names() {
    let root = env::temp_dir().join(format!("omawake-inventory-{}", std::process::id()));
    let config = BackendConfig {
        onnxruntime_library: root.join("missing-ort"),
        sherpa_library: root.join("missing-sherpa"),
        ..Default::default()
    };
    let states = inventory(&config, &root.join("missing-config"));
    assert_eq!(states.len(), 8);
    for state in states {
        assert!(state.supported);
        assert!(!state.discovered && !state.probe.ready && !state.configured);
        let value = serde_json::to_value(state).unwrap();
        for key in [
            "supported",
            "discovered",
            "source",
            "configured",
            "loadable",
            "device_accessible",
            "ready",
            "paths",
            "evidence",
            "errors",
            "remediation",
        ] {
            assert!(value.get(key).is_some(), "{key}");
        }
    }
}

#[test]
fn resolved_candidates_keep_runtime_overlay_dirs_without_ambient_loader_paths() {
    let root = env::temp_dir().join(format!("omawake-runtime-overlay-{}", std::process::id()));
    let configured = root.join("configured");
    let overlay = root.join("overlay");
    let ambient = root.join("ambient");
    for directory in [&configured, &overlay, &ambient] {
        fs::create_dir_all(directory).unwrap();
    }
    let ort = configured.join("libonnxruntime.so.1.29.0");
    let sherpa = configured.join("libsherpa-onnx-c-api.so.1.13.8");
    fs::write(&ort, b"fixture").unwrap();
    fs::write(&sherpa, b"fixture").unwrap();
    let config = BackendConfig {
        library_dirs: vec![configured.clone()],
        onnxruntime_library: ort,
        sherpa_library: sherpa,
        ..Default::default()
    };
    let report = runtime_paths::report_with(
        &config,
        &root.join("config.toml"),
        None,
        Some(&env::join_paths([&overlay]).unwrap()),
        Some(&env::join_paths([&ambient]).unwrap()),
        None,
        [None, None, None],
        |_, _| false,
        |_, _| false,
        |_, _, _, _, _, _| false,
    );
    let exact = resolve_with_locations(&config, report);
    assert!(exact.library_dirs.contains(&configured));
    assert!(exact.library_dirs.contains(&overlay));
    assert!(!exact.library_dirs.contains(&ambient));
}
