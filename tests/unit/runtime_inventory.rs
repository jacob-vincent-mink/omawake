use super::*;
use std::fs;
use std::path::PathBuf;

#[test]
fn rejected_and_preview_candidates_preserve_config_bytes() {
    let root = env::temp_dir().join(format!("omawake-runtime-inventory-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    let original = b"# retained\n[backend]\ndevice = 'cpu'\n";
    fs::write(&path, original).unwrap();
    let mut candidate = Config::load(&path).unwrap();
    candidate.backend.device = "auto".into();
    assert!(
        apply_with(&candidate, &path, true, |_, _| Probe {
            errors: vec!["provider failed".into()],
            ..Default::default()
        })
        .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    apply_with(&candidate, &path, false, |_, _| Probe {
        loadable: true,
        device_accessible: true,
        ready: true,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[test]
fn missing_app_owned_runtime_has_stable_inventory_shape() {
    let root = env::temp_dir().join(format!("omawake-runtime-missing-{}", std::process::id()));
    let config = BackendConfig {
        onnxruntime_library: root.join("missing-ort"),
        ..Default::default()
    };
    let states = inventory(&config, &root.join("config.toml"));
    assert_eq!(states.len(), 8);
    assert!(
        states
            .iter()
            .all(|state| state.supported && !state.discovered && !state.probe.ready)
    );
}

#[test]
fn resolution_keeps_configured_core_and_provider_only() {
    let root = env::temp_dir().join(format!("omawake-runtime-resolve-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let core = root.join("libonnxruntime.so.1.30.0");
    let provider = root.join("libonnxruntime_providers_openvino.so");
    fs::write(&core, b"fixture").unwrap();
    fs::write(&provider, b"fixture").unwrap();
    let config = BackendConfig {
        runtime: Runtime::Openvino,
        device: "cpu".into(),
        library_dirs: vec![root.clone()],
        onnxruntime_library: core.clone(),
        provider_library: provider.clone(),
        ..Default::default()
    };
    let resolved = resolve(&config, &root.join("config.toml"));
    assert_eq!(resolved.onnxruntime_library, core);
    assert_eq!(resolved.provider_library, provider);
    assert_eq!(resolved.library_dirs, [root]);
}

#[test]
fn pure_resolution_and_probe_failures_preserve_useful_evidence() {
    assert_eq!(name(Runtime::Default), "default");
    assert_eq!(name(Runtime::Openvino), "openvino");
    assert_eq!(name(Runtime::Cuda), "cuda");

    let root = env::temp_dir().join(format!("omawake-runtime-pure-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let core = root.join("libonnxruntime.so");
    let provider = root.join("provider.so");
    let extra = root.join("extra");
    fs::create_dir_all(&extra).unwrap();
    let mut config = BackendConfig {
        runtime: Runtime::Openvino,
        device: "npu".into(),
        onnxruntime_library: core.clone(),
        provider_library: provider.clone(),
        library_dirs: vec![root.clone()],
        ..Default::default()
    };
    let resolved = resolve_with_locations(
        &config,
        runtime_paths::RuntimeLibraryReport {
            onnxruntime_library: Some(core.clone()),
            provider_library: Some(provider.clone()),
            configured_library_dirs: vec![root.clone()],
            environment_library_dirs: vec![root.clone(), extra.clone()],
            package_library_dirs: vec![],
            effective_library_dirs: vec![],
            missing_library_dirs: vec![],
            runtime_loadable: Default::default(),
            remediation: vec![],
        },
    );
    assert_eq!(resolved.library_dirs, [root.clone(), extra]);
    assert_eq!(required(&resolved), [core.as_path(), provider.as_path()]);

    config.library_dirs = vec![PathBuf::from("relative")];
    let invalid = probe(&config, &root.join("config.toml"));
    assert!(!invalid.ready);
    assert!(!invalid.errors.is_empty());
    config.library_dirs = vec![root.clone()];
    let missing = probe(&config, &root.join("config.toml"));
    assert!(!missing.ready);
    assert!(!missing.errors.is_empty());
}

#[test]
fn isolated_probe_protocol_accepts_json_and_reports_child_failures() {
    use std::os::unix::fs::PermissionsExt;

    let root = env::temp_dir().join(format!("omawake-inventory-child-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let config = BackendConfig::default();
    let success = root.join("success.sh");
    fs::write(
        &success,
        "#!/bin/sh\nprintf '%s' '{\"loadable\":true,\"device_accessible\":true,\"ready\":true,\"evidence\":{\"versions\":[],\"provider_registration\":false,\"available_devices\":[],\"selected_device\":\"cpu\"},\"errors\":[]}'\n",
    )
    .unwrap();
    fs::set_permissions(&success, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(isolated_with_executable(&config, &success).unwrap().ready);

    let failed = root.join("failed.sh");
    fs::write(&failed, "#!/bin/sh\nexit 4\n").unwrap();
    fs::set_permissions(&failed, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        isolated_with_executable(&config, &failed)
            .unwrap_err()
            .to_string()
            .contains("terminated")
    );

    let malformed = root.join("malformed.sh");
    fs::write(&malformed, "#!/bin/sh\nprintf broken\n").unwrap();
    fs::set_permissions(&malformed, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(isolated_with_executable(&config, &malformed).is_err());
}

#[test]
fn real_ort_child_reports_cpu_evidence_and_invalid_device_when_available() {
    let Some(runtime) = env::var_os("OMAWAKE_TEST_ONNXRUNTIME") else {
        return;
    };
    let mut config = BackendConfig {
        runtime: Runtime::Default,
        device: "cpu".into(),
        onnxruntime_library: PathBuf::from(runtime),
        ..Default::default()
    };
    let ready = child(&config);
    assert!(ready.ready, "{:?}", ready.errors);
    assert_eq!(ready.evidence.versions, ["ONNX Runtime 1.30.0"]);
    assert_eq!(ready.evidence.selected_device.as_deref(), Some("cpu"));
    assert!(!ready.evidence.provider_registration);

    config.device = "npu".into();
    let invalid = child(&config);
    assert!(!invalid.ready);
    assert!(!invalid.errors.is_empty());

    config.onnxruntime_library = PathBuf::from("/definitely/missing/libonnxruntime.so");
    let missing = child(&config);
    assert!(!missing.ready);
}

#[test]
fn successful_apply_persists_only_after_a_ready_probe() {
    let root = env::temp_dir().join(format!("omawake-runtime-apply-{}", std::process::id()));
    let path = root.join("config.toml");
    let mut config = Config::default();
    config.backend.device = "cpu".into();
    let result = apply_with(&config, &path, true, |_, _| Probe {
        loadable: true,
        device_accessible: true,
        ready: true,
        ..Default::default()
    })
    .unwrap();
    assert!(result.ready);
    assert_eq!(Config::load(&path).unwrap().backend.device, "cpu");
}
