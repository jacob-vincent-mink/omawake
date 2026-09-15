use super::*;
use std::fs;

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
