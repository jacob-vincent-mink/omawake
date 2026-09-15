use super::*;
use std::fs;

#[test]
fn runtime_names_cover_every_native_provider_runtime() {
    assert_eq!(name(Runtime::Default), "default");
    assert_eq!(name(Runtime::Openvino), "openvino");
    assert_eq!(name(Runtime::Cuda), "cuda");
    assert_eq!(name(Runtime::Vulkan), "vulkan");
    assert_eq!(name(Runtime::Hip), "hip");
}

#[test]
fn rejected_and_preview_candidates_preserve_config_bytes() {
    let root =
        std::env::temp_dir().join(format!("omawake-runtime-inventory-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    let original = b"# retained\n[backend]\ndevice = 'cpu'\n";
    fs::write(&path, original).unwrap();
    let candidate = Config::load(&path).unwrap();

    assert!(
        apply_with(&candidate, &path, true, |_, _| Probe {
            errors: vec!["provider failed".into()],
            ..Default::default()
        })
        .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), original);

    let probe = Probe {
        loadable: true,
        device_accessible: true,
        ready: true,
        evidence: Evidence {
            versions: vec!["audio.cpp 0.1.0".into()],
            provider_registration: true,
            available_devices: vec!["cpu".into()],
            selected_device: Some("cpu".into()),
        },
        errors: Vec::new(),
    };
    assert!(
        apply_with(&candidate, &path, false, |_, _| probe.clone())
            .unwrap()
            .ready
    );
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[test]
fn successful_apply_is_atomic_and_persists_only_a_ready_candidate() {
    let root = std::env::temp_dir().join(format!("omawake-runtime-apply-{}", std::process::id()));
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
