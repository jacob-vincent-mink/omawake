use super::*;

#[test]
fn defaults_to_cpu_runtime() {
    let config = BackendConfig::default();
    assert_eq!(config.runtime, Runtime::Default);
    assert_eq!(config.canonical_device().unwrap(), "auto");
    assert!(
        config
            .validate_capabilities(compiled_capabilities())
            .is_ok()
    );
}

#[test]
fn validates_runtime_device_matrix() {
    for (runtime, accepted) in [
        (Runtime::Default, &["auto", "CPU"][..]),
        (Runtime::Cuda, &["auto", "GPU"][..]),
        (
            Runtime::Openvino,
            &[
                "auto",
                "npu",
                "GPU",
                "cpu",
                "auto:GPU,NPU,CPU",
                "hetero:GPU,CPU",
                "multi:NPU,CPU",
            ][..],
        ),
    ] {
        for device in accepted {
            assert!(
                canonical_device(runtime, device).is_ok(),
                "{runtime:?} {device}"
            );
        }
    }
    assert!(canonical_device(Runtime::Default, "gpu").is_err());
    assert!(canonical_device(Runtime::Cuda, "cpu").is_err());
    assert!(canonical_device(Runtime::Openvino, "hetero:GPU").is_err());
    assert!(canonical_device(Runtime::Openvino, "multi:").is_err());
    assert!(canonical_device(Runtime::Openvino, "auto:TPU").is_err());
    assert!(canonical_device(Runtime::Openvino, "future:CPU").is_err());

    let invalid_device_id = BackendConfig {
        runtime: Runtime::Default,
        device_id: 1,
        ..Default::default()
    };
    assert_eq!(
        invalid_device_id.validate_shape(),
        Err(BackendError::InvalidDeviceId)
    );
}

#[test]
fn canonicalizes_openvino_provider_syntax() {
    assert_eq!(
        canonical_device(Runtime::Openvino, " hetero:gpu, cpu ").unwrap(),
        "HETERO:GPU,CPU"
    );
    assert_eq!(
        canonical_device(Runtime::Openvino, "auto:npu,gpu").unwrap(),
        "AUTO:NPU,GPU"
    );
}

#[test]
fn acceleration_requires_a_compiled_capability() {
    let config = BackendConfig {
        runtime: Runtime::Openvino,
        device: "npu".into(),
        ..Default::default()
    };
    assert!(matches!(
        config.validate_capabilities(&["cpu"]),
        Err(BackendError::CapabilityUnavailable { .. })
    ));
    assert!(config.validate_capabilities(&["cpu", "openvino"]).is_ok());
}

#[test]
fn reports_capabilities_from_cargo_features() {
    #[cfg(all(feature = "openvino", feature = "cuda"))]
    assert_eq!(compiled_capabilities(), &["cpu", "openvino", "cuda"]);
    #[cfg(all(feature = "openvino", not(feature = "cuda")))]
    assert_eq!(compiled_capabilities(), &["cpu", "openvino"]);
    #[cfg(all(not(feature = "openvino"), feature = "cuda"))]
    assert_eq!(compiled_capabilities(), &["cpu", "cuda"]);
    #[cfg(not(any(feature = "openvino", feature = "cuda")))]
    assert_eq!(compiled_capabilities(), &["cpu"]);
}
