use super::*;

#[test]
fn defaults_to_cpu_runtime() {
    let config = BackendConfig::default();
    assert_eq!(config.runtime, Runtime::Default);
    assert_eq!(config.canonical_device().unwrap(), "cpu");
    assert!(config.validate_shape().is_ok());
}

#[test]
fn validates_runtime_device_matrix() {
    for (runtime, accepted) in [
        (Runtime::Default, &["auto", "CPU"][..]),
        (Runtime::Cuda, &["auto", "GPU"][..]),
        (Runtime::Openvino, &["auto", "npu", "GPU", "cpu"][..]),
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
    assert!(canonical_device(Runtime::Openvino, "tpu").is_err());
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
fn reports_all_supported_runtimes() {
    assert_eq!(supported_capabilities(), &["cpu"]);
}
