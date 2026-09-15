use super::*;

fn device(root: &Path, name: &str, vendor: &str, class: &str, driver: &str) {
    let path = root.join(name);
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("vendor"), vendor).unwrap();
    fs::write(path.join("class"), class).unwrap();
    fs::write(path.join("driver"), driver).unwrap();
}

#[test]
fn pci_detection_classifies_accelerators_and_keeps_stable_order() {
    let root = crate::test_support::unique_directory("hardware", "pci");
    device(&root, "0000:03:00.0", "0x10de", "0x030000", "nvidia");
    device(&root, "0000:00:0b.0", "0x8086", "0x120000", "intel_vpu");
    device(&root, "0000:00:02.0", "0x8086", "0x030000", "xe");
    device(&root, "0000:00:01.0", "0x1234", "0x020000", "other");
    let report = detect_at(&root);
    assert!(report.cuda_gpu);
    assert!(report.intel_npu);
    assert!(report.intel_gpu);
    assert!(report.vulkan_gpu);
    assert_eq!(report.devices.len(), 3);
    assert_eq!(report.devices[0].address, "0000:00:02.0");
    assert_eq!(report.devices[1].address, "0000:00:0b.0");
    assert_eq!(report.devices[2].address, "0000:03:00.0");
    assert!(report.errors.is_empty());
}

#[test]
fn pci_detection_reports_bad_roots_and_ignores_unrecognized_files() {
    let root = crate::test_support::unique_directory("hardware", "errors");
    fs::create_dir_all(root.join("bad")).unwrap();
    fs::write(root.join("bad/vendor"), "invalid").unwrap();
    fs::write(root.join("bad/class"), "invalid").unwrap();
    assert!(detect_at(&root).devices.is_empty());
    let missing = detect_at(&root.join("missing"));
    assert_eq!(missing.errors.len(), 1);
}

#[test]
fn recommendation_ranks_ready_providers_and_never_claims_model_proof() {
    let hardware = HardwareReport {
        cuda_gpu: true,
        intel_npu: true,
        intel_gpu: true,
        vulkan_gpu: true,
        ..Default::default()
    };
    let mut providers = ProviderAvailability {
        packaged_cpu: true,
        cuda: true,
        openvino_npu: true,
        openvino_gpu: true,
        vulkan: true,
    };
    let cuda = recommend(&hardware, providers);
    assert_eq!((cuda.runtime, cuda.device.as_str()), (Runtime::Cuda, "gpu"));

    providers.cuda = false;
    let npu = recommend(&hardware, providers);
    assert_eq!(
        (npu.runtime, npu.device.as_str()),
        (Runtime::Openvino, "npu")
    );

    providers.openvino_npu = false;
    assert_eq!(recommend(&hardware, providers).device, "gpu");
    providers.openvino_gpu = false;
    assert_eq!(recommend(&hardware, providers).runtime, Runtime::Vulkan);
    providers.vulkan = false;
    let cpu = recommend(&hardware, providers);
    assert_eq!(
        (cpu.runtime, cpu.device.as_str()),
        (Runtime::Default, "cpu")
    );
    assert!(cpu.provider_detected);
    assert_eq!(cpu.model_proof, "required-at-apply");
    assert!(!cpu.ready);
}

#[test]
fn recommendation_reports_detected_hardware_when_no_provider_exists() {
    let hardware = HardwareReport {
        intel_npu: true,
        ..Default::default()
    };
    let recommendation = recommend(&hardware, ProviderAvailability::default());
    assert_eq!(recommendation.runtime, Runtime::Openvino);
    assert_eq!(recommendation.device, "npu");
    assert!(recommendation.hardware_detected);
    assert!(!recommendation.provider_detected);
    assert!(
        recommendation
            .detail
            .contains("complete provider is required")
    );
}
