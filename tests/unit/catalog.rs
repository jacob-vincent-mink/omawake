use super::*;

#[test]
fn every_model_references_a_backend() {
    for model in models() {
        assert!(
            backends()
                .iter()
                .any(|backend| backend.kind == model.backend)
        );
        assert_eq!(model.archive_sha256.len(), 64);
        assert!(!model.required_files.is_empty());
    }
}

#[test]
fn lookup_and_activation_populate_config() {
    assert!(model("missing").is_none());
    let spec = model("sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01").unwrap();
    let mut config = Config::default();
    config.backend.kind = "other".into();
    config.model.directory = "/custom".into();
    spec.activate(&mut config);
    assert_eq!(config.backend.kind, "sherpa-onnx");
    assert_eq!(config.model.name, spec.id);
    assert!(config.model.directory.is_empty());
    assert_eq!(config.model.bpe_model, "bpe.model");
    assert_eq!(config.model.encoder, spec.encoder);

    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    spec.activate(&mut config);
    assert_eq!(config.model.encoder, spec.openvino_accelerator_encoder);

    config.backend.device = "gpu".into();
    spec.apply_runtime_compatibility(&mut config);
    assert_eq!(config.model.encoder, spec.openvino_accelerator_encoder);

    config.backend.device = "cpu".into();
    spec.apply_runtime_compatibility(&mut config);
    assert_eq!(config.model.encoder, spec.encoder);

    for device in ["AUTO:CPU", "HETERO:CPU,CPU", "MULTI:CPU,CPU"] {
        config.backend.device = device.into();
        spec.apply_runtime_compatibility(&mut config);
        assert_eq!(config.model.encoder, spec.encoder, "{device}");
    }
    for device in ["auto", "AUTO:CPU,NPU", "HETERO:GPU,CPU", "MULTI:NPU,GPU"] {
        config.backend.device = device.into();
        spec.apply_runtime_compatibility(&mut config);
        assert_eq!(
            config.model.encoder, spec.openvino_accelerator_encoder,
            "{device}"
        );
    }

    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    spec.apply_runtime_compatibility(&mut config);
    assert_eq!(config.model.encoder, spec.cuda_encoder);
    assert_eq!(config.model.decoder, spec.cuda_decoder);
    assert_eq!(config.model.joiner, spec.cuda_joiner);

    config.backend.runtime = Runtime::Default;
    config.backend.device = "cpu".into();
    spec.apply_runtime_compatibility(&mut config);
    assert_eq!(config.model.encoder, spec.encoder);
    assert_eq!(config.model.decoder, spec.decoder);
    assert_eq!(config.model.joiner, spec.joiner);

    config.model.directory = "/models/custom".into();
    config.model.encoder = "custom-encoder.onnx".into();
    config.backend.device = "npu".into();
    spec.apply_runtime_compatibility(&mut config);
    assert_eq!(config.model.encoder, "custom-encoder.onnx");

    config.model.directory.clear();
    config.model.encoder = spec.encoder.into();
    config.model.decoder = "custom-decoder.onnx".into();
    config.backend.runtime = Runtime::Cuda;
    spec.apply_runtime_compatibility(&mut config);
    assert_eq!(config.model.encoder, spec.encoder);
    assert_eq!(config.model.decoder, "custom-decoder.onnx");
    assert_eq!(config.model.joiner, spec.joiner);
}
