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
}
