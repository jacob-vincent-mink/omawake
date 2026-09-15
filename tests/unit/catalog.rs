use super::*;

#[test]
fn catalog_pins_a_complete_originally_sourced_profile() {
    let spec = model(DEFAULT_MODEL_ID).unwrap();
    assert_eq!(models().len(), 2);
    assert_eq!(spec.backend, "audiocpp");
    assert_eq!(spec.license, "MIT");
    assert_eq!(spec.license_status, "verified");
    assert_eq!(spec.assets.len(), 2);
    assert_eq!(spec.total_size(), 61_647_652);
    for asset in spec.assets {
        assert_eq!(asset.sha256.len(), 64);
        assert!(asset.url.starts_with("https://"));
        assert!(asset.source_url.starts_with("https://"));
        assert_eq!(asset.source_revision.len(), 40);
        assert_eq!(asset.license, "MIT");
    }
    assert_eq!(
        spec.assets[1].source_url,
        "https://github.com/snakers4/silero-vad"
    );
    assert!(spec.assets[1].url.contains(SILERO_REVISION));
    assert_eq!(spec.converted_source_revision, AUDIOCPP_GGUF_REVISION);

    let openvino = model(OPENVINO_MODEL_ID).unwrap();
    assert_eq!(openvino.backend, "openvino-genai");
    assert_eq!(openvino.languages, ["en"]);
    assert!(!openvino.multilingual);
    assert_eq!(openvino.assets.len(), 13);
    assert_eq!(openvino.total_size(), 84_208_878);
    assert_eq!(openvino.source_revision, OPENAI_WHISPER_REVISION);
    assert_eq!(openvino.converted_source_revision, OPENVINO_WHISPER_REVISION);
    assert!(openvino.assets[..12]
        .iter()
        .all(|asset| asset.source_revision == OPENVINO_WHISPER_REVISION));
    assert_eq!(openvino.assets[12].source_revision, SILERO_REVISION);
    assert!(model("missing").is_none());
}

#[test]
fn default_and_activation_are_the_qualified_audio_cpp_profile() {
    let spec = model(DEFAULT_MODEL_ID).unwrap();
    let mut config = Config::default();
    assert_eq!(config.backend.kind, "audiocpp");
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(config.model.name, DEFAULT_MODEL_ID);
    assert_eq!(config.model.verifier, spec.verifier);
    assert_eq!(config.model.vad, spec.vad);
    assert!(!format!("{config:?}").contains("sherpa"));

    config.backend.kind = "other".into();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.model.directory = "/custom".into();
    config.model.encoder = "old.onnx".into();
    spec.activate(&mut config);
    assert_eq!(config.backend.kind, "audiocpp");
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(config.backend.device, "cpu");
    assert_eq!(config.model.name, spec.id);
    assert!(config.model.directory.is_empty());
    assert_eq!(config.model.verifier, "moonshine-streaming-tiny-q8_0.gguf");
    assert_eq!(config.model.vad, "silero_vad_16k.safetensors");
    assert!(config.model.encoder.is_empty());
    assert_eq!(
        config
            .backend
            .options
            .get("audiocpp.asr_family")
            .map(String::as_str),
        Some("moonshine_asr")
    );
}
