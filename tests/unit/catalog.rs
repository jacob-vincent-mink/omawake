use std::path::PathBuf;

use super::*;

#[test]
fn catalog_pins_a_complete_originally_sourced_profile() {
    let spec = model(DEFAULT_MODEL_ID).unwrap();
    assert_eq!(backends().len(), 3);
    assert!(backends().iter().all(|backend| backend.built));
    assert_eq!(models().len(), 4);
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
    assert_eq!(openvino.license, "Apache-2.0");
    assert_eq!(
        openvino.source_url,
        "https://huggingface.co/openai/whisper-base.en"
    );
    assert_eq!(openvino.languages, ["en"]);
    assert!(!openvino.multilingual);
    assert_eq!(openvino.assets.len(), 13);
    assert_eq!(openvino.total_size(), 84_208_878);
    assert_eq!(openvino.source_revision, OPENAI_WHISPER_REVISION);
    assert_eq!(
        openvino.converted_source_revision,
        OPENVINO_WHISPER_REVISION
    );
    assert!(
        openvino.assets[..12]
            .iter()
            .all(|asset| asset.source_revision == OPENVINO_WHISPER_REVISION)
    );
    assert_eq!(openvino.assets[12].source_revision, SILERO_REVISION);
    assert_eq!(openvino.notices.len(), 3);
    assert_eq!(openvino.notices[0].license, "Apache-2.0");
    assert_eq!(openvino.notices[1].license, "Apache-2.0");
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
    let fresh = toml::to_string(&config).unwrap();
    for removed in ["sherpa", "onnxruntime", "omawake-onnx", "zipformer"] {
        assert!(!fresh.contains(removed), "fresh config retained {removed}");
    }

    config.backend.kind = "other".into();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.model.directory = "/custom".into();
    spec.activate(&mut config);
    assert_eq!(config.backend.kind, "audiocpp");
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(config.backend.device, "cpu");
    assert_eq!(config.model.name, spec.id);
    assert!(config.model.directory.is_empty());
    assert_eq!(config.model.verifier, "moonshine-streaming-tiny-q8_0.gguf");
    assert_eq!(config.model.vad, "silero_vad_16k.safetensors");
    assert_eq!(
        config
            .backend
            .options
            .get("audiocpp.asr_family")
            .map(String::as_str),
        Some("moonshine_asr")
    );

    let provider = PathBuf::from("/opt/audiocpp-cuda/libaudiocpp.so");
    let mut accelerated = Config::default();
    accelerated.backend.runtime = Runtime::Cuda;
    accelerated.backend.device = "gpu".into();
    accelerated.backend.device_id = 2;
    accelerated.backend.library = provider.clone();
    accelerated.backend.library_dirs = vec![PathBuf::from("/opt/audiocpp-cuda")];
    accelerated.backend.fallback = crate::backend::Fallback::Cpu;
    spec.activate(&mut accelerated);
    assert_eq!(accelerated.backend.runtime, Runtime::Cuda);
    assert_eq!(accelerated.backend.device, "gpu");
    assert_eq!(accelerated.backend.device_id, 2);
    assert_eq!(accelerated.backend.library, provider);
    assert_eq!(accelerated.backend.fallback, crate::backend::Fallback::Cpu);

    let openvino = model(OPENVINO_MODEL_ID).unwrap();
    openvino.activate(&mut accelerated);
    assert_eq!(accelerated.backend.kind, "openvino-genai");
    assert_eq!(accelerated.backend.runtime, Runtime::Openvino);
    assert_eq!(accelerated.backend.device, "cpu");
    assert!(accelerated.backend.library.as_os_str().is_empty());
    assert!(accelerated.backend.library_dirs.is_empty());
}

#[test]
fn every_backend_has_a_complete_default_and_rejects_foreign_formats() {
    for (backend, runtime, devices) in [
        ("audiocpp", Runtime::Default, vec!["cpu", "auto"]),
        ("audiocpp", Runtime::Cuda, vec!["gpu"]),
        ("audiocpp", Runtime::Vulkan, vec!["gpu"]),
        ("audiocpp", Runtime::Hip, vec!["gpu"]),
        (
            "openvino-genai",
            Runtime::Openvino,
            vec!["cpu", "gpu", "npu"],
        ),
        ("whispercpp", Runtime::Default, vec!["cpu"]),
    ] {
        for device in devices {
            let spec = default_model(backend, runtime, device).unwrap();
            assert!(spec.downloadable);
            assert!(spec.assets.iter().any(|asset| asset.path == spec.vad));
            assert!(
                spec.verifier == "." || spec.assets.iter().any(|asset| asset.path == spec.verifier)
            );
            for other in models().iter().filter(|other| other.backend != backend) {
                assert!(!other.compatible_with(backend, runtime, device));
            }
        }
    }
    for (backend, runtime, device) in [
        ("missing", Runtime::Default, "cpu"),
        ("whispercpp", Runtime::Cuda, "gpu"),
        ("audiocpp", Runtime::Openvino, "cpu"),
        ("openvino-genai", Runtime::Default, "cpu"),
        ("whispercpp", Runtime::Default, "npu"),
    ] {
        assert!(default_model(backend, runtime, device).is_err());
    }
    let whisper = model(WHISPER_MODEL_ID).unwrap();
    assert_eq!(whisper.total_size(), 148_849_309);
    assert!(whisper.assets[0].url.contains(WHISPER_REVISION));
    assert!(whisper.assets[1].url.contains(WHISPER_VAD_REVISION));
}

#[test]
fn whisper_activation_preserves_its_provider_and_defaults_follow_the_backend() {
    let mut config = Config::default();
    config.backend.kind = "whispercpp".into();
    config.backend.library = "/opt/whisper/libwhisper.so".into();
    config
        .backend
        .options
        .insert("audiocpp.asr_family".into(), "stale".into());
    let spec = setup_model(&config).unwrap();
    assert_eq!(spec.id, WHISPER_MODEL_ID);
    spec.activate(&mut config);
    assert_eq!(
        config.backend.library,
        PathBuf::from("/opt/whisper/libwhisper.so")
    );
    assert!(!config.backend.options.contains_key("audiocpp.asr_family"));
    assert_eq!(setup_model(&config).unwrap().id, WHISPER_MODEL_ID);
    config.backend.kind = "openvino-genai".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    assert_eq!(setup_model(&config).unwrap().id, OPENVINO_MODEL_ID);
    let before = serde_json::to_value(model(DEFAULT_MODEL_ID).unwrap()).unwrap();
    assert!(before.get("name").is_none());
    assert!(before.get("asr_family").is_none());
}

#[test]
fn multilingual_openvino_profile_exposes_the_model_language_set() {
    let spec = model(OPENVINO_MULTILINGUAL_MODEL_ID).unwrap();
    assert_eq!(spec.backend, "openvino-genai");
    assert!(spec.multilingual);
    // The profile exposes every language the pinned model itself supports:
    // Omaspeak only exposes the capability, the model makes the promise.
    assert_eq!(
        spec.languages,
        crate::engine::openvino_genai::WHISPER_MODEL_LANGUAGES
    );
    assert_eq!(spec.languages.len(), 99);
    assert!(spec.languages.contains(&"fr"));
    assert!(spec.source_url.contains("openai/whisper-base"));
    assert!(
        spec.converted_source_url
            .contains("OpenVINO/whisper-base-int8-ov")
    );
    assert_eq!(spec.assets.len(), 12);
    assert!(
        spec.assets
            .iter()
            .all(|asset| !asset.url.contains("base.en"))
    );
    let encoder = spec
        .assets
        .iter()
        .find(|a| a.path == "openvino_encoder_model.bin")
        .unwrap();
    assert_eq!(encoder.size, 23_097_456);
}
