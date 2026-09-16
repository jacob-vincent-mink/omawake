use serde::Serialize;

use crate::backend::Runtime;
use crate::config::Config;

pub const WHISPER_MODEL_ID: &str = "whisper-base.en-ggml-silero-v6.2.0";

pub const DEFAULT_MODEL_ID: &str = "moonshine-streaming-tiny-q8_0-silero-v6.2.1";

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BackendSpec {
    pub kind: &'static str,
    pub name: &'static str,
    pub built: bool,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelAsset {
    pub role: &'static str,
    pub path: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub source_url: &'static str,
    pub source_revision: &'static str,
    pub license: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LicenseNotice {
    pub path: &'static str,
    pub license: &'static str,
    pub copyright: &'static str,
    pub source_url: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub backend: &'static str,
    #[serde(skip)] // Presentation is not part of the schema-1 install identity.
    pub name: &'static str,
    /// Native provider family, distinct from the complete profile description.
    #[serde(skip)] // Derived from the pinned profile; preserve existing manifests.
    pub asr_family: &'static str,
    pub family: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    pub license_status: &'static str,
    pub license_url: &'static str,
    pub source_url: &'static str,
    pub source_revision: &'static str,
    pub languages: &'static [&'static str],
    pub multilingual: bool,
    pub converted_source_url: &'static str,
    pub converted_source_revision: &'static str,
    pub downloadable: bool,
    pub verifier: &'static str,
    pub vad: &'static str,
    pub sample_rate: i32,
    pub probe_audio: Option<&'static str>,
    pub assets: &'static [ModelAsset],
    pub notices: &'static [LicenseNotice],
}

const BACKENDS: &[BackendSpec] = &[
    BackendSpec {
        kind: "audiocpp",
        name: "audio.cpp",
        built: true,
        description: "Integrated audio.cpp C ABI provider with native Silero VAD and Moonshine ASR",
    },
    BackendSpec {
        kind: "openvino-genai",
        name: "OpenVINO GenAI",
        built: true,
        description: "External complete OpenVINO installation using the official GenAI Whisper C API",
    },
    BackendSpec {
        kind: "whispercpp",
        name: "whisper.cpp",
        built: true,
        description: "External comparison provider using the public whisper.cpp C ABI",
    },
];

const MOONSHINE_REVISION: &str = "f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a";
const AUDIOCPP_GGUF_REVISION: &str = "6d5436fc85f7a20c2e9f4e472b7f3a532f686444";
const SILERO_REVISION: &str = "7e30209a3e901f9842f81b225f3e93d8199902b1";
const OPENAI_WHISPER_REVISION: &str = "911407f4214e0e1d82085af863093ec0b66f9cd6";
const OPENVINO_WHISPER_REVISION: &str = "3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74";
pub const OPENVINO_MODEL_ID: &str = "whisper-base.en-int8-ov-silero-v6.2.1";

const AUDIOCPP_ASSETS: &[ModelAsset] = &[
    ModelAsset {
        role: "wake phrase verifier",
        path: "moonshine-streaming-tiny-q8_0.gguf",
        url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/6d5436fc85f7a20c2e9f4e472b7f3a532f686444/Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf",
        size: 60_407_904,
        sha256: "e9a342a07327f4e1745874f137f45350697e91699a91e4eb2ac60c223718f8c3",
        source_url: "https://huggingface.co/moonshine-ai/moonshine-streaming-tiny",
        source_revision: MOONSHINE_REVISION,
        license: "MIT",
    },
    ModelAsset {
        role: "voice activity detector",
        path: "silero_vad_16k.safetensors",
        url: "https://raw.githubusercontent.com/snakers4/silero-vad/7e30209a3e901f9842f81b225f3e93d8199902b1/src/silero_vad/data/silero_vad_16k.safetensors",
        size: 1_239_748,
        sha256: "c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1",
        source_url: "https://github.com/snakers4/silero-vad",
        source_revision: SILERO_REVISION,
        license: "MIT",
    },
];

const AUDIOCPP_NOTICES: &[LicenseNotice] = &[
    LicenseNotice {
        path: "LICENSES/Moonshine-MIT.txt",
        license: "MIT",
        copyright: "Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI)",
        source_url: "https://github.com/moonshine-ai/moonshine/blob/main/LICENSE",
    },
    LicenseNotice {
        path: "LICENSES/Silero-VAD-MIT.txt",
        license: "MIT",
        copyright: "Copyright (c) 2020-present Silero Team",
        source_url: "https://github.com/snakers4/silero-vad/blob/7e30209a3e901f9842f81b225f3e93d8199902b1/LICENSE",
    },
];

const OPENVINO_ASSETS: &[ModelAsset] = &[
    ModelAsset {
        role: "verifier metadata",
        path: "config.json",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/config.json",
        size: 1_272,
        sha256: "ae9ba0d02f244aa48df4d8637fcf3ce517229fb9148ecef13d44ac83b7e2a6c3",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "generation metadata",
        path: "generation_config.json",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/generation_config.json",
        size: 1_526,
        sha256: "7eb6f9dca9df06ca5ae8ed43ee5b05b200af22fedbb3fe7e0f10ea354e44b641",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "Whisper decoder weights",
        path: "openvino_decoder_model.bin",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_decoder_model.bin",
        size: 52_439_472,
        sha256: "fa97c0aa3989311aca9eeaa72997d2d14ae112b0c8a54d055a4d7dca88bca1e9",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "Whisper decoder graph",
        path: "openvino_decoder_model.xml",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_decoder_model.xml",
        size: 564_428,
        sha256: "427586ba26761013ac9cbedbec132b574ff0d522542e2c0b6e5b885b8c7dce34",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "detokenizer weights",
        path: "openvino_detokenizer.bin",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_detokenizer.bin",
        size: 749_930,
        sha256: "7045e2ab69c216fa4ef1d129fcce4d36bcf5583f01b823279278b52626238e0c",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "detokenizer graph",
        path: "openvino_detokenizer.xml",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_detokenizer.xml",
        size: 9_699,
        sha256: "15699b574239b9d4e6ce9e37a97db2b179d3907f4afb8ee7f7950b61bb9f6902",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "Whisper encoder weights",
        path: "openvino_encoder_model.bin",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_encoder_model.bin",
        size: 23_097_456,
        sha256: "f2efb087f58680a7d7cc9916a3ab8712e776ddf579b7dcce38945da08441609b",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "Whisper encoder graph",
        path: "openvino_encoder_model.xml",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_encoder_model.xml",
        size: 295_834,
        sha256: "79dc09241718475ca14277bb16766cfb688b279f412cae8972b1c1857863ae3a",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "tokenizer weights",
        path: "openvino_tokenizer.bin",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_tokenizer.bin",
        size: 1_926_439,
        sha256: "8c9def49b61ff1cdd929b1c4b035e6714f69157587ada9cbb61cb15d6248ea2b",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "tokenizer graph",
        path: "openvino_tokenizer.xml",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/openvino_tokenizer.xml",
        size: 27_011,
        sha256: "e23e9eb65cd4e0cfd7425a6a329dda7a961453fcdcec6ba4d8e8d0614b239e9a",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "audio preprocessor metadata",
        path: "preprocessor_config.json",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/preprocessor_config.json",
        size: 356,
        sha256: "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "tokenizer metadata",
        path: "tokenizer.json",
        url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/resolve/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/tokenizer.json",
        size: 3_855_707,
        sha256: "287537d5be89a39bd18e7e3875ad9900faa668493fb759392b8f52a492eca5db",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        source_revision: OPENVINO_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "voice activity detector",
        path: "silero_vad_16k.safetensors",
        url: "https://raw.githubusercontent.com/snakers4/silero-vad/7e30209a3e901f9842f81b225f3e93d8199902b1/src/silero_vad/data/silero_vad_16k.safetensors",
        size: 1_239_748,
        sha256: "c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1",
        source_url: "https://github.com/snakers4/silero-vad",
        source_revision: SILERO_REVISION,
        license: "MIT",
    },
];

const OPENVINO_NOTICES: &[LicenseNotice] = &[
    LicenseNotice {
        path: "LICENSES/OpenAI-Whisper-Model-Apache-2.0.txt",
        license: "Apache-2.0",
        copyright: "Whisper base.en model weights published by OpenAI",
        source_url: "https://huggingface.co/openai/whisper-base.en/blob/911407f4214e0e1d82085af863093ec0b66f9cd6/README.md",
    },
    LicenseNotice {
        path: "LICENSES/OpenVINO-Whisper-Conversion-Apache-2.0.txt",
        license: "Apache-2.0",
        copyright: "OpenVINO Whisper INT8 conversion published by Intel Corporation",
        source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov/blob/3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74/README.md",
    },
    LicenseNotice {
        path: "LICENSES/Silero-VAD-MIT.txt",
        license: "MIT",
        copyright: "Copyright (c) 2020-present Silero Team",
        source_url: "https://github.com/snakers4/silero-vad/blob/7e30209a3e901f9842f81b225f3e93d8199902b1/LICENSE",
    },
];

const WHISPER_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
const WHISPER_VAD_REVISION: &str = "9ffd54a1e1ee413ddf265af9913beaf518d1639b";
const WHISPER_ASSETS: &[ModelAsset] = &[
    ModelAsset {
        role: "wake phrase verifier",
        path: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.en.bin",
        size: 147_964_211,
        sha256: "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
        source_url: "https://huggingface.co/openai/whisper-base.en",
        source_revision: OPENAI_WHISPER_REVISION,
        license: "Apache-2.0",
    },
    ModelAsset {
        role: "voice activity detector",
        path: "ggml-silero-v6.2.0.bin",
        url: "https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v6.2.0.bin",
        size: 885_098,
        sha256: "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987",
        source_url: "https://huggingface.co/ggml-org/whisper-vad",
        source_revision: WHISPER_VAD_REVISION,
        license: "MIT",
    },
];

const WHISPER_NOTICES: &[LicenseNotice] = &[
    OPENVINO_NOTICES[0],
    LicenseNotice {
        path: "LICENSES/Silero-VAD-MIT.txt",
        license: "MIT",
        copyright: "Copyright (c) 2020-present Silero Team",
        source_url: "https://huggingface.co/ggml-org/whisper-vad/blob/9ffd54a1e1ee413ddf265af9913beaf518d1639b/README.md",
    },
];

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: DEFAULT_MODEL_ID,
        backend: "audiocpp",
        name: "Moonshine Streaming Tiny",
        asr_family: "moonshine_asr",
        family: "silero-vad+moonshine-asr",
        description: "Silero VAD 6.2.1 with Moonshine Streaming Tiny Q8_0 phrase verification",
        license: "MIT",
        license_status: "verified",
        license_url: "https://github.com/moonshine-ai/moonshine/blob/main/LICENSE",
        source_url: "https://huggingface.co/moonshine-ai/moonshine-streaming-tiny",
        source_revision: MOONSHINE_REVISION,
        languages: &["en"],
        multilingual: false,
        converted_source_url: "https://huggingface.co/audio-cpp/audio.cpp-gguf",
        converted_source_revision: AUDIOCPP_GGUF_REVISION,
        downloadable: true,
        verifier: "moonshine-streaming-tiny-q8_0.gguf",
        vad: "silero_vad_16k.safetensors",
        sample_rate: 16_000,
        probe_audio: None,
        assets: AUDIOCPP_ASSETS,
        notices: AUDIOCPP_NOTICES,
    },
    ModelSpec {
        id: OPENVINO_MODEL_ID,
        backend: "openvino-genai",
        name: "Whisper Base.en · OpenVINO",
        asr_family: "whisper",
        family: "silero-vad+whisper-base.en-int8",
        description: "English Silero VAD 6.2.1 with OpenVINO Whisper Base.en INT8 verification",
        license: "Apache-2.0",
        license_status: "verified-origin-and-conversion",
        license_url: "https://huggingface.co/openai/whisper-base.en/blob/911407f4214e0e1d82085af863093ec0b66f9cd6/README.md",
        source_url: "https://huggingface.co/openai/whisper-base.en",
        source_revision: OPENAI_WHISPER_REVISION,
        languages: &["en"],
        multilingual: false,
        converted_source_url: "https://huggingface.co/OpenVINO/whisper-base.en-int8-ov",
        converted_source_revision: OPENVINO_WHISPER_REVISION,
        downloadable: true,
        verifier: ".",
        vad: "silero_vad_16k.safetensors",
        sample_rate: 16_000,
        probe_audio: None,
        assets: OPENVINO_ASSETS,
        notices: OPENVINO_NOTICES,
    },
    ModelSpec {
        id: WHISPER_MODEL_ID,
        backend: "whispercpp",
        name: "Whisper Base.en · whisper.cpp",
        asr_family: "whisper",
        family: "silero-vad+whisper-base.en-ggml",
        description: "English Whisper Base.en with Silero VAD 6.2.0 through whisper.cpp 1.9.3 (CPU)",
        license: "Apache-2.0 AND MIT",
        license_status: "verified-origin-and-conversion",
        license_url: "https://huggingface.co/openai/whisper-base.en/blob/911407f4214e0e1d82085af863093ec0b66f9cd6/README.md",
        source_url: "https://huggingface.co/openai/whisper-base.en",
        source_revision: OPENAI_WHISPER_REVISION,
        languages: &["en"],
        multilingual: false,
        converted_source_url: "https://huggingface.co/ggerganov/whisper.cpp",
        converted_source_revision: WHISPER_REVISION,
        downloadable: true,
        verifier: "ggml-base.en.bin",
        vad: "ggml-silero-v6.2.0.bin",
        sample_rate: 16_000,
        probe_audio: None,
        assets: WHISPER_ASSETS,
        notices: WHISPER_NOTICES,
    },
];

pub fn backends() -> &'static [BackendSpec] {
    BACKENDS
}

pub fn models() -> &'static [ModelSpec] {
    MODELS
}

pub fn model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|item| item.id == id)
}

/// Resolve the maintained default for a backend/runtime/device combination.
/// Compatibility is a format/adapter contract, not proof of device availability.
pub fn default_model(
    backend: &str,
    runtime: Runtime,
    device: &str,
) -> anyhow::Result<&'static ModelSpec> {
    let id = match backend {
        "audiocpp" => DEFAULT_MODEL_ID,
        "openvino-genai" => OPENVINO_MODEL_ID,
        "whispercpp" => WHISPER_MODEL_ID,
        _ => anyhow::bail!("no catalog default for backend {backend:?}"),
    };
    let spec = model(id).expect("catalog default exists");
    anyhow::ensure!(
        spec.compatible_with(backend, runtime, device),
        "backend {backend:?} has no compatible default for {runtime:?} / {device}"
    );
    Ok(spec)
}

/// Keep a selected compatible catalog model; otherwise use the backend default.
pub fn setup_model(config: &Config) -> anyhow::Result<&'static ModelSpec> {
    if let Some(spec) = model(&config.model.name)
        && spec.compatible_with(
            &config.backend.kind,
            config.backend.runtime,
            &config.backend.device,
        )
    {
        return Ok(spec);
    }
    default_model(
        &config.backend.kind,
        config.backend.runtime,
        &config.backend.device,
    )
}

impl ModelSpec {
    pub fn compatible_with(self, backend: &str, runtime: Runtime, device: &str) -> bool {
        if self.backend != backend || crate::backend::canonical_device(runtime, device).is_err() {
            return false;
        }
        match backend {
            "audiocpp" => matches!(
                runtime,
                Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
            ),
            "openvino-genai" => runtime == Runtime::Openvino,
            "whispercpp" => runtime == Runtime::Default,
            _ => false,
        }
    }

    pub fn total_size(self) -> u64 {
        self.assets.iter().map(|asset| asset.size).sum()
    }

    pub fn activate(self, config: &mut Config) {
        let preserve_provider = self.compatible_with(
            &config.backend.kind,
            config.backend.runtime,
            &config.backend.device,
        );
        if !preserve_provider {
            config.backend.library.clear();
            config.backend.library_dirs.clear();
            config.backend.options.clear();
            config.backend.fallback = Default::default();
            config.backend.runtime = match self.backend {
                "openvino-genai" => Runtime::Openvino,
                _ => Runtime::Default,
            };
            config.backend.device = "cpu".into();
            config.backend.device_id = 0;
        }
        config.backend.kind = self.backend.into();
        config.backend.options.remove("audiocpp.asr_family");
        if self.backend == "audiocpp" {
            config
                .backend
                .options
                .insert("audiocpp.asr_family".into(), self.asr_family.into());
        }
        config.model.name = self.id.into();
        config.model.directory.clear();
        config.model.verifier = self.verifier.into();
        config.model.vad = self.vad.into();
        config.model.sample_rate = self.sample_rate;
    }
}

#[cfg(test)]
#[path = "../tests/unit/catalog.rs"]
mod tests;
