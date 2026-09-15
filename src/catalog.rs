use serde::Serialize;

use crate::backend::Runtime;
use crate::config::Config;

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
    pub family: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    pub license_status: &'static str,
    pub license_url: &'static str,
    pub source_url: &'static str,
    pub source_revision: &'static str,
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

const BACKENDS: &[BackendSpec] = &[BackendSpec {
    kind: "audiocpp",
    name: "audio.cpp",
    built: true,
    description: "Integrated audio.cpp C ABI provider with native Silero VAD and Moonshine ASR",
}];

const MOONSHINE_REVISION: &str = "f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a";
const AUDIOCPP_GGUF_REVISION: &str = "6d5436fc85f7a20c2e9f4e472b7f3a532f686444";
const SILERO_REVISION: &str = "7e30209a3e901f9842f81b225f3e93d8199902b1";

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

const MODELS: &[ModelSpec] = &[ModelSpec {
    id: DEFAULT_MODEL_ID,
    backend: "audiocpp",
    family: "silero-vad+moonshine-asr",
    description: "Silero VAD 6.2.1 with Moonshine Streaming Tiny Q8_0 phrase verification",
    license: "MIT",
    license_status: "verified",
    license_url: "https://github.com/moonshine-ai/moonshine/blob/main/LICENSE",
    source_url: "https://huggingface.co/moonshine-ai/moonshine-streaming-tiny",
    source_revision: MOONSHINE_REVISION,
    converted_source_url: "https://huggingface.co/audio-cpp/audio.cpp-gguf",
    converted_source_revision: AUDIOCPP_GGUF_REVISION,
    downloadable: true,
    verifier: "moonshine-streaming-tiny-q8_0.gguf",
    vad: "silero_vad_16k.safetensors",
    sample_rate: 16_000,
    probe_audio: None,
    assets: AUDIOCPP_ASSETS,
    notices: AUDIOCPP_NOTICES,
}];

pub fn backends() -> &'static [BackendSpec] {
    BACKENDS
}

pub fn models() -> &'static [ModelSpec] {
    MODELS
}

pub fn model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|item| item.id == id)
}

impl ModelSpec {
    pub fn total_size(self) -> u64 {
        self.assets.iter().map(|asset| asset.size).sum()
    }

    pub fn activate(self, config: &mut Config) {
        config.backend.kind = self.backend.into();
        config.backend.runtime = Runtime::Default;
        config.backend.device = "cpu".into();
        config.backend.device_id = 0;
        config.backend.fallback = Default::default();
        config
            .backend
            .options
            .insert("audiocpp.asr_family".into(), "moonshine_asr".into());
        config.model.name = self.id.into();
        config.model.directory.clear();
        config.model.verifier = self.verifier.into();
        config.model.vad = self.vad.into();
        config.model.sample_rate = self.sample_rate;
        config.model.encoder.clear();
        config.model.decoder.clear();
        config.model.joiner.clear();
        config.model.tokens.clear();
        config.model.bpe_model.clear();
    }
}

#[cfg(test)]
#[path = "../tests/unit/catalog.rs"]
mod tests;
