use serde::Serialize;

use crate::config::Config;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BackendSpec {
    pub kind: &'static str,
    pub name: &'static str,
    pub built: bool,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RequiredFile {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub backend: &'static str,
    pub family: &'static str,
    pub description: &'static str,
    pub archive_url: &'static str,
    pub archive_size: u64,
    pub archive_sha256: &'static str,
    pub archive_root: &'static str,
    pub encoder: &'static str,
    pub decoder: &'static str,
    pub joiner: &'static str,
    pub tokens: &'static str,
    pub bpe_model: &'static str,
    pub required_files: &'static [RequiredFile],
}

const BACKENDS: &[BackendSpec] = &[BackendSpec {
    kind: "sherpa-onnx",
    name: "sherpa-onnx",
    built: true,
    description: "In-process ONNX keyword spotting through the official Rust crate",
}];

const KWS_FILES: &[RequiredFile] = &[
    RequiredFile {
        path: "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        size: 4_807_159,
        sha256: "1e721676515bcd42a186979733981213c66c80db680e1cc582dfedf3be76e678",
    },
    RequiredFile {
        path: "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        size: 277_985,
        sha256: "e40ff43297abe815e8898494c17e71bba2152d9d40fa3eb803f75d0f7533329a",
    },
    RequiredFile {
        path: "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        size: 163_380,
        sha256: "eae9da0c7e1e6c6a3f4cc42d167899c388f6c6701b94cb96320e4f55df79624c",
    },
    RequiredFile {
        path: "tokens.txt",
        size: 5_006,
        sha256: "fd2ded4050a55d2b1578870ba8697d02371980217806b7558bd0a5cc60f3ba53",
    },
    RequiredFile {
        path: "bpe.model",
        size: 244_837,
        sha256: "c8a2a0129c4ab8e463164c142f82d25649661b122c8cd0b7aab5c9e80b90ad24",
    },
    RequiredFile {
        path: "test_wavs/0.wav",
        size: 212_044,
        sha256: "6bc58a4efdf20daac252b6b1502632601a71efe0308f6757dc1eda34891a7e4f",
    },
    RequiredFile {
        path: "test_wavs/1.wav",
        size: 534_924,
        sha256: "5143a6ba93c4b274e2c4ac22deb75c2c48936c853f0519add1de828b6c79cc5a",
    },
    RequiredFile {
        path: "test_wavs/test_keywords.txt",
        size: 52,
        sha256: "fd9c3b504ab922bc96ce8613eff02193f8a5526fcf74a14df436c7af74ad649d",
    },
];

const MODELS: &[ModelSpec] = &[ModelSpec {
    id: "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
    backend: "sherpa-onnx",
    family: "zipformer-kws",
    description: "GigaSpeech Zipformer streaming keyword spotter (3.3M parameters)",
    archive_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2",
    archive_size: 17_626_723,
    archive_sha256: "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a",
    archive_root: "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
    encoder: "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    decoder: "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    joiner: "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    tokens: "tokens.txt",
    bpe_model: "bpe.model",
    required_files: KWS_FILES,
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
    pub fn activate(self, config: &mut Config) {
        config.backend.kind = self.backend.into();
        config.model.name = self.id.into();
        config.model.directory.clear();
        config.model.encoder = self.encoder.into();
        config.model.decoder = self.decoder.into();
        config.model.joiner = self.joiner.into();
        config.model.tokens = self.tokens.into();
        config.model.bpe_model = self.bpe_model.into();
    }
}

#[cfg(test)]
#[path = "../tests/unit/catalog.rs"]
mod tests;
