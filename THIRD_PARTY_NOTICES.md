# Third-party notices

The MIT license in `LICENSE` applies to Omawake's project-authored source.

Omawake vendors a minimal set of public whisper.cpp v1.9.3 C ABI declarations
from commit `371b5a7561823ab2bb32142d2751e35e7534727b`. These declarations let every
release build load an exact compatible `libwhisper` at runtime; no whisper.cpp
implementation is copied or statically linked. The declarations remain under
the upstream MIT license in `vendor/whispercpp-1.9.3/LICENSE`.

Linux release archives include the official ONNX Runtime 1.30.0 CPU library
under its MIT license. Each archive carries ONNX Runtime's `LICENSE` and
`ThirdPartyNotices.txt`. Optional OpenVINO and CUDA provider packages retain
their own upstream notices.

The separately installed GigaSpeech Zipformer keyword model weights are marked
`Apache License 2.0` in the README supplied by the
[ModelScope publisher](https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/summary).
Omawake records this as a publisher declaration: the upstream sherpa model
index and GitHub release do not include a separate model-specific license
notice. The pinned archive has SHA-256
`f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a`.
Omawake does not relicense or redistribute the model.

The archive's two probe WAVs are derived from the
[LibriSpeech ASR corpus, SLR12](https://www.openslr.org/12/), prepared by Vassil
Panayotov with the assistance of Daniel Povey and distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- `test_wavs/0.wav` is test-clean utterance `1089-134686-0002`.
- `test_wavs/1.wav` is test-clean utterance `1221-135766-0001`.

The publisher converted the original LibriSpeech recordings to 16 kHz PCM WAV
and renamed them in the model archive. Those format and filename changes do not
change the recordings' CC BY 4.0 terms. Omawake downloads the WAVs as part of
the separately installed archive and does not include them in release archives.

The algorithmic reference for Omawake's independently authored keyword search
is [icefall PR #1428](https://github.com/k2-fsa/icefall/pull/1428), specifically
icefall commit `aac7df064a6d1529f3bf4acccc6c550bd260b7b3`. Icefall is licensed
under Apache-2.0. Omawake does not include icefall or sherpa source code.

Release archives contain `RUST-DEPENDENCIES.txt`, generated with cargo-about
from the locked Rust dependency graph.
