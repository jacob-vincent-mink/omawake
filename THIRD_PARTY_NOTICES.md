# Third-party notices

The MIT license in `LICENSE` applies to Omawake's project-authored source.

Linux release archives include the official ONNX Runtime 1.30.0 CPU library
under its MIT license. Each archive carries ONNX Runtime's `LICENSE` and
`ThirdPartyNotices.txt`. Optional OpenVINO and CUDA provider packages retain
their own upstream notices.

The separately installed GigaSpeech Zipformer keyword model is marked
`Apache License 2.0` in the publisher README contained in Omawake's pinned
archive (`sha256:f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a`).
Omawake does not relicense the model.

The algorithmic reference for Omawake's independently authored keyword search
is [icefall PR #1428](https://github.com/k2-fsa/icefall/pull/1428), specifically
icefall commit `aac7df064a6d1529f3bf4acccc6c550bd260b7b3`. Icefall is licensed
under Apache-2.0. Omawake does not include icefall or sherpa source code.

Release archives contain `RUST-DEPENDENCIES.txt`, generated with cargo-about
from the locked Rust dependency graph.
