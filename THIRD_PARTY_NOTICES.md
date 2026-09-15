# Third-party notices

The MIT license in `LICENSE` applies to Omawake's project-authored source.

Linux release archives include the official ONNX Runtime 1.30.0 CPU library
under its MIT license. Each archive carries ONNX Runtime's `LICENSE` and
`ThirdPartyNotices.txt`. Optional OpenVINO and CUDA provider packages retain
their own upstream notices.

The GigaSpeech Zipformer keyword model is distributed under Apache-2.0
according to its publisher model card. The model is installed separately and
is not relicensed by Omawake. The training recipe is icefall PR #1428; icefall
is Apache-2.0.

Release archives contain `RUST-DEPENDENCIES.txt`, generated with cargo-about
from the locked Rust dependency graph.
