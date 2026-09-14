# Third-party notices

The MIT license in `LICENSE` applies to Omawake's project-authored source. It
does not relicense the components and model files described below.

## Omarchy-derived mark

The open corner frame in `assets/omawake-mark*.svg` is adapted from the
[Omarchy](https://github.com/basecamp/omarchy) icon, copyright David Heinemeier
Hansson, under the MIT License. See `licenses/OMARCHY-LICENSE`.

## sherpa-onnx

`native/sherpa` builds a modified sherpa-onnx 1.13.8 companion library.
sherpa-onnx is copyright its contributors and licensed under Apache-2.0. The
exact upstream commit and modification notice are in `native/sherpa/NOTICE`;
the license is in `licenses/SHERPA-ONNX-LICENSE`.

The companion DSO statically incorporates native dependencies from sherpa's
checksum-pinned build. Their notices are retained under `licenses/native` and
are copied into every release archive:

- kaldi-native-fbank, kaldi-decoder, kaldifst, simple-sentencepiece, and
  OpenFST: Apache-2.0 notices
- KissFFT: BSD-3-Clause
- nlohmann/json: MIT
- Eigen: primarily MPL-2.0, with Apache-2.0, BSD, and MINPACK notices for files
  carrying those terms

Eigen 5.0.1 source is available from the upstream checksum-pinned archive at
`https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz`
(SHA-256 `e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec`).
Omawake does not modify Eigen. The reproducible sherpa build and Omawake's
sherpa patch are in `native/sherpa`.

## ONNX Runtime

Linux release archives bundle the official CPU ONNX Runtime library under its
MIT License. Each archive carries ONNX Runtime's `LICENSE` and
`ThirdPartyNotices.txt` files alongside the native libraries.

## Rust dependencies

Release archives contain a `RUST-DEPENDENCIES.txt` file generated from the
locked, shipped dependency graph with cargo-about. CI rejects dependencies
outside the repository's explicit permissive-license allowlist.

## Wake-word models

Models are not part of the Omawake source license or release archive. The
current GigaSpeech KWS publisher archive lacks a clear standalone license grant
for the model weights. Omawake must not bundle, mirror, or automatically
download those weights until their terms are verified. A user may configure a
model obtained under rights they have separately established.
