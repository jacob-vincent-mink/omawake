# Third-party notices

The MIT license in `LICENSE` applies to Omawake's project-authored source.

Omawake release builds dynamically load the public C ABI of
[audio.cpp](https://github.com/0xShug0/audio.cpp) at commit
`e9ff20042ec85af960a720368c6927cda19ad65f`. audio.cpp is licensed under
Apache-2.0. Omawake does not invoke the audio.cpp CLI. Release archives that
contain `libaudiocpp` also contain its upstream license and notices.

The pinned compact provider statically retains these audio.cpp build
components, whose exact license texts are included in release archives:

- ggml, MIT;
- cJSON, MIT;
- libyaml, MIT;
- the PocketFFT-derived speech FFT implementation, BSD-3-Clause; and
- tokenizer code conservatively attributed to llama.cpp, MIT.

The default provider is built with `AUDIOCPP_DEPLOYMENT_BUILD=OFF`; provider
CLI, server, test, and model-management executables are not distributed.

The default model profile is installed separately and contains:

- Moonshine Streaming Tiny from original revision
  `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`, licensed under MIT by Useful
  Sensors, Inc. (dba Moonshine AI). Its Q8_0 GGUF comes from the audio.cpp GGUF
  repository at revision `6d5436fc85f7a20c2e9f4e472b7f3a532f686444`.
- Silero VAD 6.2.1 from the original `snakers4/silero-vad` repository at
  revision `7e30209a3e901f9842f81b225f3e93d8199902b1`, licensed under MIT by the
  Silero Team.

Setup records the exact URLs, revisions, byte sizes, and SHA-256 checksums in
the installed `PROVENANCE.json` and `.omawake-model.json` files. It also writes
the full required MIT notices beneath the model profile's `LICENSES/`
directory. Models are not included in the Omawake release archive.

The optional direct OpenVINO profile downloads Intel's INT8 conversion of
OpenAI Whisper `base.en` at revision
`3b292a83752fbfcad0bd6384bcf71d0b1fc4fe74`. The original OpenAI model weights
at revision `911407f4214e0e1d82085af863093ec0b66f9cd6` and the converted OpenVINO
model both declare Apache-2.0. Setup writes the Apache-2.0 terms and the two
distinct source revisions beside the installed model. The OpenVINO runtime is
supplied separately by the user and is not included in Omawake releases.

Omawake also vendors a minimal set of public whisper.cpp v1.9.3 C ABI
declarations from commit `371b5a7561823ab2bb32142d2751e35e7534727b` for a
comparison provider. No whisper.cpp implementation is copied or statically
linked. The declarations remain under the upstream MIT license in
`vendor/whispercpp-1.9.3/LICENSE`.

Release archives contain `RUST-DEPENDENCIES.txt`, generated with cargo-about
from the locked Rust dependency graph.

The experimental trainable encoder uses Intel's `openvino` and `openvino-sys`
Rust wrappers (Apache-2.0) with runtime linking. Their license appears in the
locked Rust dependency notices. This does not bundle the OpenVINO runtime.
Whisper preprocessing and the tiny classifier are independently implemented;
no sherpa-onnx implementation is included. The numeric test fixtures are
project-generated signals and reference feature values, not speech recordings.
