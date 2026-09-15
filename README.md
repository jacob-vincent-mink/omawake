<p align="center">
  <img alt="Omawake: a stylized ear" src="assets/omawake-mark-on-light.svg" width="160">
</p>

# Omawake

Omawake is a local wake-word daemon written in Rust. It captures audio with
CPAL, runs a streaming ONNX keyword model, and maps stable wake-word IDs to
direct argument-vector actions. Recorded and live audio use the same detector,
including sample-rate conversion and streaming state.

The keyword pipeline is owned by Omawake. It implements the model's published
Kaldi feature contract, Zipformer transducer state handling, modified
Aho-Corasick context graph, and keyword beam search directly in Rust. Only ONNX
Runtime is loaded dynamically; there is no sherpa library, patch, or ABI.

## Install

Linux x86-64 and aarch64 releases include the official ONNX Runtime 1.30.0 CPU library.
Verify and unpack a release, then run setup:

```bash
sha256sum --check omawake-0.0.1-rc-linux-x86_64.sha256
tar -xJf omawake-0.0.1-rc-linux-x86_64.tar.xz
cd omawake-0.0.1-rc-linux-x86_64
./omawake setup
```

The model is installed separately. The current catalog entry retains its
publisher filename,
`sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`, but Omawake does not use
the sherpa runtime. The model card identifies the weights as Apache-2.0. Setup
verifies the pinned archive and each extracted asset.

```bash
omawake setup all
omawake setup check
```

Setup downloads the pinned archive from its publisher by default. An existing
copy can be supplied with `--archive /path/to/model.tar.bz2`.

## Use

```bash
omawake test --audio test.wav --json
omawake benchmark --warmup 2 --iterations 10 test.wav > benchmark.json
omawake daemon
omawake status --json
omawake pause
omawake resume
omawake stop
```

The recognizer remains loaded while the daemon releases the microphone before
an action and reopens it after the configured cooldown. Actions do not pass
through a shell.

```bash
omawake wake-word add --id computer --phrase Computer -- notify-send "Wake word heard"
omawake wake-word remove computer
```

## Runtime selection

`backend.runtime` accepts `default`, `openvino`, and `cuda`.

| Runtime | Devices |
|---|---|
| `default` | `auto`, `cpu` |
| `openvino` | `auto`, `cpu`, `gpu`, `npu` |
| `cuda` | `auto`, `gpu` |

The release-owned ORT core is found next to the executable under `lib/`.
OpenVINO and CUDA use official V2 execution-provider plugins registered through
ORT's plugin API. Provider packages contain the provider DSO and vendor
dependencies, not another ORT core. Select a prepared package directory with:

```bash
omawake setup runtime --runtime openvino --device npu --dir /opt/omawake-openvino --apply
omawake setup runtime --runtime cuda --device gpu --dir /opt/omawake-cuda --apply
```

Omawake selects the registered device explicitly. OpenVINO NPU uses the float
graph set, specializes the encoder batch to one and the decoder/joiner batch to
at least eight, and binds concrete host outputs. OpenVINO compiled models are
cached under
`${XDG_CACHE_HOME:-$HOME/.cache}/omawake/openvino/<device>/compiled`.

`omawake setup runtime --json` reports the discovered ORT core, optional
provider, exposed devices, and probe errors. `fallback = "cpu"` permits an
accelerator initialization failure to fall back to CPU and records that choice
in status and benchmark JSON.

See [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md) and [RUNTIME.md](RUNTIME.md)
for package layout details.

## Build

```bash
cargo build --release --locked
cargo test --locked
```

Building needs stable Rust, `pkg-config`, and ALSA development headers. The
binary has no link-time dependency on ONNX Runtime. A source build needs an ORT
1.30.0 library through `backend.onnxruntime_library`, `OMAWAKE_ONNXRUNTIME_LIBRARY`,
`OMAWAKE_LIBRARY_PATH`, or one of the package-relative library directories.

## Model and algorithm provenance

The supported GigaSpeech Zipformer model was trained with the icefall keyword
spotting recipe introduced by [icefall PR #1428](https://github.com/k2-fsa/icefall/pull/1428).
Omawake's decoder is an independent Rust implementation of the algorithm
published in that work: the pinned
[`keywords_search`](https://github.com/k2-fsa/icefall/blob/aac7df064a6d1529f3bf4acccc6c550bd260b7b3/egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py#L962)
behavior and
[`ContextGraph`](https://github.com/k2-fsa/icefall/blob/aac7df064a6d1529f3bf4acccc6c550bd260b7b3/icefall/context_graph.py)
automaton specification. It does not copy or compile sherpa implementation
source.

Omawake and ONNX Runtime are MIT licensed. Icefall is Apache-2.0, and the
pinned model archive's publisher README identifies the model as Apache License
2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
