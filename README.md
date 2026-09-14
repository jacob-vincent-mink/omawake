# Omawake

Omawake is a local wake-word daemon written in Rust. It compiles configured phrases with the model's tokenizer, captures audio through CPAL, and maps stable wake-word IDs to direct argument-vector actions. sherpa-onnx is the first registered backend (in-process ONNX keyword spotting); the setup and catalog layers are backend-neutral.

The current CPU build supports multiple wake words, recorded and live input, `pause`/`resume`/`stop`, JSON status and schema output, and validated config mutation. The recognizer stays loaded while the microphone stream is released before every action and reopened after cooldown.

## Build

```bash
cargo build --release
cargo test
```

The default binary uses sherpa-onnx's implicit static link mode and reports the
`cpu` capability. To build the OpenVINO flavor, point sherpa-onnx at a shared
library directory that was built with the ONNX Runtime OpenVINO execution
provider, then enable Omawake's feature:

```bash
SHERPA_ONNX_LIB_DIR=/absolute/path/to/sherpa-onnx/lib \
  cargo build --release --features openvino
```

The feature selects sherpa-onnx's shared link mode and makes Omawake report the
`openvino` compiled capability. The build deliberately fails unless
`SHERPA_ONNX_LIB_DIR` contains `libonnxruntime_providers_openvino.so`; the
ordinary sherpa-onnx shared release is CPU-only. The supplied native libraries,
their OpenVINO plugins, and loader paths must remain available at runtime. A
successful build or accepted configuration does not establish device placement;
inspect runtime provider evidence before treating NPU or GPU placement as
verified.

[`native/openvino`](native/openvino/README.md) contains the reproducible build
for the pinned ONNX Runtime 1.29, OpenVINO 2026.2.1, and sherpa-onnx 1.13.8
stack used by the hardware benchmark, including both required source patches.

Copy `config.example.toml` to `${XDG_CONFIG_HOME:-$HOME/.config}/omawake/config.toml`, then verify everything with:

```bash
omawake setup check
```

## Setup

Run `omawake setup` in a terminal to open the guided setup. Use the arrow keys and
Enter to choose **Full setup**, **Runtime**, **Model**, or **Check**. Full setup walks
through a compatible inference runtime and device, shows every downloadable model
with its install status, backend, family, description, and download size, then
installs the model and launcher and enables and starts the user service. The focused setup commands are
interactive too:

```bash
omawake setup runtime   # choose a compiled runtime and compatible device
omawake setup model     # browse, download if needed, and activate a model
```

When input is redirected or piped, these commands stay noninteractive and print
the runtime/model catalog with commands that scripts can run. Use `--json` by
itself for a machine-readable model catalog; model actions such as `--download`,
`--set`, and `--verify` are separate invocations.

The equivalent one-command, noninteractive network install is:

```bash
omawake setup all
```

It downloads and installs the default GigaSpeech Zipformer KWS model, writes `backend.kind` / `model.name` into the config, installs the launcher and the systemd service (started unless `--no-start`), then prints `setup check` results. It is idempotent — re-running it reports `already-installed` and does not re-download. Options:

```bash
omawake setup all --model <id>          # another catalog model
omawake setup all --archive <path>      # use an already-downloaded, pinned archive instead of fetching
omawake setup all --no-start
omawake setup all --progress-format json
```

Model management:

```bash
omawake setup model --list                          # catalog: id, backend, installed/available, description
omawake setup model --download sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
omawake setup model --verify sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
omawake setup model --set sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01   # verify an existing install, then activate it
```

`--download` verifies and installs the model (activating it in the config unless `--no-activate`); `--verify` only checks an installed model; `--set` verifies and then activates. `--archive <path>` installs from a local archive you already downloaded — it is still size + SHA256 verified against the pinned catalog entry. Model files land under `${XDG_DATA_HOME:-$HOME/.local/share}/omawake/models/<id>/`, or set `model.directory` to point elsewhere. Archives and extracted assets are verified by size and SHA256 (archive, then every required file), and installs are atomic: extraction goes to a staging directory and is renamed into place, with the previous model rolled back if activation fails.

Diagnostics and service management:

```bash
omawake setup check              # config, backend, model, engine, wake words, launcher, service
omawake setup runtime            # registered backends, compiled capabilities, runtime/device matrix
omawake setup systemd --status   # or: --uninstall, --no-start (no flag installs + starts)
omawake setup menu --status      # or: --uninstall (no flag installs the launcher)
```

```bash
omawake audio-devices
omawake test --audio test.wav --json
omawake benchmark --warmup 2 --iterations 10 test.wav another.wav > benchmark.json
omawake daemon
omawake status --json
omawake pause
omawake resume
omawake stop
```

`benchmark` loads the configured detector once, never opens a microphone or runs
actions, and emits JSON with the model load time, detections and timing for every
measured file iteration, audio duration, real-time factor, and p50/p95 summaries.
The [OpenVINO benchmark harness](scripts/benchmark-openvino.sh) prepares isolated
cold/hot CPU, GPU, and NPU lanes. See the
[Dell XPS 16 report](benchmarks/openvino-dell-xps-2026-09-14.md) for the tested
OpenVINO 2026.2.1 and ONNX Runtime 1.29.0 results and model compatibility note.

Add and remove mappings without editing TOML:

```bash
omawake wake-word add --id computer --phrase Computer -- notify-send "Wake word heard"
omawake wake-word remove computer
# `word` is a shorter alias:
omawake word remove computer
```

Removing the final mapping is allowed so configurations can be rebuilt incrementally. Starting the detector still requires at least one enabled mapping.

Actions are executed directly. Omawake does not insert a shell. Configure `sh -lc` explicitly when shell evaluation is intentional.

## Backends

`backend.runtime` accepts `default`, `openvino`, and `cuda`. Valid device values are:

| Runtime | Devices |
|---|---|
| `default` | `auto`, `cpu` |
| `cuda` | `auto`, `gpu` |
| `openvino` | `auto`, `npu`, `gpu`, `cpu`, `AUTO:...`, `HETERO:...`, `MULTI:...` |

The current build registers sherpa-onnx as the first (in-process ONNX) backend. The distributed CPU build reports only the `cpu` compiled capability. It rejects unavailable acceleration before model creation when `fallback = "error"`. With `fallback = "cpu"`, it emits a warning and exposes the fallback in status and test output. OpenVINO requires the `openvino` Cargo feature and an OpenVINO-enabled shared sherpa/ONNX Runtime stack. CUDA requires a separately linked release flavor. Merely selecting an accelerated runtime in TOML never counts as verified placement.

For OpenVINO, an empty `backend.provider_config` makes Omawake atomically write a user-private config below `$XDG_STATE_HOME/omawake/cache/openvino/<device>/provider.config`. It includes the uppercase canonical `device_type` and a separate compiled-model `cache_dir` for that device. Exact `NPU` selection also defaults `enable_qdq_optimizer=True` and `disable_dynamic_shapes=True`; `[backend.options]` can override those values and pass through other single-line OpenVINO options such as `ProfilingFilePrefix`. Option keys accept ASCII letters, digits, `.`, `_`, and `-`. Omawake manages `cache_dir`, and a `device_type` option must exactly match the selected canonical device.

Provider-specific device properties belong in OpenVINO's inline JSON
`load_config` value. For example, the Panther Lake NPU used in the benchmark
needed this platform override with the installed compiler:

```toml
[backend.options]
load_config = '{"NPU":{"NPU_PLATFORM":"5010"}}'
```

Set `backend.provider_config` to use an existing file instead. Absolute paths are used directly; relative paths resolve beside Omawake's application config. Omawake verifies that the path names a regular file and passes its absolute canonical path to sherpa without rewriting it.

See [DEMO.md](DEMO.md) for the reproducible live-capture demonstration and measured result.
