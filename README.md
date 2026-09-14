# Omawake

Omawake is a local wake-word daemon written in Rust. It compiles configured phrases with the model's tokenizer, captures audio through CPAL, and maps stable wake-word IDs to direct argument-vector actions. sherpa-onnx is the first registered backend (in-process ONNX keyword spotting); the setup and catalog layers are backend-neutral.

The current CPU build supports multiple wake words, recorded and live input, `pause`/`resume`/`stop`, JSON status and schema output, and validated config mutation. The recognizer stays loaded while the microphone stream is released before every action and reopened after cooldown.

## Build

```bash
cargo build --release
cargo test
```

Copy `config.example.toml` to `${XDG_CONFIG_HOME:-$HOME/.config}/omawake/config.toml`, then verify everything with:

```bash
omawake setup check
```

## Setup

`omawake setup` is a group of subcommands that install the model, write the config, register the desktop menu entry, and install the user systemd service. The one-command network install:

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
omawake daemon
omawake status --json
omawake pause
omawake resume
omawake stop
```

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

The current build registers sherpa-onnx as the first (in-process ONNX) backend. The distributed CPU build reports only the `cpu` compiled capability. It rejects unavailable acceleration before model creation when `fallback = "error"`. With `fallback = "cpu"`, it emits a warning and exposes the fallback in status and test output. OpenVINO and CUDA require separately linked release flavors; merely selecting them in TOML never counts as verified placement.

See [DEMO.md](DEMO.md) for the reproducible live-capture demonstration and measured result.
