# Omawake

Omawake is a local wake-word daemon written in Rust. It embeds sherpa-onnx keyword spotting, compiles configured phrases with the model's SentencePiece tokenizer, captures audio through CPAL, and maps stable wake-word IDs to direct argument-vector actions.

The current CPU build supports multiple wake words, recorded and live input, `pause`/`resume`/`stop`, JSON status and schema output, and validated config mutation. The recognizer stays loaded while the microphone stream is released before every action and reopened after cooldown.

## Build

```bash
cargo build --release
cargo test
```

Copy `config.example.toml` to `${XDG_CONFIG_HOME:-$HOME/.config}/omawake/config.toml`. Download and extract the [GigaSpeech Zipformer KWS model](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) beneath `${XDG_DATA_HOME:-$HOME/.local/share}/omawake/models/`, or set `model.directory` to its absolute path.

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
```

Actions are executed directly. Omawake does not insert a shell. Configure `sh -lc` explicitly when shell evaluation is intentional.

## Backends

`backend.runtime` accepts `default`, `openvino`, and `cuda`. Valid device values are:

| Runtime | Devices |
|---|---|
| `default` | `auto`, `cpu` |
| `cuda` | `auto`, `gpu` |
| `openvino` | `auto`, `npu`, `gpu`, `cpu`, `AUTO:...`, `HETERO:...`, `MULTI:...` |

The distributed CPU build reports only the `cpu` compiled capability. It rejects unavailable acceleration before model creation when `fallback = "error"`. With `fallback = "cpu"`, it emits a warning and exposes the fallback in status and test output. OpenVINO and CUDA require separately linked release flavors; merely selecting them in TOML never counts as verified placement.

See [DEMO.md](DEMO.md) for the reproducible live-capture demonstration and measured result.
