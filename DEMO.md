# Omawake demo

Validated on 2026-09-13 with `sherpa-onnx` 1.13.8 and the GigaSpeech 3.3M KWS model.

## Recorded routing

A config with `Lovely Child -> touch /tmp/omawake-action-lovely-child` and `Forever -> touch /tmp/omawake-action-forever` was run with:

```bash
omawake --config /tmp/omawake-demo.toml test \
  --audio MODEL/test_wavs/1.wav --execute --json
```

The pure-Rust compiler emitted:

```text
▁LOVE LY ▁CHI L D @lovely-child
▁FOR E VER @forever
```

The detector returned both stable IDs and both direct actions exited 0.

## Live daemon path

The live test created a temporary PipeWire null sink, routed Omawake's CPAL stream to its monitor, and played the same fixture into the sink. The daemon opened `pulse` at 44.1 kHz stereo; sherpa created its in-process 16 kHz resampler. It then logged:

```text
loaded wake-word model in 305 ms
armed on pulse (44100 Hz, 2 channel(s))
detected lovely-child; action exited 0
armed on pulse (44100 Hz, 2 channel(s))
detected forever; action exited 0
armed on pulse (44100 Hz, 2 channel(s))
stopped
```

Distinct marker files confirmed both actions. Re-arming opened a fresh microphone and recognizer stream without reloading the 305 ms model. `stop` removed the control socket. The original default microphone was restored and the temporary PipeWire module was unloaded.

## Capability behavior

The CPU binary with `runtime = "openvino"`, `device = "npu"`, and `fallback = "error"` exited before model creation with `requires openvino`. With `fallback = "cpu"`, it warned visibly, decoded both IDs, and reported:

```json
{"effective_runtime":"default","fallback_used":true}
```

## 2026-09-13 — setup proof

Ran one-command setup from a locally supplied, pinned archive in an isolated
environment:

```bash
omawake setup all --archive /tmp/recon/kws-model.tar.bz2
```

It succeeded and was idempotent (a second run reported `already-installed` with no re-download). Testing the bundled fixture then produced a `lovely-child` detection with a 338 ms model load:

```bash
omawake test --audio "$HOME/.local/share/omawake/models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav" --json
```

Automatic model download is disabled because the upstream model license is
unclear. Supply an archive obtained under rights you have verified.
