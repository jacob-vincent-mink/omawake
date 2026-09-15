# Benchmark evidence

Most artifacts dated 2026-09-14 record the former sherpa-adapter
implementation; a few are low-level accelerator experiments. They are
retained as historical engineering evidence.

The current direct Rust KWS release evidence is:

- [`cuda-gb10-ort130-2026-09-15.md`](cuda-gb10-ort130-2026-09-15.md):
  ONNX Runtime 1.30 CUDA Plugin EP on NVIDIA GB10.
- [`openvino-dell-xps-ort130-2026-09-15.md`](openvino-dell-xps-ort130-2026-09-15.md):
  ONNX Runtime 1.30 OpenVINO Plugin EP on Intel CPU, iGPU, and NPU.

Each current report names the exact application commit and runtime artifacts
tested. Audio was piped through the WAV input path; neither proof used live
capture or playback.
