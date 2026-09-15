# Changelog

## Unreleased

## 0.0.1-rc.2 - 2026-09-15

- Added an independent Rust implementation of the published icefall
  keyword-spotting model contract.
- Updated the packaged CPU runtime to ONNX Runtime 1.30.0 and optional
  accelerators to its V2 execution-provider plugin interface.
- Added Linux aarch64 release archives and direct CPU parity validation in
  release CI.
- Use the model's FP32 accelerator graphs on Intel GPU and NPU. NPU also uses a
  fixed beam of at least eight to preserve the reference CPU detections.
- Serialize OpenVINO GPU compilation to prevent a reproduced Intel compiler
  crash during first-time cache preparation.
- Keep successful native provider diagnostics inside worker processes while
  retaining them when an operation fails.
- Updated rustls to 0.23.45 for RUSTSEC-2026-0285.

## 0.0.1-rc - 2026-09-14

- Added local wake-word detection from live capture or WAV files.
- Added multiple wake-word mappings to direct argument-vector actions.
- Added foreground daemon control, on-demand tests, and JSON benchmarks.
- Added guided and scriptable setup for models, runtimes, and diagnostics.
- Added one runtime-neutral executable with packaged CPU support and external
  OpenVINO and CUDA runtime selection.
- Added strict setup-time OpenVINO GPU/NPU model compilation and cache checks.
- Keep registered accelerator libraries mapped until process exit so vendor worker
  cleanup cannot call into an unloaded provider after inference completes.
- Added Intel CPU, iGPU, NPU, and NVIDIA GB10 validation evidence.

Known limits: release artifacts target Linux x86-64 and aarch64 with glibc 2.35
or newer; accelerator stacks are external. The separately downloaded catalog
model is publisher-declared Apache-2.0.
