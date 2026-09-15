# Changelog

## Unreleased

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

Known limits: the release artifact targets Linux x86-64 with glibc 2.34 or
newer; accelerator stacks are external; the catalog model is bring-your-own
because its upstream model license is unclear.
