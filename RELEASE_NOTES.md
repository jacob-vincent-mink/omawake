# Omawake 0.0.1-rc.1

This release candidate provides local wake-word detection from live audio or WAV
files, multiple phrase-to-action mappings, foreground daemon controls, and
guided setup. Linux x86-64 and aarch64 archives include the default CPU
runtime. The same executable can use externally installed OpenVINO or CUDA
stacks selected during setup. Explicit OpenVINO GPU and NPU setup compiles and
verifies the fixed-shape wake-word model cache before the configuration is
applied.

Accelerator libraries remain mapped until process exit while sessions and runtime
objects are still destroyed normally. This prevents late vendor worker cleanup
from calling into an unloaded execution-provider library.

The wake-word model is not bundled. Guided setup downloads a checksum-pinned
publisher archive whose included README marks the model as Apache License 2.0;
`--archive` also accepts a previously downloaded copy.

The model integration is owned by Omawake: feature extraction, streaming
state, context-graph traversal, beam search, and keyword finalization are an
independent Rust implementation of the original icefall algorithm and model
contract. No sherpa runtime, patch, ABI, or implementation source is included.

Validated configurations include default and OpenVINO CPU, Intel iGPU and NPU,
and NVIDIA GB10 CUDA using the official ORT 1.30 Plugin EP. Accelerator support
requires matching external runtime libraries. See `INSTALL.md` and `RUNTIME.md`
for the exact setup. Benchmark artifacts recorded before the direct Rust KWS
implementation are identified as historical in `benchmarks/README.md`.
