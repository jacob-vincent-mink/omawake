# Omawake 0.0.1-rc

This first preview provides local wake-word detection from live audio or WAV
files, multiple phrase-to-action mappings, foreground daemon controls, and
guided setup. The Linux x86-64 archive includes the default CPU runtime. The
same executable can use externally installed OpenVINO or CUDA stacks selected
during setup. Explicit OpenVINO GPU and NPU setup compiles and verifies the
fixed-shape wake-word model cache before the configuration is applied.

Accelerator libraries remain mapped until process exit while sessions and runtime
objects are still destroyed normally. This prevents late vendor worker cleanup
from calling into an unloaded execution-provider library.

The wake-word model is not bundled or downloaded automatically because the
upstream GigaSpeech model license is unclear. Obtain the pinned archive under
terms you have verified and pass it to setup with `--archive`.

Validated configurations include default and OpenVINO CPU, Intel iGPU and NPU,
and NVIDIA GB10 CUDA. Accelerator support requires matching external runtime
libraries. See `INSTALL.md`, `RUNTIME.md`, and the benchmark reports for the
exact setup and evidence.
