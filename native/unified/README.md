# Unified CPU, OpenVINO, and CUDA runtime

[`build.sh`](build.sh) produces the shared native stack used by Omawake's Linux
x86_64 release workflow. It builds one ONNX Runtime with its CPU engine,
OpenVINO Execution Provider, and CUDA Execution Provider, then links one shared
sherpa-onnx layer against that runtime. The source commits, OpenVINO package,
and local patches are pinned identically to the OpenVINO-only builder.

The build requires Linux x86_64, CMake 3.28 or newer, a CUDA 12.8-compatible
toolkit, cuDNN 9, and the prerequisites listed by the OpenVINO builder. By
default it reads CUDA from `/usr/local/cuda`, cuDNN from `/usr`, and targets the
NVIDIA architectures used by Turing through Blackwell:

```bash
OMA_BUILD_JOBS=10 native/unified/build.sh
source /tmp/oma-native-unified/runtime/env.sh
cargo build --release --all-features
```

Override `CUDA_HOME`, `CUDNN_HOME`, or the semicolon-separated
`OMA_CUDA_ARCHITECTURES` when building in another toolchain image. The output
contract is `${OMA_NATIVE_ROOT:-/tmp/oma-native-unified}/runtime/{lib,include,env.sh}`.
The builder verifies the cuDNN header and library below `CUDNN_HOME`; the
generated `env.sh` includes the detected cuDNN library directory.

The release archive contains one Omawake executable, the base sherpa/ONNX
Runtime libraries needed for CPU execution, and both provider DSOs. CUDA and
OpenVINO remain runtime dependencies: selecting CUDA requires a compatible
NVIDIA driver plus CUDA/cuDNN libraries; selecting OpenVINO requires a
compatible OpenVINO installation and device driver. Missing accelerator
dependencies do not prevent the packaged executable from starting on CPU
because ONNX Runtime loads a provider DSO only when that provider is selected.

The release workflow verifies the archive's relative loader path and asserts
that the same executable advertises exactly `cpu`, `openvino`, and `cuda`.
