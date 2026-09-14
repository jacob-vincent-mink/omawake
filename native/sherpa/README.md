# Provider-neutral sherpa-onnx build

[`build.sh`](build.sh) builds the patched sherpa-onnx C API against an external
ONNX Runtime SDK. It does not clone, build, patch, or copy ONNX Runtime, CUDA,
OpenVINO, or execution-provider libraries. The resulting sherpa library can
register compatible provider DSOs at process startup and use the same retained
`OrtEnv` for CPU, OpenVINO, and CUDA sessions.

The build is pinned to sherpa-onnx `v1.13.8` at
`11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`. It applies the checksum-pinned Oma
runtime patch in [`patches`](patches). TTS is disabled in this build, so the
resulting companion contains only the wake-word code Omawake uses.

Provide an ONNX Runtime 1.29 shared-library SDK. An official extracted archive
has the expected layout:

```text
/opt/onnxruntime-linux-x64-1.29.0/
├── include/onnxruntime_c_api.h
├── include/onnxruntime_cxx_api.h
└── lib/libonnxruntime.so
```

Build sherpa against it with:

```bash
OMA_ORT_ROOT=/opt/onnxruntime-linux-x64-1.29.0 \
  OMA_BUILD_JOBS=10 \
  native/sherpa/build.sh
source /tmp/oma-native-sherpa/runtime/env.sh
```

For SDKs with another layout, set both `OMA_ORT_INCLUDE_DIR` to the directory
containing `onnxruntime_cxx_api.h` and `OMA_ORT_LIB_DIR` to the directory
containing `libonnxruntime.so`. `OMA_NATIVE_ROOT` changes the build/output root.
The output contains the patched sherpa C API library and header only.

Release CI downloads that official CPU archive, verifies SHA256
`c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba`,
and invokes this script against it. The release package stages the ORT soname
chain and `libsherpa-onnx-c-api.so` under `lib/`. It leaves
`libonnxruntime_providers_*` out so OpenVINO and CUDA remain user-supplied setup
backends. The installed sherpa libraries use `$ORIGIN` to resolve companion
libraries in the same directory.

The application must load the selected `libonnxruntime` before loading
`libsherpa-onnx-c-api`, validate ORT's API version, then retain the Oma runtime
handle for the entire lifetime of every sherpa engine. Provider libraries are
registered by explicit path. Destroy streams, generated results, and
recognizers before destroying the runtime handle; its destructor
unregisters provider libraries in reverse order and releases the retained
environment.

The patched C ABI reports `SherpaOnnxGetOmaRuntimeAbiVersion(void) == 1`. It exposes
runtime creation, provider registration, device discovery, error reporting,
and destruction. OpenVINO and CUDA both use `GetEpDevices` plus
`SessionOptionsAppendExecutionProvider_V2`; neither depends on a provider being
compiled into the ORT core.

Provider DSOs remain external runtime inputs:

- CPU needs only the chosen core ORT library.
- CUDA uses the standalone CUDA plugin registered as `CUDAExecutionProvider`,
  plus compatible CUDA, cuDNN, and driver libraries.
- OpenVINO uses a provider library compatible with the selected ORT core, its
  matching `libonnxruntime_providers_shared`, and the OpenVINO runtime and
  driver libraries.

An OpenVINO plugin must advertise the requested CPU, GPU, or NPU through
`GetEpDevices`. `AUTO` selects the advertised
`OpenVINOExecutionProvider.AUTO` device. A physical accelerator reported by
ORT is insufficient if the provider plugin does not expose a corresponding EP
device. Put OpenVINO properties in `load_config`, scoped to the selected
device, for example:

```toml
[backend.options]
load_config = '{"NPU":{"CACHE_DIR":"/var/tmp/oma-openvino-cache","NPU_PLATFORM":"5010"}}'
```
