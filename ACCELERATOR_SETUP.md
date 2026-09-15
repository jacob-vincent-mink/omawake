# Accelerator setup

CPU works from an unpacked Omawake release. Acceleration uses official ONNX
Runtime V2 execution-provider plugin packages compatible with the bundled ORT
1.30.0 core.

An accelerator directory should contain the provider DSO and its vendor
dependencies. It must not contain another `libonnxruntime.so`. Official
OpenVINO Python packages use
`libonnxruntime_providers_openvino_plugin.so`; official CUDA packages use
`libonnxruntime_providers_cuda.so`.

```bash
omawake setup runtime --json
omawake setup runtime --runtime openvino --device cpu --dir /opt/omawake-openvino --apply
omawake setup runtime --runtime openvino --device gpu --dir /opt/omawake-openvino --apply
omawake setup runtime --runtime openvino --device npu --dir /opt/omawake-openvino --apply
omawake setup runtime --runtime cuda --device gpu --dir /opt/omawake-cuda --apply
omawake setup check
```

## Intel CPU, integrated GPU, and NPU

The official [`onnxruntime-ep-openvino`](https://pypi.org/project/onnxruntime-ep-openvino/)
1.7.0 wheel supplies the V2 provider, OpenVINO runtime, and CPU/GPU/NPU plugins
for Linux x86-64. Download and unpack it without installing a second ONNX
Runtime core:

```bash
runtime="$HOME/.local/share/omawake/runtimes/openvino-1.7.0"
mkdir -p "$runtime" /tmp/omawake-openvino-download
python -m pip download --only-binary=:all: --no-deps \
  --dest /tmp/omawake-openvino-download onnxruntime-ep-openvino==1.7.0
python -m zipfile -e \
  /tmp/omawake-openvino-download/onnxruntime_ep_openvino-1.7.0-*.whl \
  "$runtime"

omawake setup runtime --runtime openvino --device cpu \
  --dir "$runtime/onnxruntime_ep_openvino" --apply
omawake setup runtime --runtime openvino --device gpu \
  --dir "$runtime/onnxruntime_ep_openvino" --apply
omawake setup runtime --runtime openvino --device npu \
  --dir "$runtime/onnxruntime_ep_openvino" --apply
```

The wheel is tagged `manylinux_2_28_x86_64`; it is not an aarch64 package.
Intel GPU and NPU use also requires the matching device driver. For example,
Arch Linux provides those dependencies through `openvino-intel-gpu-plugin` and
`openvino-intel-npu-plugin`. Setup must enumerate the requested device before
it saves the selection.

## NVIDIA CUDA

Download the official standalone CUDA Plugin EP matching the machine and CUDA
major. For example, for CUDA 13 on Linux x86-64:

```bash
archive=cuda_ep_cuda13_0.1.0_linux-x64.tar.gz
curl -fLO "https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.1.0/$archive"
printf '%s  %s\n' \
  5fa5cc5b19843809707818302771e4d16b740df069ca63908b28245e5f6b8398 \
  "$archive" | sha256sum -c -
runtime="$HOME/.local/share/omawake/runtimes/cuda13"
mkdir -p "$runtime"
tar -C "$runtime" -xzf "$archive"

omawake setup runtime --runtime cuda --device gpu --dir "$runtime" --apply
```

Other official 0.1.0 Linux assets are:

| CUDA | Architecture | Archive | SHA-256 |
|---|---|---|---|
| 12 | x86-64 | `cuda_ep_cuda12_0.1.0_linux-x64.tar.gz` | `dc34a4450e1b352671235205fb7d865c56ae61c7f8631df33ae2a369d4d1dcab` |
| 13 | aarch64 | `cuda_ep_cuda13_0.1.0_linux-aarch64.tar.gz` | `d02f9d438df1ad2cfc770e9eb93094710c1713d6a23f6bdc0f532ed83eb4b5f4` |

Install the NVIDIA driver, matching CUDA toolkit, and cuDNN separately. Add
their library directories to `backend.library_dirs` if the system loader does
not already find them. `omawake setup runtime --json` reports missing provider
dependencies without changing the active configuration.

For OpenVINO, install the matching Intel GPU or NPU driver. The provider's
device list must expose the selected hardware. Omawake supplies a provider
`load_config` with a persistent cache directory. It requests accuracy mode on
GPU and NPU and f32 inference precision on GPU. Extra V2 provider options may
be added under
`[backend.options]`; `load_config` and `reshape_input` are reserved because
Omawake generates them from the selected device and model graph.

The calibrated int8 graphs are used on default CPU, OpenVINO CPU, and CUDA.
OpenVINO GPU and NPU use all three float graphs because the quantized graphs
lose keyword accuracy on these devices. Omawake also raises the NPU beam width
to at least eight, passes concrete shapes through OpenVINO's `reshape_input`
option, and pads the decoder/joiner batch to that fixed beam width. Every graph
output consumed by Rust is bound to CPU-accessible memory so provider-owned GPU
or NPU buffers do not escape their run boundary.

CUDA selection uses `backend.device_id` and the GPU exposed by the registered
CUDA plugin. Add CUDA and cuDNN library directories to `backend.library_dirs`
when the system loader does not already know them.

Provider readiness proves registration and device access. `omawake test` or
the setup model-cache probe proves that the model compiles and executes.
