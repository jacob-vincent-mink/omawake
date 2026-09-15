# Accelerator setup

Omawake 0.0.1-rc ships one runtime-neutral executable with a ready-to-use
default CPU stack. Acceleration is optional and external. Setup only discovers,
probes, and records native libraries the user already installed; it never
downloads or installs OpenVINO, CUDA, ONNX Runtime, or a driver.

Unlike Omaspeak's direct OpenVINO path, Omawake runs keyword spotting through
its patched sherpa-onnx C API. An accelerator bundle therefore needs four
pieces from the same ABI-compatible stack:

- ONNX Runtime 1.29 core;
- `libonnxruntime_providers_shared.so` and the selected provider DSO;
- `libsherpa-onnx-c-api.so` built by [`native/sherpa/build.sh`](native/sherpa/build.sh)
  against that exact ONNX Runtime SDK;
- the matching Intel or NVIDIA runtime and driver libraries.

`omawake setup runtime --json` shows all runtime/device rows, discovered paths,
probe evidence, and remediation. Only `--apply` writes a selection.

## Default CPU

The release archive and `-bin-rc` package need no inference runtime package.
After supplying the licensed model archive, verify the file path without a
microphone:

```bash
omawake setup all --archive /path/to/sherpa-onnx-kws-model.tar.bz2
omawake test --audio /path/to/test.wav --json
```

The release's ONNX Runtime 1.29 CPU core and patched sherpa library are found
beside the executable or under `/usr/lib/omawake` when packaged.

## Intel integrated GPU and NPU with OpenVINO

On Arch Linux and Omarchy, install OpenVINO plus only the device plugin you
intend to use:

```bash
# Provider build tools (one time)
sudo pacman -S --needed base-devel cmake git patchelf python

# Intel integrated GPU
sudo pacman -S --needed openvino openvino-intel-gpu-plugin

# Intel NPU (includes compiler and NPU driver dependencies)
sudo pacman -S --needed openvino openvino-intel-npu-plugin
```

Log out and back in if permissions changed, then confirm the device node:

```bash
ls -l /dev/dri/renderD*   # integrated GPU
ls -l /dev/accel/accel*  # NPU
```

Omawake requires ONNX Runtime's OpenVINO provider DSO, which the OpenVINO
package itself does not supply. Build the provider from the exact ONNX Runtime
1.29.0 tag against the installed OpenVINO CMake package:

```bash
git clone --branch v1.29.0 --depth 1 --recursive \
  https://github.com/microsoft/onnxruntime.git onnxruntime-openvino-1.29.0
cd onnxruntime-openvino-1.29.0
./build.sh --config Release --build_dir "$PWD/build-openvino" --parallel \
  --build_shared_lib --skip_tests --compile_no_warning_as_error \
  --use_openvino NPU \
  --cmake_extra_defines OpenVINO_DIR=/usr/lib/cmake/openvino \
  FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
  onnxruntime_BUILD_UNIT_TESTS=OFF
```

Build the patched sherpa library against that result, then assemble a private
runtime directory. These commands run from an Omawake source checkout:

```bash
ort_source=/absolute/path/to/onnxruntime-openvino-1.29.0
ort_build="$ort_source/build-openvino/Release"
OMA_ORT_INCLUDE_DIR="$ort_source/include/onnxruntime/core/session" \
OMA_ORT_LIB_DIR="$ort_build" \
OMA_NATIVE_ROOT="$HOME/.cache/omawake-sherpa-openvino" \
  native/sherpa/build.sh

runtime_root="$HOME/.local/share/omawake/runtimes/openvino-1.29.0"
mkdir -p "$runtime_root/lib"
cp -a "$ort_build"/libonnxruntime.so* "$runtime_root/lib/"
cp -a "$ort_build"/libonnxruntime_providers_shared.so "$runtime_root/lib/"
cp -a "$ort_build"/libonnxruntime_providers_openvino.so "$runtime_root/lib/"
cp -a "$HOME/.cache/omawake-sherpa-openvino/runtime/lib/libsherpa-onnx-c-api.so" \
  "$runtime_root/lib/"
```

Install the wake model before applying the accelerator so setup can compile it
immediately. Then select either Intel device:

```bash
omawake setup runtime --runtime openvino --device gpu \
  --dir "$runtime_root" --apply

# Or, for NPU:
omawake setup runtime --runtime openvino --device npu \
  --dir "$runtime_root" --apply
```

For GPU and NPU, setup runs the pinned test WAV through the detector in an
isolated, no-fallback process and requires a nonempty compiled model blob before
it saves the selection. Blobs live below
`${XDG_CACHE_HOME:-$HOME/.cache}/omawake/openvino/<device>/compiled`. Intel GPU
compiler crashes are contained in the child and retried a bounded number of
times; setup still fails without a complete detector pass. If runtime setup
precedes the model, model setup performs this compilation before activation.

Validate with direct audio input:

```bash
omawake setup check
omawake test --audio /path/to/test.wav --json
```

The [Dell XPS OpenVINO report](benchmarks/openvino-dell-xps-2026-09-14.md)
records CPU, integrated GPU, and NPU placement, detection parity, and timings.

## NVIDIA GPU with CUDA

Install an NVIDIA driver, CUDA, and cuDNN, then download the official ONNX
Runtime 1.29 archive matching the CUDA major. ONNX Runtime 1.29 uses cuDNN 9:

```bash
# CUDA 13, Linux x86-64
curl -fLO https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-gpu_cuda13-1.29.0.tgz
printf '%s  %s\n' \
  844c64acfc43ab9423215c26493055ea229268e28283146cc644ecef0bdae048 \
  onnxruntime-linux-x64-gpu_cuda13-1.29.0.tgz | sha256sum -c -
tar -xzf onnxruntime-linux-x64-gpu_cuda13-1.29.0.tgz

# CUDA 12 alternative: SHA-256
# 4ca594a0da83927befbd73fe020d7f569be151d70bb4fe9741ad405f4882e2ad
```

Build sherpa against the extracted SDK and assemble the runtime directory:

```bash
ort_root="$PWD/onnxruntime-linux-x64-gpu_cuda13-1.29.0"
OMA_ORT_ROOT="$ort_root" \
OMA_NATIVE_ROOT="$HOME/.cache/omawake-sherpa-cuda" \
  native/sherpa/build.sh

runtime_root="$HOME/.local/share/omawake/runtimes/cuda-1.29.0"
mkdir -p "$runtime_root/lib"
cp -a "$ort_root"/lib/libonnxruntime.so* "$runtime_root/lib/"
cp -a "$ort_root"/lib/libonnxruntime_providers_shared.so "$runtime_root/lib/"
cp -a "$ort_root"/lib/libonnxruntime_providers_cuda.so "$runtime_root/lib/"
cp -a "$HOME/.cache/omawake-sherpa-cuda/runtime/lib/libsherpa-onnx-c-api.so" \
  "$runtime_root/lib/"

omawake setup runtime --runtime cuda --device gpu \
  --dir "$runtime_root" --apply
omawake setup check
omawake test --audio /path/to/test.wav --json
```

A system CUDA/cuDNN install normally exposes its libraries to the loader. For
an isolated install, add every dependency directory after Apply and rerun the
check:

```bash
omawake config set backend.library_dirs \
  "$runtime_root/lib:/usr/local/cuda/lib64:/absolute/path/to/cudnn/lib"
omawake setup check
```

The [GB10 CUDA report](benchmarks/cuda-gb10-2026-09-14.md) records exact
detections, provider probing, process-specific Nsight kernels, and timings.

## Services

Runtime and model setup leave the user service absent. Test on demand first.
Run `omawake setup systemd` only when you want an always-running microphone
daemon.

Official runtime references:

- [OpenVINO Linux installation](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-linux.html)
- [OpenVINO NPU device requirements](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
