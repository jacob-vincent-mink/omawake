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
omawake setup runtime --runtime openvino --device gpu --dir /opt/omawake-openvino --apply
omawake setup runtime --runtime openvino --device npu --dir /opt/omawake-openvino --apply
omawake setup runtime --runtime cuda --device gpu --dir /opt/omawake-cuda --apply
omawake setup check
```

For OpenVINO, install the matching Intel GPU or NPU driver. The provider's
device list must expose the selected hardware. Omawake supplies a provider
`load_config` with a persistent cache directory and requests accuracy mode on
NPU. Extra V2 provider options may be added under
`[backend.options]`; `load_config` and `reshape_input` are reserved because
Omawake generates them from the selected device and model graph.

The calibrated int8 graphs are used on CPU, GPU, and CUDA. Intel NPU is the one
model-specific exception: it uses all three float graphs because the quantized
graphs execute but lose keyword accuracy on that device. Omawake also raises
the NPU beam width to at least eight, passes concrete shapes through OpenVINO's
`reshape_input` option, and pads the decoder/joiner batch to that fixed beam
width. Concrete CPU output tensors keep provider-owned output buffers from
escaping their run binding.

CUDA selection uses `backend.device_id` and the GPU exposed by the registered
CUDA plugin. Add CUDA and cuDNN library directories to `backend.library_dirs`
when the system loader does not already know them.

Provider readiness proves registration and device access. `omawake test` or
the setup model-cache probe proves that the model compiles and executes.
