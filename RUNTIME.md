Omawake ships one executable and a CPU runtime under lib/: ONNX Runtime 1.29.0
and sherpa-onnx 1.13.8 with the extended sherpa API for keyword spotting.
The extended sherpa library is required for the exception-safe KWS C boundary,
but default CPU inference does not create or retain a provider-plugin runtime.

Run `omawake setup runtime --json` to inspect runtime/device availability.
OpenVINO and CUDA are external: supply matching ORT core, patched sherpa,
provider DSO and Intel/NVIDIA dependencies. Setup never installs these files.

`omawake setup runtime --runtime openvino --device npu --dir /absolute/runtime`
previews and probes a candidate. Add `--apply` to save after a successful probe.
The guided setup offers a separate Apply/Cancel review. Model installation is
separate and subject to its license policy. Only `setup systemd` installs or
starts the optional service. Runtime readiness does not prove model placement.
