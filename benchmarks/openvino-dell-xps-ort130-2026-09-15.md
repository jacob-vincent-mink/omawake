# Intel OpenVINO Plugin EP validation — 2026-09-15

Omawake's release KWS code through commit
`eb69aba6d6ae4115968c891a8724a2845049cabd` was exercised on a Dell XPS with
an Intel Core Ultra X7 358H, Arc B390 iGPU (`8086:b080`), and Series 3 NPU
(`8086:b03e`). The candidate executable reported `0.0.1-rc.2` and had SHA-256
`1ef3353e6ba48363bf5a821334934e034d26fd0e74c2cd4472c485421c664834`.

The test used ONNX Runtime 1.30.0, the official OpenVINO Plugin EP 1.7.0, and
OpenVINO 2026.3.1. The ORT core SHA-256 was
`245a6f8c38127551057a1cd1ffd59f0a186a227ade4f3492dea2494eb565542e`;
the provider SHA-256 was
`00830d945b40a658e90cce381c362b15145d6a4224dd00da8eb41eded3688b8b`.
Audio entered through the publisher WAV fixtures. No microphone or playback
was used.

Every cold and hot iteration on all four lanes returned the exact expected
IDs: `0.wav` produced `light-up`; `1.wav` produced `lovely-child`, then
`forever`. This is exact parity over the three positive fixture detections. It
does not substitute for a larger false-accept and false-reject evaluation.

Cold is one measured iteration in a fresh process and per-lane cache root. Hot
is a second process with one warmup and three measured iterations. Times are
milliseconds; the file columns show their individual p50 values.

| Lane | Phase | Load | `0.wav` | `1.wav` |
|---|---|---:|---:|---:|
| default CPU | cold | 235.729 | 72.690 | 177.756 |
| default CPU | hot | 307.646 | 119.972 | 170.918 |
| OpenVINO CPU | cold | 1,399.306 | 313.538 | 579.198 |
| OpenVINO CPU | hot | 474.844 | 225.070 | 572.996 |
| OpenVINO iGPU | cold | 18,324.754 | 3,252.817 | 362.200 |
| OpenVINO iGPU | hot | 259.012 | 129.262 | 323.733 |
| OpenVINO NPU | cold | 5,028.894 | 192.918 | 344.431 |
| OpenVINO NPU | hot | 1,300.965 | 123.957 | 291.709 |

OpenVINO CPU is slower than ORT's default CPU provider because the same small
streaming graphs take a different partitioning, kernel, and dispatch path. The
iGPU and NPU also pay provider and transfer overhead while the decoder loop
remains host-driven. Accelerator lanes use FP32 graphs to preserve detections;
CPU lanes use the publisher INT8 graphs. These costs explain why more capable
hardware does not imply lower latency for this workload.

All OpenVINO lanes reported their requested runtime and device with
`fallback_used=false`. NPU busy time increased by 356,162 microseconds in the
cold phase and 1,122,564 microseconds in the hot phase. All eight benchmark
stderr files were empty.

## Setup-time compilation

A separate empty-cache runtime setup selected
`OpenVINOExecutionProvider:GPU:45184`, compiled the three FP32 graphs on its
first worker attempt, and produced three model blobs totaling 36,827,136
bytes. Omawake limits OpenVINO GPU compilation to one compiler thread to avoid
the Intel OpenCL compiler crash reproduced before this release. Three clean
cold-cache trials, including the release candidate binary, completed on their
first attempt; `coredumpctl` recorded no new Omawake crash.

An empty-cache NPU setup selected
`OpenVINOExecutionProvider:NPU:45118`, compiled three blobs totaling
13,173,659 bytes, and increased NPU busy time by 127,548 microseconds. A later
process imported that cache, detected all three expected IDs, increased NPU
busy time by 359,555 microseconds, and left every compiled-blob SHA-256
unchanged. Its stderr was empty. This proves that model compilation happened
during setup and that normal inference reused the prepared cache.

Compact structured evidence is retained in
[`openvino-dell-xps-ort130-results-2026-09-15`](openvino-dell-xps-ort130-results-2026-09-15/).
Compiled model blobs, OpenCL compiler caches, and OpenVINO profiles remain
off-repository.
