# Current OpenVINO proof: Dell XPS 16

Validated on 2026-09-14 after the native adapter cleanup and setup-cache
implementation. This run used the release-mode `0.0.1-rc` Omawake executable
and one sherpa C adapter for the packaged default CPU runtime and the
user-supplied OpenVINO runtime. It read the official GigaSpeech 3.3M KWS test
WAVs directly and never opened a microphone. Compact JSON, cache hashes, setup
output, provider evidence, and device counters are retained in the
[RC setup-cache evidence directory](openvino-dell-xps-rc-setup-cache-results-2026-09-14/).
The [native adapter evidence](openvino-dell-xps-current-results-2026-09-14/)
retains the lower-level library contract checks.

## Artifact identity

- sherpa-onnx upstream: `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`
- extended runtime patch: `b462ad2f88bbd5811df8180d40e4bd798569b0a1aa58fd9aaa4a38394e293f86`
- KWS exception-safety patch: `ad895ce231ec7ce2d1e9575450b6493c88d2272b8d3bbded8ad4a24a26d8944a`
- Omawake executable: `f100e57e4169fbfeefabe47d10a41457e8467332091317e32e516fbabf1e7444`
- staged sherpa C adapter: `b854cdee22547ff4e2728f160caeb3d42b2ab81424ae168f8b8c13700867d884`
- OpenVINO ONNX Runtime 1.29.0: `d381bf42df6bffc4910bc9f653cb77206bad055157da1bf30d5c5d7a1e977b03`
- OpenVINO provider: `73c0ee3cd80fbf802aa376f21951afa8302d81915a90a1ba7dacf68a2faa2b01`

The adapter exports `SherpaOnnxGetExtendedApiVersion`; the retired
product-specific marker is absent. Its native lifetime/registration contract
test passed in OpenVINO CPU provider mode, including idempotent registration,
conflicting-path rejection, spotter-retained provider lifetime, clean
re-registration, and the exception-safe failure when accelerated KWS creation
has no retained runtime. The host did not have `patchelf`, so the compile-only build first
used the documented no-op shim. The staged proof copy then had the build-time
absolute suffix removed from its dynamic string table; `readelf -d` verified
the resulting `RUNPATH` is exactly `$ORIGIN`. The original compiler output and
staged hashes are both recorded in `artifact-fingerprints.txt`. Shipping builds
still require real `patchelf` through `native/sherpa/build.sh`.

## Hardware and method

- Dell XPS 16 DA16260
- Intel Core Ultra X7 358H, 16 cores
- Intel Panther Lake Arc B390 GPU (`8086:b080`)
- Intel Core Ultra Series 3 NPU (`8086:b03e`)
- Linux 7.2.3-arch1-1-ptl, x86-64
- OpenVINO 2026.2.1

Cold import is one measured pass in a fresh process. The GPU and NPU lanes
import the cache prepared by setup; the default and OpenVINO CPU lanes load in
their usual way. Hot is another process that performs two warmups and measures
ten passes per WAV. This is a functional comparison rather than a stable
performance study. Times are milliseconds. The p50/p95 values combine both
WAV lengths, so the per-file values below are clearer for comparison.

| Lane | Phase | Load | p50 | p95 | p50 RTF | p95 RTF |
|---|---:|---:|---:|---:|---:|---:|
| default CPU | cold import | 314.677 | 60.779 | 148.326 | 0.00887 | 0.00917 |
| default CPU | hot | 304.368 | 57.547 | 145.555 | 0.00863 | 0.00871 |
| OpenVINO CPU | cold import | 1601.980 | 262.147 | 544.226 | 0.03256 | 0.03957 |
| OpenVINO CPU | hot | 669.897 | 311.274 | 582.520 | 0.03436 | 0.04103 |
| OpenVINO iGPU | cold import | 407.214 | 285.456 | 446.841 | 0.01708 | 0.06745 |
| OpenVINO iGPU | hot | 376.302 | 113.531 | 292.159 | 0.01692 | 0.01748 |
| OpenVINO NPU | cold import | 240.022 | 207.203 | 230.126 | 0.01377 | 0.03128 |
| OpenVINO NPU | hot | 211.883 | 79.378 | 197.697 | 0.01174 | 0.01197 |

The aggregate p50/p95 pair mixes a 6.625-second WAV with a 16.715-second WAV.
These ten-iteration per-file hot values are the clearer comparison:

| Lane | 6.625 s WAV p50 / p95 | 16.715 s WAV p50 / p95 |
|---|---:|---:|
| default CPU | 57.18 / 57.55 ms | 143.68 / 145.57 ms |
| OpenVINO CPU | 229.74 / 311.27 ms | 561.07 / 597.42 ms |
| OpenVINO iGPU | 111.63 / 113.53 ms | 283.98 / 295.46 ms |
| OpenVINO NPU | 77.77 / 79.38 ms | 194.47 / 197.78 ms |

Default CPU uses ONNX Runtime's CPU execution provider. OpenVINO CPU uses the
same physical processor through ONNX Runtime's OpenVINO provider and Intel's
CPU plugin, so it has different graph partitioning, kernels, scheduling, and
dispatch costs. Its per-file median was about 3.9 to 4.0 times slower here. The
iGPU and NPU lanes offload only the encoder while decoder and joiner work
remains on CPU. For this small streaming network, device dispatch, copies, and
provider boundaries are a large part of elapsed time; the iGPU was therefore
about 1.95 to 1.98 times slower than default CPU even though it performed the
requested accelerator work. GPU and NPU also use the FP32 encoder required for
detection parity, while both CPU lanes retain the smaller INT8 encoder.

NPU model compilation took 4.18 seconds during setup. The first measured NPU
process then imported the prepared cache, so its cold-import timings do not
include model compilation.

Every measured iteration matched the default CPU detection set: `0.wav`
detected `light-up`, while `1.wav` detected `lovely-child` and `forever`. This
is exact 3/3 positive-fixture parity. It does not measure false accepts or
accuracy over broad speech and noise.

Each OpenVINO phase produced an ORT profile containing
`OpenVINOExecutionProvider`, and the isolated setup inventory independently
reported the requested CPU, GPU, and NPU devices accessible. The NPU busy
counter increased by 90,421 microseconds during setup, 79,335 microseconds
during the first direct inference, 236,753 microseconds during the cold-import
benchmark, and 1,960,396 microseconds during the hot benchmark. Together these
establish real provider and NPU placement. Sherpa routes the encoder through
OpenVINO; its decoder and joiner remain on CPU.

A final clean ONNX Runtime 1.29.0 teardown check against the installed OpenVINO
2026.3.1 stack, after process-lifetime accelerator-library pinning, read the
official `0.wav` directly and detected `light-up`. Its profile recorded 21
`OpenVINOExecutionProvider` events, and NPU busy time increased by 84,197
microseconds. The process exited 0 with no coredump. Both prepared NPU cache
blobs retained their exact size, mtime, and SHA-256, so this run imported the
setup cache rather than compiling during inference. The compact result is in
`openvino-dell-xps-rc-setup-cache-results-2026-09-14/clean-ort-npu-teardown-proof.json`.

## Setup contract

The final proof began with empty, isolated XDG config, state, and cache roots.
Applying the NPU runtime performed a real inference during setup and created
one 11,895,398-byte compiled blob under
`$XDG_CACHE_HOME/omawake/openvino/npu/compiled`. New processes then performed
direct inference, parity checks, and both benchmark phases without changing
that blob's hash.

Applying the iGPU runtime likewise compiled its fixed graph during setup. The
Intel OpenCL compiler terminated with signal 11 on the first three child
attempts while progressively filling its cache; attempt 4 of the bounded five
completed and created one 26,254,143-byte blob. The retry policy applies only
to signal terminations. Ordinary failures stop immediately, and a fifth signal
failure is final. Fresh-process iGPU benchmarks kept the prepared blob
byte-for-byte unchanged and produced OpenVINO execution-provider profiles.

Every setup and inference step used `fallback = "error"`. The setup check
reported the NPU model cache ready. The retained SHA-256 manifests prove NPU
and iGPU cache reuse, and all 88 measured file iterations across the four
lanes returned the exact expected positive-fixture detections.
