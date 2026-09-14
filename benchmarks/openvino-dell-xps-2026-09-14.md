# Current OpenVINO proof: Dell XPS 16

Validated on 2026-09-14 after the native adapter cleanup. This run used one
current Omawake executable and one freshly built sherpa C adapter for the
packaged default CPU runtime and the user-supplied OpenVINO runtime. It read the
official GigaSpeech 3.3M KWS test WAVs directly and never opened a microphone.
Compact JSON and setup evidence is retained in the
[current evidence directory](openvino-dell-xps-current-results-2026-09-14/).

## Artifact identity

- sherpa-onnx upstream: `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`
- extended runtime patch: `b462ad2f88bbd5811df8180d40e4bd798569b0a1aa58fd9aaa4a38394e293f86`
- KWS exception-safety patch: `ad895ce231ec7ce2d1e9575450b6493c88d2272b8d3bbded8ad4a24a26d8944a`
- Omawake executable: `5fde4c0b32830371c7cac591f87e4b42a1c58e979559d88fe5be70d77cbfa3f1`
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

Cold is one measured pass in a fresh process and cache. Hot is a new process
that reuses the cache, performs one warmup, and measures two passes per WAV.
The small sample is a functional comparison rather than a stable performance
study. Times are milliseconds. The p50/p95 values combine both WAV lengths, so
per-file values in the retained JSON are more useful for detailed comparison.

| Lane | Phase | Load | p50 | p95 | p50 RTF | p95 RTF |
|---|---:|---:|---:|---:|---:|---:|
| default CPU | cold | 308.320 | 60.026 | 146.884 | 0.00879 | 0.00906 |
| default CPU | hot | 303.434 | 58.804 | 146.342 | 0.00876 | 0.00888 |
| OpenVINO CPU | cold | 1572.650 | 257.136 | 547.265 | 0.03274 | 0.03881 |
| OpenVINO CPU | hot | 622.543 | 215.590 | 537.577 | 0.03207 | 0.03254 |
| OpenVINO iGPU | cold | 1220.122 | 284.371 | 565.861 | 0.01701 | 0.08541 |
| OpenVINO iGPU | hot | 374.709 | 112.565 | 284.240 | 0.01687 | 0.01701 |
| OpenVINO NPU | cold | 228.647 | 235.590 | 3615.891 | 0.01409 | 0.54579 |
| OpenVINO NPU | hot | 221.127 | 79.562 | 187.286 | 0.01120 | 0.01201 |

Every measured iteration matched the default CPU detection set: `0.wav`
detected `light-up`, while `1.wav` detected `lovely-child` and `forever`. This
is exact 3/3 positive-fixture parity. It does not measure false accepts or
accuracy over broad speech and noise.

Each OpenVINO phase produced an ORT profile containing
`OpenVINOExecutionProvider`, and the isolated setup inventory independently
reported the requested CPU, GPU, and NPU devices accessible. The NPU busy
counter increased by 259,521 microseconds during the cold run and 549,476
microseconds during the hot run. Together these establish real provider and
NPU placement. Sherpa routes the encoder through OpenVINO; its decoder and
joiner remain on CPU.

## Setup contract

The packaged default CPU runtime was discovered with source `package` before a
config existed. Its scripted preview returned `ready=true` and left the config
absent; `--apply` then persisted the exact validated paths. The OpenVINO NPU
preview also left the config absent, `--apply` persisted it, and a subsequent
inventory reported OpenVINO `auto`, `cpu`, `gpu`, and `npu` ready. The retained
preview, apply, and inventory JSON are the direct command output.
