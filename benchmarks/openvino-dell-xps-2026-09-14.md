# OpenVINO benchmark: Dell XPS 16

Measured on 2026-09-14 with the official GigaSpeech 3.3M KWS model and its two
test WAVs. The raw evidence is retained locally in the ignored artifact run
`benchmark-artifacts/run-20260914T050440Z-981876`.

## System

- Dell XPS 16 DA16260
- Intel Core Ultra X7 358H, 16 cores
- Intel Panther Lake Arc B390 GPU (`8086:b080`)
- Intel Core Ultra Series 3 NPU (`8086:b03e`)
- Linux 7.2.3-arch1-1-ptl, x86-64
- Rust 1.98.1 and sherpa-onnx 1.13.8
- ONNX Runtime 1.29.0 OpenVINO EP with official OpenVINO 2026.2.1

The shared ONNX Runtime provider DSO was rebuilt against OpenVINO 2026.2.1 and
contained the local zero-element tensor patch. Runtime loader tracing confirms
that the final KWS harness loaded that DSO and its 2026.2.1 OpenVINO libraries.
KWS did not exercise or need the zero-element fix; that fix was required for the
Omaspeak Supertonic graph.

## Method and result

The [benchmark harness](../scripts/benchmark-openvino.sh) used exact `cpu`,
`gpu`, and `npu` device selections, separate state/cache directories per lane,
and separate profiling prefixes per phase. Cold is the first process against a
fresh lane cache with zero warmups and one measured iteration per WAV. Hot
reuses that cache; its two warmups per WAV and ten measured iterations per WAV
all use one newly loaded engine. The load value measures backend creation,
including provider graph compilation, but excludes process startup and keyword
compilation. Times are milliseconds; RTF is inference time divided by audio
duration.

| Lane | Phase | Load | p50 | p95 | p50 RTF | p95 RTF |
|---|---:|---:|---:|---:|---:|---:|
| default CPU | cold | 372.247 | 71.144 | 173.605 | 0.01039 | 0.01074 |
| default CPU | hot | 370.163 | 69.308 | 174.739 | 0.01038 | 0.01046 |
| OpenVINO CPU | cold | 1744.473 | 289.803 | 657.883 | 0.03936 | 0.04374 |
| OpenVINO CPU | hot | 652.073 | 264.416 | 660.373 | 0.03827 | 0.03991 |
| OpenVINO iGPU | cold | 1448.610 | 283.091 | 612.611 | 0.01694 | 0.09247 |
| OpenVINO iGPU | hot | 404.936 | 140.995 | 292.341 | 0.01745 | 0.02102 |
| OpenVINO NPU | cold | 215.027 | 220.114 | 4050.786 | 0.01317 | 0.61144 |
| OpenVINO NPU | hot | 204.260 | 72.955 | 180.395 | 0.01059 | 0.01097 |

All eight phases matched the default CPU result in every measured iteration: `0.wav`
detected `light-up`, while `1.wav` detected `lovely-child` and `forever`. This
is exact 3/3 wake-word detection parity on the supplied positive fixtures. The
two-file set does not measure false accepts or estimate accuracy on broader
speech and noise.

Every accelerated profile contains `OpenVINOExecutionProvider`; no provider
fallback or device-unavailable message occurred. The NPU busy counter increased
by 240,000 microseconds during cold and 1,933,536 microseconds during hot. These
are runtime evidence independent of the requested device. Benchmark JSON keeps
`placement_verified=false` because configuration alone is not proof.
Sherpa routes the encoder session through OpenVINO in this KWS architecture;
decoder and joiner execution remains on CPU.

## Model compatibility

In earlier compatibility runs, the catalog's fully INT8 model preserved all
detections on both CPU lanes, but GPU/NPU provider processes returned zero
detections. An initial fresh-cache INT8 GPU process also crashed. Disabling the
NPU QDQ optimizer did not restore any detections, while its busy counter still
increased.

Using the FP32 encoder with the INT8 decoder and joiner restored all three
detections and produced a stable fresh-cache GPU run. The harness therefore
uses `fp32-encoder` for GPU and NPU by default and retains INT8 for both CPU
lanes. The incompatible INT8 evidence is preserved in runs
`run-20260914T045935Z-970102` and `run-20260914T050411Z-980895`; the focused
FP32-encoder confirmation is `run-20260914T050330Z-979607`.
