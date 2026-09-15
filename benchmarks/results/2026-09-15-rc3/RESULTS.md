# Omawake v0.0.1-rc.3 hardware evidence

The Linux release source at `938341e4898b5cfaea36d0c93910bced80b5b94b`
was exercised entirely with WAV input. No microphone was opened and mapped
actions were disabled. The local x86-64 binary SHA-256 was
`f9c2d2cc89683d5d1ce2b7e279e05457ba269a544f797149dbfa5ae652ef7354`.

## Intel local results

The host was an Intel Core Ultra X7 358H with an Arc B390 integrated GPU and a
Series 3 NPU. OpenVINO used the complete 2026.3.0 runtime installed for
Voxtype. The default provider was audio.cpp commit `e9ff200` on CPU.

The positive set contains 60 Piper-generated `Hey Jarvis` clips from three
voices with varied prosody. The clean negative set is a deterministic,
checksum-pinned 100-file LibriSpeech subset totaling 0.250689 hours. The files
are not redistributed by this repository. Every backend received the same
files and used exact normalized whole-phrase matching.

| Provider | Device | Cold model load | Positive recall | Precision | F1 | Positive p50 / p95 | Positive RTF | False activations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| audio.cpp + Moonshine Tiny Q8_0 | CPU | 113 ms | 59/60 (98.3%) | 100% | 99.2% | 66 / 74 ms | 0.0304 | 0 / 0.250689 h |
| OpenVINO Whisper Base.en INT8 | CPU | 367 ms | 51/60 (85.0%) | 100% | 91.9% | 201 / 213 ms | 0.0916 | 0 / 0.250689 h |
| OpenVINO Whisper Base.en INT8 | iGPU | 225 ms | 50/60 (83.3%) | 100% | 90.9% | 63 / 79 ms | 0.0317 | 0 / 0.250689 h |
| OpenVINO Whisper Base.en INT8 | NPU | 277 ms | 51/60 (85.0%) | 100% | 91.9% | 69 / 90 ms | 0.0330 | 0 / 0.250689 h |

OpenVINO CPU and NPU produced identical prediction fingerprints. The iGPU
differed by one positive clip, a 1.7 percentage-point recall change, so device
placement did not create a large accuracy loss within the OpenVINO profile.
The OpenVINO profile itself trails the default Moonshine profile on this set;
most misses decoded `Hey Jarvis` as `Hate Jarvis`. The faster iGPU and NPU
numbers therefore compare execution devices for the same OpenVINO model, while
the default CPU row uses a different verifier and is not a pure runtime
comparison.

Setup compiled and verified two iGPU cache artifacts (82,997,248 bytes) and
three NPU cache artifacts (357,552,967 bytes) before activation. Placement
reports disabled fallback and confirmed that every OpenVINO graph initialized
on the requested device.

## NVIDIA GB10 CUDA result

The same source commit built and passed all tests natively on aarch64, then
loaded a complete CUDA 13.0 audio.cpp provider on an NVIDIA GB10 with driver
580.173.02. Setup reported `cuda:0`, ready device and model proof, and no
fallback. A file-only test and every measured benchmark iteration detected
`hey-jarvis`; no action was executed. `nvidia-smi pmon` observed the exact
Omawake worker PID using 232–311 MiB and 1–4% SM.

| Cold load | Cold inference | Hot p50 | Hot p95 | Hot p50 RTF |
|---:|---:|---:|---:|---:|
| 1,470 ms | 1,447 ms | 791 ms | 800 ms | 0.399 |

SGLang was concurrently using about 92–95% of the GB10 SM capacity, so this is
functionality and placement evidence rather than an uncontended performance
ceiling. The complete evidence archive is retained locally as
`/tmp/omawake-rc3-cuda-evidence.tar.gz`, SHA-256
`0da501a5b18215f04f36994ebced2c6dcdd662043711253a0f3a5c9ef25bdd82`.

## Reproducibility and limits

The local positive and negative manifests have SHA-256 values
`be1b76f89051fcac96c6f1ad7ef4a1cf23707e66da919227c660b5260537ea42`
and `ec09b5c216d3b1331a56878c1f02f391c271287f4083334f3e6b7e5122d3593a`.
Exact report hashes and prediction fingerprints are in `metrics.json`.

These sets establish basic recall, clean-speech rejection, timing, and device
placement for the release candidate. They do not replace the longer
speech-negative, adverse-audio, far-field, or idle-power gates described in
the qualification plan.
