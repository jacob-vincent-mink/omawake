# Trained encoder CPU/iGPU/NPU parity check

Date: 2026-09-16. Dell XPS with Intel integrated GPU and NPU, using the installed
OpenVINO 2026.3.1 stack and the feature-branch release build.

This is a small functional check, not general accuracy or ambient false-trigger
qualification. The experimental accelerator label remains appropriate.

## Method

The existing CPU-trained head and threshold were held fixed. Four previously
held-out human recordings (two positives, two negatives) were identified by the
active head's validation fingerprints. No retraining or threshold tuning was
performed. Rechecking these clips measures device parity, not new independent
validation evidence.

The application's production embedding worker ran its actual VAD, feature
extraction, OpenVINO encoder, and pooling path on each device. Its reported
`EXECUTION_DEVICES` values were CPU, NPU, and GPU.0. Returned embeddings were
scored with the saved head's normalized logistic classifier. The public
`omawake benchmark` command then independently verified the same decisions using
the application's classifier on all three devices (two passes per clip).

Only WAV files were used. No microphone, playback, actions, service changes, or
active configuration edits occurred. Worker caches and test configurations were
isolated in private temporary directories. Silero VAD and the tiny classifier
remain on CPU; the Whisper encoder executes on the selected device. Whisper's
text decoder is not used.

## Results

| Encoder device | Fresh application-cache startup | Warm median encoder time | Largest score difference from CPU | Correct decisions |
| --- | ---: | ---: | ---: | ---: |
| CPU (2 threads) | 495 ms | 283.6 ms | — | 4/4 |
| NPU | 5,709 ms | 35.8 ms | 0.003033 | 4/4 |
| Intel iGPU (`GPU.0`) | 1,052 ms | 14.9 ms | 0.000868 | 4/4 |

Startup includes compilation/loading and the worker's initialization inference;
it is not a clean measure of compiler time alone. Driver-level caches were not
purged. Warm encoder timing is the median across the four clips on the second
pass, excluding VAD, IPC, and classifier overhead. This is one short run on a
working desktop, not a statistically robust performance benchmark.

The fixed threshold was 0.638432. Minimum score-to-threshold distances were
0.064521 on CPU, 0.066054 on NPU, and 0.065389 on iGPU. Both repeated passes
produced the same scores on each device. Maximum absolute embedding component
differences from CPU were 0.002723 on NPU and 0.002149 on iGPU. The public command
reported `trained-whisper-encoder`, verified placement, and no fallback for all
three devices; each pass accepted both positives and rejected both negatives.

Local raw evidence: `/tmp/omawake-encoder-devices-1ia4zlg0/`. Recordings, personal
configuration, and trained weights are intentionally not included in the repo.

## Reproduction and limits

Use a temporary config containing only the trained word and its unchanged head.
Copy its model/backend profile, set an absolute model directory and the requested
OpenVINO device (`cpu`, `gpu`, or `npu`), and isolate XDG cache/runtime directories.
Run `omawake --config TEMP_CONFIG benchmark POSITIVE_1.wav POSITIVE_2.wav
NEGATIVE_1.wav NEGATIVE_2.wav --warmup 0 --iterations 2` on each device. Inspect
placement evidence, fallback status, and per-clip decisions. Never use `--execute`.

Before a general recommendation, measure more speakers, confusable negatives,
noise/distance conditions, sustained ambient speech, energy usage, and repeated
startup/warm timing. The existing CPU-trained head worked unchanged here; this
does not establish that every model export, driver, NPU, or trained head will
preserve its decision margins.

## Separate training and deployment follow-up

The application now supports `word train WORD --dataset MANIFEST
--training-device cpu --run-device npu --json` without requiring an NPU-named
engine profile. The reverse direction (`npu` training, `cpu` deployment) was also
run using the same saved recordings, without `--apply`. Source model/runtime
assets came from an isolated CPU profile in both cases.

| Training features | Calibration/validation | Misses | False activations | Minimum held-out margin |
| --- | --- | ---: | ---: | ---: |
| CPU (verified) | NPU (verified) | 0/2 | 0/2 | 0.063816 |
| NPU (verified) | CPU (verified) | 0/2 | 0/2 | 0.065164 |

Both commands reported `applied: false`, no actions, and separate actual training
and deployment execution devices. No active user config, head, or source dataset
was modified. Local evidence: `/tmp/omawake-cross-device-training-9_2rf1i4/`.
These are repeated checks of the saved evaluation clips, not new independent
accuracy evidence. Automated tests additionally cover destination load failures,
encoder-contract mismatch, and device-dependent features that cannot satisfy
validation; none may activate or replace the prior detector.

## Opt-in history and correction workflow (2026-09-16)

A temporary file-fed live stream exercised the real trained encoder on NPU,
using an existing negative **training** clip. The test deliberately lowered the
threshold to 0.01 to force a capture; this is not an accuracy measurement or a
recommended operating threshold. The classifier scored the clip 0.166334 and
saved one private PCM WAV event with NPU placement, score, threshold, head, and
encoder metadata. Processing the same input through the offline file-test path
produced no history. The captured event was labeled false-positive.

A real `word train --feedback --training-device cpu --run-device npu` candidate
then included the correction in training, extracted training features on CPU,
and calibrated/validated on NPU. The candidate passed the existing two-positive,
two-negative held-out split: zero misses, zero false activations, minimum margin
0.064015, calibrated threshold 0.638166. It was **not applied**. The reused split
and forced trigger demonstrate plumbing and deployment validation, not improved
ambient accuracy. No microphone, playback, or wake-word action was used, and
all new config, history, and cache files were isolated under `/tmp`.

The corresponding automated suite passed 316 tests with 90.38% line coverage.
It covers threshold override/reset, disabled-by-default and bounded private
history, live/offline separation, replay via a fake player, labels, training-only
feedback, deduplication and relabeling, evaluation-overlap rejection, guided
feedback collection, and deployment validation with an explicit threshold.
Formatting and Clippy with warnings denied also passed.
