# Enrollment CPU functional proof

Date: 2026-09-15. Feature branch: `feature/wake-word-onboarding`.
This is a functional prototype proof, not a general accuracy qualification.
The release branches and real user configuration were not modified.

## What ran

All audio was read from files. No microphone capture, speaker playback, or
wake-word actions ran. Tests used independent XDG config/data/cache/state/runtime
directories and the installed OpenVINO CPU stack. The application was a debug
build. Intel GPU/NPU and NVIDIA execution for the **new head pipeline** are not
claimed by this proof.

- OpenVINO GenAI Whisper transcribed one synthetic “Hey Jarvis” clip as
  “Hate Jarvis.” `word onboard` proposed that exact spelling. Preview left config
  unchanged; an explicit `--accept-alias` and `--apply` stored it.
- The direct OpenVINO Whisper encoder produced finite 512-dimensional
  embeddings. `EXECUTION_DEVICES` returned `CPU`.
- Training used 4 recordings, calibration another 4, and validation another 4:
  each split had two positives and two negatives. Positive recordings were
  pre-existing Piper-generated “Hey Jarvis” examples; negatives were separate
  two-second LibriSpeech excerpts. Normalized audio fingerprints were disjoint.
- Local held-out results: 2/2 positives detected, 2/2 negatives rejected, minimum
  score-to-threshold margin 0.35149. The entire 12-clip training preview took
  about 5.9 seconds, including startup, in this debug build.
- Applying the head retained the old word/command/aliases and pinned its engine
  profile. A transcript-only “Computer” word remained on the default profile.
  Both groups then consumed file audio concurrently: the positive produced only
  the trained word ID; the tested negative produced no detections.
- An additional US-voice positive outside the training/calibration/validation
  recordings was also detected by the trained head.
- `word train --reuse-recordings --json` selected the retained labeled session
  and reproduced local validation without changing the active config.

The head's encoder contract was
`whisper-base-en-v1-dd9057704283405868faecec3763f8bb0ce7533bae7e81c3118fa047ddcc04c1`.
It includes actual encoder XML/BIN and VAD bytes plus preprocessing/segmentation
versions. Matching dimensions alone are not sufficient for compatibility.

## Independent preprocessing check

The committed chirp fixture was compared over all 240,000 Whisper input feature
values against an independent NumPy/OpenAI-filter reference. Maximum absolute
error was approximately 0.00001055, below the 0.0001 test tolerance. Mandatory
CI tests include the reference values; they do not silently skip when a local
research directory is absent. Pooling tests separately verify that the padded
encoder tail does not enter a short-clip embedding.

## Behavioral regression coverage

The automated tests exercise actual terminal arrow-key approval and cancellation,
file-only preview/apply, named profile selection, failed recording retries,
retention/removal, contaminated training splits, concurrent config edits,
immutable head artifacts, incompatible encoder rejection, shared-head scoring,
concurrent engine passes, stream reuse, controlled worker startup/IPC failures,
and cancellation. Protocol tests use normal error responses and exits, not
intentional native crashes. The repository's >90% line-coverage gate is unchanged.

## Limits and next qualification

Twelve short clips are enough to demonstrate the lifecycle, not to estimate a
background false-activation rate. Synthetic positives and audiobook negatives
also differ in recording conditions and may allow shortcuts unrelated to the
wake phrase. Qualification needs independently collected positives, close
phonetic negatives, unseen speakers, noise/distance variation, and long ambient
speech before claiming production accuracy. Do not tune against the held-out set
and reuse it as an independent score.

Accelerator qualification must independently verify preprocessing/embedding
parity, device placement, head accuracy, cache preparation, and cold/hot timing.
The existing transcription-provider accelerator results do not transfer
implicitly to this encoder/head pipeline. This work remains on a feature branch
and does not block a release of the existing transcription path.
