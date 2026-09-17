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

## Guided dataset collection follow-up

The onboarding flow now creates the training dataset from microphone prompts:
five initial positive examples are reused, followed by five more positives and
ten other-speech examples, split 6/2/2 per class. Tests inject audio at the
capture boundary and exercise the complete collection, training, final review,
activation, and optional retention transaction. They also cover cancellation
before recording, during negative recording, and at final Apply; encoder failure;
unseparable examples; and concurrent config edits. Automated tests do not open
the microphone. The PTY regression verifies the recognition-method selection
alongside transcript review and Apply/cancel.

At this follow-up, all 291 tests passed with 90.18% line coverage; formatting and
Clippy passed. The existing real CPU retained-dataset preview was rerun using the
released audio.cpp library from the local release-verification tree (the former
`/usr/lib/omawake/libaudiocpp.so` installation was no longer present). It again
reported CPU execution, zero misses and zero false activations on the four
held-out clips, with minimum margin 0.35148865. This remains a small file-based
proof, not qualification of the twenty-recording enrollment target or a real
microphone accuracy measurement.

## Segmentation failure and retention regression

A real CPU check concatenated two spoken phrase clips with one second of silence
and substituted that WAV into a twelve-clip dataset. The VAD reported two speech
segments. Training rejected the sample, left the isolated config unchanged, and
preserved all twelve labeled WAVs and the manifest because retention was chosen.
Evidence from the local run: `/tmp/omawake-segmentation-proof-cehvafik`.

Guided collection now applies the same segmentation check to each new clip and
to reused transcript examples, retrying only the offending clip. Tests cover a
split reused clip followed by silence and a valid retry, plus zero/two-segment
failures and model-load failure after the complete dataset has been saved.
Explicit retention also survives final activation cancellation. The regression
suite passed 295 tests with 90.34% line coverage; Clippy and the release build
passed. Tests use injected microphone audio; the real check uses file audio.

## Optional Omaspeak assistance

Real Omaspeak file-output calls were checked with M1 at speed 0.9 and F1 at 1.1,
using `say --no-play --out ... --voice ... --speed ... -- TEXT`. Both produced
44.1 kHz WAVs without playback. The F1 check used an isolated runtime directory
and CPU config, producing 1.271 seconds of audio. Evidence is in
`/tmp/omaspeak-assisted-native-n0yq7543`. These checks establish interface
compatibility, not correct pronunciation or improved wake-word accuracy;
pronunciation approval remains the user's explicit playback/review step.

Tests cover opt-in installation and decline, checksum rejection, keeping native
libraries and licenses with the executable, silent synthesis, voice approval,
synthetic provenance, rejecting synthetic calibration/validation, failure
fallback with saved checkpoints, and resuming with fresh human evaluation data.
No test installs into the real home directory, records the microphone, or plays
sound. PTY checks cover resume cancellation and incompatible-encoder rejection
before recording.

The completed assisted-enrollment suite passed 304 tests (281 library, 23 CLI)
with 90.07% line coverage against the unchanged 90.01% gate. Formatting, Clippy
with warnings denied, and the release build passed.
