# Model support assessment and plan

Assessed 2026-09-15 against omawake `af25056`, omaspeak `86064f6`, and the
packaged audio.cpp revision `e9ff20042ec85af960a720368c6927cda19ad65f`.
This is a planning change; no models, runtime providers, or user configuration
were installed or changed. New candidates below have not been run in Omawake.

## Review order and scope

Read [PRIORITIES.md](PRIORITIES.md) first for the ranked backlog, dependencies,
release cuts, deferred experiments and recommended explicit declines. It
supersedes the ordering below. This assessment preserves the technical research;
a capability appearing here is not a commitment to implement it.

## Recommendation

Keep Moonshine Streaming Tiny as the English default, add downloadable
whisper.cpp assets immediately, then offer a small curated choice of larger
Moonshine and multilingual Whisper models. Make every supported backend resolve
to a complete, downloadable, tested profile. Rank alternatives by wake-word
performance, memory and power, rather than general transcription leaderboards.

Use the companion [catalog and setup contract](MODEL-CATALOG-CONTRACT.md) for
the selection UI, provenance, downloads and backend compatibility. The paired
Omaspeak worktree contains the TTS and voice-capability plan.

## Findings in the current code

| Area | Finding | Implication |
|---|---|---|
| `src/catalog.rs` | Two profiles: Moonshine Tiny + Silero and OpenVINO Whisper Base.en + Silero; three advertised backends | `whispercpp` has no downloadable/default profile |
| `scripts/build-default-audiocpp-provider.sh` | Pinned provider builds `AUDIOCPP_MODELS=moonshine_asr` | Adding a different ASR family to the catalog alone cannot enable it |
| `ModelSpec::activate` | Always writes `audiocpp.asr_family=moonshine_asr` for audio.cpp | Store the actual provider family on each profile |
| `src/app.rs::choose_model_with` | Every catalog entry is selectable; active/install state, license and size already shown | Add compatibility filtering, readable names, search and bounded scrolling |
| `runtime_selection_candidate` | Maps runtimes to two hardcoded IDs and chooses backend from runtime | Introduce default resolution by backend, runtime, device and language; preserve explicit whisper.cpp selection |
| `src/engine/openvino_genai/mod.rs` | Worker reports a fixed `WHISPER_BASE_EN_PROFILE` | Model identity/language reporting must come from the selected profile |
| `src/engine/whisper/adapter.c` | Forces English and CPU; uses whisper.cpp's own VAD API | Multilingual config needs to reach the C adapter; use GGML VAD weights for this backend |
| `src/phrase.rs` | Unicode NFKC/case normalization, then alphanumeric token runs | Test languages without spaces and combining marks before claiming multilingual wake matching |
| `src/setup/model.rs` | Pinned files, hashes, sizes, staging, verification and rollback already exist | Extend this installer; preserve its atomic behavior |

The current implementation detects phrases from speech transcripts; these ASR
models are phrase verifiers, not interchangeable trained keyword-spotting heads.
Enrollment/head work exists in other branches and is not part of these main
snapshots. Evaluate that work under W13 in PRIORITIES.md before planning integration; do
not assume it has shipped or automatically belongs in the next release.

## Default coverage

| Backend / runtime | Default profile | Canonical artifact source | Decision |
|---|---|---|---|
| audio.cpp / CPU | Moonshine Streaming Tiny Q8_0 + Silero 6.2.1 | [audio.cpp conversion](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Moonshine-Streaming-GGUF), [Silero upstream](https://github.com/snakers4/silero-vad) | Keep existing pinned profile |
| audio.cpp / CUDA | Same complete profile | Same | Keep; validate against the selected CUDA provider |
| audio.cpp / Vulkan | Same complete profile | Same | Keep; GPU availability alone is not a performance recommendation |
| audio.cpp / HIP | Same complete profile | Same | Intended default; qualify on actual AMD hardware before calling HIP recommended |
| OpenVINO GenAI / CPU | Whisper Base.en INT8 + Silero 6.2.1 | [Intel conversion](https://huggingface.co/OpenVINO/whisper-base.en-int8-ov), Silero upstream | Keep |
| OpenVINO GenAI / GPU | Same complete profile | Same | Keep with device-specific proof |
| OpenVINO GenAI / NPU | Same complete profile | Same | Keep with compile/cache/placement proof |
| whisper.cpp / CPU | **Add** Whisper Base.en GGML + Silero v6.2.0 GGML | [whisper.cpp maintainer models](https://huggingface.co/ggerganov/whisper.cpp), [ggml-org VAD](https://huggingface.co/ggml-org/whisper-vad/tree/main) | First missing default to implement and qualify |

Use `ggml-base.en.bin` initially for whisper.cpp and pin an exact revision and
SHA-256 for it and `ggml-silero-v6.2.0.bin`. If that VAD version fails the pinned
whisper.cpp 1.9.3 adapter, qualify the upstream v5.1.2 artifact instead and make
the version explicit in the profile. Do not substitute audio.cpp's safetensors
VAD. Additional whisper.cpp acceleration is outside the current adapter contract.

Existing audio.cpp CPU and Vulkan evidence is in
[the Vulkan report](../benchmarks/results/2026-09-15-vulkan/RESULTS.md): Tiny had
59/60 synthetic positive recall on both, with warm inference p50 72 ms on CPU
versus 119 ms on the Intel iGPU. That small sample supports retaining Tiny and
avoiding automatic GPU preference; it does not establish production recall or
power usage. See also [the rc.3 evidence](../benchmarks/results/2026-09-15-rc3/RESULTS.md).

## Curated expansion

| Priority | Candidate | Source / approximate download | Purpose and required work |
|---|---|---|---|
| P0 | Whisper Base.en GGML + GGML Silero | Maintainer repositories above; calculate full pinned size when importing | Close the missing-backend default; add activation/runtime selection support |
| P1 | Moonshine Streaming Small Q8_0 | [audio.cpp package directory](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Moonshine-Streaming-GGUF), about 301 MB before VAD | Same-family accuracy candidate for difficult phrases; compare with Tiny |
| P1 | Moonshine Streaming Medium Q8_0 | Same directory, about 316 MB before VAD | Benchmark alongside Small; expose whichever earns a meaningful accuracy/latency advantage |
| P1 | Whisper Base multilingual INT8 IR | [OpenVINO/whisper-base-int8-ov](https://huggingface.co/OpenVINO/whisper-base-int8-ov) | First multilingual OpenVINO candidate; wire language config and profile metadata |
| P1 | Whisper Tiny multilingual INT8 IR | [OpenVINO/whisper-tiny-int8-ov](https://huggingface.co/OpenVINO/whisper-tiny-int8-ov), repository about 49 MB | Lower-memory comparison; qualify exact language and device combinations |
| P2 | Whisper Small multilingual / Small.en INT8 IR | [multilingual](https://huggingface.co/OpenVINO/whisper-small-int8-ov), [English](https://huggingface.co/OpenVINO/whisper-small.en-int8-ov/tree/main), repositories about 257 MB | Accuracy-oriented option only if wake evaluation justifies cost |
| P2 | Qwen3-ASR 0.6B audio.cpp GGUF | [audio.cpp Qwen3 docs](https://github.com/0xShug0/audio.cpp/blob/e9ff20042ec85af960a720368c6927cda19ad65f/docs/models/qwen3.md) and upstream model package spec | Multilingual audio.cpp option; needs family build, language handling, complete assets and short-utterance benchmarking |

Sizes from public directory listings are approximate decimal MB, not measured
memory requirements. Tiny's current complete pinned download is about 61.6 MB.
New asset hashes and exact dependency totals are implementation work, not
verified download manifests in this plan. Keep current revision pins unchanged.

For a non-English setup, resolve a language-compatible default after its
qualification. Never label an English-only model recommended for that language.
Before qualification, explain the limitation and offer a supported backend
switch rather than silently using English or promising every language in a
model card.

## Capabilities that fit Omawake

| Capability | Natural use | Priority and boundary |
|---|---|---|
| Multilingual ASR + explicit language hints | Wake phrases in the user's language, accents and custom names | High; fixed language for short phrases is easier to validate than auto-detection |
| Streaming ASR / VAD | Earlier phrase decisions and interruption of spoken assistant output | High-value experiment; require finalized phrase evidence, deduplication and measured end-to-end latency |
| Cloning, design and style via Omaspeak | Generate varied wake-phrase samples and confusing near-matches for evaluation or future head training | High-value offline tooling; save generator/model/seed provenance and retain independent real-speaker holdouts |
| Forced alignment | Find exact phrase boundaries in enrollment/evaluation recordings | Medium; optional offline preparation, not every wake event; alignment assumes a transcript and is not proof the phrase was spoken |
| Denoise/enhancement | Improve noisy enrollment samples or evaluate a noisy-room frontend | Medium experiment; measure missed wakes and false wakes before enabling on live audio |
| Diarization / speaker information | Annotate multi-speaker evaluation recordings; investigate personalized wake behavior | Later; diarization labels speakers, it does not authenticate a user or prove a replay is live |

Streaming support must be checked in the pinned C ABI and the actual model
session; an upstream streaming tag may describe buffered chunks rather than a
low-latency online recognizer. Prefer keeping heavy generation, alignment and
speaker models out of the always-running wake worker.

For voice interaction, coordinate Omaspeak playback and Omawake pause/resume
ownership so generated speech cannot wake the assistant repeatedly. True
barge-in requires playback-reference echo handling and an evaluation set with
simultaneous speaker output; VAD or diarization alone is insufficient.

## Technical implementation notes (subject to priorities)

1. **Complete defaults and catalog metadata.** Implement the common contract,
   whisper.cpp profile, explicit backend choice, actual ASR family and per-profile
   language identity. Retain old IDs and config values. CI checks every public
   backend/runtime has a complete default declaration and validates installed
   defaults in isolated homes without microphone capture or executing actions.
2. **Make setup scale.** Replace the unfiltered model menu with the contract's
   list/search/details UI. CLI and interactive setup use the same resolver.
   Test unavailable families, language mismatch, cancelled download, corrupt
   artifacts and failed provider probes; active config must remain unchanged.
3. **Add same-family choices.** Import pinned Moonshine Small/Medium and Whisper
   Tiny/Base multilingual artifacts; fix English assumptions in workers and
   phrase handling. Use the existing `evaluation` machinery for comparisons.
4. **Qualify recommendations.** Compare real voices, accents, names, near-match
   phrases, long unrelated speech, silence, TV/music and playback echo. Report
   false activations/hour with test duration and uncertainty, recall, onset and
   endpoint latency p50/p95, warm/cold load, RSS, device memory and idle/active
   power. Predeclare acceptable regression budgets against current defaults.
   Short synthetic-only tests cannot promote a new default.
5. **Prototype Qwen and optional workflows.** Add a provider with the chosen ASR
   family, first prove offline C ABI execution, then assess streaming and
   Omaspeak-generated evaluation datasets separately.

No application tests were run for this documentation-only assessment. Existing
benchmark results are cited as prior evidence, not rerun results. Model-card
availability was checked with web browsing; no candidate weights were downloaded.
