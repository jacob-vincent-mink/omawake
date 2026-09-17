# Priorities and scope decisions

This is the proposed disposition of all next steps identified in the model
assessment and shared catalog contract, including related enrollment branches.
It does not claim to inventory every possible future feature. Open issues and
reviews were checked on 2026-09-15; neither repository had additional open
issues or review comments to incorporate. These are recommendations for review,
not decisions already accepted or declined by the maintainer.

This document supersedes the earlier assessment's ordering. Feature research
and model availability remain in [the assessment](MODEL-SUPPORT-PLAN.md);
shared mechanics remain in [the catalog contract](MODEL-CATALOG-CONTRACT.md).
Implementation is not authorized merely by merging these planning documents.

## Product boundary

Omawake detects configured wake phrases and dispatches configured actions. Its
primary measures are missed wakes, false activations, response time, reliability
and continuous resource use. ASR is an internal means to verify phrases.
General transcription, speaker identity and audio generation are separate
products, even when the same runtime supports them.

Every supported backend still needs a good, complete, downloadable default.
Scope reduction does not weaken that requirement. Unqualified hardware is
labelled experimental until tested; it cannot silently fall back and claim
support. Keep the existing working defaults while closing coverage gaps.

Priority meanings: **P0** completes the support contract; **P1** improves the
core product after P0; **P2** is a bounded experiment with a promotion gate.
**Defer** has no scheduled implementation until its stated trigger occurs.
**Decline** recommends excluding the capability from this repository's product.
Order within each table is execution order; effort is relative S/M/L, not a
calendar estimate. Each row can be accepted or declined independently except
where dependencies are stated.

## Ranked implementation backlog

| Rank / ID | Decision | Next step and owner | Dependency / effort | Completion or stop condition |
|---|---|---|---|---|
| 1 / W01 | P0: pursue | Record release model/provider/device baselines and define acceptable wake regressions in Omawake | None / M | Versioned real-speech and negative-audio evaluation inputs, thresholds and provenance; document missing device evidence |
| 2 / W02 | P0: pursue | Complete the whisper.cpp CPU default from canonical Whisper + GGML Silero downloads | W01 / M | Exact pins, hashes, licenses, all required assets, explicit backend selection and clean-home file-only detection pass |
| 3 / W03 | P0: pursue | Central default resolver, actual family/profile/language metadata, compatibility and default-coverage assertions | W02 can develop alongside schema / M | All three backend kinds resolve correctly; current IDs/config survive; selected backend is not overwritten by runtime inference |
| 4 / W04 | P0: pursue | Curated setup list: friendly name, compatible choices, active/install state, download size and details | W03 / M | CLI and wizard agree, no unsupported family is offered as runnable; narrow-terminal scrolling works |
| 5 / W05 | P0: pursue | Preserve atomic download/activation; add missing download locking, cancellation and disk preflight | W03 / M | Corruption, insufficient space, competing installs and failed probes leave the active model/config intact; implement only protections not already present |
| 6 / W06 | P0: pursue | Qualify defaults on advertised CPU/CUDA/Vulkan/OpenVINO devices and obtain HIP evidence | W01–W05 / L, hardware dependent | Complete per-device evidence; untested combinations explicitly experimental; no automatic GPU recommendation without benefit |
| 7 / W07 | P1: pursue | Reviewed catalog import/check tooling, offline local import and readable diagnostics | P0 / M | Immutable file/dependency manifests and reproducible import; broken default URLs detected without changing user pins |
| 8 / W08 | P1: pursue | Benchmark Moonshine Small and Medium against Tiny | W01, W03 / M | Promote at most the variants with meaningful wake benefit; leave both out if neither earns its footprint |
| 9 / W09 | P1: pursue | First non-English wake profile: multilingual OpenVINO Whisper Base, Tiny as comparison | W01, W03 / L | Explicit language reaches inference and status; phrase matching works on chosen language corpus; no blanket multilingual claim |
| 10 / W10 | P1: pursue | Harden playback pause/resume ownership with Omaspeak and the consuming harness | P0 / M, joint integration | TTS cannot retrigger wakes; nested pauses, cancellation and crashes release ownership correctly; no echo-canceller project |
| 11 / W11 | P1: pursue when needed | Add model search, language filters and grouped precision variants | W08/W09 grow the list / S–M | Existing short list remains usable; add controls only for real catalog choices |
| 12 / W12 | P2: bounded experiment | Measure streaming phrase verification using an already supported family | W01, W10 / M | Keep only if endpoint latency improves without extra false triggers, duplicate actions or unacceptable power; no speculative ASR family expansion |
| 13 / W13 | P2: bounded experiment | Assess existing enrollment/trainable-head branches against the default pipeline | W01 / L | Compare on independent real-speaker holdouts; catalog changes integrate only if a head demonstrably improves wake detection |

W06 is a release gate, not a promise that access to every device is available.
W08/W09 may collect evidence before it finishes, but must not promote an
unqualified model/device pair. The first release stops after P0; P1 additions
can ship independently once qualified. P2 is not required for either release.

## Deferred and external work

| ID | Disposition | Item | Trigger / destination |
|---|---|---|---|
| W14 | Defer | Whisper Small / Small.en models | Real phrase failures persist after W08/W09, and smaller supported profiles cannot meet the target |
| W15 | Defer | Qwen3-ASR and additional ASR families | Demonstrated language/accuracy gap on hardware the existing families cannot serve; first run one bounded offline comparison |
| W16 | Defer | Additional whisper.cpp acceleration | A concrete hardware need not served by current backends; requires a new device qualification and maintained default |
| W17 | External developer tooling | Omaspeak-generated positive/near-match corpora | Maintainer evaluation scripts may call Omaspeak; do not add generation or voice-design UI/models to Omawake |
| W18 | External developer tooling | Forced alignment and denoise for dataset preparation | Existing external tools may prepare annotated fixtures; Omawake consumes standard audio/labels and records provenance |
| W19 | Defer | Live denoise/enhancement | Reproducible noisy-room failure and a bounded test showing improvement over input-device/PipeWire remedies; account for power/latency |
| W20 | Defer | Signed remote catalog refresh | Release-bundled catalogs measurably cannot keep maintained defaults available; first prefer a normal catalog patch release |
| W21 | Conditional maintenance | Project-hosted model conversions/mirrors | Canonical compatible artifacts fail availability/quality/redistribution requirements; keep recipe/manifest in Git and large weights in immutable artifact hosting |
| W22 | P1 maintenance | Transfer resume and verified-file reuse | Add around actual large downloads; preserve immutable pins and complete-file hashing, with interruption tests |

External tooling is not a hidden product backlog: do not ship its runtime,
model picker, or UI inside Omawake. W17/W18 are optional maintainer aids, and
are not prerequisites for a usable default or a user enrollment flow.

## Recommended explicit declines

| ID | Decline in Omawake | Reason / boundary |
|---|---|---|
| W23 | General dictation, transcription, captions and meeting recording | Other speech/transcription tools own these workflows; expose wake events/actions |
| W24 | Speaker identification, authentication and diarization features | Wake phrase matching does not require an identity system; exclude biometric enrollment and speaker-routing UI |
| W25 | Cloning, voice design, voice conversion or speech generation | Omaspeak or external authoring tools own speech output; no generative models in the wake service |
| W26 | Media cleanup, source separation, music/SFX/video generation | No direct contribution to wake detection |
| W27 | Conversation state, LLM reasoning and dialogue orchestration | Consuming applications own behavior after the wake event |
| W28 | Full-duplex conversation / built-in echo cancellation project | Pause ownership is in scope; true barge-in requires an audio/harness integration project and is not part of model support |
| W29 | Unfiltered Hugging Face browser, generic audio.cpp task runner or all-family provider bundle | Increases download/support surface without improving wake reliability |
| W30 | Separate catalog service, framework or shared runtime daemon | Embedded reviewed data and existing workers are sufficient for two small applications |

These declines do not prevent maintainers using external tools to investigate
wake quality, or a future explicitly scoped proposal supported by new evidence.
They remove the features from the default implementation queue now.

## Small review and delivery units

1. **Coverage:** W01–W03, including pinned whisper.cpp assets and migration tests.
2. **Setup:** W04–W05, with shared catalog contract conformance and rollback tests.
3. **Release evidence:** W06; publish supported versus experimental combinations.
4. **Core improvements:** separate PRs for tooling, measured model alternatives,
   language support and pause ownership (W07–W11, W22).
5. **Experiments:** W12/W13 require an evidence report and a promote-or-stop
   decision before product integration; deferred/declined rows create no work
   automatically.

Validate documentation links and consistency for this planning update.
Implementation PRs must run the acceptance checks attached to their rows; this
priority document does not claim new model or hardware validation.
