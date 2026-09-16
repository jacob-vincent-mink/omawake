# Model support completion goal

Tracking date: 2026-09-16. This implements the user-approved priorities and
subsequent completion sequence. The historical assessment is not a claim that
all features or hardware are qualified.

## Goal and completion rule

Finish release coverage/setup and its outstanding acceptance gates, then deliver
core lifecycle reliability, gated streaming, measured preset/model alternatives,
pinned catalog maintenance and a first qualified non-English profile per app.
Every delivery has tests or a reproducible evidence report and a reviewed PR.
Do not mark quality gates passed from synthesis validity or ASR transcripts alone.
Do not promote hardware from runtime compatibility or one successful inference.

HIP remains pending and unqualified because hardware is unavailable. It is not
a release blocker for explicitly qualified combinations. Optional experiments
(cloning, delivery instructions, enrollment, additional native conversions) are
outside this goal; deferred and declined scope stays outside the active queue.

## Ordered joint delivery sequence

1. Reconcile acceptance status, evidence and supported versus experimental
   device claims. Audit fresh setup, activation and cache-failure rollback.
2. Finish available release checks. Obtain human listening ratings, real-room
   wake recordings and controlled resource/power measurements before closing
   their gates. Record unavailable evidence explicitly; continue independent work.
3. W10/S06: bounded requests, cancellation, request identity and owned playback
   pauses. Nested owners, client disconnects and crashes must not leak state,
   resume another owner's pause or let TTS retrigger wake actions.
4. S07: inspect the pinned API, then test incremental Supertonic output. Promote
   only with improved first-audio latency, bounded buffering, prompt cancellation,
   no seams/underruns and correct file export. Record a stop decision otherwise.
5. S08/S09: preserve legacy Supertonic voice identity while introducing only the
   named-voice/adapter fields needed by Kokoro. Pin all frontend/voice resources,
   provide preview and fresh installation proof, and compare speech quality.
6. W08: compare Moonshine Small/Medium with Tiny on identical corpora and declare
   quality/resource tradeoffs. Keep only variants that justify their footprint.
7. W07/S10: reproducible pinned catalog checks/imports, offline import and asset
   availability checks that never silently replace user pins.
8. W09/S10: one explicitly tested non-English profile per app, with language
   reaching inference and status, complete resources and language-specific evidence.
   Use benchmark and language requirements to choose the candidate before download.
9. W11/W22/S11: filters and transfer resume only when actual catalog/download
   scale warrants them. Record the decision; verified reuse must preserve hashes.

Each item ends with an implementation and acceptance result, or its specified
promote/defer/stop decision. Human-dependent gates remain open until evidence
arrives; missing evidence is not a pass. Core work can proceed while collecting
those inputs, without declaring the first release fully qualified.

## Current external inputs

- Human ratings of the existing English listening packs (intelligibility,
  pronunciation, missing words, naturalness and voice consistency).
- Independent real-room wake positives, human near-matches, distance and acoustic
  playback/TV/music recordings with labels and provenance.
- A controlled measurement window and readable energy counters for attributed
  resource/power comparisons. Shared-host GPU telemetry is not application power.
- A selected language and suitable independent evaluation/listening data before
  promoting the first non-English profile. No blanket multilingual claim.

## Omawake acceptance status

| IDs | Status | Evidence / remaining acceptance |
|---|---|---|
| W01 | Partial | Versioned real speech, generated adverse/near-match and 10.739-hour negative evidence; real-room, controlled regression and continuous idle power remain open |
| W02/W03 | Implemented | Complete pinned whisper.cpp default and backend resolver; migration/default tests and file-only proof |
| W04 | Implemented | Friendly compatible rows, disabled explanations, compatible preselection, narrow-terminal navigation and family discovery; CLI catalog stays stable |
| W05 | Implemented; final audit pending | Locking, cancellation, disk preflight, hash verification and rollback tests; map cache/probe failures to acceptance before closure |
| W06 | Partial | CPU, Vulkan, Intel GPU/NPU and GB10 CUDA evidence exists; evidence depth differs by pair, HIP pending, no blanket hardware promotion |
| W07–W11/W22 | Next core work | Joint lifecycle first, then measured variants, maintenance and qualified language; filters/resume conditional on scale |
| W08 | Done — both larger variants deferred (stop-promote, 2026-09-16) | [Moonshine Small/Medium vs Tiny](../benchmarks/results/2026-09-16-moonshine-variants/RESULTS.md) on identical corpora: identical false activations (10 per variant over 10.739 h, 0/h for `light-up`), identical clean behavior, only degraded-`forever` recall differs; 3–5× CPU and 2.5–3.5× memory cost → neither earns its footprint. Pinned opt-in profiles retained for the W14 trigger |
| W11/W22 | Deferred — scale not warranted (recorded 2026-09-16) | Pinned omawake catalog remains 3 curated model bundles and the largest single artifact is ~148 MB (whisper.cpp ggml-base.en); search/filter controls and transfer-resume stay out until an actual catalog or download-scale need appears. Any future resume work must preserve immutable pins and complete-file hashing with interruption tests |

Current evidence: [defaults and gates](MODEL-DEFAULTS.md),
[installer protections](INSTALLER-PROTECTIONS.md),
[broader qualification](../benchmarks/results/2026-09-16-qualification/RESULTS.md),
[CUDA qualification](../benchmarks/results/2026-09-16-cuda/RESULTS.md).
GB10 CUDA processed 5,557 negative clips with no activations and no fallback;
this does not replace real-room qualification or a matched performance gate.

Initial lifecycle audit: `Command::Pause`/`Resume` update a single daemon boolean.
Owned pause leases must preserve manual pause semantics and support independent
owners before cross-app playback integration can meet W10.
