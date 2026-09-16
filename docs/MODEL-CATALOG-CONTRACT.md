# Shared model catalog and setup contract

Proposed 2026-09-15 for Omawake and Omaspeak. This document is mirrored in both
planning worktrees so either plan is reviewable independently. It specifies new
behavior; the illustrative commands and fields are not implemented yet.

## Scope and implementation priority

[PRIORITIES.md](PRIORITIES.md) controls which parts of this contract enter the
implementation queue in this repository. This document describes a compatible
long-term data shape, not a requirement to build every field and command now.
First implement complete defaults, compatibility, readable listings and atomic
installation. Add search/variant grouping when the curated list needs them;
add clone/style fields only with an approved family that consumes them.

Remote catalog refresh, provider editions and advanced voice authoring are
deferred. Generic task runners, unfiltered model browsers and a separate shared
catalog service are recommended declines. Maintained project-hosted artifacts
remain a fallback when canonical sources cannot supply a good default.

## Design

Ship a curated, versioned catalog with each application. Setup should work from
that catalog offline and download only the selected model's pinned assets. Use
upstream catalogs and Hugging Face to prepare reviewed additions, not as a live,
unfiltered menu of everything tagged ASR or TTS.

Keep four separate concepts: model family, artifact variant, execution backend,
and voice profile. A model may have several GGUF/IR variants, and a backend may
run several families. Neither a `.gguf` suffix nor an OpenVINO tag establishes
compatibility with an application's adapter.

## Setup presentation

1. Detect available providers/devices; preserve a valid explicit selection.
   Offer the packaged CPU default on a fresh setup. Resolve a recommendation
   for the selected backend/device and language, with advanced backend choices.
2. Show a short list headed **Recommended**, then **Other compatible models**.
   Default to supported models; a separate **Show experimental/unavailable**
   view explains missing provider families or dependencies.
3. Use friendly model names. Each row shows language, remaining download size,
   installed/active state and a concise benefit. Show capability labels such as
   “Preset voices”, “Clone a voice” or “Voice design” only when applicable.
4. Search by name/language/capability; group precision variants under a family.
   Scroll within terminal height, keep selection visible, and support narrow
   terminals and a plain non-interactive listing. Disabled entries need a
   focusable details view so their remediation can actually be read.
5. Details show total/incremental download, disk/cache needs, RAM/VRAM estimate
   or “not measured”, model/voice source, license, backend compatibility and
   evaluation status. Keep paths, hashes and low-level options in details.
6. Download and verify into staging, load/probe with fallback disabled, prepare
   required caches, then activate. A backend switch gets an explicit summary of
   the new model and voice. Cancel/failure preserves the active configuration.

Illustrative rows after candidate qualification (sizes are illustrative where
marked; production values come from the pinned manifest):

```text
Wake-word model                     English · audio.cpp / CPU

  > Moonshine Tiny      Recommended · 59 MiB download · English
    Moonshine Small     Accuracy candidate · ~287 MiB · English
    Moonshine Medium    Accuracy candidate · ~301 MiB · English

  / Search   Enter Details / select   Esc Back
```

```text
Speech model                        English · audio.cpp / CPU

  > Supertonic 3        Recommended · 10 preset voices · Installed
    Kokoro 82M          More preset voices · ~181 MiB + language resources
    Qwen3 CustomVoice   Expressive delivery · Download size in details

  Filter: All / Preset voices / Clone / Design
```

Preserve current commands and add filters/details rather than a second installer.
Proposed consistent surface:

```text
<app> models list --backend <kind> --language en [--capability clone] [--json]
<app> models info <id> [--json]
<app> setup model --model <id> [--download-only]
<app> setup model --recommended --backend <kind> --device <device> --language en
```

Current `models`/`setup model` spellings remain aliases where applicable. JSON
lists report stable IDs, compatibility reasons, status, sizes and capabilities;
non-interactive setup never invents an answer to a required model-license gate.

## Catalog schema and resolution

Start with checked-in data compiled/embedded in the existing Rust catalog
modules. Avoid a shared runtime service or new shared crate initially; keep the
schema and fixture conformance tests aligned between repositories.

| Entity | Required data |
|---|---|
| Model identity | Stable ID, display name, family/variant, purpose, languages, precision and format |
| Adapter profile | Backend kind, exact provider family, task/mode, sample rate, asset roles, request/session/load defaults, minimum ABI/runtime, required build families and resources |
| Files | Explicit relative paths, byte sizes, SHA-256, immutable download revision/URL, source and conversion provenance, file-level license/notice |
| Voice support | Preset IDs and language mapping, default voice, reference-audio/transcript requirements, design/style support |
| Dependencies | Tokenizer/vocoder/codec, voice embeddings, VAD, dictionaries or phonemizer data, with independent provenance |
| Qualification | Backend/runtime/device/platform/language matrix; candidate, supported or recommended; evidence path, provider hash/revision and test date |
| Requirements | Download/install/cache sizes, measured memory or unknown, license acceptance/access requirements |

Resolve `recommended(backend, runtime, device, language, capability)` centrally.
Every supported backend must have at least one default profile with all files,
a usable preset voice where relevant, and a successful qualification record.
Each advertised runtime/device combination must either have a qualified default
or be explicitly experimental/unavailable. No “bring your own model” default.
User-requested features such as clone/design have separate capability defaults;
they do not replace a backend's ordinary TTS default.

Compatibility is an intersection of catalog adapter support, installed provider
families, runtime/device support, required resources and a model-backed probe.
Before download, use catalog data plus audio.cpp registry enumeration. After
loading in the existing isolated worker, use `audiocpp_model_supports`, language,
speaker-reference/style queries and option introspection. The pinned
[C ABI](https://github.com/0xShug0/audio.cpp/blob/e9ff20042ec85af960a720368c6927cda19ad65f/include/audiocpp.h)
already exposes these. They establish capability, not output quality or actual
hardware placement. Record those separately.

`model_specs/*.json` from the same audio.cpp revision are useful maintainer
inputs for package filenames and task options. Do not execute upstream setup
scripts or import mutable `main` download references into release catalogs.
Do not assume the root audio.cpp GGUF repository's “other” license applies to
every individual model or voice.

## Asset maintenance

Prefer (1) the model author's compatible artifact, (2) the runtime maintainer's
conversion, (3) a reviewed community conversion with reproducible provenance,
then (4) our own reproducible conversion. A canonical ONNX model that the
existing adapter can execute is better than an unnecessary IR conversion.

For our conversions, keep the source revision, converter revision, dependency
lock, conversion command, asset manifest, licenses and quality checks in Git.
Publish large immutable weights as release assets or in a project-owned model
repository; keep tiny essential resources in Git when practical. Contribute
the conversion upstream when possible. A required default must remain
maintained even if no upstream package is suitable. An emergency mirror must
preserve permitted redistribution, exact bytes, provenance and notices.

Implement a maintainer import tool that resolves the upstream revision,
downloads selected files, computes size/hash, records dependencies and emits a
reviewable catalog diff. Scheduled verification detects deleted assets, changed
requirements and broken default recipes without changing user pins. A later
catalog refresh can use signed versioned releases with a known-good fallback;
it is not required for the first iteration.

Keep existing atomic installers. Add per-profile download locking, verified
file reuse, bounded/resumable transfers where supported, cancellation and a
disk-space preflight covering download + staging + existing model + compiled
caches. Resume only against the pinned object and verify the complete hash.
Model files are immutable; compiled caches are separate and keyed by model
hashes, provider/runtime version, device/driver, shapes and compile options.

## Release gates

- Schema checks: unique IDs, complete required roles, immutable sources, valid
  sizes/hashes, notices, default coverage and no candidate accidentally marked
  recommended.
- Setup tests: backend/family/language filtering, invalid selections, old-ID
  migration, unavailable resources, offline listing/import, terminal scrolling,
  interrupted/corrupt downloads and activation rollback.
- Native integration: every recommended backend/model profile succeeds from an
  isolated empty home using the release provider, produces meaningful output,
  and reports actual device placement with fallback disabled. Hardware not
  exercised remains explicitly unqualified.
- Quality/performance: project-specific corpora plus cold/warm timings, memory,
  and device evidence. Accepting a session or producing nonempty PCM is not
  sufficient to call a default good.
