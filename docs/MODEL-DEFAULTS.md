# Model defaults: first implementation slice

Implements default coverage and selection work from approved priorities W02/W03
([planning PR](https://github.com/jacob-vincent-mink/omawake/pull/2)). It does not
promote new hardware recommendations or additional ASR families.

`setup all` without `--model` now uses the active compatible catalog model or
the selected backend's default. An explicit `--model` still takes precedence.
`setup model` prints a download hint for that backend. Selecting CPU runtime
preserves an explicitly configured whisper.cpp backend. Selecting CUDA, Vulkan,
HIP or OpenVINO resolves the appropriate provider and model together.

| Backend | Maintained default | Runtime contract |
|---|---|---|
| audio.cpp | Moonshine Streaming Tiny Q8_0 + Silero 6.2.1 | CPU, CUDA, Vulkan, HIP |
| OpenVINO GenAI | Whisper Base.en INT8 + Silero 6.2.1 | OpenVINO CPU/GPU/NPU |
| whisper.cpp | Whisper Base.en GGML + Silero 6.2.0 GGML | CPU, exact libwhisper 1.9.3 ABI |

Runtime contracts describe adapter compatibility, not qualified hardware.
Existing IDs and installation manifests remain valid. Friendly names and native
family hints do not change schema-1 installation identity or file hashes.

## whisper.cpp setup

Supply a complete whisper.cpp 1.9.3 library installation through
`OMAWAKE_WHISPER_LIBRARY` or `backend.library`. Model downloads come from
revision-pinned `ggerganov/whisper.cpp` and `ggml-org/whisper-vad` repositories.
The total bundle is 148,849,309 bytes; both files are verified before activation.

```sh
omawake setup model --download whisper-base.en-ggml-silero-v6.2.0 --no-activate
omawake config set backend.kind whispercpp
omawake config set backend.library /absolute/path/to/libwhisper.so.1.9.3
omawake setup model --set whisper-base.en-ggml-silero-v6.2.0
```

Use `--source-dir` with the exact pinned files for offline installation.
Runtime discovery executes a bounded isolated ABI probe without requiring model
files. It reports `loadable`, not `ready`; model-backed proof is still required.
An incompatible or incomplete library fails before activation.

## Baseline and promotion gates (W01)

Existing reference evidence is in
[rc.3 results](../benchmarks/results/2026-09-15-rc3/RESULTS.md) and
[Vulkan results](../benchmarks/results/2026-09-15-vulkan/RESULTS.md). The added
whisper.cpp file-only smoke report is in
[default coverage evidence](../benchmarks/results/model-defaults/RESULTS.md).

Before promoting a different default, use versioned/hash-pinned real positive,
near-match and speech-negative recordings, with independent speakers, adverse
audio and playback echo. Compare the candidate and current default on identical
inputs, hardware, threads and caches. For setup-only changes with the same
model/provider, require identical detections, no additional false activations
or duplicate actions, and no fallback. Predeclare a maximum 10% increase in
warm p95 inference and peak memory, checked over three repeated runs before
calling a regression; record cold load separately. A changed model needs an
explicit quality/resource tradeoff review rather than inheriting these claims.

The existing 60 synthetic positives and 15-minute clean negative set are smoke
baselines, not continuous-listening qualification. The [broader qualification run](../benchmarks/results/2026-09-16-qualification/RESULTS.md)
extends multi-speaker, adverse, negative and offline VAD evidence. Real-room
recordings and idle-power qualification remain outstanding W01 work. HIP remains unqualified here. W04 compatibility filtering and W06 device qualification remain incomplete.
Provider-family checks and installer guards are now documented in
[installer protections](INSTALLER-PROTECTIONS.md).

## Setup list navigation (W04)

Setup selectors now size their viewport to the terminal height, keep the
highlighted row visible, and show its position in the list. Long details wrap
within the terminal width; arrow-key navigation brings subsequent rows into
view. Resizing recomputes the layout. Zero-size terminals use an 80×24 fallback.
The CLI catalog and JSON interfaces are unchanged. The model picker now disables rows incompatible with the selected backend/runtime/device, explains the required backend, and starts on a compatible row. Invalid or disabled selections are rejected before installation. CLI catalog listing remains complete and stable for discovery. Provider-family checks
are described in [installer protections](INSTALLER-PROTECTIONS.md).

## Subsequent implementation

See [provider and installer boundaries](INSTALLER-PROTECTIONS.md) for the next
implemented slice, its tests and remaining limits.

[Default corpus checks](../benchmarks/results/2026-09-16-default-corpus/RESULTS.md)
record the new file-only evidence and qualification limits.

[Broader qualification evidence](../benchmarks/results/2026-09-16-qualification/RESULTS.md)
records release-build comparisons, generated adverse/near-match corpora, Intel
GPU/NPU checks and remaining promotion gates.
