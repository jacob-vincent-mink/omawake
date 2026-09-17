# Moonshine Small/Medium vs Tiny wake comparison, 2026-09-16

W08 evidence: the three pinned Moonshine Streaming Q8_0 variants (Tiny, Small,
Medium) ran the identical corpora through the identical released provider
library, differing only in the verifier GGUF. The decision question is whether
either larger variant earns its footprint; promotion requires meaningful wake
benefit, and "leave both out" is an acceptable outcome.

## Controlled inputs

- **Provider**: released `omawake-bin 0.0.2` packaged `libaudiocpp.so.0.1.0`,
  sha256 `29b506ea…a325c10` — byte-identical to the library recorded by the
  2026-09-16 wake qualification, so the provider is not a variable. audio.cpp
  CPU backend (Silero VAD + `moonshine_asr`), `fallback = "error"`, no
  re-exec, per-file fresh process through the native inference worker.
- **Variants**: pinned profiles verified at load by exact file name, size and
  sha256 (`backend.options.audiocpp.asr_variant`, default `tiny`):
  Tiny 60,407,904 B `e9a342a0…`, Small 300,621,248 B `61036888…`,
  Medium 315,583,648 B `cc242a59…` (audio-cpp/audio.cpp-gguf@`6d5436fc`).
- **Corpora** (identical across arms; manifests with per-clip hashes in `runs/`):
  adverse 44 clips (48 expected light-up/lovely-child/forever events, clean +
  echo-120–240 ms + white-noise 0/10 dB variants), near-match 12 clips
  (light-bulb/line-up/computer-family negatives), synthetic 60 Piper
  "hey jarvis" positives, and the full 5,557-clip / 10.739 h LibriSpeech
  negative corpus used by the earlier full-negative run.
- **Configuration**: only wake words configured per manifest keyword
  (`light-up` + `lovely-child` + `forever`, or `hey-jarvis`); profiles in
  `configs/`. Shared host; CPU backend, default thread budget. Timing figures
  are not uncontended regression-gate measurements.

## Accuracy

| Manifest | Metric | Tiny | Small | Medium |
|---|---|---|---|---|
| adverse (48 events) | true positives | 36 | 41 | 43 |
| adverse | false negatives | 12 | 7 | 5 |
| adverse | false positives / wrong-id | 0 / 0 | 0 / 0 | 0 / 0 |
| near-match (12) | false positives | 0 | 0 | 0 |
| synthetic (60) | true positives | 59 | 59 | 60 |
| full negative (5,557 / 10.739 h) | false activations | 10 | 10 | 10 |
| full negative | keyword attribution | 9 `forever` + 1 `lovely-child` | 9 `forever` + 1 `lovely-child` | 9 `forever` + 1 `lovely-child` |
| full negative | `light-up` false activations | **0** | **0** | **0** |

Every adverse miss in every arm is the word `forever` under
`white-noise-0db` or `echo-120-240ms` degradation. `light-up` and
`lovely-child` were detected by all three variants in all adverse conditions,
and clean-condition behavior is identical. The recall spread between variants
is therefore confined to the hardest synthetic degradations of a single
keyword; it is not a clean-speech or near-match benefit.

## Resources (same 16.7 s clip, 5 iterations after warmup)

| Variant | Model load | Per-pass elapsed (RTF) | Peak process RSS | Model file |
|---|---|---|---|---|
| Tiny | 133 ms | ~565 ms (0.034) | ~150 MiB | 58 MiB |
| Small | 409 ms | ~1.93 s (0.115) | ~376 MiB | 287 MiB |
| Medium | 315 ms | ~2.77 s (0.166) | ~536 MiB | 301 MiB |

Evaluate-mode aggregate RTF over the small manifests agrees (0.035 / 0.108 /
0.152). Small and Medium cost roughly 3–5× CPU and 2.5–3.5× resident memory
of Tiny for continuous verification.

## Decision

**Neither Small nor Medium is promoted; both stay deferred (stop-promote
decision recorded 2026-09-16).** All three variants produced the *identical*
false-activation behavior on the 10.739 h negative corpus — exactly 10
activations, all attributable to the generic `forever` (9) and `lovely-child`
(1) keyword probes, with zero `light-up` false activations per hour for
every variant, matching the earlier CUDA full-negative run. Clean-speech,
near-match and synthetic behavior is likewise identical across arms except
for degraded-condition `forever` recall (adverse: 36 → 41 → 43 of 48).

Against that, Small and Medium cost 3.3–4.9× CPU, 2.5–3.5× resident memory,
and ~5× model footprint for continuous verification, and would add
runtime/device re-qualification surface. The adverse-condition recall gain
is confined to one keyword under synthetic degradation on a smoke-sized set,
which does not meet the "meaningful wake benefit" bar. The pinned Small and
Medium profiles remain available (opt-in via
`backend.options.audiocpp.asr_variant`) for future re-evaluation if real
phrase failures persist after W09, which is exactly the deferred W14 trigger.

Timing note: the full-negative runs were executed concurrently on a shared
host, so their RTF/p95 figures are contended; the resource table above uses
the uncontended sequential benchmark runs.

## Raw evidence

`runs/` holds every evaluate/benchmark JSON plus per-run maximum-RSS capture;
`configs/` holds the exact per-variant profiles.
