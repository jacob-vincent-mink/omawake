# Language profile: Spanish (W09)

Omawake's first explicitly qualified non-English wake profile. The default
remains English (Moonshine Streaming Tiny / Whisper Base.en); Spanish ships
as a separately selected, pinned profile.

## Curated profile

| Item | Value |
|---|---|
| Catalog ID | `whisper-base-int8-ov-silero-v6.2.1` |
| Backend | `openvino-genai` (runtime `openvino`) |
| Verifier | multilingual OpenVINO Whisper Base INT8 |
| Converted source | `OpenVINO/whisper-base-int8-ov@0606293f…` |
| Original source | `openai/whisper-base` (Apache-2.0) |
| VAD | Silero VAD 6.2.1 (pinned `silero_vad_16k.safetensors`) |
| Qualified languages | `es` only |

Install and select:

```sh
omawake setup model --download whisper-base-int8-ov-silero-v6.2.1
omawake config set backend.kind openvino-genai
omawake config set backend.runtime openvino
omawake config set model.name whisper-base-int8-ov-silero-v6.2.1
omawake config set model.language es
omawake wake-word add --id enciende-luz --phrase "enciende la luz" -- <command>
```

## Language semantics

- `model.language` accepts a token from the pinned model's own
  `generation_config.json` language set (99 Whisper languages). Accepting a
  token validates configuration only — it is not a qualification claim.
- `languages` in the catalog profile, status and evidence stays the curated
  qualification claim: `es`. The English-only profile rejects any language.
- The explicit language reaches inference through a derived
  `generation_config.<lang>.json` in the worker-owned cache directory. The
  pinned model tree is never modified. The pinned OpenVINO GenAI
  2026.3.1.0 C wrapper rejects every language through its
  `set_language` entry point (status -17) while accepting the same field via
  `create_from_json`; this is recorded as an upstream limitation, and the
  engine fails loudly if the derived config cannot be validated.
- Status (`omawake status --json`), daemon details and evaluation reports
  include the configured language next to the model name.

## Evidence snapshot (2026-09-16)

Kokoro-synthesized Spanish positives (3/3 detected), near-match and
distractor rejection, and 0 false activations over 0.32 h of FLEURS
`es_419` read-speech negatives: see
[2026-09-16-spanish/RESULTS.md](../benchmarks/results/2026-09-16-spanish/RESULTS.md).
Real-room recordings, longer negatives, more speakers and human listening
remain open. No other language is claimed.
