# Spanish wake profile (multilingual Whisper, OpenVINO), 2026-09-16

W09 evidence: the first qualified non-English wake profile. The curated
`whisper-base-int8-ov-silero-v6.2.1` catalog profile (Silero VAD 6.2.1 +
multilingual OpenVINO Whisper Base INT8) runs with an explicit
`model.language = "es"` that reaches both inference and status. Spanish is the
only language claimed; no blanket multilingual qualification is made even
though the pinned model itself accepts 99 language tokens.

## Language reaching inference and status

- `model.language` is a new config key (restart required). The engine accepts
  only tokens of the pinned model's own `generation_config.json`
  (`lang_to_id`, 99 languages) and only for the multilingual profile;
  the English-only profile rejects any language with a clear error. Status
  (`omawake status --json`), daemon details and evaluation reports carry the
  configured language alongside the model name.
- **Pinned-runtime finding**: the OpenVINO GenAI 2026.3.1.0 C wrapper's
  `ov_genai_whisper_generation_config_set_language` poisons the config object
  — every later `validate`/`generate` call on it fails with status -17, even
  for `"en"`, with the unmodified config validating cleanly. The same
  language loads fine through `create_from_json`. The engine therefore bakes
  the explicit language into a derived
  `generation_config.<lang>.json` inside the worker-owned cache directory
  (the verified model tree is never modified), builds the config from that
  file, and passes it to every generate call. This is recorded as an upstream
  runtime limitation, not worked around silently.

## Evidence

Positive and near-match clips were synthesized with Omaspeak's Kokoro 82M
Spanish voice `ef_dora` (Apache-2.0), a maintainer-evaluation use of
Omaspeak consistent with W17 boundaries. Negatives are the first 100
dev clips of FLEURS Latin-American Spanish (`es_419`,
`google/fleurs@70bb2e84…`, CC-BY-4.0, converted to 16 kHz mono PCM16 with
ffmpeg), staged with per-clip SHA-256 in
[negative.manifest.json](negative.manifest.json).

| Set | Clips | Result |
|---|---|---|
| "enciende la luz" positives | 3 | 3/3 detected |
| "apaga la luz" near-match | 1 | not detected (transcribed "Apakah la luz?") |
| "enciende el ventilador" distractor | 1 | not detected |
| Spanish negatives | 100 (0.320 h) | **0 false activations** (0.0/h) |

Sample WAVs are in [samples/](samples/) (`pos1–3` detected, `neg1–2`
rejected). The evaluate report is [negative-eval.json](negative-eval.json)
(RTF 0.061, peak RSS ≈ 407 MiB, per-clip hashes).

Wake word configuration: `enciende-luz` = "enciende la luz"; profile in
[configs/profile-es.toml](configs/profile-es.toml).

## Limits

- Smoke-scale: three positives, one distractor family and 19 minutes of
  read-speech negatives do not qualify continuous-listening false-activation
  rates, speakers, devices or adverse conditions; real-room recordings,
  longer Spanish negatives and human listening remain open.
- The English default remains untouched; Spanish ships as an explicitly
  selected profile.
- No claim is made for any other language despite the model supporting 99.
