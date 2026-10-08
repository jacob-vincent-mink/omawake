# Cloud qualification handoff

Cloud HTTP and WebSocket fixture tests pass locally. Paid-account access,
recognition quality, false-wake rates and real-room microphone behavior need
independent qualification. No live vendor requests were made during development.

## Configure a provider and its key

```sh
omawake setup cloud --provider deepgram --realtime
omawake cloud credential install --stdin < /private/deepgram-key
omawake cloud credential check
```

`setup cloud` without `--provider` offers guided cloud configuration. Deepgram
HTTP remains the default; `--realtime` opts into streaming all session audio,
including silence and unrelated speech. `--realtime=false` selects HTTP clips.
OpenAI-compatible supports HTTP clips only. Wake definitions and actions remain
local and are preserved by setup.

The installer writes an owned regular file with mode 0600 and saves its absolute
path, never the key, in config. Both CLI and daemon read it directly. Environment
keys take precedence. Restart an existing daemon after configuration changes.
`cloud credential check --daemon` checks the service process's environment and
the saved file without displaying secrets; it does not prove a changed config
has been reloaded. Credential/setup checks never record or upload audio.

## Qualify file recognition without running actions

```sh
omawake cloud smoke --audio /path/wake.wav
python3 benchmarks/cloud/qualify.py --binary ./omawake --config /path/config.toml \
  --audio /path/wake.wav --audio /path/unrelated.wav --audio /path/silence.wav \
  --out-dir /tmp/deepgram-realtime-qualification --accept-charges
```

Smoke uploads the WAV through the selected transport, reports final detections
and latency, and never executes commands. In HTTP mode silence produces no
request; in realtime mode all provided audio is transmitted. The corpus driver
writes `report.json` and a human review sheet. For Omawake always supply `--audio`.
Prepare recordings with the exact configured phrase, aliases, near misses,
unrelated speech, quiet/background noise and longer passages. Compare expected
IDs, false positives, misses and finalization latency. Run separately for Deepgram
HTTP, Deepgram realtime and the intended OpenAI-compatible service.

For live qualification, use harmless local commands and explicitly enable live
execution only after reviewing file results. Exercise pause/cancel/resume,
connection interruption and a slow provider. Verify interim transcripts never
trigger actions and completed utterances trigger once, with no replay or stale
responses after resume. Record the provider model, language, account access,
build/version, architecture, microphone/environment and every unresolved issue.
A successful request alone does not qualify detection quality.
