# Opt-in cloud transcription

Set `backend.kind = "deepgram"` or `"openai-compatible"` to replace local ASR
with an HTTP transcriber, or opt into Deepgram realtime streaming. Phrase normalization, aliases, whole-phrase matching,
commands, wake-pause ownership and cooldown policy remain local. Remote mode
requires runtime=default, device=remote, device_id=0 and fallback=error.
No local inference model or native provider is loaded.

```toml
[backend]
kind = "deepgram"
runtime = "default"
device = "remote"
fallback = "error"

[backend.cloud]
model = "nova-3"
api_key_env = "DEEPGRAM_API_KEY"
timeout_seconds = 60
max_audio_seconds = 30
vad_threshold = 0.01
endpoint_milliseconds = 500

[model]
language = "en"

[[wake_words]]
id = "assistant"
phrase = "hey computer"
enabled = true
command = ["notify-send", "Wake phrase heard"]
```

OpenAI-compatible defaults are API root `https://api.openai.com/v1`, model
`gpt-4o-mini-transcribe`, and environment variable `OPENAI_API_KEY`. Deepgram
uses `https://api.deepgram.com/v1`, `nova-3`, and `DEEPGRAM_API_KEY`.
`backend.cloud.base_url`, `model`, `api_key_env`, `api_key_file`, `realtime`, `timeout_seconds`,
`max_audio_seconds`, `vad_threshold`, and `endpoint_milliseconds` are editable
with `config set`. Named engine profiles also carry their own cloud config.
An empty model.language uses the provider default (Deepgram realtime defaults to English; choose an explicit supported language or `multi` for multilingual recognition).
Existing phrase definitions need not be recreated when changing providers.

By default (`realtime = false`), providers receive **completed utterance clips**. Audio is resampled to 16 kHz, gated locally using configurable 20 ms RMS
energy frames, with 200 ms pre-roll and 100–2000 ms silence endpointing. It is
an energy gate, not Silero or a trained keyword detector; background noise can
cause uploads. Utterances are capped at `min(max_audio_seconds,30)` seconds.
Silence alone is not uploaded. Every voiced segment can contain unrelated
speech and is uploaded in this opt-in mode; only phrase matching is local.
This mode does not provide local-keyword-first cloud verification.

Deepgram receives a mono linear16 WAV at `/listen`; OpenAI-compatible receives
a multipart WAV at `/audio/transcriptions`. Only complete response transcripts
are matched. There are no actions from partial transcripts. ElevenLabs Scribe Realtime and Soniox remain future adapters; HTTP compatibility
does not imply their protocols.

A separate supervised process performs network work. The live session does
not wait for HTTP in the capture callback or control loop. At most three bounded
utterances are pending (one active plus two queued); overflow/errors fail the
session without silently dropping or replaying speech. Pause/session disposal
kills and reaps its worker, discards queued work and prevents stale responses
from triggering after resume. Parent-death cleanup covers forced termination.
File testing waits for completed results and uses the same local matching.

Credentials are environment references or a private `api_key_file`, not config
secrets. `cloud credential install --stdin` configures a shared CLI/daemon file;
otherwise configure the daemon
process environment before starting/restarting it. `setup check` validates config,
phrases and credential availability without recording/uploading audio. HTTPS is
required except explicit loopback HTTP; URLs cannot contain credentials, queries
or fragments, and redirects are disabled. Response JSON is bounded to 1 MiB;
HTTP bodies and transport URLs are omitted from errors. Request deadlines are
1–300 seconds. There are no automatic retries or alternate-provider fallbacks.

Use `test --audio FILE.wav --json` before microphone use. Loopback tests verify
bounded WAV upload, provider auth/routes, local whole-phrase matching, silence,
malformed/provider failure handling and error redaction. Cloud recognition quality,
false-wake rates, paid account access and real-room operation remain unqualified.

References: [Deepgram prerecorded transcription](https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded),
[OpenAI file transcription](https://developers.openai.com/api/docs/guides/speech-to-text).

## Deepgram realtime

`omawake setup cloud --provider deepgram --realtime` or
`config set backend.cloud.realtime true` selects the WebSocket route at `/listen`.
The HTTPS API root is converted to WSS; only loopback development endpoints may
use WS. Audio is resampled to 16 kHz mono linear16 and sent in bounded 100 ms
frames. **All session audio is sent, including silence and unrelated speech**;
`vad_threshold` applies only to HTTP clip mode. Provider VAD uses the configured
`endpoint_milliseconds`; finalized segments are accumulated until `speech_final`
or the corresponding `UtteranceEnd`. Interim text and repeated finalized ranges
never trigger wake actions. `max_audio_seconds` bounds an unfinished utterance;
exceeding it ends the session. A long-lived session may contain many utterances.

A supervised process owns the connection and network threads. The live audio
queue holds at most 32 frames (3.2 seconds), plus bounded pipe/worker buffers;
overflow fails the session. Response messages/transcripts are bounded. CloseStream
flushes file-session finals, keepalive/ping maintain idle sessions, and provider
errors/disconnects/timeouts end the session without automatic reconnect or audio
replay. Pause destroys the session/worker; resume creates a fresh connection.
No Tokio, D-Bus or inference runtime is added.

Use `cloud smoke --audio FILE.wav` to measure recognition without executing
commands. [Qualification instructions](CLOUD-QUALIFICATION.md) provide a repeatable
corpus driver and independent live-review checklist.

Protocol: [Deepgram live audio](https://developers.deepgram.com/reference/speech-to-text/listen-streaming),
[final segments and endpointing](https://developers.deepgram.com/docs/understand-endpointing-interim-results).
