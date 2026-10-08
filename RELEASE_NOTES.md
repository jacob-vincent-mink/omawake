# Omawake 0.1.3-rc.2

Adds guided cloud setup, private API-key files shared by CLI/daemon, offline
credential checks, and explicit file smoke tests that never execute wake actions.
A repeatable qualification driver records results for an independent collaborator.

Deepgram now supports opt-in WebSocket recognition. Provider VAD endpoints final
segments; utterance accumulation and duplicate suppression prevent interim or
repeated results from triggering actions. Network work runs in a supervised
process with bounded queues, keepalive/heartbeats and cancellation cleanup.
Disconnections, timeouts and overflow end the session without reconnect/replay.

HTTP utterance transcription remains available through Deepgram and
OpenAI-compatible APIs. Local whole-phrase matching, aliases and cooldown policy
apply to both transports. Realtime sends all session audio, including silence and
unrelated speech; HTTP sends locally gated voiced clips.

Paid-account access, recognition accuracy and real-room false-wake/latency
qualification remain pending. See `docs/CLOUD-QUALIFICATION.md` and `docs/CLOUD.md`.
Consumer-events/D-Bus and the unrelated audio.cpp ABI change remain separate.
No Tokio or new inference runtime is bundled; ONNX Runtime remains excluded.
Existing local defaults and previously published RCs remain unchanged.

## Omawake 0.1.2-rc.2

This candidate reopens setup with the saved runtime, device, and model selected
while keeping `r` for hardware recommendations. It improves native provider
discovery, exposes schema-advertised provider options, and reports native audio
worker failures for diagnosis. The cross-app inference suite now exercises
CPU, Vulkan, and CUDA paths.

## Omawake 0.1.2-rc.1

This release candidate introduces a single tabbed setup menu. Left and Right
move between runtime, device, model, microphone, and final review; Up and Down
highlight choices, Space selects, and Enter advances or applies. Press `r` to
restore detected recommendations and jump to review. Model downloads,
compilation, config changes, and the optional launcher install start only after
Apply. `omawake setup --recommended` remains available for one-command setup.

Setup now preserves an active daemon's config ownership and custom config
paths during edits and service changes. The terminal E2E suite exercises
navigation, cancellation, provider failures, and full Apply using isolated
homes; CI saves the smoke-suite screen frames as an artifact.

## Omawake 0.1.1

This update fixes `setup runtime --dir /usr` with an explicit OpenVINO CPU or
NPU device on Arch installations where OpenVINO plugins live in
`/usr/lib/openvino` and core and GenAI libraries live in `/usr/lib`. Runtime
discovery now retains both directories. The default GenAI provider lookup also
includes the plugin directory.

## Omawake 0.1.0

This release accepts both split NPU compiler packages and self-contained
OpenVINO 2026.4 NPU plugins, checking any separate compiler libraries beside
the chosen plugin. Its OpenVINO GenAI Whisper path passed a
file-only NPU test with the isolated 2026.4 C build: a Kokoro WAV synthesized
by Omaspeak was transcribed and detected with an exact transcript alias.

If an existing configuration pins a 2026.3 GenAI bundle, set
`backend.library` and `backend.library_dirs` to the new installation before
testing it. The package's optional `openvino-genai` dependency pins the
matching OpenVINO runtime version.

## Omawake 0.0.3

This release completes the model-support roadmap:

- pinned catalog URL health checks (`setup model --check-urls`) and sharper
  offline-import diagnostics that name expected and observed values;
- a Spanish wake profile on the multilingual Whisper Base INT8 OpenVINO row,
  with `model.language` reaching both inference and status (smoke-scale
  evidence; real-room and listening gates stay open);
- Moonshine Small and Medium evaluated against Tiny on identical corpora:
  identical false-activation behavior at 3-5x the cost, so both stay
  deferred behind pinned opt-in `asr_variant` profiles;
- connection-owned playback pauses: TTS playback holds the wake daemon
  paused and cannot retrigger wake actions, with nested manual pauses,
  client disconnects and crashes releasing ownership safely;
- installation and activation audits closed: download rollback, cache
  acceptance mapping, and catalog verification with exact provenance;
- service refresh, transcript aliases and `--show-transcripts` diagnostics
  from 0.0.2 carry forward unchanged.

The runtime-neutral Rust executable includes a pinned CPU audio.cpp provider.
Setup downloads verified Moonshine Streaming Tiny Q8_0 and Silero VAD models.
Accelerator providers remain external: audio.cpp for CUDA, Vulkan, or HIP, and
OpenVINO GenAI for Intel CPU, GPU, or NPU. OpenVINO accelerator caches are
compiled during setup before activation. Setup does not install vendor runtimes
or enable a systemd service.

The archives include project and dependency license notices. Downloaded models
retain their provenance and licenses beside their installed assets.

The [rc.3 hardware results](benchmarks/results/2026-09-15-rc3/RESULTS.md)
record file-only CPU, OpenVINO CPU/iGPU/NPU, and NVIDIA GB10 CUDA validation.
The [Intel Vulkan results](benchmarks/results/2026-09-15-vulkan/RESULTS.md)
add a same-model CPU/iGPU comparison with equal recall on the positive set.
These measurements identify their tested source revision and models; they are
not an accuracy guarantee for arbitrary wake phrases.
