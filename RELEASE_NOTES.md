# Omawake 0.0.3

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
