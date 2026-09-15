# Omawake 0.0.2

This follow-up to 0.0.1 improves wake-phrase configuration and setup:

- exact per-wake-word transcript aliases for uncommon names and repeatable ASR
  spelling variants, without global fuzzy matching;
- opt-in transcript diagnostics with `test --show-transcripts` to understand
  missed phrases before adding an alias;
- validation that rejects phrases and aliases containing only punctuation.

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
