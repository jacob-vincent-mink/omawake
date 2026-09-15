# Omawake 0.0.1

This release introduces the greenfield native-provider architecture:

- a runtime-neutral Rust executable;
- a compact, pinned CPU audio.cpp C ABI provider in each Linux archive;
- verified Moonshine Streaming Tiny Q8_0 and Silero VAD 6.2.1 model setup;
- direct external audio.cpp CUDA, Vulkan, and HIP provider selection;
- direct external OpenVINO GenAI CPU, GPU, and NPU selection with setup-time
  accelerator cache compilation;
- guided and focused setup with atomic rollback and no implicit service or
  vendor-runtime installation;
- fire-and-forget action launching with detached I/O and background child
  reaping;
- provider-neutral exact phrase-verifier evaluation reports.
- exact per-wake-word transcript aliases for coined names and repeatable ASR
  spelling variants without global fuzzy matching.

The release archives contain project, Rust dependency, audio.cpp, ggml, cJSON,
libyaml, PocketFFT, and conservative llama tokenizer notices. Models are
downloaded during setup and retain their own provenance and licenses beside the
installed assets.

File-only rc.3 evidence covers the packaged CPU provider, OpenVINO CPU/iGPU/NPU
on Intel Core Ultra hardware, and CUDA on an NVIDIA GB10. It records setup-time
accelerator caches, placement, cold load, warm timing, positive recall, and a
clean-speech negative subset in the
[rc.3 hardware results](benchmarks/results/2026-09-15-rc3/RESULTS.md).
