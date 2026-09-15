# Changelog

## 0.0.2 - Unreleased

- Added exact per-wake-word transcript aliases for uncommon names and repeatable
  ASR spelling variants without enabling global fuzzy matching.
- Added opt-in transcript diagnostics to file and microphone tests.
- Refresh an already-active service after CLI config, runtime, model, or
  wake-word changes; restore the previous config and restart on failure.
- Recommend discovered accelerator providers in the order CUDA, Intel NPU,
  Intel GPU, Vulkan, then CPU, while preserving existing manual selections.
- Added file-only Intel Vulkan validation and a shared-provider build example.
- Reject phrases and aliases that normalize to no letters or numbers.

## 0.0.1 - 2026-09-15

- Promoted the native-provider architecture after the rc.3 CPU, Intel
  CPU/iGPU/NPU, and NVIDIA CUDA qualification pass.
- Fixed guided runtime changes so the library prompt and saved configuration
  cannot reuse paths from a previously selected provider.
- Made wake-word actions fire-and-forget with detached standard streams and
  background child reaping so long-running commands do not block detection.
- Hardened release packaging checks around provider discovery and isolated
  test environments.

## 0.0.1-rc.3 - 2026-09-15

- Replaced the prototype inference path with native audio.cpp and direct
  OpenVINO GenAI providers.
- Added a compact packaged CPU provider and external CUDA, Vulkan, HIP, and
  Intel CPU/GPU/NPU provider discovery.
- Added a provider-neutral exact phrase-verifier evaluation report.
- Added setup-time OpenVINO GPU/NPU cache compilation and transactional setup.
- Completed provider and model license/provenance packaging.
- Preserved accelerator runtime, device, library, and fallback choices when a
  compatible model is activated.
- Fixed focused runtime setup to probe the complete candidate configuration,
  including its selected catalog model, before committing it.
- Expanded runtime and configuration discovery, including CUDA, Vulkan, HIP,
  OpenVINO, provider options, and explicit accelerator device indexes.
- Restored the previous config, desktop launcher, and active service after a
  late full-setup failure, and hardened generated launcher and systemd paths.
- Removed unused audio configuration fields and made missing launchers an
  optional setup-check result.

## 0.0.1-rc.2 - 2026-09-15

- Added local wake-phrase detection for live capture and WAV files.
- Added multiple phrase-to-action mappings using direct argument vectors.
- Added foreground daemon control, on-demand testing, and JSON benchmarks.
- Added guided and scriptable model/runtime setup with no implicit service or
  vendor-runtime installation.
- Added Linux x86-64 and aarch64 release packaging.

Known limits: releases target glibc 2.35 or newer. Accelerator stacks are
external and are accepted only after their provider and device probe succeeds.
