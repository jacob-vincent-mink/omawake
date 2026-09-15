# Changelog

## Unreleased

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
