# Moonshine CUDA corpus on GB10, 2026-09-16

Release-build Omawake `50b3b33` ran on the user-provided
`jacob@promaxgb10-d666`: ARM64 Ubuntu 24.04.5, NVIDIA GB10, driver 580.173.02,
CUDA 13.0.88. The existing native provider is audio.cpp
`e9ff20042ec85af960a720368c6927cda19ad65f`, built for Moonshine/Silero with CUDA
graphs and `121a-real`. Every completed evaluation verified explicit CUDA
session creation with fallback disabled. `nvidia-smi` independently observed
Omawake CUDA processes. Exact binaries, providers and model hashes are in
[environment.json](environment.json).

The model is the pinned Moonshine Streaming Tiny Q8_0 + Silero 6.2.1 default.
The application was rebuilt natively from public committed source in an
isolated directory; existing user-space ALSA linker metadata supplied the
missing development pkg-config information. XDG runtime/cache/state directories
were isolated. All tests were file-only and no actions were executed. Live
configuration/services remained untouched; SGLang stayed resident on the GPU.

## Detection results

| Corpus | Result |
|---|---:|
| Real speech, clean | 12/12 expected events |
| Generated white noise, 10 dB SNR | 11/12 |
| Generated white noise, 0 dB SNR | 6/12 |
| Generated echo, 120/240 ms | 8/12 |
| Synthetic “hey Jarvis”, three Piper voices | 59/60 |
| Synthetic near-matches, M1/F1 | 0/12 clips activated |
| LibriSpeech negative smoke, 15.04 minutes | 0/100 clips activated |

No lane produced an extra or duplicate activation. The adverse source set is
eleven recordings from ten LibriSpeech speaker IDs; corruptions are generated,
not real rooms or playback recordings. CUDA detected one more 0 dB event than
the prior Intel CPU run, but the hardware/provider build differs; this small
observation does not establish an accuracy advantage or replace the adverse
quality gate.

Individual reports and resources: [adverse](wake-adverse.json),
[near-match](wake-near.json), [synthetic](wake-synthetic.json),
[negative smoke](wake-negative.json). Input manifests are the unchanged pins
from the [broader qualification](../2026-09-16-qualification/RESULTS.md) and
[default corpus](../2026-09-16-default-corpus/RESULTS.md) reports.

## Full negative corpus

The same hash-pinned **5,557 recordings / 10.738754 hours** used in the CPU run
completed on CUDA with **zero false activations** for `light up` and `computer`.
Placement was verified and fallback stayed disabled. The run took
556.27 wall seconds and 724.53 process CPU seconds;
sampled process-group RSS peaked at 700.1 MiB. These are accelerated
file-processing resources, not real-time microphone duty or idle power.

[Aggregate report/resources](full-negative.json) records the raw-report hash.
The [unchanged full manifest](../2026-09-16-qualification/long-negative.manifest.json)
retains every input hash. Full per-file results are retained in the local/remote
qualification artifacts. This is candidate-only device evidence; zero observed
events in a finite, correlated read-speech corpus does not imply a zero real-world
false-activation rate or extend the result to arbitrary wake phrases.

## Resources and reproduction

Sampled process-group RSS peaked at 638.3 MiB across the four small-corpus runs.
It sums resident pages every 200 ms, potentially double-counting shared pages,
and excludes device memory. One-second `nvidia-smi` samples observed Omawake
processes using 11–342 MiB of GPU memory; GB10's unified memory means these
numbers must not be summed as independent physical allocations.

The combined wake/speech telemetry window recorded 10.30–29.05 W whole-GPU
power across 78 samples. SGLang was resident: these are shared-device readings,
not application-attributed energy or idle power. [Telemetry](gpu.csv) and
[process observations](processes.csv) are retained. No uncontended speedup or
new automatic accelerator recommendation is claimed.

Use the committed [adverse config](wake-adverse.toml),
[negative config](wake-negative.toml) or [synthetic config](wake-synthetic.toml)
with the pinned assets, adjusting filesystem paths only, then run
`scripts/measure-command.py --out /new/result -- /path/omawake --config CONFIG evaluate MANIFEST`.
The adverse and near-match manifests reproduce the earlier generated corpus;
the negative/synthetic manifest paths can be restored from the pinned fixtures.

Real-room/TV/music, distance/playback, multilingual positives, controlled repeat
performance and microphone-idle power remain open. **HIP stays pending because
the maintainer has no HIP hardware.** This evidence applies to this GB10 provider
build, not every CUDA architecture or GPU.
