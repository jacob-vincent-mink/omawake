# Broader default qualification, 2026-09-16

These file-only checks extend W01/W06. They do **not** promote a new default or
finish continuous-listening qualification. Live configuration, services and
caches were left unchanged; no microphone or playback device was opened.

Optimized reference `af25056` and candidate `5f25f0e` use the same installed
Moonshine Streaming Tiny Q8_0/Silero provider, two CPU threads and fallback
`error`. [Environment and hashes](environment.json) identify binaries, configs,
provider and installed-model manifests. Parallel qualification lanes used
separate CPU affinities on a shared Intel host; thermal state, memory bandwidth
and unrelated workloads were uncontrolled. Timing is descriptive, not a passed
10% regression gate.

## Real speech with generated adverse conditions

Eleven LibriSpeech recordings from ten speaker IDs contain twelve expected
phrase events: `light up`, `lovely child`, and `forever`. Reference transcripts
supply phrase-level labels, without event timestamps. Each recording appears
clean, with Gaussian white noise at 10 dB and 0 dB whole-clip SNR, and with echoes
at 120 ms (gain 0.45) and 240 ms (gain 0.25). Common attenuation prevents clipping
while preserving the mix. This is generated corruption, not recorded rooms,
TV/music, microphone distance or acoustic playback echo.

| Backend / device | Clean | Noise 10 dB | Noise 0 dB | Generated echo |
|---|---:|---:|---:|---:|
| Moonshine CPU reference | 12/12 | 11/12 | 5/12 | 8/12 |
| Moonshine CPU candidate | 12/12 | 11/12 | 5/12 | 8/12 |
| Whisper OpenVINO GPU candidate | 12/12 | 12/12 | 8/12 | 12/12 |
| Whisper OpenVINO NPU candidate | 12/12 | 12/12 | 8/12 | 12/12 |

Reference/candidate Moonshine keyword sequences are identical on every clip;
all four lanes have zero extra, wrong-ID or duplicate detections. OpenVINO
reports selected-device initialization with CPU fallback disabled; Silero VAD
still runs on CPU by design. The stronger Whisper results here are an observation
on this small corpus, not enough evidence to switch defaults.

[Full evaluations](evaluations.json), [source pins](source-positive.manifest.json)
and [generated pins/recipes](adverse.manifest.json) retain individual misses.
To reproduce, restore the source WAVs under their manifest-relative filenames,
then run Python 3.11+:

```sh
python3 scripts/build-adverse-corpus.py /path/source-positive.manifest.json \
  --out /new/adverse-corpus
python3 scripts/measure-command.py --out /new/measurement -- \
  /path/omawake --config /path/isolated.toml evaluate /new/adverse-corpus/manifest.json
```

Use the same model/provider pins, fallback `error`, and the three listed phrases.
For a relocated OpenVINO executable, explicitly set `OMAWAKE_AUDIOCPP_LIBRARY`
to the packaged CPU provider for VAD. The first GPU attempt without this path
failed before inference; the recorded completed run includes it.

## Long speech-negative corpus

Moonshine CPU candidate processed **5,557 LibriSpeech test-clean/test-other clips,
10.738754 hours**, with **zero false activations** for `light up` and `computer`.
Fallback was disabled and CPU placement verified. The reference transcripts
contain no configured phrase in this set; the two known `light up` positives
(`1089-134686-0002`, `6128-63240-0023`) are excluded. This is candidate-only long-run
evidence, not a full-length reference/candidate comparison or proof for arbitrary
phrases. Zero events in this finite, correlated read-speech corpus does not imply
a zero real-world false-activation rate.

[Aggregate report and resource measurements](long-negative.json) and
[all file/hash pins](long-negative.manifest.json) retain corpus identity. Full
per-file results remain in the local qualification archive; their SHA-256 is in
the aggregate. The run took 1,621.38 wall seconds and 2,879.20 process CPU seconds;
sampled process-group RSS peaked at 272.82 MiB. This accelerated file run is not
real-time continuous microphone listening or a power measurement.

## Near matches

Six [authored confusables](../../near-match-cases.json), each synthesized with
Supertonic 3 M1 and F1, produced zero activations for `light up` and `computer` on
reference/candidate Moonshine CPU and candidate OpenVINO GPU/NPU. Examples include
`light bulb`, `line up`, `computerized`, `compute` and `commuter`. The total is only
28.29 seconds: this is a synthetic smoke check, not independent human negative
speech. Labels come from the synthesis prompt, not a listening annotation.

[Near-match pins](near-match.manifest.json) and full evaluations retain inputs
and outcomes. Generation used Omaspeak's `scripts/measure-model-corpus.py` with
this corpus, one iteration, then FFmpeg mono PCM16/16 kHz resampling. Model
license is Open RAIL-M; this manifest does not assert a separate output license.

## Offline silence and noise-floor checks

Ten minutes each of PCM16/16 kHz silence and seeded Gaussian noise with RMS
approximately -60 dBFS produced zero verifier transcripts and zero detections.
A known positive control produced the expected transcript and `light-up` event,
confirming diagnostics were enabled. [Resource and diagnostic evidence](idle.json)
retains exact output and fixture hashes.

Silence consumed 10.80 process CPU seconds over 7.05 wall seconds; noise consumed
9.07 CPU seconds over 6.13 wall seconds, each processing 600 seconds of audio.
Sampled group RSS peaked at 141.3 and 143.6 MiB respectively.

These are accelerated file-processing checks of VAD gating. They do not measure
real-time microphone capture, daemon scheduling or idle watts. `wait4` records
process CPU time and maximum process RSS; sampled process-group RSS sums resident
pages every 200 ms and can double-count shared pages. It excludes device memory
and is not unique physical memory.

## Remaining gates and execution limits

Human near-matches, independent room/distance/playback recordings, TV/music and
multilingual positives remain needed. More speakers, repeated device/resource
comparisons and real-time idle measurements are required before hardware or
default promotion. This corpus is too small to establish adverse-audio quality.

RAPL energy counters are unreadable and noninteractive sudo requires a password;
no watts are claimed. The host is on AC with a full battery, so battery readings
cannot substitute. Configured CUDA hosts could not be reached (SANDERS DNS;
BENSON SSH timeout); no local NVIDIA or AMD/HIP device is available. Existing
CUDA/Vulkan evidence is unchanged, and HIP remains unqualified.

An isolated Omaspeak NPU cache copy hit the `/tmp` user quota; a subsequent
cache-status process received SIGBUS with a loader-only stack. Its exact cause
is unresolved. The same binary hash and complete copied caches worked from the
workspace filesystem. Only the incomplete test copy was removed. No application
cache corruption or application-code defect is established by that failed run.

## Reproduce the offline idle fixtures

The files use seed 20260916, 600 seconds, mono PCM16 at 16 kHz. Generate them in
a new directory; then run `omawake test --audio FILE --show-transcripts` with
isolated XDG directories and the recorded Moonshine configuration. The positive
control is `1089-134686-0002-clean.wav` from the generated adverse set.

```python
from array import array
import random
import sys
import wave

for name in ("silence", "noise-floor"):
    rng = random.Random(20260916)
    with wave.open(name + ".wav", "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        for _ in range(600):
            samples = array("h", (round(rng.gauss(0, 32.767))
                if name == "noise-floor" else 0 for _ in range(16000)))
            if sys.byteorder != "little":
                samples.byteswap()
            wav.writeframes(samples.tobytes())
```
