# Default wake corpus checks, 2026-09-16

Two existing CPU defaults were evaluated using pinned WAV files: audio.cpp
Moonshine Streaming Tiny Q8_0/Silero 6.2.1 and whisper.cpp 1.9.3 Whisper Base.en
GGML/Silero 6.2.0. Both used two threads and fallback disabled. Configuration,
cache, state and runtime files were isolated. No microphone, playback or mapped
action ran; the file-only evaluator does not execute commands.

| Profile | Human “light up” | Synthetic “hey jarvis” | False activations on 100 negative clips |
|---|---:|---:|---:|
| Moonshine/audio.cpp CPU | 2/2 | 59/60 | 0 over 0.250689 hours |
| Whisper Base.en/whisper.cpp CPU | 2/2 | 56/60 | 0 over 0.250689 hours |

There were no duplicate or wrong-ID detections. Moonshine retains the default
recommendation; the Whisper bundle fills whisper.cpp's backend-default contract
and does not earn a global default promotion here. On the shared host, synthetic
positive file p95 was 99 ms for Moonshine and 1,654 ms for Whisper; negative
aggregate RTF was 0.042 and 0.361 respectively. These are descriptive observations
from development builds while other work ran, not a matched release performance
regression gate or an uncontended hardware benchmark.

## Corpus provenance and reproducibility

[Metrics and hashes](metrics.json) record the binaries/providers, model
provenance manifests, configurations, reports and prediction fingerprints.
The [human](human.manifest.json), [synthetic](synthetic.manifest.json), and
[negative](negative.manifest.json) manifests retain exact sample hashes,
annotations and source metadata. Numeric filenames represent staged copies;
no audio is redistributed. Human/negative clips come from the existing
LibriSpeech corpus (source/license in each manifest); the two positive clip IDs
identify the “light up” examples from the previously assembled human corpus.
The synthetic set is the existing 60-clip Piper “hey jarvis” set. Use the
negative set only with its annotated “light up” phrase, not unrelated keywords.

To reproduce, provision the catalog-pinned model/library profile, stage the
hash-matching WAVs beside the appropriate manifest, configure only its keyword
(`light-up` / `light up`, or `hey-jarvis` / `hey jarvis`), and run:

```sh
omawake --config /isolated/profile.toml evaluate /corpus/manifest.json
```

Two human positives and fifteen minutes of negatives are insufficient for
continuous-listening qualification. Real multi-speaker/near-match/adverse/echo
coverage, longer negatives, matched repeated latency/memory runs, idle power and
additional runtime/device proof remain outstanding. This does not qualify HIP
or update existing accelerator recommendations.

## Real offline installation and interruption

A separate clean-XDG smoke test copied the complete pinned Whisper/VAD bundle
through the production offline installer, invalidated only its temporary
provenance record to trigger repair, then sent SIGINT after staging began.
The installer returned cancellation, removed staging and retained the previous
weights/provenance bytes. A subsequent repair succeeded, proving the lock was
released. No live configuration was changed. See [installer evidence](installer.json).
