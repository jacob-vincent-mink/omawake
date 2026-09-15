# Wake verifier qualification

Omawake uses a voice-activity gate followed by speech recognition and
provider-independent phrase matching. A configured phrase requires no
per-phrase model training, but the selected verifier still determines language
coverage and recognition quality. Every catalog profile must therefore declare
its languages. The first audio.cpp Moonshine and OpenVINO Whisper `base.en`
profiles are English profiles.

## What must be measured

A clean speech corpus is useful for repeatable regression testing, but it is a
weak false-activation test. Release qualification has four separate lanes:

1. **Positive recall.** Use real speech plus varied multi-voice synthetic
   speech, fast and short prosody, multiple accents, and at least one phrase
   that was not used to train or select the verifier. Report events, misses,
   wrong IDs, duplicates, precision, recall, and F1.
2. **Speech negatives.** Use at least ten hours whose reference transcripts do
   not contain the configured phrases. Pin every file and exclude real phrase
   occurrences before counting negative hours.
3. **Adverse negatives.** Measure TV or podcast speech, music, room noise,
   babble, competing speakers, and far-field or reverberant audio separately.
   Do not compare a clean-corpus false-activation rate with a published noisy
   benchmark rate.
4. **Idle operation.** Feed long silence and a realistic microphone noise floor
   without opening an audio output. Record whether ASR was gated off, idle CPU
   time, resident memory, and power where the platform exposes it.

For every lane, preserve a versioned manifest, file hashes, corpus provenance,
application/config hashes, provider identity, model revision, device-placement
evidence, and a prediction fingerprint. Accelerator comparisons use the same
files and matching policy.

## Current targeted gate

The checked local positive set contains 11 LibriSpeech clips with 12 events
across `light up`, `lovely child`, and `forever`. It deliberately includes the
ASR spelling variation `for ever`. Phrase matching ignores word boundaries but
otherwise requires exact normalized characters; this lets the same acoustic
phrase survive ASR tokenization without introducing general fuzzy matching.

The long clean-negative set contains 5,557 checksum-pinned LibriSpeech clips
and 10.739 hours of speech. Two clips containing real `light up` utterances are
excluded from its negative hours. Both `light up` and the default `computer`
phrase are enabled in the run.

A separate GB10 experiment contains 60 Piper `hey jarvis` positives from three
voices with varied prosody, 2.4 hours of clean LibriSpeech negatives, and 1.7
hours of generated noise-floor audio. In that experiment Whisper Tiny missed
4/60 positives while Whisper Base missed 0/60; the VAD accepted all four missed
clips. Until reproduced through Omawake, this is model-selection evidence, not
a release result. Tiny verifier profiles must not be recommended without
passing the same positive and negative gates.

## Latency and resource reporting

Report at least:

- cold provider and model load;
- cached setup load for devices that compile models;
- first request and persistent-session hot request;
- speech-end-to-detection p50 and p95;
- VAD time, verifier time, total time, and real-time factor;
- peak resident memory;
- idle VAD duty cycle and the fraction of audio windows sent to ASR.

An accelerated verifier can have faster inference and a slower cold compile.
Omawake prepares OpenVINO GPU/NPU caches during setup, before saving the config,
so first-inference latency is not presented as setup latency.

## Later speaker enrollment

The provider boundary leaves room for an optional enrolled-speaker check after
VAD and phrase verification and before action execution. That can reject a TV
or bystander that says the correct phrase. It requires its own impostor,
same-speaker, playback, and noisy far-field evaluation and is not part of the
initial release gate.
