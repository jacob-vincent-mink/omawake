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
clips. That harness matched the substring `jarvis`, rather than requiring the
complete `hey jarvis` phrase, so its apparent parity with fixed-phrase
detectors is not a like-for-like product result. Until reproduced through
Omawake's complete phrase matcher, it is model-selection evidence only. Tiny
verifier profiles must not be recommended without passing the same positive
and negative gates.

Those 60 WAV files have now also been replayed through Omawake. With the
complete configured phrase `hey jarvis`, the default Moonshine verifier found
59/60 after adding 320 ms of bounded post-roll (98.33% recall, 99.16% F1,
aggregate real-time factor 0.031). The remaining clip transcribes as
`Page Arvis` even when the complete WAV is sent directly to Moonshine. Before
post-roll, two other clips lost the final consonant when cut exactly at
Silero's speech-end timestamp; direct ASR recovered them with trailing
context. This is why the native ring retains post-roll instead of compensating
with fuzzy phrase matching.

OpenVINO Whisper Base.en found only 51/60 when required to match the complete
`hey jarvis` phrase: all nine misses transcribed as `Hate Jarvis`. Configuring
the intentional shorter phrase `jarvis` found 60/60. This reproduces the
research harness result and confirms its boundary: changing the configured
phrase changes product behavior, so the shorter-phrase score cannot be
reported as complete-phrase recall. The checksum-pinned input manifest is
`3c30d570a2b8f7a6ed67eddc6f056cd775379aff13a57e5e4b4afdd9bbafa0de`;
the Moonshine complete-phrase, OpenVINO complete-phrase, and OpenVINO
shorter-phrase report hashes are respectively
`0e247994b811247f3473b4e1cc60169144ade6226a8ef20c03b1b9bc0c6a7039`,
`33a8e74733c776e0674c4b304c6296f21a5a8befa26a5f6904d091e7d16baaf2`,
and `c280f380029f57bd4eb82fc5e53e18758c2a4889216013955fa5b41475a205cd`.

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
