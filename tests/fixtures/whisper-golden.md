# Whisper feature regression fixture

`whisper-golden.wav` is a project-generated, 12,000-sample, 16 kHz mono chirp
plus a 1,200 Hz sinusoid, quantized to signed 16-bit PCM. It contains no speech.

`whisper-golden-mel-80frames.f32` contains the first 80 frames of each of 80 mel
bands, little-endian f32. Frames 80 through 2999 equal frame 79 in their band
because the remaining input is zero padding. The regression test checks all
240,000 resulting feature values, including that tail.

The independent reference used NumPy float64 FFT and OpenAI Whisper's original
`mel_filters.npz` 80-band table (SHA-256
`7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c`).
Specification: <https://github.com/openai/whisper/blob/main/whisper/audio.py>.
The input was padded to 480,000 samples, reflect-padded by 200 samples,
windowed with periodic Hann400 at hop160, and transformed to power spectra.
After applying the original mel table, values were log10-clamped to a range of
8, then normalized with `(log + 4) / 4`. The last STFT frame was dropped.

These generated fixtures are covered by the repository's MIT license. No
third-party source code or model weights are included in these fixture files.
