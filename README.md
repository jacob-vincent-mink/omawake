<p align="center">
  <img alt="Omawake: a stylized ear" src="assets/omawake-mark-on-light.svg" width="160">
</p>

# Omawake

Omawake is a local wake-phrase daemon written in Rust. Silero VAD gates a
bounded audio buffer, Moonshine Streaming Tiny transcribes each completed
utterance, and Omawake maps whole-phrase matches to direct argument-vector
actions. Recorded and live audio use the same detector.

The default provider is [audio.cpp](https://github.com/0xShug0/audio.cpp).
Omawake dynamically loads its public C ABI as a library; it never invokes the
audio.cpp CLI. The hidden `__audiocpp-worker` command is Omawake re-executing
its own binary for crash isolation and warm sessions.

## Install and guided setup

Linux x86-64 and aarch64 releases require glibc 2.35 or newer. Verify and
unpack a release, then start the guided terminal setup:

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omawake-0.0.3-linux-x86_64.tar.xz
cd omawake-0.0.3-linux-x86_64
./omawake setup
```

The Ratatui setup home shows the current runtime, model, microphone, wake words, and service state. Use arrow keys or `j`/`k` to move, Enter to open a choice, and Esc or `q` to go back. **Start guided setup** chooses a provider, model, and microphone, reviews the plan, and applies it. The default release provider is audio.cpp on CPU. Setup downloads and verifies these two pinned MIT-licensed assets as one model profile:

- Moonshine Streaming Tiny Q8_0, 60,407,904 bytes
- Silero VAD 6.2.1, 1,239,748 bytes

The install is atomic: both assets, their checksums, provenance manifest, and
license notices must verify before the new model directory becomes active.
Setup then initializes the provider and both models with a silent file-only
proof before saving the config.

Setup installs the optional desktop settings launcher, which opens the same setup home. It does **not** install or start a systemd service during the guided flow. From setup home, review the example `Computer` wake word and its action, choose **Teach a wake word** if recognition needs examples, test the microphone, then use **Try recognition** to listen for five seconds without executing actions. **Background service** lets you explicitly install and start the user service, stop or restart it, check its status, or uninstall it. You can also run on demand with `omawake daemon` or use `omawake setup systemd` from the shell.

The **Wake words & actions** screen adds phrases and lets you change the phrase, program, individual arguments, aliases, and enabled state. An active trained detector keeps its spoken phrase and does not use transcript aliases. To replace a trained phrase, add a new wake word, teach it, then remove the old one. If you launched the daemon directly, stop it with `omawake stop` before editing and start it again afterward; setup refuses edits it cannot apply to the running process.

**Advanced settings** covers CPU threads, cooldown, and capture queue. Changes to an active setup-managed service restart it automatically; setup rolls back the configuration if the restart fails. Guided setup, microphone selection, teaching, and the recognition test hold the running daemon paused while recording or testing. This also works for a manually launched daemon, preserves a user's existing pause, and releases the hold if setup exits. Setup leaves unrecognized local user units alone. **Run setup checks** verifies the full installation after each step.

These focused commands expose the same choices without the full guided flow:

```bash
omawake setup runtime
omawake setup model --list
omawake setup check
```

For an offline install, put both exact catalog filenames in one directory:

```bash
omawake setup all --source-dir /absolute/path/to/assets
```

`setup runtime` discovers and probes a complete `libaudiocpp` provider. On
a fresh source build, run `omawake setup` and choose the provider directory
and model together. Once the compatible model is installed, focused runtime
selection can validate and apply a different provider directory:

```bash
omawake setup runtime \
  --runtime default --device cpu --dir /absolute/path/to/provider --apply
```

CUDA, Vulkan, and HIP also accept `--device-id N` for a zero-based GPU index.

Ordinary setup does not install system runtimes. OpenVINO GenAI and audio.cpp
CUDA, Vulkan, and HIP choices ask for a complete user-supplied provider
directory. Setup probes the provider and requested device before it saves an
accelerated configuration.

On a new configuration, the guided picker recommends the first usable choice
in this order: CUDA, Intel NPU through OpenVINO, Intel GPU through OpenVINO,
Vulkan, then the packaged CPU provider. Hardware detection and provider
discovery are shown separately; only Apply runs the isolated, model-backed
proof. An existing runtime remains selected by default so setup never silently
replaces a manual choice.

## Use

The following examples assume `omawake` is on `PATH`. Prefix commands with
`./` when running directly from an unpacked release directory.

```bash
omawake test --audio test.wav --json
omawake benchmark --warmup 2 --iterations 10 test.wav > benchmark.json
omawake daemon
omawake status --json
omawake pause
omawake resume
omawake stop
```

The recognizer remains loaded while the daemon releases the microphone before
launching an action and reopens it after the configured cooldown. Actions are
started in the background with detached standard streams and reaped when they
exit, so a long-running command cannot block detection. Actions do not pass
through a shell.

```bash
omawake wake-word add --id computer --phrase Computer -- notify-send "Wake word heard"
omawake wake-word remove computer
```

When the optional user service is already active, successful `config`,
`wake-word`, focused runtime, and model activation changes restart it once.
Omawake restores the prior config and explicitly restarts the prior daemon if
the updated daemon fails to start. A custom `--config` never restarts a unit
that points at another file.

For names or coined words that the verifier spells inconsistently, use
**Teach a wake word** in `omawake setup`, or run:

```bash
omawake word onboard jarvis
# Import examples without opening the microphone:
omawake word onboard jarvis --audio example-1.wav --audio example-2.wav --json
```

The guided flow first offers **Whisper spellings**, **Trainable KWS**, or
**Trainable KWS with Omaspeak assistance**, without an `--engine` flag.
The Whisper path records examples, shows each observed spelling, and lets you
approve exact aliases with arrow keys and Enter. Nothing changes until Apply;
no wake-word actions run during onboarding. Recordings are discarded unless you
explicitly choose to keep them. Existing `add-alias` and `remove-alias` commands
remain available for manual editing.

An experimental frozen-encoder head can handle phrases that transcription does
not represent reliably. Transcript words and trained words can run together;
compatible heads share an encoder. Applying a trained head preserves aliases and
pins its model profile, so changing the default backend does not erase it.
Choose **Trainable KWS — learn from my voice** at the start, then select
the training device and the device where the finished detector will run
(CPU, or experimental Intel iGPU/NPU). Configured OpenVINO model assets are reused
and the deployment profile is created automatically. You can also switch to
training after reviewing spellings. Onboarding reuses any examples already collected, collects at least
10 wake-phrase and 10 other-speech recordings,
and prepares separate training, calibration, and held-out examples automatically.
You review validation results before Apply; no hand-written dataset is needed.
The first implementation uses the direct
OpenVINO Whisper encoder; **CPU has a functional file-based proof**. A small
[CPU/iGPU/NPU parity check](docs/validation/ENROLLMENT-ACCELERATOR-PROOF.md) also
passes; broader background-speech accuracy testing is still pending.
Optional **Omaspeak-assisted training** adds pronunciation-reviewed synthetic
examples while keeping calibration and validation human-only. If Omaspeak is
missing, onboarding offers an explicit user-local release installation. Resume a
retained session with `word onboard --dataset manifest.json` to reuse training
clips and collect fresh evaluation recordings. To add more human examples, run
`omawake word onboard WORD` and choose **Add positive and negative examples**;
saved datasets are discovered automatically. Training and deployment can use
different devices without manually creating named profiles:

```bash
omawake word train jarvis --reuse-recordings --training-device cpu --run-device npu --apply
```

Training features are extracted on the training device; the small classifier is
fitted on CPU. Calibration and held-out validation run on the deployment device
before activation. A failed destination check preserves the current detector.
See [wake-word enrollment](docs/architecture/WAKE-WORD-ENROLLMENT.md) for commands,
recording retention, profiles, retraining, and the limits of local validation.

For trained words, inspect or override the detection threshold and opt in to
local audio history when diagnosing false activations:

```bash
omawake word threshold jarvis              # show calibrated/effective threshold
omawake word threshold jarvis 0.8          # example override; tune to your recordings
omawake word threshold jarvis auto         # restore the calibrated threshold
omawake word history jarvis enable --max-events 100
omawake word history jarvis list
omawake word history jarvis play EVENT_ID
omawake word history jarvis label EVENT_ID false-positive
omawake word onboard jarvis                # offers reviewed clips during retraining
omawake word history jarvis disable        # stop capture; preserve existing clips
omawake word history jarvis clear          # delete captured clips and labels
```

History is **off by default**. It saves only live trained detections, locally,
with scores and the triggering speech clip. Oldest events expire at the limit,
including labeled events. Higher thresholds reduce activations but can miss
real wake phrases; scores are not calibrated probabilities. Labels do not
silently change the active model: guided retraining uses them as training
examples, collects fresh evaluation clips, and requires validation before Apply.

Use `evaluate` for reproducible accuracy and false-activation measurements over
a labeled WAV corpus:

```bash
omawake evaluate path/to/manifest.json > report.json
```

The manifest keeps audio paths relative to its own directory, pins every file
by SHA-256, and may label whole clips or timestamped wake-word events. Expected
phrase IDs must be enabled in the selected config. See the versioned
[manifest schema](schemas/evaluation-manifest-v1.schema.json) and
[report schema](schemas/evaluation-report-v1.schema.json).
The [rc.3 hardware results](benchmarks/results/2026-09-15-rc3/RESULTS.md)
compare default CPU, OpenVINO CPU/iGPU/NPU, and CUDA with file-only input.

## Default provider and models

The initial qualified profile is
`moonshine-streaming-tiny-q8_0-silero-v6.2.1`. The default configuration is
`backend.kind = "audiocpp"`, `runtime = "default"`, and `device = "cpu"`.
The release provider holds reusable native Silero and Moonshine sessions behind
audio.cpp's versioned public C ABI. Omawake owns buffering, endpoint handling,
phrase matching, cooldowns, and actions in Rust.

Moonshine is a speech verifier rather than a fixed keyword classifier, so the
English profile needs no per-phrase training for text phrases within its
language coverage. Phrase matching is case-insensitive,
punctuation-insensitive, Unicode-normalized, and ignores ASR word-boundary
variation while requiring the same normalized characters. See the
[qualification plan](docs/architecture/EVALUATION.md) for the positive,
false-activation, adverse-audio, and idle-cost gates.

The catalog pins:

- the original Moonshine model revision
  `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`;
- the converted GGUF repository revision
  `6d5436fc85f7a20c2e9f4e472b7f3a532f686444`;
- the exact upstream Silero file at revision
  `7e30209a3e901f9842f81b225f3e93d8199902b1`.

Installed `PROVENANCE.json`, `.omawake-model.json`, and `LICENSES/` files keep
the original and converted sources distinct. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[native-provider architecture note](docs/architecture/NATIVE-PROVIDERS.md).

## Build

```bash
cargo build --release --locked
cargo test --locked
```

Building Omawake needs stable Rust, `pkg-config`, and ALSA development headers.
The Rust executable has no link-time dependency on audio.cpp. A runnable source
build also needs a compatible `libaudiocpp` ABI 0.1 provider. The release build
script pins the audio.cpp source revision and produces the CPU provider that is
packaged with Omawake:

```bash
scripts/build-default-audiocpp-provider.sh \
  /path/to/audio.cpp /path/to/build-directory
```

Omawake is MIT licensed. audio.cpp is Apache-2.0. Moonshine Streaming Tiny and
Silero VAD are MIT licensed. The optional OpenAI Whisper `base.en` weights and
Intel OpenVINO conversion are Apache-2.0; their required notices are written
beside every installed model profile.

## Audio devices

Run `omawake setup audio` to select and test the application’s audio device.
Pinned routing requires PipeWire’s `pw-dump` and `pw-record` (Omawake) or
`pw-play` (Omaspeak), supplied by `pipewire` and `pipewire-audio` on Arch.
See [Audio device selection](docs/AUDIO-DEVICES.md) for configuration,
service restart behavior, discovery JSON, and disconnect recovery.
