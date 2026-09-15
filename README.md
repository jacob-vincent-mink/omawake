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
tar -xJf omawake-0.0.1-rc.2-linux-x86_64.tar.xz
cd omawake-0.0.1-rc.2-linux-x86_64
./omawake setup
```

Use the arrow keys and Enter to choose a provider and model, review the plan,
and apply it. The default release provider is audio.cpp on CPU. Setup downloads
and verifies these two pinned MIT-licensed assets as one model profile:

- Moonshine Streaming Tiny Q8_0, 60,407,904 bytes
- Silero VAD 6.2.1, 1,239,748 bytes

The install is atomic: both assets, their checksums, provenance manifest, and
license notices must verify before the new model directory becomes active.
Setup then initializes the provider and both models with a silent file-only
proof before saving the config.

Setup installs the optional desktop settings launcher. It does **not** install
or start a systemd service. Run on demand with `omawake daemon`, or explicitly
install the user service later with `omawake setup systemd`.

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

`setup runtime` discovers and probes a complete `libaudiocpp` provider. Point
to a provider directory explicitly when testing a source build:

```bash
omawake setup runtime \
  --runtime default --device cpu --dir /absolute/path/to/provider --apply
```

Ordinary setup does not install system runtimes. OpenVINO and other accelerated
providers only become selectable after a complete provider has been qualified;
the setup screen reports unavailable choices and their status rather than
saving an unusable configuration.

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
an action and reopens it after the configured cooldown. Actions do not pass
through a shell.

```bash
omawake wake-word add --id computer --phrase Computer -- notify-send "Wake word heard"
omawake wake-word remove computer
```

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
Silero VAD are MIT licensed; their required notices are written beside every
installed model profile.
