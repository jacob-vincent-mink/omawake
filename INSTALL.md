# Installing Omawake

Release archives contain one runtime-neutral Rust executable and the compact
CPU `libaudiocpp` provider. Linux x86-64 and aarch64 builds require glibc 2.35
or newer.

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omawake-0.0.3-linux-x86_64.tar.xz
cd omawake-0.0.3-linux-x86_64
./omawake setup
```

The setup home starts a guided flow that discovers the packaged provider,
downloads and verifies the pinned Moonshine and Silero assets, runs a silent
file-only proof, and then saves the config. The initial mapping listens for
`Computer` and runs `notify-send`; review or replace this example in **Wake
words & actions**. Full setup also installs a desktop launcher that reopens
setup home. It does not install or enable a service or a vendor runtime.

From setup home you can change the microphone, runtime, model, wake phrases,
actions, and aliases; test recognition without executing actions; and run
checks. Choose **Background service** explicitly to install/start, stop,
restart, or uninstall the systemd user service.

To install for one user:

```bash
install -Dm755 omawake "$HOME/.local/bin/omawake"
mkdir -p "$HOME/.local/lib/omawake"
cp -a lib/. "$HOME/.local/lib/omawake/"
omawake setup
```

Focused and offline setup are also available:

```bash
omawake setup runtime
omawake setup model --list
omawake setup all --source-dir /absolute/directory/with/both/catalog-assets
omawake setup check
```

Run `omawake daemon` directly for an on-demand session. The optional user
service can also be installed with `omawake setup systemd`; `--no-start` installs
and enables it without starting it. Distribution packages may install the disabled
unit from `packaging/systemd/omawake.service` without enabling it.

For acceleration, install a complete audio.cpp CUDA, Vulkan, or HIP provider,
or a complete OpenVINO GenAI runtime, then choose the runtime, device, provider
directory, and model in `omawake setup`. For focused command-line selection,
include `--runtime` and `--device` as well as `--dir`; changing the directory
alone does not select a different runtime. See the complete commands and model
prerequisites in [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md).

The guided picker recommends CUDA, Intel NPU, Intel GPU, Vulkan, or CPU in that
order when both matching hardware and a provider are detected. It reports
hardware, provider availability, and the pending model proof separately.
Existing manual selections stay preselected. If the optional service is
already active, a successful focused config, runtime, model, or wake-word edit
restarts it once; failed startup restores the previous config and daemon.
Resume a manually paused service before editing. Setup also refuses automatic
restarts when the effective systemd unit has unrecognized overrides.

## Build from source

Install stable Rust, CMake, a C++ compiler, `pkg-config`, and ALSA development
headers:

```bash
cargo build --release --locked
cargo test --all-targets --locked -- --test-threads=1
```

The Rust executable deliberately has no inference library link-time dependency.
For a runnable default provider, check out audio.cpp at
`e9ff20042ec85af960a720368c6927cda19ad65f` and run:

```bash
scripts/build-default-audiocpp-provider.sh /path/to/audio.cpp /path/to/build
```

Copy `build/bin/libaudiocpp.so.0.1.0` beside the executable under `lib/`, or
run `omawake setup` and select its complete build directory together with a
model. Focused `setup runtime --apply` requires the compatible model to be
installed already.

## Audio devices

Run `omawake setup audio` to select and test the application’s audio device.
Pinned routing requires PipeWire’s `pw-dump` and `pw-record` (Omawake) or
`pw-play` (Omaspeak), supplied by `pipewire` and `pipewire-audio` on Arch.
See [Audio device selection](docs/AUDIO-DEVICES.md) for configuration,
service restart behavior, discovery JSON, and disconnect recovery.
