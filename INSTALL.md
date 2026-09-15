# Installing Omawake

Release archives contain one runtime-neutral Rust executable and the compact
CPU `libaudiocpp` provider. Linux x86-64 and aarch64 builds require glibc 2.35
or newer.

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omawake-0.0.1-linux-x86_64.tar.xz
cd omawake-0.0.1-linux-x86_64
./omawake setup
```

The guided terminal flow discovers the packaged provider, downloads and
verifies the pinned Moonshine and Silero assets, runs a silent file-only proof,
and then saves the config. The initial mapping listens for `Computer` and runs
`notify-send`; replace it with `omawake wake-word add` and `remove`. Full setup
also installs a desktop launcher. It does not install or enable a service or a
vendor runtime.

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

Run `omawake daemon` directly for an on-demand session. Install the optional
user service only with `omawake setup systemd`; `--no-start` installs and
enables it without starting it. Distribution packages may install the disabled
unit from `packaging/systemd/omawake.service` without enabling it.

For acceleration, install a complete audio.cpp CUDA, Vulkan, or HIP provider,
or a complete OpenVINO GenAI runtime, then point `setup runtime --dir` at it.
See [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md).

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
select its complete build directory in `omawake setup runtime`.
