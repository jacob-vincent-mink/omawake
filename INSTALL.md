# Installing Omawake

Omawake 0.0.1-rc.1 supports Linux x86-64 and aarch64 with glibc 2.35 or newer.
Each release archive contains one runtime-neutral executable and a bundled CPU
runtime. OpenVINO and CUDA remain external runtime choices configured after
installation.

There are no 0.0.1-rc.1 prebuilt artifacts for macOS or Windows.

## Release archive

Download `omawake-0.0.1-rc.1-linux-x86_64.tar.xz` and `SHA256SUMS.txt` from the
[v0.0.1-rc.1 release](https://github.com/jacob-vincent-mink/omawake/releases/tag/v0.0.1-rc.1),
then verify and unpack it:

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omawake-0.0.1-rc.1-linux-x86_64.tar.xz
cd omawake-0.0.1-rc.1-linux-x86_64
./omawake --version
```

You can run Omawake from the unpacked directory. To install it for one user
while preserving runtime discovery:

```bash
install -Dm755 omawake "$HOME/.local/bin/omawake"
mkdir -p "$HOME/.local/lib/omawake"
cp -a lib/. "$HOME/.local/lib/omawake/"
```

Ensure `$HOME/.local/bin` is on `PATH`, then run the guided setup:

```bash
omawake setup
```

Setup automatically downloads and verifies the Apache-2.0 GigaSpeech wake-word
model from its [pinned publisher archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2).
You can also provide an existing archive:

```bash
omawake setup all --archive /path/to/model.tar.bz2
omawake test --audio /path/to/test.wav --json
```

Setup installs a desktop launcher. It does not install, enable, or start a
systemd service. `test` and `benchmark` load the engine on demand and exit.
`daemon` keeps it resident; `status`, `pause`, `resume`, and `stop` address that
running daemon. Install a user service only when desired with
`omawake setup systemd`.

Distribution packages may place the disabled vendor unit from
`packaging/systemd/omawake.service` under `/usr/lib/systemd/user`. Installing
that file does not enable or start the daemon.

## OpenVINO or CUDA

Install an official ONNX Runtime 1.30 V2 provider plugin and its vendor runtime,
then point setup at the plugin package root or library directory. The provider
package must not include a second ONNX Runtime core:

```bash
omawake setup runtime --runtime openvino --device npu \
  --dir /opt/intel/openvino --apply

omawake setup runtime --runtime cuda --device gpu \
  --dir /opt/omawake-cuda-runtime --apply
```

The probe must pass before configuration is saved. Omawake does not download
or install accelerator runtimes. Use `omawake setup runtime --json` for exact
library, device, and remediation details.

With OpenVINO GPU or NPU, setup compiles the installed model synchronously and
stores reusable blobs below
`${XDG_CACHE_HOME:-$HOME/.cache}/omawake/openvino/<device>`. The first
compilation can take several seconds. If runtime setup precedes model
installation, model setup completes this step before activation.
Some Intel GPU compiler versions can terminate during a fresh cold compile;
Omawake contains that work in a child process and allows up to five total
attempts while the vendor cache is populated. Only signal terminations are
retried. Setup applies the runtime only after a complete detector pass succeeds.

See [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md) for OpenVINO and CUDA package
layout details.

## Build from source

Install a Rust toolchain, `pkg-config`, and ALSA development headers, then run:

```bash
git clone https://github.com/jacob-vincent-mink/omawake.git
cd omawake
cargo build --release --locked
cargo test --locked
```

The source-built executable is runtime-neutral and does not contain ONNX
Runtime. Copy the release `lib/` directory beside the executable, or configure
an exact ONNX Runtime 1.30 core path, before running setup. Optional accelerator
setup adds a provider plugin and vendor libraries while continuing to use that
same core.
