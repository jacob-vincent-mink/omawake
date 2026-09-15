# Installing Omawake

Omawake 0.0.1-rc supports Linux x86-64. The release archive contains one
runtime-neutral executable and a bundled CPU runtime. OpenVINO and CUDA remain
external runtime choices configured after installation.

There are no 0.0.1-rc prebuilt artifacts for aarch64, macOS, or Windows. CUDA has
also been validated from source on aarch64 NVIDIA GB10 hardware.

## Release archive

Download `omawake-0.0.1-rc-linux-x86_64.tar.xz` and `SHA256SUMS.txt` from the
[v0.0.1-rc release](https://github.com/jacob-vincent-mink/omawake/releases/tag/v0.0.1-rc),
then verify and unpack it:

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omawake-0.0.1-rc-linux-x86_64.tar.xz
cd omawake-0.0.1-rc-linux-x86_64
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

The current catalog does not automatically download its GigaSpeech wake-word
model because the upstream model license is unclear. Setup asks for a local
copy of the [pinned sherpa-onnx archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2)
obtained under terms you have verified:

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

Install the vendor runtime and an ABI-matched ONNX Runtime 1.29 provider stack,
then point setup at its root or library directory:

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

See [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md) for tested Arch/Omarchy Intel
iGPU and NPU packages and complete OpenVINO and CUDA runtime bundle recipes.

## Build from source

Install a Rust toolchain, `pkg-config`, and ALSA development headers, then run:

```bash
git clone https://github.com/jacob-vincent-mink/omawake.git
cd omawake
cargo build --release --locked
cargo test --locked
```

The source-built executable is runtime-neutral and does not contain Omawake's
CPU runtime or extended sherpa companion. Copy the release `lib/` directory
beside the executable before running setup; accelerator setup can then replace
the ONNX Runtime core/provider while retaining Omawake's companion library.
