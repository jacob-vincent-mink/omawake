<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/omawake-mark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/omawake-mark-on-light.svg">
    <img alt="Omawake: a stylized ear" src="assets/omawake-mark-on-light.svg" width="160">
  </picture>
</p>

# Omawake

Omawake is a local wake-word daemon written in Rust. It compiles configured phrases with the model's tokenizer, captures audio through CPAL, and maps stable wake-word IDs to direct argument-vector actions. sherpa-onnx is the first registered backend (in-process ONNX keyword spotting); the setup and catalog layers are backend-neutral.

Omawake supports multiple wake words, recorded and live input, `pause`/`resume`/`stop`, JSON status and schema output, and validated config mutation. The recognizer stays loaded while the microphone stream is released before every action and reopened after cooldown.

## Install

The 0.0.1-rc release supports Linux x86-64 with glibc 2.34 or newer and ALSA.
It ships one executable plus a ready-to-use CPU runtime. Download the archive
and `SHA256SUMS.txt` from the
[GitHub release](https://github.com/jacob-vincent-mink/omawake/releases/tag/v0.0.1-rc),
then verify and unpack it:

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omawake-0.0.1-rc-linux-x86_64.tar.xz
cd omawake-0.0.1-rc-linux-x86_64
./omawake setup runtime --json
./omawake setup
```

Move the application to its final location before setup because a desktop
launcher or an explicitly requested service records the executable path. See
[INSTALL.md](INSTALL.md) for a per-user installation, source builds, and
external OpenVINO or CUDA setup.
[ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md) gives complete Intel iGPU/NPU and
NVIDIA CUDA runtime bundle recipes.

> **A model is required and is not bundled.** The sole 0.0.1-rc catalog model is
> the [`sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2)
> archive published with sherpa-onnx. Omawake does not download it because its model
> license is unclear. Inspect the pinned size, hash, and terms with
> `omawake setup model --json`, obtain it under rights you have verified, then
> supply it with `omawake setup all --archive /path/to/model.tar.bz2`.

## Build

```bash
cargo build --release
cargo test
```

Run the source build as `target/release/omawake`. Building requires stable
Rust, a C/C++ toolchain, `pkg-config`, and ALSA development headers. The release
archive supplies the native CPU libraries; a source checkout requires a
compatible external stack or a copied release `lib/` directory.

Omawake does not link ONNX Runtime or sherpa-onnx at build time. The same Rust
executable supports CPU, OpenVINO, and CUDA:

```bash
cargo build --release
omawake setup runtime --runtime openvino --device npu \
  --dir /absolute/path/to/runtime-or-sdk --apply
```

The normal Linux release archive includes a working CPU default under `lib/`:
the official ONNX Runtime 1.29.0 CPU library and its matching extended
sherpa-onnx 1.13.8 C API. An unpacked release therefore needs only a model for
CPU inference. OpenVINO and CUDA setup points the same executable at a
user-supplied, ABI-matched ONNX Runtime core, patched sherpa library, provider
plugin, and vendor runtime. Setup accepts a flat library directory or an SDK
root with libraries under `lib`, `lib64`, or OpenVINO's
`runtime/lib/intel64[/Release]` layout. The release does not bundle acceleration
provider plugins. A successful setup does not establish device placement;
inspect runtime provider evidence before treating NPU or GPU placement as
verified.

[`native/sherpa`](native/sherpa/README.md) builds the patched sherpa-onnx
library against a user-supplied ONNX Runtime 1.29 SDK. CUDA and OpenVINO
providers remain external runtime inputs; Omawake does not build or bundle
their libraries.
The [GB10 CUDA validation](benchmarks/cuda-gb10-2026-09-14.md) records direct
WAV accuracy, Nsight kernel placement, GPU telemetry, and a CPU comparison.

Copy `config.example.toml` to `${XDG_CONFIG_HOME:-$HOME/.config}/omawake/config.toml`, then verify everything with:

```bash
omawake setup check
```

## Setup

Run `omawake setup` in a terminal to open the guided setup. Use the arrow keys and
Enter to choose **Full setup**, **Runtime**, **Model**, or **Check**. Full setup walks
through a compatible inference runtime and device, shows every model with its
install status, backend, family, description, license status, and archive size,
then asks for a licensed local archive when automatic download is unavailable.
It installs the selected model and launcher. It does not install or start a service. If a
service supplied by the user or a package is already active, setup restarts it
after successfully applying the new configuration. The focused setup commands
are interactive too:

```bash
omawake setup runtime   # CPU works from a release; acceleration accepts external libraries
omawake setup runtime --runtime openvino --device npu --dir /opt/oma-runtime --apply
omawake setup model     # browse, download if needed, and activate a model
```

When input is redirected or piped, these commands stay noninteractive and print
the runtime/model catalog with commands that scripts can run. Use `--json` by
itself for a machine-readable model catalog; model actions such as `--download`,
`--set`, and `--verify` are separate invocations.

The equivalent one-command, noninteractive setup from a model archive you are
licensed to use is:

```bash
omawake setup all --archive /path/to/model.tar.bz2
```

It verifies and installs the model, writes `backend.kind` / `model.name` into
the config, installs the launcher, then prints `setup check` results. It does
not install, enable, or start a service. An already-active service is restarted
after a successful apply; an inactive service is left inactive. Options:

```bash
omawake setup all --model <id>          # another catalog model
omawake setup all --archive <path>      # required until a catalog model has verified download terms
omawake setup all --progress-format json
```

Model management:

```bash
omawake setup model --list                          # includes download and license status
omawake setup model --download sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --archive /path/to/model.tar.bz2
omawake setup model --verify sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
omawake setup model --set sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01   # verify an existing install, then activate it
```

`--download` verifies and installs the model (activating it in the config unless `--no-activate`); `--verify` only checks an installed model; `--set` verifies and then activates. The current GigaSpeech KWS archive does not have sufficiently clear model-license terms, so Omawake will not fetch it automatically. Supply a local archive obtained under rights you have established with `--archive <path>`; it is still size + SHA256 verified against the pinned catalog entry. The guided TUI lets you select the model and enter that local archive path. Model files land under `${XDG_DATA_HOME:-$HOME/.local/share}/omawake/models/<id>/`, or set `model.directory` to point elsewhere. `.omawake-model.json` records that the archive was user supplied together with its pinned hash and catalog license status. Archives and extracted assets are verified by size and SHA256 (archive, then every required file), and installs are atomic: extraction goes to a staging directory and is renamed into place, with the previous model rolled back if activation fails.

Diagnostics and service management:

```bash
omawake setup check              # config, backend, model, engine, wake words, launcher, optional service status
omawake setup runtime            # supported runtimes, runtime loadability, library paths, device matrix
omawake setup systemd --status   # or: --uninstall, --no-start (no flag installs + starts)
omawake setup menu --status      # or: --uninstall (no flag installs the launcher)
```

Run `omawake daemon` in the foreground. If you explicitly want a systemd user
service, `omawake setup systemd` installs, enables, and starts it; pass
`--no-start` to install and enable it without starting it. Packages and other
downstream integrations can instead provide their own service definition.

```bash
omawake audio-devices
omawake test --audio test.wav --json
omawake benchmark --warmup 2 --iterations 10 test.wav another.wav > benchmark.json
omawake daemon
omawake status --json
omawake pause
omawake resume
omawake stop
```

`benchmark` loads the configured detector once, never opens a microphone or runs
actions, and emits JSON with the model load time, detections and timing for every
measured file iteration, audio duration, real-time factor, and p50/p95 summaries.
The [OpenVINO benchmark harness](scripts/benchmark-openvino.sh) prepares isolated
cold/hot CPU, GPU, and NPU lanes. See the
[Dell XPS 16 report](benchmarks/openvino-dell-xps-2026-09-14.md) for the tested
OpenVINO 2026.2.1 and ONNX Runtime 1.29.0 results and model compatibility note.

Add and remove mappings without editing TOML:

```bash
omawake wake-word add --id computer --phrase Computer -- notify-send "Wake word heard"
omawake wake-word remove computer
# `word` is a shorter alias:
omawake word remove computer
```

Removing the final mapping is allowed so configurations can be rebuilt incrementally. Starting the detector still requires at least one enabled mapping.

Actions are executed directly. Omawake does not insert a shell. Configure `sh -lc` explicitly when shell evaluation is intentional.

## Backends

`backend.runtime` accepts `default`, `openvino`, and `cuda`. Valid device values are:

| Runtime | Devices |
|---|---|
| `default` | `auto`, `cpu` |
| `cuda` | `auto`, `gpu` |
| `openvino` | `auto`, `npu`, `gpu`, `cpu` |

The executable exposes all three runtime choices and loads an external, patched
sherpa-onnx C API plus its exact ONNX Runtime dynamically. It rejects a missing,
version-mismatched, or unpatched stack before model creation. OpenVINO and CUDA
also require a matching execution-provider plugin and the requested hardware
device. With `fallback = "cpu"`, an accelerator initialization failure emits a
warning and is visible in status and test output. Selecting a runtime in TOML
alone never counts as verified placement.

Native vendor libraries can be declared without changing the process-wide shell
environment:

```toml
[backend]
library_dirs = ["/opt/intel/openvino/runtime/lib/intel64", "/opt/oma/runtime/lib"]
```

Relative entries resolve beside Omawake's config file. `OMAWAKE_LIBRARY_PATH`
adds a path-list overlay after configured directories. Exact configured
libraries win over environment and directory discovery. Directory search order
is configured directories, the app environment overlay, then package
directories; the ambient loader remains a final discovery fallback. This makes
the bundled CPU stack automatic while configured external acceleration stacks
win deterministically. Omawake discovers package libraries in the executable
directory, its `lib/` child, or `../lib/omawake`. Before an
engine command (`test`, `benchmark`, or `daemon`) loads the provider on Linux,
Omawake validates these app-owned directories and re-executes itself once with
them prepended to `LD_LIBRARY_PATH`. A sentinel prevents re-exec loops. Config
commands and setup discovery never re-exec, so `omawake setup runtime --json`
can always report configured, environment, package, effective, and missing paths.
Every native check runs in an isolated process with a 20-second timeout. It
verifies ORT 1.29.0, sherpa 1.13.8, the extended sherpa API version, keyword API
exports, and ABI compatibility. Default CPU does not create a plugin runtime;
acceleration additionally requires provider registration and device access.
`setup runtime --json` reports an `inventory`
entry for every runtime/device with support, discovery source, configuration,
loadability, device access, readiness, resolved paths, evidence, errors, and
remediation. Runtime readiness does not prove model execution placement.

Explicit runtime/device/directory arguments preview and probe a candidate. Add
`--apply` to save it after a successful probe. The guided flow probes before its
Apply/Cancel review. Failure and cancellation leave the config unchanged.
Packaged CPU paths stay implicit; external selections pin exact libraries and
dependency directories. Ambient `LD_LIBRARY_PATH` is discovery context only and
is never saved. Setup does not install native runtimes; model setup and explicit
`setup systemd` remain separate.

The release package supplies the CPU ONNX Runtime and patched sherpa C API.
These settings replace that default with an external matching stack for
OpenVINO or CUDA and locate its provider and vendor dependencies. Explicit
`omawake setup systemd` writes only Omawake's effective config, app environment,
and package-relative paths into the unit. It never copies the caller's ambient
`LD_LIBRARY_PATH`.

For CUDA, an empty `backend.provider_config` makes Omawake write a private
provider config below `$XDG_CACHE_HOME/omawake/cuda/device-<id>`. It maps
`backend.device_id` to ONNX Runtime's `device_id` and passes validated
`[backend.options]` entries through as CUDA EP V2 options. Omawake manages the
`device_id` entry and defaults `cudnn_conv_algo_search` to `HEURISTIC`; an
explicit backend option can override the search mode. Set
`backend.provider_config` to an existing file for full manual control; relative
paths resolve beside Omawake's application config.
This provider-file behavior requires the tracked sherpa patch built by
[`native/sherpa/build.sh`](native/sherpa/build.sh) against the selected ONNX
Runtime SDK.

For OpenVINO, an empty `backend.provider_config` makes Omawake atomically write a user-private config below `$XDG_CACHE_HOME/omawake/openvino/<device>/provider.config`. It includes the uppercase `device_type` used by sherpa to select the registered EP device and a typed `load_config` JSON object. Omawake puts the compiled models in the sibling `compiled/` directory and defaults `NPU_QDQ_OPTIMIZATION` to `"YES"` for NPU. Put OpenVINO device properties in `backend.options.load_config`. Session controls such as `ProfilingFilePrefix`, `GraphOptimizationLevel`, and `SessionConfig.*` remain top-level. Unknown option keys fail during setup. A `device_type` option must match the selected device.

When an explicit GPU or NPU selection and a verified catalog model meet during
full, runtime, or model setup, Omawake runs the pinned probe WAV through the detector
in an isolated, no-fallback process. Setup succeeds only after OpenVINO writes a
nonempty compiled `.blob` under
`$XDG_CACHE_HOME/omawake/openvino/<device>/compiled`. If an external compiler
terminates by signal, setup allows up to five isolated attempts and still requires
a complete inference before applying the configuration. Runtime-only setup can record a
validated accelerator selection before a model exists; `setup check` then reports the
cache as incomplete until model setup performs the compilation. This keeps
device compilation out of the first `test` or daemon inference.

For catalog-managed models, the setup and config commands select the FP32
encoder for OpenVINO GPU, NPU, and `auto`. Exact CPU retains the smaller INT8 encoder.
CUDA selects the FP32 encoder, decoder, and joiner so model work reaches the
GPU instead of falling back for quantized operators. An explicit custom model
directory or model file remains untouched. The FP32 encoder avoids the lost
detections observed with the INT8 encoder on Intel accelerators.

Provider-specific device properties belong in OpenVINO's inline JSON
`load_config` value. For example, the Panther Lake NPU used in the benchmark
needed this platform override with the installed compiler:

```toml
[backend.options]
load_config = '{"NPU":{"NPU_PLATFORM":"5010"}}'
```

Set `backend.provider_config` to use an existing file instead. Absolute paths are used directly; relative paths resolve beside Omawake's application config. Omawake verifies that the path names a regular file and passes its absolute canonical path to sherpa without rewriting it.

See [DEMO.md](DEMO.md) for the reproducible live-capture demonstration and measured result.

## License

Omawake source is licensed under the [MIT License](LICENSE). Bundled runtime
components and downloaded models remain under their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md). Model terms and download status
are part of the setup catalog and are not covered by Omawake's MIT license.
