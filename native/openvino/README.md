# Native OpenVINO runtime

Omawake's `openvino` feature needs a shared sherpa-onnx build whose ONNX Runtime
contains the OpenVINO Execution Provider. The ordinary sherpa-onnx release is
CPU-only. [`build.sh`](build.sh) creates the tested stack from pinned sources:

- ONNX Runtime `v1.29.0` at `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`
- sherpa-onnx `v1.13.8` at `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`
- Intel OpenVINO `2026.2.1.21919.ede283a88e3`, verified by SHA-256

The OpenVINO archive SHA-256 is
`cf7a3eb84a1edbd852f719a4ba8c15dbf02a3744ab61488b6cae44747d90bf78`.

The ORT patch SHA-256 is
`6d6ec445dc761208aded9c7d786e9112d2920a1509e2683c86c6d2bca8fa499f`;
the sherpa patch SHA-256 is
`785d5f491cf195c3b39a855066867eb5d9e6dd7ed8a61abae3c379167e326909`.

The build applies the tracked ORT zero-element tensor fix required by
Supertonic and the tracked sherpa patch that lets Supertonic choose OpenVINO
per component. Those patches are part of the tested runtime contract even when
an application does not use Supertonic itself. OpenVINO 2026.2.1 is pinned
because the OpenVINO 2026.3.1 combination tested on this host crashed during
process teardown after NPU keyword inference.

The script downloads source and binaries into `${OMA_NATIVE_ROOT}` (default
`/tmp/oma-native-openvino`), performs full native builds, and assembles headers
and shared libraries below `${OMA_NATIVE_ROOT}/runtime`. It does not install
files into the system. The native build is large and can take a substantial
amount of time and disk space. Prerequisites are Python 3, a C/C++ toolchain,
CMake, Make, Git, curl, tar, and the build dependencies required by ONNX
Runtime.

```bash
OMA_BUILD_JOBS=10 native/openvino/build.sh
source /tmp/oma-native-openvino/runtime/env.sh
cargo build --release --features openvino
```

`build.sh` is safe to rerun at the pinned commits and detects already-applied
patches. It stops if an existing source checkout is at another commit or a
patch no longer applies. Set `OMA_NATIVE_ROOT` to use another absolute build
directory.

ORT's `--use_openvino NPU` build selection enables the OpenVINO EP rather than
restricting the assembled provider to one device. This exact provider build was
tested with explicit `CPU`, Intel `GPU`, and Intel `NPU` device selections.

At runtime, keep the environment from `runtime/env.sh`. For the Panther Lake
5010 NPU tested here, pass the platform property through ORT's inline JSON:

```toml
[backend.options]
load_config = '{"NPU":{"NPU_PLATFORM":"5010"}}'
```

This build enables the provider; a successful build alone does not prove model
placement. The hardware benchmark records provider profiles and the kernel NPU
busy-time counter separately.
