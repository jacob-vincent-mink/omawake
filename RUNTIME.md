# Runtime contract

Omawake releases own the ONNX Runtime core. Linux x86-64 and aarch64 archives contain
ONNX Runtime 1.30.0 under `lib/`; the Rust executable resolves and loads it at
startup and has no `DT_NEEDED` entry for ONNX Runtime.

The `default` runtime uses ORT's CPU execution provider. The `openvino` and
`cuda` runtimes register official V2 execution-provider plugin packages with
`RegisterExecutionProviderLibrary` and select the requested device through the
ORT 1.30 device API. Provider packages must not include another ORT core.

Discovery order is an exact configured path, `OMAWAKE_ONNXRUNTIME_LIBRARY` or
`OMAWAKE_PROVIDER_LIBRARY`, `backend.library_dirs`, `OMAWAKE_LIBRARY_PATH`, and
package-relative directories. On Linux Omawake re-executes once with its
app-owned dependency directories in `LD_LIBRARY_PATH` before loading inference
code.

```bash
omawake setup runtime --json
omawake setup runtime --runtime openvino --device npu --dir /opt/omawake-openvino --apply
```

Setup probes ORT 1.30.0, provider registration, and matching device exposure in
an isolated child process. Model setup additionally executes the catalog probe
WAV for explicit OpenVINO GPU/NPU selections and requires a compiled cache
artifact before applying the candidate.

The model, feature extraction, transducer state, context graph, and keyword
decoder are implemented by Omawake itself. No companion inference library is
required.
