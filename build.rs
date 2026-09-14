use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if env::var_os("CARGO_FEATURE_OPENVINO").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }

    let directory = env::var_os("SHERPA_ONNX_LIB_DIR").unwrap_or_else(|| {
        panic!(
            "the `openvino` feature requires SHERPA_ONNX_LIB_DIR to point to a shared sherpa-onnx runtime built with the ONNX Runtime OpenVINO execution provider"
        )
    });
    let provider = Path::new(&directory).join("libonnxruntime_providers_openvino.so");
    if !provider.is_file() {
        panic!(
            "the `openvino` feature requires {}; the ordinary sherpa-onnx shared release is CPU-only",
            provider.display()
        );
    }
}
