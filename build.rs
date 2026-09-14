use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    let openvino = env::var_os("CARGO_FEATURE_OPENVINO").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    if (!openvino && !cuda) || env::var_os("DOCS_RS").is_some() {
        return;
    }

    let directory = env::var_os("SHERPA_ONNX_LIB_DIR").unwrap_or_else(|| {
        panic!(
            "accelerated runtime features require SHERPA_ONNX_LIB_DIR to point to a compatible shared sherpa-onnx runtime"
        )
    });
    let directory = Path::new(&directory);
    for (enabled, feature, library) in [
        (openvino, "openvino", "libonnxruntime_providers_openvino.so"),
        (cuda, "cuda", "libonnxruntime_providers_cuda.so"),
    ] {
        let provider = directory.join(library);
        if enabled && !provider.is_file() {
            panic!(
                "the `{feature}` feature requires {}; the ordinary sherpa-onnx shared release is CPU-only",
                provider.display()
            );
        }
    }
}
