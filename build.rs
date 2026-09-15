use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {description}: {error}"));
    assert!(status.success(), "{description} failed with {status}");
}

fn whisper_root() -> Option<PathBuf> {
    env::var_os("WHISPER_CPP_ROOT").map(PathBuf::from)
}

fn version(root: &Path) -> String {
    let cmake = fs::read_to_string(root.join("CMakeLists.txt"))
        .expect("read upstream whisper.cpp CMakeLists.txt");
    ["MAJOR", "MINOR", "PATCH"]
        .map(|part| {
            let prefix = format!("set(WHISPER_VERSION_{part} ");
            cmake
                .lines()
                .find_map(|line| line.strip_prefix(&prefix))
                .and_then(|value| value.strip_suffix(')'))
                .expect("parse upstream whisper.cpp version")
        })
        .join(".")
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(omawake_whisper_adapter)");
    println!("cargo:rerun-if-env-changed=WHISPER_CPP_ROOT");
    println!("cargo:rerun-if-changed=src/engine/whisper/adapter.c");
    let Some(root) = whisper_root() else {
        println!(
            "cargo:warning=WHISPER_CPP_ROOT is unset; whisper.cpp provider support is disabled"
        );
        return;
    };
    for required in [
        root.join("include/whisper.h"),
        root.join("ggml/include/ggml.h"),
    ] {
        assert!(
            required.is_file(),
            "missing upstream header {}",
            required.display()
        );
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let object = output.join("whisper_adapter.o");
    let archive = output.join("libomawake_whisper_adapter.a");
    run(
        Command::new(env::var_os("CC").unwrap_or_else(|| "cc".into()))
            .arg("-std=c11")
            .arg("-fPIC")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Werror")
            .arg(format!("-DOMA_WHISPER_ABI_VERSION=\"{}\"", version(&root)))
            .arg("-I")
            .arg(root.join("include"))
            .arg("-I")
            .arg(root.join("ggml/include"))
            .arg("-c")
            .arg("src/engine/whisper/adapter.c")
            .arg("-o")
            .arg(&object),
        "whisper.cpp adapter compilation",
    );
    run(
        Command::new(env::var_os("AR").unwrap_or_else(|| "ar".into()))
            .arg("crs")
            .arg(&archive)
            .arg(&object),
        "whisper.cpp adapter archive",
    );
    println!("cargo:rustc-cfg=omawake_whisper_adapter");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=omawake_whisper_adapter");
    println!("cargo:rustc-link-lib=dl");
    println!(
        "cargo:rerun-if-changed={}",
        root.join("CMakeLists.txt").display()
    );
}
