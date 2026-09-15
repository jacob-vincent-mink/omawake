use std::env;
use std::path::PathBuf;
use std::process::Command;

const WHISPER_VERSION: &str = "1.9.3";
const WHISPER_ABI_HEADER: &str = "vendor/whispercpp-1.9.3/whisper_abi.h";

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {description}: {error}"));
    assert!(status.success(), "{description} failed with {status}");
}

fn main() {
    println!("cargo:rerun-if-changed=src/engine/whisper/adapter.c");
    println!("cargo:rerun-if-changed={WHISPER_ABI_HEADER}");
    println!("cargo:rerun-if-changed=vendor/whispercpp-1.9.3/LICENSE");
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
            .arg(format!("-DOMA_WHISPER_ABI_VERSION=\"{WHISPER_VERSION}\""))
            .arg("-I")
            .arg("vendor/whispercpp-1.9.3")
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
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=omawake_whisper_adapter");
    println!("cargo:rustc-link-lib=dl");
}
