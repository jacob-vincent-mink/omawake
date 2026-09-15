use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(unix)]
use std::thread;
#[cfg(unix)]
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::net::UnixListener;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[cfg(unix)]
use omawake::config::Config;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn sandbox() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omawake-cli-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(args)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn fake_systemctl(root: &Path, exit: i32) -> PathBuf {
    let bin = root.join(format!("bin-{exit}"));
    fs::create_dir_all(&bin).unwrap();
    let program = bin.join("systemctl");
    fs::write(
        &program,
        format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$OMAWAKE_SYSTEMCTL_LOG\"\nexit {exit}\n"),
    )
    .unwrap();
    #[cfg(unix)]
    fs::set_permissions(&program, fs::Permissions::from_mode(0o755)).unwrap();
    bin
}

fn run_with_path(root: &Path, args: &[&str], path: &Path) -> Output {
    let owned_libraries = root.join("owned-libraries");
    let ambient_libraries = root.join("ambient-libraries");
    fs::create_dir_all(&owned_libraries).unwrap();
    fs::create_dir_all(&ambient_libraries).unwrap();
    Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(args)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMAWAKE_SYSTEMCTL_LOG", root.join("systemctl.log"))
        .env("OMAWAKE_LIBRARY_PATH", &owned_libraries)
        .env("LD_LIBRARY_PATH", &ambient_libraries)
        .env("PATH", path)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[cfg(target_os = "linux")]
fn compile_shared(root: &Path, name: &str, source: &str) -> PathBuf {
    let source_path = root.join(format!("{name}.c"));
    let library_path = root.join(format!("lib{name}.so"));
    fs::write(&source_path, source).unwrap();
    let output = Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(&source_path)
        .arg("-o")
        .arg(&library_path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    library_path
}

#[cfg(unix)]
#[test]
fn guided_setup_accepts_arrow_keys_and_enter_in_a_real_pty() {
    if Command::new("script").arg("--version").output().is_err() {
        return;
    }
    let root = sandbox();
    let ort = compile_shared(
        &root,
        "guided-onnxruntime",
        r#"
#include <stdint.h>
typedef struct { const void *(*GetApi)(uint32_t); const char *(*GetVersionString)(void); } OrtApiBase;
static const char *version(void) { return "1.29.0"; }
static const OrtApiBase base = {0, version};
const OrtApiBase *OrtGetApiBase(void) { return &base; }
"#,
    );
    let sherpa = compile_shared(
        &root,
        "guided-sherpa-onnx-c-api",
        r#"
#include <stdint.h>
const char *SherpaOnnxGetVersionStr(void) { return "1.13.8"; }
const char *SherpaOnnxGetOnnxruntimeVersionStr(void) { return "1.29.0"; }
int32_t SherpaOnnxGetExtendedApiVersion(void) { return 1; }
void *SherpaOnnxCreateOrtRuntime(void) { return 0; }
void SherpaOnnxDestroyOrtRuntime(void *p) { (void)p; }
int32_t SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(void *r, const char *n, const char *p) { return r && n && p; }
int32_t SherpaOnnxOrtRuntimeHasExecutionProviderDevice(void *r, const char *e, const char *d) { return r && e && d; }
const char *SherpaOnnxOrtRuntimeGetLastError(const void *r) { (void)r; return "fixture"; }
const void *SherpaOnnxCreateKeywordSpotter(const void *p) { return p; }
void SherpaOnnxDestroyKeywordSpotter(const void *p) { (void)p; }
const void *SherpaOnnxCreateKeywordStream(const void *p) { return p; }
void SherpaOnnxOnlineStreamAcceptWaveform(const void *p, int32_t r, const float *s, int32_t n) { (void)p; (void)r; (void)s; (void)n; }
int32_t SherpaOnnxIsKeywordStreamReady(const void *p, const void *s) { return p && s; }
void SherpaOnnxDecodeKeywordStream(const void *p, const void *s) { (void)p; (void)s; }
const void *SherpaOnnxGetKeywordResult(const void *p, const void *s) { return p && s ? p : 0; }
void SherpaOnnxDestroyKeywordResult(const void *p) { (void)p; }
void SherpaOnnxDestroyOnlineStream(const void *p) { (void)p; }
void SherpaOnnxOnlineStreamInputFinished(const void *p) { (void)p; }
"#,
    );
    let config_path = root.join("config/omawake/config.toml");
    let mut config = Config::default();
    config.backend.onnxruntime_library = ort;
    config.backend.sherpa_library = sherpa;
    config.save(&config_path).unwrap();
    let binary = env!("CARGO_BIN_EXE_omawake");
    assert!(!binary.contains(['\'', '"', ' ']));
    let mut child = Command::new("script")
        .args(["-qec", &format!("{binary} setup"), "/dev/null"])
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("TERM", "xterm-256color")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    // Wait for raw mode, choose Runtime, accept Default, choose CPU, then keep discovery.
    thread::sleep(Duration::from_millis(750));
    input.write_all(b"\x1b[B\r").unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(150));
    input.write_all(b"\r").unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(150));
    input.write_all(b"\x1b[B\r").unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(150));
    input.write_all(b"\r").unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(150));
    input.write_all(b"\x1b[A\r").unwrap();
    drop(input);
    let deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            panic!("guided setup did not finish after PTY input");
        }
        thread::sleep(Duration::from_millis(25));
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let terminal = stdout(&output);
    assert!(terminal.contains("Omawake setup"));
    assert!(terminal.contains("Inference runtime"));
    assert!(terminal.contains("Inference device"));
    assert!(terminal.contains("Review runtime"));
    assert!(terminal.contains("runtime configured: default / cpu"));
    assert_eq!(Config::load(&config_path).unwrap().backend.device, "cpu");
}

#[test]
fn word_alias_add_remove_and_empty_configuration_round_trip() {
    let root = sandbox();
    let remove = run(&root, &["word", "remove", "computer"]);
    assert!(remove.status.success(), "{}", stderr(&remove));
    assert!(stdout(&remove).contains("removed wake word: computer"));
    assert_eq!(
        stdout(&run(&root, &["word", "list", "--json"])).trim(),
        "[]"
    );

    let add = run(
        &root,
        &[
            "wake-word",
            "add",
            "--id",
            "computer",
            "--phrase",
            "Computer",
            "--",
            "true",
        ],
    );
    assert!(add.status.success(), "{}", stderr(&add));
    assert!(stdout(&add).contains("added wake word: computer"));
    assert!(stdout(&run(&root, &["word", "list"])).contains("computer"));
    assert!(run(&root, &["word", "remove", "computer"]).status.success());
    assert!(!run(&root, &["word", "remove", "missing"]).status.success());
}

#[test]
fn setup_discovery_and_remediation_commands() {
    let root = sandbox();
    for args in [
        &["setup", "runtime"][..],
        &["setup", "runtime", "--json"],
        &["setup", "model", "--list"],
        &["setup", "model", "--json"],
        &["setup", "model"],
    ] {
        let output = run(&root, args);
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(!stdout(&output).is_empty());
    }
    let runtime = run(&root, &["setup", "runtime", "--json"]);
    let runtime: serde_json::Value = serde_json::from_slice(&runtime.stdout).unwrap();
    assert!(runtime["supported_capabilities"].is_array());
    assert!(runtime["libraries"]["runtime_loadable"].is_object());
    assert!(runtime["libraries"]["effective_library_dirs"].is_array());
    for args in [
        &["setup"][..],
        &["setup", "check", "--json"],
        &["setup", "model", "--verify", "missing"],
        &["setup", "all", "--model", "missing"],
        &["setup", "systemd", "--status"],
    ] {
        assert!(!run(&root, args).status.success());
    }
    assert!(run(&root, &["setup", "menu"]).status.success());
    assert!(run(&root, &["setup", "menu", "--status"]).status.success());
    assert!(
        run(&root, &["setup", "menu", "--uninstall"])
            .status
            .success()
    );
    assert!(!run(&root, &["setup", "menu", "--status"]).status.success());

    let mixed_json = run(
        &root,
        &[
            "setup",
            "model",
            "--json",
            "--download",
            "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
        ],
    );
    assert!(!mixed_json.status.success());
    assert!(mixed_json.stdout.is_empty());
    assert!(stderr(&mixed_json).contains("cannot be used with"));
}

#[test]
fn setup_runtime_directory_rejects_unloadable_cpu_and_cuda_stacks_without_persisting() {
    for (runtime, expected_runtime, provider) in [
        ("default", omawake::backend::Runtime::Default, None),
        (
            "cuda",
            omawake::backend::Runtime::Cuda,
            Some("libonnxruntime_providers_cuda.so"),
        ),
    ] {
        let root = sandbox();
        let bundle = root.join(format!("{runtime}-sdk"));
        let libraries = bundle.join("lib64");
        fs::create_dir_all(&libraries).unwrap();
        fs::write(libraries.join("libonnxruntime.so.1.29.0"), b"fixture").unwrap();
        fs::write(libraries.join("libsherpa-onnx-c-api.so.1.13.8"), b"fixture").unwrap();
        if let Some(provider) = provider {
            fs::write(libraries.join(provider), b"fixture").unwrap();
        }

        let output = run(
            &root,
            &[
                "setup",
                "runtime",
                "--runtime",
                runtime,
                "--device",
                if runtime == "cuda" { "gpu" } else { "cpu" },
                "--dir",
                bundle.to_str().unwrap(),
            ],
        );
        assert!(!output.status.success());
        assert!(stderr(&output).contains("runtime candidate rejected; config unchanged"));
        assert!(!root.join("config/omawake/config.toml").exists());
        let _ = expected_runtime;
    }
}

#[cfg(target_os = "linux")]
#[test]
fn direct_wav_detection_and_benchmark_use_the_external_runtime() {
    let root = sandbox();
    let ort = compile_shared(
        &root,
        "onnxruntime",
        r#"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { const void *(*GetApi)(uint32_t); const char *(*GetVersionString)(void); } OrtApiBase;
static const char *version(void) { return "1.29.0"; }
static const OrtApiBase base = {0, version};
const OrtApiBase *OrtGetApiBase(void) { return &base; }
"#,
    );
    let sherpa = compile_shared(
        &root,
        "sherpa-onnx-c-api",
        r#"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { const char *keyword; const char *tokens; const char *const *tokens_arr; int32_t count; float *timestamps; float start_time; const char *json; } Result;
static int runtime, spotter, stream, ready;
static const char *tokens[] = {"WAKE"};
static float timestamps[] = {0.1f};
static Result result = {"computer", "WAKE", tokens, 1, timestamps, 0.1f, "{}"};
const char *SherpaOnnxGetVersionStr(void) { return "1.13.8"; }
const char *SherpaOnnxGetOnnxruntimeVersionStr(void) { return "1.29.0"; }
int32_t SherpaOnnxGetExtendedApiVersion(void) { return 1; }
void *SherpaOnnxCreateOrtRuntime(void) { return &runtime; }
void SherpaOnnxDestroyOrtRuntime(void *p) { (void)p; }
int32_t SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(void *r, const char *n, const char *p) { return r && n && p && !strstr(p, "rejected"); }
int32_t SherpaOnnxOrtRuntimeHasExecutionProviderDevice(void *r, const char *ep, const char *device) { return r && ep && device; }
const char *SherpaOnnxOrtRuntimeGetLastError(const void *r) { (void)r; return "fake error"; }
const void *SherpaOnnxCreateKeywordSpotter(const void *c) {
    if (c) {
        system("for device in gpu npu; do cache=\"$XDG_CACHE_HOME/omawake/openvino/$device/compiled\"; if [ -d \"$cache\" ]; then printf cache > \"$cache/fixture.blob\"; fi; done");
    }
    return c ? &spotter : 0;
}
void SherpaOnnxDestroyKeywordSpotter(const void *p) { (void)p; }
const void *SherpaOnnxCreateKeywordStream(const void *p) { return p ? &stream : 0; }
int32_t SherpaOnnxIsKeywordStreamReady(const void *p, const void *s) { return p && s && ready; }
void SherpaOnnxDecodeKeywordStream(const void *p, const void *s) { (void)p; (void)s; ready = 0; }
const Result *SherpaOnnxGetKeywordResult(const void *p, const void *s) { return p && s ? &result : 0; }
void SherpaOnnxDestroyKeywordResult(const Result *r) { (void)r; }
void SherpaOnnxDestroyOnlineStream(const void *s) { (void)s; }
void SherpaOnnxOnlineStreamAcceptWaveform(const void *s, int32_t rate, const float *samples, int32_t count) { ready = s && rate > 0 && samples && count > 0; }
void SherpaOnnxOnlineStreamInputFinished(const void *s) { ready = s != 0; }
"#,
    );

    let model = root.join("model");
    fs::create_dir_all(&model).unwrap();
    for file in ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"] {
        fs::write(model.join(file), b"fixture").unwrap();
    }
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bpe.model"),
        model.join("bpe.model"),
    )
    .unwrap();
    let probe_wav = model.join("test_wavs/0.wav");
    fs::create_dir_all(probe_wav.parent().unwrap()).unwrap();
    let mut writer = hound::WavWriter::create(
        &probe_wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    writer.write_sample(1_i16).unwrap();
    writer.finalize().unwrap();
    let config_path = root.join("config/omawake/config.toml");
    let mut config = Config::default();
    config.model.directory = model.display().to_string();
    config.model.encoder = "encoder.onnx".into();
    config.model.decoder = "decoder.onnx".into();
    config.model.joiner = "joiner.onnx".into();
    config.backend.onnxruntime_library = ort;
    config.backend.sherpa_library = sherpa.clone();
    config.wake_words[0].command = vec!["true".into()];
    config.save(&config_path).unwrap();

    let cpu_discovery = run(&root, &["setup", "runtime"]);
    assert!(cpu_discovery.status.success(), "{}", stderr(&cpu_discovery));
    assert!(stdout(&cpu_discovery).contains("default   auto, cpu                 runtime ready"));

    config.backend.sherpa_library = compile_shared(
        &root,
        "unpatched-sherpa-onnx-c-api",
        r#"
const char *SherpaOnnxGetVersionStr(void) { return "1.13.8"; }
const char *SherpaOnnxGetOnnxruntimeVersionStr(void) { return "1.29.0"; }
"#,
    );
    config.save(&config_path).unwrap();
    let unpatched = run(&root, &["setup", "runtime"]);
    assert!(unpatched.status.success(), "{}", stderr(&unpatched));
    assert!(stdout(&unpatched).contains("default   auto, cpu                 runtime not found"));
    assert!(stdout(&unpatched).contains("patched sherpa-onnx 1.13.8"));
    config.backend.sherpa_library = sherpa;

    for (runtime, device, library) in [
        (
            omawake::backend::Runtime::Cuda,
            "gpu",
            "onnxruntime_providers_cuda",
        ),
        (
            omawake::backend::Runtime::Openvino,
            "npu",
            "onnxruntime_providers_openvino",
        ),
    ] {
        config.backend.runtime = runtime;
        config.backend.device = device.into();
        config.backend.provider_library =
            compile_shared(&root, library, "int provider_entry(void) { return 1; }");
        config.save(&config_path).unwrap();
        let discovery = run(&root, &["setup", "runtime"]);
        assert!(discovery.status.success(), "{}", stderr(&discovery));
        assert!(stdout(&discovery).contains("runtime ready"));
    }
    let probe_contents = fs::read(&probe_wav).unwrap();
    fs::remove_file(&probe_wav).unwrap();
    let deferred = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "openvino",
            "--device",
            "npu",
            "--apply",
        ],
    );
    assert!(deferred.status.success(), "{}", stderr(&deferred));
    assert!(stderr(&deferred).contains("model cache preparation deferred"));
    assert!(
        !root
            .join("cache/omawake/openvino/npu/compiled/fixture.blob")
            .exists()
    );
    let deferred_config = Config::load(&config_path).unwrap();
    assert_eq!(
        deferred_config.backend.runtime,
        omawake::backend::Runtime::Openvino
    );
    assert_eq!(deferred_config.backend.device, "npu");
    let deferred_check = run(&root, &["setup", "check", "--json"]);
    assert!(!deferred_check.status.success());
    let deferred_checks: serde_json::Value =
        serde_json::from_slice(&deferred_check.stdout).unwrap();
    let cache_check = deferred_checks
        .as_array()
        .unwrap()
        .iter()
        .find(|check| check["name"] == "model-cache")
        .unwrap();
    assert_eq!(cache_check["ok"], false);

    fs::write(&probe_wav, probe_contents).unwrap();
    let prepared = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "openvino",
            "--device",
            "npu",
            "--apply",
        ],
    );
    assert!(prepared.status.success(), "{}", stderr(&prepared));
    assert!(stderr(&prepared).contains("preparing OpenVINO NPU model cache"));
    assert!(stderr(&prepared).contains("prepared OpenVINO NPU model cache"));
    let cache_artifact = root.join("cache/omawake/openvino/npu/compiled/fixture.blob");
    assert_eq!(fs::read(&cache_artifact).unwrap(), b"cache");
    let prepared_config = Config::load(&config_path).unwrap();
    assert_eq!(
        prepared_config.backend.runtime,
        omawake::backend::Runtime::Openvino
    );
    assert_eq!(prepared_config.backend.device, "npu");
    let prepared_check = run(&root, &["setup", "check", "--json"]);
    let prepared_checks: serde_json::Value =
        serde_json::from_slice(&prepared_check.stdout).unwrap();
    let cache_check = prepared_checks
        .as_array()
        .unwrap()
        .iter()
        .find(|check| check["name"] == "model-cache")
        .unwrap();
    assert_eq!(cache_check["ok"], true);

    let prepared_gpu = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "openvino",
            "--device",
            "gpu",
            "--apply",
        ],
    );
    assert!(prepared_gpu.status.success(), "{}", stderr(&prepared_gpu));
    assert!(stderr(&prepared_gpu).contains("preparing OpenVINO GPU model cache"));
    assert!(stderr(&prepared_gpu).contains("prepared OpenVINO GPU model cache"));
    assert_eq!(
        fs::read(root.join("cache/omawake/openvino/gpu/compiled/fixture.blob")).unwrap(),
        b"cache"
    );
    let gpu_config = Config::load(&config_path).unwrap();
    assert_eq!(gpu_config.backend.device, "gpu");
    let gpu_check = run(&root, &["setup", "check", "--json"]);
    let gpu_checks: serde_json::Value = serde_json::from_slice(&gpu_check.stdout).unwrap();
    let cache_check = gpu_checks
        .as_array()
        .unwrap()
        .iter()
        .find(|check| check["name"] == "model-cache")
        .unwrap();
    assert_eq!(cache_check["ok"], true);

    config.backend.runtime = omawake::backend::Runtime::Openvino;
    config.backend.device = "npu".into();
    config.backend.provider_library = compile_shared(
        &root,
        "rejected-openvino-provider",
        "int provider_entry(void) { return 1; }",
    );
    config.save(&config_path).unwrap();
    let rejected = run(&root, &["setup", "runtime"]);
    assert!(rejected.status.success(), "{}", stderr(&rejected));
    assert!(stdout(&rejected).contains("external runtime not found"));
    let before = fs::read(&config_path).unwrap();
    let failed = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "openvino",
            "--device",
            "npu",
            "--apply",
        ],
    );
    assert!(!failed.status.success());
    assert!(stderr(&failed).contains("register execution-provider library"));
    assert_eq!(fs::read(&config_path).unwrap(), before);

    config.backend.runtime = omawake::backend::Runtime::Default;
    config.backend.device = "auto".into();
    config.backend.provider_library = PathBuf::new();
    config.save(&config_path).unwrap();

    let original = fs::read(&config_path).unwrap();
    let preview = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "default",
            "--device",
            "cpu",
        ],
    );
    assert!(preview.status.success(), "{}", stderr(&preview));
    assert!(stdout(&preview).contains("\"applied\": false"));
    assert_eq!(fs::read(&config_path).unwrap(), original);
    let applied = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "default",
            "--device",
            "cpu",
            "--apply",
        ],
    );
    assert!(applied.status.success(), "{}", stderr(&applied));
    assert!(stdout(&applied).contains("\"applied\": true"));
    assert_eq!(Config::load(&config_path).unwrap().backend.device, "cpu");

    let wav = root.join("input.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    writer.write_sample(1_i16).unwrap();
    writer.finalize().unwrap();

    let detected = run(
        &root,
        &[
            "test",
            "--audio",
            wav.to_str().unwrap(),
            "--execute",
            "--json",
        ],
    );
    assert!(detected.status.success(), "{}", stderr(&detected));
    assert!(stdout(&detected).contains("computer"));
    let benchmark = run(
        &root,
        &[
            "benchmark",
            "--warmup",
            "0",
            "--iterations",
            "1",
            wav.to_str().unwrap(),
        ],
    );
    assert!(benchmark.status.success(), "{}", stderr(&benchmark));
    let report: serde_json::Value = serde_json::from_slice(&benchmark.stdout).unwrap();
    assert!(report.is_object());
    assert!(stdout(&benchmark).contains("computer"));
}

#[test]
fn setup_all_help_keeps_service_installation_explicit() {
    let root = sandbox();
    let all = run(&root, &["setup", "all", "--help"]);
    assert!(all.status.success(), "{}", stderr(&all));
    let help = stdout(&all);
    assert!(help.contains("Does not install a service"));
    assert!(help.contains("omawake setup systemd"));
    assert!(!help.contains("--no-start"));

    let setup = run(&root, &["setup", "--help"]);
    assert!(setup.status.success(), "{}", stderr(&setup));
    assert!(stdout(&setup).contains("optional systemd user service"));
    assert!(!run(&root, &["setup", "all", "--no-start"]).status.success());
}

#[test]
fn config_commands_cover_supported_keys_and_errors() {
    let root = sandbox();
    assert_eq!(
        stdout(&run(&root, &["config", "get", "backend.kind"])).trim(),
        "sherpa-onnx"
    );
    assert!(run(&root, &["config", "get", "--json"]).status.success());
    assert!(run(&root, &["config", "schema"]).status.success());
    assert!(run(&root, &["config", "schema", "--json"]).status.success());

    for (key, value) in [
        ("backend.kind", "future-backend"),
        ("backend.runtime", "default"),
        ("backend.device", "cpu"),
        ("backend.threads", "3"),
        ("backend.fallback", "cpu"),
        ("backend.device_id", "0"),
        ("backend.provider_config", "provider.json"),
        ("backend.library_dirs", "/tmp"),
        ("model.name", "custom"),
        ("model.directory", "/tmp/model"),
        ("model.sample_rate", "16000"),
        ("model.keywords_score", "2.0"),
        ("model.keywords_threshold", "0.5"),
        ("audio.device", "test"),
        ("audio.channels", "mono"),
        ("audio.buffer_milliseconds", "100"),
        ("daemon.cooldown_milliseconds", "500"),
        ("daemon.queue_capacity", "4"),
    ] {
        let output = run(&root, &["config", "set", key, value]);
        assert!(output.status.success(), "{key}: {}", stderr(&output));
        assert!(
            run(&root, &["config", "unset", key]).status.success(),
            "{key}"
        );
    }
    for args in [
        &["config", "get", "missing.key"][..],
        &["config", "set", "missing.key", "x"],
        &["config", "unset", "missing.key"],
        &["config", "set", "backend.runtime", "bogus"],
        &["config", "set", "backend.fallback", "bogus"],
        &["config", "set", "backend.threads", "0"],
    ] {
        assert!(!run(&root, args).status.success());
    }

    assert!(
        run(&root, &["config", "set", "backend.runtime", "cuda"])
            .status
            .success()
    );
    assert!(
        run(
            &root,
            &["config", "set", "backend.provider_config", "cuda.config",],
        )
        .status
        .success()
    );
    assert!(
        run(&root, &["config", "set", "backend.runtime", "default"])
            .status
            .success()
    );
    let saved = Config::load(&root.join("config/omawake/config.toml")).unwrap();
    assert_eq!(saved.backend.runtime, omawake::backend::Runtime::Default);
    assert!(saved.backend.provider_config.is_empty());
}

#[test]
fn runtime_discovery_reports_invalid_paths_without_reexec_and_engine_use_rejects_them() {
    let root = sandbox();
    let set = run(
        &root,
        &[
            "config",
            "set",
            "backend.library_dirs",
            "missing-provider-libraries",
        ],
    );
    assert!(set.status.success(), "{}", stderr(&set));

    let discovery = run(&root, &["setup", "runtime", "--json"]);
    assert!(discovery.status.success(), "{}", stderr(&discovery));
    let value: serde_json::Value = serde_json::from_slice(&discovery.stdout).unwrap();
    let expected = root.join("config/omawake/missing-provider-libraries");
    assert_eq!(
        value["libraries"]["configured_library_dirs"][0],
        expected.display().to_string()
    );
    assert_eq!(
        value["libraries"]["missing_library_dirs"][0],
        expected.display().to_string()
    );
    assert!(
        !value["libraries"]["remediation"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let engine = run(&root, &["benchmark", "/missing.wav"]);
    assert!(!engine.status.success());
    assert!(stderr(&engine).contains("native library directories must be absolute existing"));
}

#[test]
fn runtime_commands_fail_cleanly_without_hardware_model_or_daemon() {
    let root = sandbox();
    assert!(run(&root, &["status"]).status.success());
    assert!(run(&root, &["status", "--json"]).status.success());
    assert!(!run(&root, &["test"]).status.success());
    assert!(
        !run(&root, &["test", "--audio", "/missing.wav"])
            .status
            .success()
    );
    assert!(!run(&root, &["benchmark", "/missing.wav"]).status.success());
    for command in ["pause", "resume", "stop"] {
        assert!(!run(&root, &[command]).status.success());
    }
    assert!(!run(&root, &["daemon"]).status.success());
}

#[cfg(unix)]
#[test]
fn client_commands_exchange_framed_messages_with_a_running_daemon() {
    for (args, state) in [
        (&["status"][..], "armed"),
        (&["status", "--json"][..], "armed"),
        (&["pause"][..], "paused"),
        (&["resume"][..], "armed"),
        (&["stop"][..], "stopping"),
    ] {
        let root = sandbox();
        let runtime = root.join("run/omawake");
        fs::create_dir_all(&runtime).unwrap();
        let listener = match UnixListener::bind(runtime.join("control.sock")) {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("{error}"),
        };
        let expected_state = state.to_owned();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            BufReader::new(stream.try_clone().unwrap())
                .read_line(&mut request)
                .unwrap();
            let request: serde_json::Value = serde_json::from_str(&request).unwrap();
            let response = serde_json::json!({
                "protocol": 1,
                "id": request["id"],
                "type": "state",
                "state": expected_state,
                "details": {"backend": {"kind": "fake"}}
            });
            writeln!(stream, "{response}").unwrap();
        });
        let output = run(&root, args);
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains(state));
        server.join().unwrap();
    }
}

#[test]
fn malformed_config_is_reported_before_runtime_commands() {
    let root = sandbox();
    let config = root.join("config/omawake/config.toml");
    fs::create_dir_all(config.parent().unwrap()).unwrap();
    fs::write(config, "invalid = [").unwrap();
    assert!(!run(&root, &["status"]).status.success());
}

#[test]
fn systemd_lifecycle_uses_user_manager_and_propagates_failures() {
    let root = sandbox();
    let success = fake_systemctl(&root, 0);
    assert!(
        run_with_path(&root, &["setup", "systemd", "--no-start"], &success)
            .status
            .success()
    );
    let unit = fs::read_to_string(root.join("config/systemd/user/omawake.service")).unwrap();
    assert!(unit.contains(&root.join("owned-libraries").display().to_string()));
    assert!(!unit.contains(&root.join("ambient-libraries").display().to_string()));
    assert!(
        run_with_path(&root, &["setup", "systemd"], &success)
            .status
            .success()
    );
    assert!(
        run_with_path(&root, &["setup", "systemd", "--status"], &success)
            .status
            .success()
    );
    let failure = fake_systemctl(&root, 1);
    assert!(
        !run_with_path(&root, &["setup", "systemd", "--status"], &failure)
            .status
            .success()
    );
    assert!(
        run_with_path(&root, &["setup", "systemd", "--uninstall"], &success)
            .status
            .success()
    );
    assert!(!root.join("config/systemd/user/omawake.service").exists());
    let calls = fs::read_to_string(root.join("systemctl.log")).unwrap();
    for expected in [
        "--user daemon-reload",
        "--user enable omawake.service",
        "--user restart omawake.service",
        "--user is-active --quiet omawake.service",
        "--user status omawake.service --no-pager",
        "--user disable --now omawake.service",
    ] {
        assert!(
            calls.lines().any(|call| call == expected),
            "missing {expected:?} in {calls:?}"
        );
    }
    assert!(
        !run_with_path(&root, &["setup", "systemd"], &failure)
            .status
            .success()
    );
}
