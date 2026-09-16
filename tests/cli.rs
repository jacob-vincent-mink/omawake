use std::fs;
#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(unix)]
use std::thread;
#[cfg(unix)]
use std::time::{Duration, Instant};

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
    let bin = path.join("test-bin");
    fs::create_dir_all(&bin).unwrap();
    let systemctl = bin.join("systemctl");
    fs::write(&systemctl, "#!/bin/sh\nexit 3\n").unwrap();
    #[cfg(unix)]
    fs::set_permissions(&systemctl, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn test_path(root: &Path) -> std::ffi::OsString {
    let mut paths = vec![root.join("test-bin")];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    std::env::join_paths(paths).unwrap()
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
        .env("PATH", test_path(root))
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

#[cfg(unix)]
#[test]
fn guided_setup_wraps_words_and_accepts_arrow_keys_in_a_real_pty() {
    if Command::new("script").arg("--version").output().is_err() {
        return;
    }
    let root = sandbox();
    let binary = env!("CARGO_BIN_EXE_omawake");
    assert!(!binary.contains(['\'', '"', ' ']));
    let mut child = Command::new("script")
        .args(["-qec", &format!("{binary} setup"), "/dev/null"])
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("TERM", "xterm-256color")
        .env(
            "OMAWAKE_AUDIOCPP_LIBRARY",
            root.join("missing-libaudiocpp.so"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    thread::sleep(Duration::from_millis(750));
    // Runtime flow, default audio.cpp, CPU, then current discovery.
    for keys in [b"\x1b[B\r".as_slice(), b"\r", b"\r", b"\r"] {
        input.write_all(keys).unwrap();
        input.flush().unwrap();
        thread::sleep(Duration::from_millis(150));
    }
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
    assert!(!output.status.success());
    let terminal = stdout(&output);
    assert!(terminal.contains("Omawake setup"));
    assert!(terminal.contains("Inference runtime"));
    assert!(terminal.contains("Inference device"));
    assert!(terminal.contains("Runtime libraries"));
    assert!(terminal.contains("missing-libaudiocpp.so"));
    assert!(!root.join("config/omawake/config.toml").exists());
}

#[cfg(unix)]
fn build_fake_whisper(root: &Path, version: Option<&str>) -> PathBuf {
    let library = root.join("native/libwhisper.so.1");
    fs::create_dir_all(library.parent().unwrap()).unwrap();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut command = Command::new("cc");
    command
        .args([
            "-std=c11", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        ])
        .arg("-I")
        .arg(manifest.join("vendor/whispercpp-1.9.3"));
    if let Some(version) = version {
        command.arg(format!("-DOMAWAKE_FAKE_VERSION=\"{version}\""));
    }
    let status = command
        .arg(manifest.join("tests/fixtures/whisper_fake.c"))
        .arg("-o")
        .arg(&library)
        .status()
        .unwrap();
    assert!(status.success());
    library
}

#[cfg(unix)]
fn write_fake_whisper_config(root: &Path, library: PathBuf) -> (PathBuf, PathBuf) {
    let model = root.join("models/fake-whisper");
    fs::create_dir_all(&model).unwrap();
    fs::write(model.join("verifier.bin"), b"safe fake verifier").unwrap();
    fs::write(model.join("vad.bin"), b"safe fake vad").unwrap();
    let mut config = Config::default();
    config.backend.kind = "whispercpp".into();
    config.backend.device = "cpu".into();
    config.backend.library_dirs = vec![library.parent().unwrap().to_owned()];
    config.backend.library = library;
    config.model.name = "fake-whisper".into();
    config.model.directory = model.display().to_string();
    config.model.verifier = "verifier.bin".into();
    config.model.vad = "vad.bin".into();
    let config_path = root.join("config/omawake/config.toml");
    config.save(&config_path).unwrap();

    let wave = root.join("speech.wav");
    let mut writer = hound::WavWriter::create(
        &wave,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..16_000 {
        writer.write_sample(1_000_i16).unwrap();
    }
    writer.finalize().unwrap();
    (config_path, wave)
}

#[cfg(unix)]
#[test]
fn whisper_library_worker_reuses_one_model_and_session_for_file_requests() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (_, wave) = write_fake_whisper_config(&root, library);
    let log = root.join("whisper.log");
    let output = Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args([
            "benchmark",
            wave.to_str().unwrap(),
            "--warmup",
            "0",
            "--iterations",
            "2",
        ])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMAWAKE_FAKE_WHISPER_LOG", &log)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let events = fs::read_to_string(log).unwrap();
    assert_eq!(
        events
            .lines()
            .filter(|line| *line == "verifier-load")
            .count(),
        1
    );
    assert_eq!(events.lines().filter(|line| *line == "vad-load").count(), 1);
    assert_eq!(
        events.lines().filter(|line| *line == "transcribe").count(),
        2
    );

    let diagnostic = run(
        &root,
        &[
            "test",
            "--audio",
            wave.to_str().unwrap(),
            "--show-transcripts",
        ],
    );
    assert!(diagnostic.status.success(), "{}", stderr(&diagnostic));
    assert!(stderr(&diagnostic).contains("verifier transcript: \"computer\""));
    let conflicting = run(
        &root,
        &[
            "test",
            "--audio",
            wave.to_str().unwrap(),
            "--show-transcripts",
            "--json",
        ],
    );
    assert!(!conflicting.status.success());
}

#[cfg(unix)]
#[test]
fn whisper_evaluation_runs_in_the_isolated_native_json_worker() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (_, wave) = write_fake_whisper_config(&root, library);
    let manifest = root.join("evaluation.json");
    let checksum = omawake::evaluation::sha256_bytes(&fs::read(&wave).unwrap());
    fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "corpus": {
                "id": "safe-fake-whisper",
                "version": "1",
                "license_spdx": "CC0-1.0",
                "source": "generated test signal"
            },
            "clips": [{
                "id": "positive",
                "path": wave.file_name().unwrap().to_str().unwrap(),
                "sha256": checksum,
                "split": "test",
                "expected": [{"keyword_id": "computer"}]
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let output = run(&root, &["evaluate", manifest.to_str().unwrap()]);
    assert!(output.status.success(), "{}", stderr(&output));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["evaluation"], "omawake-phrase-verifier-accuracy");
    assert_eq!(report["backend"]["backend_kind"], "whispercpp");
    assert_eq!(report["files"][0]["id"], "positive");
    assert_eq!(report["summary"]["true_positives"], 1);
}

#[cfg(unix)]
#[test]
fn whisper_library_worker_reaps_and_restarts_after_a_normal_pre_stream_exit() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (_, wave) = write_fake_whisper_config(&root, library);
    let log = root.join("whisper.log");
    let exit_marker = root.join("worker-exited-once");
    let output = Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(["test", "--audio", wave.to_str().unwrap(), "--json"])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMAWAKE_FAKE_WHISPER_LOG", &log)
        .env("OMAWAKE_FAKE_WHISPER_EXIT_ONCE", &exit_marker)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(exit_marker.is_file());
    let events = fs::read_to_string(log).unwrap();
    assert_eq!(
        events
            .lines()
            .filter(|line| *line == "verifier-load")
            .count(),
        2
    );
    assert_eq!(events.lines().filter(|line| *line == "vad-load").count(), 2);
    assert_eq!(
        events.lines().filter(|line| *line == "transcribe").count(),
        1
    );
}

#[cfg(unix)]
#[test]
fn whisper_library_worker_rejects_development_abi_with_a_normal_error() {
    let root = sandbox();
    let library = build_fake_whisper(&root, Some("1.9.3-dev"));
    let (_, wave) = write_fake_whisper_config(&root, library);
    let output = run(&root, &["test", "--audio", wave.to_str().unwrap()]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("unsupported libwhisper ABI version 1.9.3-dev"),
        "{}",
        stderr(&output)
    );
}

#[cfg(unix)]
#[test]
fn whisper_library_can_be_selected_from_an_absolute_environment_path() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (config_path, wave) = write_fake_whisper_config(&root, library.clone());
    let mut config = Config::load(&config_path).unwrap();
    config.backend.library.clear();
    config.backend.library_dirs.clear();
    config.save(&config_path).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(["test", "--audio", wave.to_str().unwrap()])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMAWAKE_WHISPER_LIBRARY", &library)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).contains("computer"));

    let relative = Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(["test", "--audio", wave.to_str().unwrap()])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMAWAKE_WHISPER_LIBRARY", "relative/libwhisper.so")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!relative.status.success());
    assert!(stderr(&relative).contains("must be an absolute file path"));
}

#[cfg(unix)]
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
            "--alias",
            "Come pewter",
            "--alias",
            "Compute her",
            "--",
            "true",
        ],
    );
    assert!(add.status.success(), "{}", stderr(&add));
    assert!(stdout(&add).contains("added wake word: computer"));
    assert!(stdout(&run(&root, &["word", "list"])).contains("computer"));
    let listed: serde_json::Value =
        serde_json::from_str(&stdout(&run(&root, &["word", "list", "--json"]))).unwrap();
    assert_eq!(
        listed[0]["aliases"],
        serde_json::json!(["Come pewter", "Compute her"])
    );
    assert!(
        run(&root, &["word", "add-alias", "computer", "Come pooter"])
            .status
            .success()
    );
    assert!(
        run(&root, &["word", "remove-alias", "computer", "Come pooter"])
            .status
            .success()
    );
    assert!(
        !run(&root, &["word", "remove-alias", "computer", "missing"])
            .status
            .success()
    );
    assert!(run(&root, &["word", "remove", "computer"]).status.success());
    assert!(!run(&root, &["word", "remove", "missing"]).status.success());
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
fn setup_model_catalog_dispatches_listing_selection_and_verification_errors() {
    let root = sandbox();
    let list = run(&root, &["setup", "model", "--list"]);
    assert!(list.status.success(), "{}", stderr(&list));
    assert!(stdout(&list).contains("moonshine-streaming-tiny-q8_0-silero-v6.2.1"));

    let no_action = run(&root, &["setup", "model"]);
    assert!(no_action.status.success(), "{}", stderr(&no_action));
    assert!(stdout(&no_action).contains("Download the default"));

    for args in [
        &[
            "setup",
            "model",
            "--verify",
            "moonshine-streaming-tiny-q8_0-silero-v6.2.1",
        ][..],
        &[
            "setup",
            "model",
            "--set",
            "moonshine-streaming-tiny-q8_0-silero-v6.2.1",
        ],
        &["setup", "model", "--download", "unknown-profile"],
    ] {
        let output = run(&root, args);
        assert!(!output.status.success(), "{args:?}");
        assert!(!stderr(&output).is_empty());
    }
    assert!(!root.join("config/omawake/config.toml").exists());
}

#[test]
fn setup_inspection_recovers_invalid_pre_release_config_without_weakening_normal_parsing() {
    let root = sandbox();
    let config_path = root.join("config/omawake/config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    let invalid = b"[backend]\nremoved_option = \"\"\n";
    fs::write(&config_path, invalid).unwrap();

    for args in [
        &["setup", "runtime", "--json"][..],
        &["setup", "model", "--json"],
    ] {
        let output = run(&root, args);
        assert!(output.status.success(), "{args:?}: {}", stderr(&output));
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap();
        assert!(stderr(&output).contains("successful setup apply will replace it"));
        assert_eq!(fs::read(&config_path).unwrap(), invalid);
    }

    let check = run(&root, &["setup", "check", "--json"]);
    assert!(!check.status.success());
    let checks: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
    assert_eq!(checks[0]["name"], "config");
    assert_eq!(checks[0]["ok"], false);
    assert_eq!(fs::read(&config_path).unwrap(), invalid);

    let normal = run(&root, &["config", "get", "--json"]);
    assert!(!normal.status.success());
    assert!(stderr(&normal).contains("unknown field `removed_option`"));
    assert_eq!(fs::read(&config_path).unwrap(), invalid);
}

#[test]
fn successful_explicit_setup_repairs_invalid_config_and_failure_restores_it() {
    let invalid = b"[backend]\nremoved_option = \"\"\n";

    let failed_root = sandbox();
    let failed_config = failed_root.join("config/omawake/config.toml");
    fs::create_dir_all(failed_config.parent().unwrap()).unwrap();
    fs::write(&failed_config, invalid).unwrap();
    let failed_bin = fake_systemctl(&failed_root, 1);
    let failed = run_with_path(
        &failed_root,
        &["setup", "systemd", "--no-start"],
        &failed_bin,
    );
    assert!(!failed.status.success());
    assert_eq!(fs::read(&failed_config).unwrap(), invalid);

    let root = sandbox();
    let config_path = root.join("config/omawake/config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    fs::write(&config_path, invalid).unwrap();
    let bin = fake_systemctl(&root, 0);
    let applied = run_with_path(&root, &["setup", "systemd", "--no-start"], &bin);
    assert!(applied.status.success(), "{}", stderr(&applied));
    assert!(stderr(&applied).contains("successful setup apply will replace it"));
    assert_eq!(Config::load(&config_path).unwrap().backend.kind, "audiocpp");
    assert!(
        !fs::read_to_string(config_path)
            .unwrap()
            .contains("removed_option")
    );
}

#[test]
fn config_commands_cover_supported_keys_and_errors() {
    let root = sandbox();
    assert_eq!(
        stdout(&run(&root, &["config", "get", "backend.kind"])).trim(),
        "audiocpp"
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
        ("backend.library_dirs", "/tmp"),
        ("model.name", "custom"),
        ("model.directory", "/tmp/model"),
        ("model.sample_rate", "16000"),
        ("audio.device", "test"),
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
        run(&root, &["config", "set", "backend.runtime", "default"])
            .status
            .success()
    );
    let saved = Config::load(&root.join("config/omawake/config.toml")).unwrap();
    assert_eq!(saved.backend.runtime, omawake::backend::Runtime::Default);
    assert!(saved.backend.library.as_os_str().is_empty());
}

#[cfg(unix)]
#[test]
fn cli_config_and_wake_word_edits_refresh_only_the_active_default_service() {
    let root = sandbox();
    let config_path = root.join("config/omawake/config.toml");
    Config::default().save(&config_path).unwrap();
    let active = fake_systemctl(&root, 0);

    let set = run_with_path(
        &root,
        &["config", "set", "daemon.queue_capacity", "4"],
        &active,
    );
    assert!(set.status.success(), "{}", stderr(&set));
    assert!(stderr(&set).contains("active daemon restarted"));
    let log = fs::read_to_string(root.join("systemctl.log")).unwrap();
    assert_eq!(log.matches("try-restart").count(), 1, "{log}");
    assert_eq!(log.matches("is-active").count(), 2, "{log}");

    fs::write(root.join("systemctl.log"), "").unwrap();
    let alias = run_with_path(
        &root,
        &["word", "add-alias", "computer", "compute her"],
        &active,
    );
    assert!(alias.status.success(), "{}", stderr(&alias));
    let log = fs::read_to_string(root.join("systemctl.log")).unwrap();
    assert_eq!(log.matches("try-restart").count(), 1, "{log}");

    fs::write(root.join("systemctl.log"), "").unwrap();
    let inactive = fake_systemctl(&root, 3);
    let unset = run_with_path(
        &root,
        &["config", "unset", "daemon.queue_capacity"],
        &inactive,
    );
    assert!(unset.status.success(), "{}", stderr(&unset));
    let log = fs::read_to_string(root.join("systemctl.log")).unwrap();
    assert_eq!(log.matches("is-active").count(), 1, "{log}");
    assert_eq!(log.matches("try-restart").count(), 0, "{log}");

    fs::write(root.join("systemctl.log"), "").unwrap();
    let custom = root.join("custom.toml");
    Config::default().save(&custom).unwrap();
    let custom_edit = run_with_path(
        &root,
        &[
            "--config",
            custom.to_str().unwrap(),
            "config",
            "set",
            "daemon.queue_capacity",
            "3",
        ],
        &active,
    );
    assert!(custom_edit.status.success(), "{}", stderr(&custom_edit));
    assert_eq!(fs::read_to_string(root.join("systemctl.log")).unwrap(), "");
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
    assert_eq!(value["provider"]["kind"], "audiocpp");
    assert_eq!(value["provider"]["loadable"], false);
    assert_eq!(value["provider"]["ready"], false);
    assert_eq!(value["configured_provider"]["probe"]["ready"], false);
    assert!(
        value["configured_provider"]["probe"]["errors"][0]
            .as_str()
            .unwrap()
            .contains(&expected.display().to_string())
    );

    let engine = run(&root, &["benchmark", "/missing.wav"]);
    assert!(!engine.status.success());
    assert!(stderr(&engine).contains(&expected.display().to_string()));

    let missing = root.join("missing-complete-provider");
    let configured = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "default",
            "--device",
            "cpu",
            "--dir",
            missing.to_str().unwrap(),
        ],
    );
    assert!(!configured.status.success());
    assert!(stderr(&configured).contains(&missing.display().to_string()));
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
    assert!(!unit.contains("LD_LIBRARY_PATH"));
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
    assert!(!root.join("config/systemd/user/omawake.service").exists());
}

#[cfg(unix)]
#[test]
fn onboarding_previews_spelling_variants_and_applies_only_reviewed_aliases() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (path, wave) = write_fake_whisper_config(&root, library);
    let mut config = Config::load(&path).unwrap();
    config.wake_words[0].phrase = "Unusual phrase".into();
    config.save(&path).unwrap();
    let before = fs::read(&path).unwrap();
    let args = [
        "wake-word",
        "onboard",
        "computer",
        "--audio",
        wave.to_str().unwrap(),
        "--json",
    ];
    let output = run(&root, &args);
    assert!(output.status.success(), "{}", stderr(&output));
    let preview: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(preview["applied"], false);
    assert_eq!(preview["actions_executed"], false);
    assert_eq!(fs::read(&path).unwrap(), before);
    let alias = preview["review"]["proposals"][0]["text"].as_str().unwrap();
    assert!(!alias.is_empty());
    let mut rejected = args.to_vec();
    rejected.extend(["--apply", "--accept-alias", "not observed"]);
    let output = run(&root, &rejected);
    assert!(!output.status.success());
    assert_eq!(fs::read(&path).unwrap(), before);
    let mut apply = args.to_vec();
    apply.extend(["--apply", "--accept-alias", alias, "--keep-recordings"]);
    let output = run(&root, &apply);
    assert!(output.status.success(), "{}", stderr(&output));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["applied"], true);
    assert_eq!(Config::load(&path).unwrap().wake_words[0].aliases, [alias]);
    let retained = Path::new(report["recordings"].as_str().unwrap());
    assert!(retained.join("manifest.json").is_file());
    assert!(retained.join("sample-001.wav").is_file());
    assert_eq!(
        fs::read_dir(root.join("cache/omawake/onboarding"))
            .unwrap()
            .count(),
        0
    );
    let listed = run(&root, &["word", "recordings", "computer", "--json"]);
    assert!(listed.status.success());
    let sessions: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    let session = sessions[0]["session"].as_str().unwrap();
    let removed = run(
        &root,
        &["word", "recordings", "computer", "--remove", session],
    );
    assert!(removed.status.success(), "{}", stderr(&removed));
    assert!(!retained.exists());
    assert!(wave.is_file());
    assert_eq!(Config::load(&path).unwrap().wake_words[0].aliases, [alias]);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn onboarding_selects_named_profile_and_keeps_prior_heads() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (path, wave) = write_fake_whisper_config(&root, library);
    let mut config = Config::load(&path).unwrap();
    config.engines.insert(
        "spellings".into(),
        omawake::config::EngineProfile {
            backend: config.backend.clone(),
            model: config.model.clone(),
        },
    );
    config.backend.kind = "unavailable-default".into();
    config.wake_words[0].phrase = "Unusual name".into();
    let command = config.wake_words[0].command.clone();
    config.wake_words[0].enrollment = Some(omawake::enrollment::artifact::EnrollmentBinding {
        threshold: None,
        history: Default::default(),
        active: true,
        heads: [(
            "previous-encoder".into(),
            PathBuf::from("previous-head.json"),
        )]
        .into(),
    });
    config.save(&path).unwrap();
    let args = [
        "word",
        "onboard",
        "computer",
        "--engine",
        "spellings",
        "--audio",
        wave.to_str().unwrap(),
        "--json",
        "--apply",
        "--accept-alias",
        "computer",
    ];
    let output = run(&root, &args);
    assert!(output.status.success(), "{}", stderr(&output));
    let saved = Config::load(&path).unwrap();
    let word = &saved.wake_words[0];
    assert_eq!(word.engine.as_deref(), Some("spellings"));
    assert_eq!(word.command, command);
    assert!(!word.uses_trained_head());
    assert_eq!(
        word.enrollment.as_ref().unwrap().heads["previous-encoder"],
        PathBuf::from("previous-head.json")
    );
    assert_eq!(saved.backend.kind, "unavailable-default");
    let before = fs::read(&path).unwrap();
    let output = run(
        &root,
        &[
            "word",
            "onboard",
            "computer",
            "--engine",
            "missing",
            "--audio",
            wave.to_str().unwrap(),
            "--apply",
        ],
    );
    assert!(!output.status.success());
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn onboarding_pty_reviews_spellings_and_cancels_without_mutation() {
    use std::io::Read;
    use std::sync::{Arc, Mutex};
    for (cancel, create_new) in [(false, false), (true, false), (false, true)] {
        let root = sandbox();
        let library = build_fake_whisper(&root, None);
        let (path, wave) = write_fake_whisper_config(&root, library);
        let mut config = Config::load(&path).unwrap();
        config.wake_words[0].phrase = "Unusual name".into();
        config.engines.insert(
            "alternate".into(),
            omawake::config::EngineProfile {
                backend: config.backend.clone(),
                model: config.model.clone(),
            },
        );
        config.save(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let binary = env!("CARGO_BIN_EXE_omawake");
        assert!(!binary.contains(['\'', '"', ' ']));
        assert!(!wave.to_str().unwrap().contains(['\'', '"', ' ']));
        let mut child = Command::new("script")
            .args([
                "-qec",
                &format!("{binary} word onboard --audio {}", wave.display()),
                "/dev/null",
            ])
            .env("XDG_CONFIG_HOME", root.join("config"))
            .env("XDG_DATA_HOME", root.join("data"))
            .env("XDG_CACHE_HOME", root.join("cache"))
            .env("XDG_STATE_HOME", root.join("state"))
            .env("XDG_RUNTIME_DIR", root.join("run"))
            .env("PATH", test_path(&root))
            .env("TERM", "xterm-256color")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        let mut output = child.stdout.take().unwrap();
        let transcript = Arc::new(Mutex::new(Vec::new()));
        let copied = Arc::clone(&transcript);
        let reader = thread::spawn(move || {
            let mut b = [0; 4096];
            while let Ok(n) = output.read(&mut b) {
                if n == 0 {
                    break;
                }
                copied.lock().unwrap().extend_from_slice(&b[..n]);
            }
        });
        let final_keys = if cancel { "\x1b[B\r" } else { "\r" };
        let mut steps = vec![(
            "Wake-word onboarding",
            if create_new { "\x1b[B\r" } else { "\r" },
        )];
        if create_new {
            steps.extend([
                ("New word ID", "new-word\r"),
                ("Wake phrase", "Strange phrase\r"),
                ("Action arguments", "[\"/usr/bin/true\"]\r"),
            ]);
        }
        steps.extend([
            ("Recognition engine", "\x1b[B\r"),
            ("Heard:", "\x1b[A\r"),
            ("Recognition method", "\r"),
            ("Enrollment recordings", "\x1b[B\r"),
            ("Apply wake-word onboarding", final_keys),
        ]);
        for (screen, keys) in steps {
            let deadline = Instant::now() + Duration::from_secs(8);
            loop {
                if String::from_utf8_lossy(&transcript.lock().unwrap()).contains(screen) {
                    break;
                }
                if Instant::now() > deadline {
                    let _ = child.kill();
                    panic!(
                        "missing PTY screen {screen}: {}",
                        String::from_utf8_lossy(&transcript.lock().unwrap())
                    );
                }
                thread::sleep(Duration::from_millis(10));
            }
            input.write_all(keys.as_bytes()).unwrap();
            input.flush().unwrap();
        }
        drop(input);
        let deadline = Instant::now() + Duration::from_secs(5);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() > deadline {
                child.kill().unwrap();
                panic!("onboarding PTY did not terminate");
            }
            thread::sleep(Duration::from_millis(10));
        }
        let status = child.wait().unwrap();
        reader.join().unwrap();
        if cancel {
            assert!(!status.success());
            assert_eq!(fs::read(&path).unwrap(), before);
            assert!(!root.join("data/omawake/enrollments").exists());
        } else {
            assert!(status.success());
            let saved = Config::load(&path).unwrap();
            let word = saved.wake_words.last().unwrap();
            assert_eq!(word.aliases, ["computer"]);
            assert_eq!(word.engine.as_deref(), Some("alternate"));
            if create_new {
                assert_eq!(word.command, ["/usr/bin/true"]);
                assert_eq!(word.id, "new-word");
            } else {
                assert_eq!(word.command, config.wake_words[0].command);
            }
        }
        assert!(wave.is_file());
        assert_eq!(
            fs::read_dir(root.join("cache/omawake/onboarding"))
                .unwrap()
                .count(),
            0
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn onboarding_new_word_with_file_input_preserves_other_words() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (path, wave) = write_fake_whisper_config(&root, library);
    let old = Config::load(&path).unwrap();
    let output = run(
        &root,
        &[
            "word",
            "onboard",
            "strange",
            "--phrase",
            "Strange name",
            "--audio",
            wave.to_str().unwrap(),
            "--apply",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let config = Config::load(&path).unwrap();
    assert_eq!(config.wake_words.len(), 2);
    assert_eq!(config.wake_words[0].phrase, old.wake_words[0].phrase);
    assert_eq!(config.wake_words[1].command[0], "notify-send");
    let bytes = fs::read(&path).unwrap();
    let output = run(
        &root,
        &[
            "word",
            "onboard",
            "computer",
            "--phrase",
            "cannot replace",
            "--audio",
            wave.to_str().unwrap(),
            "--apply",
        ],
    );
    assert!(!output.status.success());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let output = run(&root, &["word", "recordings", "strange", "--json"]);
    assert!(output.status.success());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn resumed_onboarding_can_cancel_and_rejects_incompatible_encoder_before_recording() {
    for cancel in [true, false] {
        let root = sandbox();
        let library = build_fake_whisper(&root, None);
        let (config, wave) = write_fake_whisper_config(&root, library);
        let before = fs::read(&config).unwrap();
        let manifest = root.join("resume.json");
        let split = serde_json::json!([
            {"audio":wave,"positive":true},{"audio":wave,"positive":true},
            {"audio":wave,"positive":false},{"audio":wave,"positive":false}
        ]);
        fs::write(
            &manifest,
            serde_json::to_vec(
                &serde_json::json!({"training":split,"calibration":split,"validation":split}),
            )
            .unwrap(),
        )
        .unwrap();
        let output = Command::new("/usr/bin/python3")
            .arg("-c")
            .arg(
                r#"
import os,pty,select,sys,time,signal
binary,manifest,cancel=sys.argv[1:]
pid,fd=pty.fork()
if pid==0:
 os.execv(binary,[binary,'word','onboard','computer','--engine','default','--dataset',manifest])
data=b'';answered=False;deadline=time.monotonic()+12
while time.monotonic()<deadline:
 ready,_,_=select.select([fd],[],[],0.05)
 if ready:
  try: chunk=os.read(fd,65536)
  except OSError: chunk=b''
  data+=chunk
  if not answered and b'Resume enrollment' in data:
   os.write(fd,b'\x1b' if cancel=='yes' else b'\r');answered=True
 done,status=os.waitpid(pid,os.WNOHANG)
 if done:
  assert answered,data.decode(errors='replace')
  assert os.waitstatus_to_exitcode(status)!=0,data.decode(errors='replace')
  expected=b'configuration unchanged' if cancel=='yes' else b'initialize training encoder'
  assert expected in data,data.decode(errors='replace')
  assert b'Record fresh examples' not in data
  sys.exit(0)
os.kill(pid,signal.SIGTERM);os.waitpid(pid,0)
raise AssertionError(data.decode(errors='replace'))
"#,
            )
            .arg(env!("CARGO_BIN_EXE_omawake"))
            .arg(&manifest)
            .arg(if cancel { "yes" } else { "no" })
            .env("XDG_CONFIG_HOME", root.join("config"))
            .env("XDG_DATA_HOME", root.join("data"))
            .env("XDG_CACHE_HOME", root.join("cache"))
            .env("XDG_STATE_HOME", root.join("state"))
            .env("XDG_RUNTIME_DIR", root.join("run"))
            .env("TERM", "xterm-256color")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(fs::read(config).unwrap(), before);
        assert!(!root.join("data/omawake/heads").exists());
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn microphone_onboarding_exposes_training_before_capture_without_engine_flag() {
    for configured in [false, true] {
        let root = sandbox();
        let library = build_fake_whisper(&root, None);
        let (config, _) = write_fake_whisper_config(&root, library);
        if configured {
            let mut value: toml::Value =
                toml::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
            let mut backend = value["backend"].clone();
            backend["runtime"] = toml::Value::String("openvino".into());
            backend["device"] = toml::Value::String("cpu".into());
            let profile = toml::Value::Table(toml::map::Map::from_iter([
                ("backend".into(), backend),
                ("model".into(), value["model"].clone()),
            ]));
            value.as_table_mut().unwrap().insert(
                "engines".into(),
                toml::Value::Table(toml::map::Map::from_iter([("my-encoder".into(), profile)])),
            );
            fs::write(&config, toml::to_string(&value).unwrap()).unwrap();
        }
        let before = fs::read(&config).unwrap();
        let output = Command::new("/usr/bin/python3")
            .arg("-c")
            .arg(
                r#"
import os,pty,select,sys,time,signal
pid,fd=pty.fork()
if pid==0:
 os.execv(sys.argv[1],[sys.argv[1],'word','onboard','computer'])
data=b'';answered=False;training_answered=False;deployment_answered=False;deadline=time.monotonic()+12
while time.monotonic()<deadline:
 ready,_,_=select.select([fd],[],[],0.05)
 if ready:
  try: data+=os.read(fd,65536)
  except OSError: pass
  if not answered and b'Trainable KWS with Omaspeak assistance' in data:
   os.write(fd,b'\x1b[B\r');answered=True
  if not training_answered and b'Training device' in data:
   os.write(fd,b'\r');training_answered=True
  if not deployment_answered and b'Finished detector device' in data:
   os.write(fd,b'\r');deployment_answered=True
 done,status=os.waitpid(pid,os.WNOHANG)
 if done:
  assert answered,data.decode(errors='replace')
  assert os.waitstatus_to_exitcode(status)!=0
  expected=b'initialize training encoder' if sys.argv[2]=='yes' else b'No recordings were collected'
  assert expected in data,data.decode(errors='replace')
  assert b'Record example 1' not in data
  sys.exit(0)
os.kill(pid,signal.SIGTERM);os.waitpid(pid,0)
raise AssertionError(data.decode(errors='replace'))
"#,
            )
            .arg(env!("CARGO_BIN_EXE_omawake"))
            .arg(if configured { "yes" } else { "no" })
            .env("XDG_CONFIG_HOME", root.join("config"))
            .env("XDG_DATA_HOME", root.join("data"))
            .env("XDG_CACHE_HOME", root.join("cache"))
            .env("XDG_STATE_HOME", root.join("state"))
            .env("XDG_RUNTIME_DIR", root.join("run"))
            .env("TERM", "xterm-256color")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(fs::read(config).unwrap(), before);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn saved_training_dataset_is_discoverable_and_cancellable_without_a_path_flag() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (config, wave) = write_fake_whisper_config(&root, library);
    let before = fs::read(&config).unwrap();
    let session = root.join("data/omawake/enrollments/computer/saved-session");
    fs::create_dir_all(&session).unwrap();
    let split = serde_json::json!([
        {"audio":wave,"positive":true},{"audio":wave,"positive":true},
        {"audio":wave,"positive":false},{"audio":wave,"positive":false}
    ]);
    let manifest = session.join("manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(
            &serde_json::json!({"training":split,"calibration":split,"validation":split}),
        )
        .unwrap(),
    )
    .unwrap();
    let saved_before = fs::read(&manifest).unwrap();
    for mode in ["extend", "fresh", "resume"] {
        let output = Command::new("/usr/bin/python3").arg("-c").arg(r#"
import os,pty,select,sys,time,signal
pid,fd=pty.fork()
if pid==0:
 os.execv(sys.argv[1],[sys.argv[1],'word','onboard','computer'])
data=b'';stage=0;deadline=time.monotonic()+12
while time.monotonic()<deadline:
 ready,_,_=select.select([fd],[],[],0.05)
 if ready:
  try: data+=os.read(fd,65536)
  except OSError: pass
  if stage==0 and b'Add positive and negative examples' in data:
   os.write(fd,b'\x1b[B\r' if sys.argv[2]=='fresh' else b'\r');stage=1;data=b''
  if stage==1 and (b'Recognition method' if sys.argv[2]=='fresh' else b'Choose saved dataset') in data:
   os.write(fd,b'\r' if sys.argv[2]=='resume' else b'\x1b');stage=2
 done,status=os.waitpid(pid,os.WNOHANG)
 if done:
  assert stage==2,data.decode(errors='replace')
  assert os.waitstatus_to_exitcode(status)!=0
  assert b'configuration unchanged' in data,data.decode(errors='replace')
  sys.exit(0)
os.kill(pid,signal.SIGTERM);os.waitpid(pid,0)
raise AssertionError(data.decode(errors='replace'))
"#)
            .arg(env!("CARGO_BIN_EXE_omawake"))
            .arg(mode)
            .env("XDG_CONFIG_HOME", root.join("config"))
            .env("XDG_DATA_HOME", root.join("data"))
            .env("XDG_CACHE_HOME", root.join("cache"))
            .env("XDG_STATE_HOME", root.join("state"))
            .env("XDG_RUNTIME_DIR", root.join("run"))
            .env("TERM", "xterm-256color")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(fs::read(&config).unwrap(), before);
        assert_eq!(fs::read(&manifest).unwrap(), saved_before);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn threshold_and_history_commands_are_explicit_private_and_reversible() {
    use omawake::enrollment::{
        artifact::{self, EnrollmentBinding},
        head::{Example, Head},
        history::{self, Event, HistoryConfig, Label},
    };
    let root = sandbox();
    let mut config = Config::default();
    let config_file = root.join("config/omawake/config.toml");
    let split = |label: &str| {
        (0..4)
            .map(|i| Example {
                id: format!("{label}-{i}"),
                values: vec![if i < 2 { 1.0 } else { -1.0 }, 0.1],
                positive: i < 2,
            })
            .collect::<Vec<_>>()
    };
    let mut head = Head::train("test", &split("train"), &split("cal")).unwrap();
    head.validate_held_out(&split("held")).unwrap();
    let artifact = artifact::install(&root.join("heads"), &head).unwrap();
    config.wake_words[0].enrollment = Some(EnrollmentBinding {
        heads: [("test".into(), artifact)].into(),
        ..Default::default()
    });
    config.save(&config_file).unwrap();
    for value in ["0.9", "auto"] {
        let out = run(&root, &["word", "threshold", "computer", value]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let saved = Config::load(&config_file).unwrap();
        assert_eq!(
            saved.wake_words[0].enrollment.as_ref().unwrap().threshold,
            if value == "auto" { None } else { Some(0.9) }
        );
    }
    let before = fs::read(&config_file).unwrap();
    for value in ["0", "1.1", "NaN", "nonsense"] {
        assert!(
            !run(&root, &["word", "threshold", "computer", value])
                .status
                .success()
        );
        assert_eq!(fs::read(&config_file).unwrap(), before);
    }
    assert!(
        run(&root, &["word", "threshold", "computer"])
            .status
            .success()
    );
    assert!(
        !run(&root, &["word", "threshold", "unknown"])
            .status
            .success()
    );
    assert!(
        run(
            &root,
            &["word", "history", "computer", "enable", "--max-events", "2"]
        )
        .status
        .success()
    );
    let saved = Config::load(&config_file).unwrap();
    assert!(
        saved.wake_words[0]
            .enrollment
            .as_ref()
            .unwrap()
            .history
            .enabled
    );
    let paths = omawake::paths::AppPaths {
        config_file: config_file.clone(),
        data_dir: root.join("data/omawake"),
        cache_dir: root.join("cache/omawake"),
        state_dir: root.join("state/omawake"),
        runtime_dir: root.join("run/omawake"),
    };
    let event = Event {
        id: String::new(),
        word_id: "computer".into(),
        created_ms: 0,
        score: 0.9,
        threshold: 0.8,
        encoder_contract: "test".into(),
        head: "test-head".into(),
        device: "CPU".into(),
        label: Label::Unreviewed,
        audio: PathBuf::new(),
    };
    let id = history::record(
        &paths,
        &HistoryConfig {
            enabled: true,
            max_events: 2,
        },
        event,
        &[1000; 1600],
    )
    .unwrap()
    .unwrap();
    let out = run(&root, &["word", "history", "computer", "list", "--json"]);
    assert!(out.status.success());
    let events: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(events[0]["label"], "unreviewed");
    assert!(
        run(&root, &["word", "history", "computer", "list"])
            .status
            .success()
    );
    assert!(
        run(
            &root,
            &[
                "word",
                "history",
                "computer",
                "label",
                &id,
                "false-positive"
            ]
        )
        .status
        .success()
    );
    assert_eq!(
        history::list(&paths, "computer").unwrap()[0].label,
        Label::FalsePositive
    );
    let player = root.join("test-bin/pw-play");
    fs::write(&player, "#!/bin/sh\n[ -f \"$1\" ]\n").unwrap();
    fs::set_permissions(&player, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        run(&root, &["word", "history", "computer", "play", &id])
            .status
            .success()
    );
    assert!(
        !run(&root, &["word", "history", "computer", "play", "unknown"])
            .status
            .success()
    );
    assert!(
        run(&root, &["word", "history", "computer", "disable"])
            .status
            .success()
    );
    assert!(
        !Config::load(&config_file).unwrap().wake_words[0]
            .enrollment
            .as_ref()
            .unwrap()
            .history
            .enabled
    );
    assert_eq!(history::list(&paths, "computer").unwrap().len(), 1);
    assert!(
        run(&root, &["word", "history", "computer", "clear"])
            .status
            .success()
    );
    assert!(history::list(&paths, "computer").unwrap().is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn daemon_recovers_a_pinned_microphone_and_keeps_controls_responsive() {
    let root = sandbox();
    let library = build_fake_whisper(&root, None);
    let (config_path, _) = write_fake_whisper_config(&root, library);
    let mut config = Config::load(&config_path).unwrap();
    config.audio.device = "pipewire:test-mic".into();
    config.wake_words[0].command = vec!["true".into()];
    config.save(&config_path).unwrap();
    let inventory = root.join("inventory.json");
    fs::write(&inventory, "[]").unwrap();
    let dump = root.join("test-bin/pw-dump");
    fs::write(&dump, "#!/bin/sh\ncat \"$OMAWAKE_TEST_INVENTORY\"\n").unwrap();
    fs::set_permissions(&dump, fs::Permissions::from_mode(0o755)).unwrap();
    let recorder = root.join("test-bin/pw-record");
    fs::write(&recorder, "#!/usr/bin/python3\nimport sys,time,os\nopen(os.environ['OMAWAKE_TEST_RECORDER_PID'],'w').write(str(os.getpid()))\nwhile True:\n sys.stdout.buffer.write(bytes(1024));sys.stdout.buffer.flush();time.sleep(0.016)\n").unwrap();
    fs::set_permissions(&recorder, fs::Permissions::from_mode(0o755)).unwrap();
    let pid_file = root.join("recorder.pid");
    struct Daemon(std::process::Child);
    impl Drop for Daemon {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut daemon = Daemon(
        Command::new(env!("CARGO_BIN_EXE_omawake"))
            .args(["--config", config_path.to_str().unwrap(), "daemon"])
            .env("XDG_CONFIG_HOME", root.join("config"))
            .env("XDG_DATA_HOME", root.join("data"))
            .env("XDG_CACHE_HOME", root.join("cache"))
            .env("XDG_STATE_HOME", root.join("state"))
            .env("XDG_RUNTIME_DIR", root.join("run"))
            .env("PATH", test_path(&root))
            .env("OMAWAKE_TEST_INVENTORY", &inventory)
            .env("OMAWAKE_TEST_RECORDER_PID", &pid_file)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let wait_state = |wanted: &str| {
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            let out = run(&root, &["status", "--json"]);
            let state: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            if state["state"] == wanted {
                return state;
            }
            assert!(Instant::now() < deadline, "did not reach {wanted}: {state}");
            thread::sleep(Duration::from_millis(50));
        }
    };
    let unavailable = wait_state("audio_unavailable");
    assert_eq!(
        unavailable["details"]["audio"]["requested"],
        "pipewire:test-mic"
    );
    assert_eq!(unavailable["details"]["audio"]["available"], false);
    assert!(run(&root, &["pause"]).status.success());
    wait_state("paused");
    assert!(run(&root, &["resume"]).status.success());
    wait_state("audio_unavailable");
    let connected = r#"[{"type":"PipeWire:Interface:Node","info":{"props":{"media.class":"Audio/Source","node.name":"test-mic"}}}]"#;
    fs::write(&inventory, connected).unwrap();
    let armed = wait_state("armed");
    assert_eq!(armed["details"]["audio"]["effective"], "pipewire:test-mic");
    let deadline = Instant::now() + Duration::from_secs(2);
    while !pid_file.exists() {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(10));
    }
    fs::write(&inventory, "[]").unwrap();
    let pid: i32 = fs::read_to_string(&pid_file)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(unsafe { libc::kill(pid, libc::SIGTERM) }, 0);
    wait_state("audio_unavailable");
    fs::write(&inventory, connected).unwrap();
    wait_state("armed");
    config.audio.device = "pipewire:saved-but-not-active".into();
    config.save(&config_path).unwrap();
    let status = wait_state("armed");
    assert_eq!(status["details"]["audio"]["requested"], "pipewire:test-mic");
    assert_eq!(
        status["details"]["audio"]["saved"],
        "pipewire:saved-but-not-active"
    );
    assert_eq!(status["details"]["audio"]["restart_required"], true);
    assert!(run(&root, &["stop"]).status.success());
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(status) = daemon.0.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        assert!(Instant::now() < deadline, "daemon did not stop");
        thread::sleep(Duration::from_millis(25));
    }
}
