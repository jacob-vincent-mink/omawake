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
