use std::fs;
#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

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
        "omawake-onnx"
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
            &[
                "config",
                "set",
                "backend.provider_library",
                "/tmp/provider.so",
            ],
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
    assert!(saved.backend.provider_library.as_os_str().is_empty());
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
