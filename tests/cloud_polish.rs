use omawake::config::Config;
use std::os::unix::fs::PermissionsExt;
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
fn root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("omawake-polish-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}
fn cli(root: &std::path::Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_omawake"));
    c.arg("--config")
        .arg(root.join("config.toml"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env_remove("DEEPGRAM_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("ELEVENLABS_API_KEY")
        .env_remove("CARTESIA_API_KEY")
        .stdin(Stdio::null());
    c
}
#[test]
fn cloud_setup_key_handoff_and_offline_checks() {
    let root = root("credentials");
    let out = cli(&root)
        .args(["setup", "cloud", "--provider", "deepgram"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let config = Config::load(&root.join("config.toml")).unwrap();
    assert_eq!(config.backend.device, "remote");
    assert_eq!(config.backend.kind, "deepgram");
    let out = cli(&root)
        .args(["cloud", "credential", "check"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"credential_available\":false"));
    let mut child = cli(&root)
        .args(["cloud", "credential", "install", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"private-fixture-secret\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!String::from_utf8_lossy(&out.stderr).contains("private-fixture-secret"));
    let config = Config::load(&root.join("config.toml")).unwrap();
    let key = PathBuf::from(&config.backend.cloud.api_key_file);
    assert_eq!(
        fs::metadata(&key).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(
        !fs::read_to_string(root.join("config.toml"))
            .unwrap()
            .contains("private-fixture-secret")
    );
    let out = cli(&root)
        .args(["cloud", "credential", "check"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"credential_available\":true"));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("private-fixture-secret"));
    let out = cli(&root)
        .args(["setup", "check", "--json"])
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&out.stdout).contains("private-fixture-secret"));
    fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(
        !cli(&root)
            .args(["cloud", "credential", "check"])
            .status()
            .unwrap()
            .success()
    );
    fs::remove_file(&key).unwrap();
    std::os::unix::fs::symlink("config.toml", &key).unwrap();
    assert!(
        !cli(&root)
            .args(["cloud", "credential", "check"])
            .status()
            .unwrap()
            .success()
    );
    let out = cli(&root)
        .args(["config", "schema", "--json"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("backend.cloud.api_key_file"));
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn invalid_cloud_setup_never_overwrites_config_or_accepts_secret_arguments() {
    let root = root("invalid");
    let path = root.join("config.toml");
    Config::default().save(&path).unwrap();
    let before = fs::read(&path).unwrap();
    for args in [
        vec!["setup", "cloud", "--provider", "bad"],
        vec![
            "setup",
            "cloud",
            "--provider",
            "deepgram",
            "--base-url",
            "http://example.com",
        ],
        vec![
            "setup",
            "cloud",
            "--provider",
            "deepgram",
            "--api-key-env",
            "BAD\nKEY",
        ],
        vec!["setup", "cloud"],
        vec!["cloud", "credential", "install"],
    ] {
        assert!(!cli(&root).args(args).output().unwrap().status.success());
        assert_eq!(fs::read(&path).unwrap(), before);
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn realtime_setup_is_explicit_preserves_selection_and_rejects_other_providers() {
    let root = root("realtime-setup");
    let path = root.join("config.toml");
    assert!(
        cli(&root)
            .args(["setup", "cloud", "--provider", "deepgram", "--realtime"])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(Config::load(&path).unwrap().backend.cloud.realtime);
    assert!(
        cli(&root)
            .args([
                "setup",
                "cloud",
                "--provider",
                "deepgram",
                "--model",
                "nova-3"
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(Config::load(&path).unwrap().backend.cloud.realtime);
    assert!(
        cli(&root)
            .args([
                "setup",
                "cloud",
                "--provider",
                "deepgram",
                "--realtime=false"
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(!Config::load(&path).unwrap().backend.cloud.realtime);
    assert!(
        !cli(&root)
            .args([
                "setup",
                "cloud",
                "--provider",
                "openai-compatible",
                "--realtime"
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(Config::load(&path).unwrap().backend.kind, "deepgram");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn daemon_credential_check_handles_stopped_and_wrong_process_services() {
    let root = root("daemon-check");
    let bin = root.join("bin");
    fs::create_dir_all(&bin).unwrap();
    let script = bin.join("systemctl");
    assert!(
        cli(&root)
            .args(["setup", "cloud", "--provider", "deepgram"])
            .output()
            .unwrap()
            .status
            .success()
    );
    let path = std::env::join_paths(
        std::iter::once(bin.clone())
            .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
    )
    .unwrap();
    for body in [
        "#!/bin/sh\necho 0\n".to_owned(),
        format!("#!/bin/sh\necho {}\n", std::process::id()),
    ] {
        fs::write(&script, body).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let out = cli(&root)
            .args(["cloud", "credential", "check", "--daemon"])
            .env("PATH", &path)
            .output()
            .unwrap();
        assert!(!out.status.success());
    }
    let mut daemon = Command::new("python3")
        .args(["-c", "import time; time.sleep(20)", "--config"])
        .arg(root.join("config.toml"))
        .arg("daemon")
        .env("DEEPGRAM_API_KEY", "daemon-fixture-secret")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    fs::write(&script, format!("#!/bin/sh\necho {}\n", daemon.id())).unwrap();
    let out = cli(&root)
        .args(["cloud", "credential", "check", "--daemon"])
        .env("PATH", &path)
        .output()
        .unwrap();
    daemon.kill().unwrap();
    daemon.wait().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"credential_available\":true"));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("daemon-fixture-secret"));
    fs::remove_dir_all(root).unwrap();
}
