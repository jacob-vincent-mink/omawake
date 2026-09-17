use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "omawake-audio-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("bin")).unwrap();
        let result = Self { root };
        result.script(
            "pw-dump",
            "#!/bin/sh\ncat \"$AUDIO_FIXTURE/inventory.json\"\n",
        );
        fs::write(result.root.join("inventory.json"), r#"[{"type":"PipeWire:Interface:Node","info":{"props":{"media.class":"Audio/Source","node.name":"other-device","node.description":"USB device"}}},{"type":"PipeWire:Interface:Node","info":{"props":{"media.class":"Audio/Source","node.name":"test-device","node.description":"USB device"}}}]"#).unwrap();
        result
    }
    fn script(&self, name: &str, source: &str) {
        let p = self.root.join("bin").join(name);
        fs::write(&p, source).unwrap();
        fs::set_permissions(p, fs::Permissions::from_mode(0o755)).unwrap();
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_omawake"))
            .arg("--config")
            .arg(self.root.join("config.toml"))
            .args(args)
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            )
            .env("AUDIO_FIXTURE", &self.root)
            .output()
            .unwrap()
    }
    fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn device_configuration_discovery_and_schema_are_model_free() {
    let f = Fixture::new();
    f.ok(&["config", "set", "audio.device", "pipewire:test-device"]);
    assert_eq!(
        f.ok(&["config", "get", "audio.device"]).trim(),
        "pipewire:test-device"
    );
    let report: serde_json::Value =
        serde_json::from_str(&f.ok(&["audio-devices", "--detailed", "--json"])).unwrap();
    assert_eq!(report["selected"], "pipewire:test-device");
    assert!(
        report["devices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["selector"] == "pipewire:test-device" && d["available"] == true)
    );
    let schema: serde_json::Value =
        serde_json::from_str(&f.ok(&["config", "schema", "--json"])).unwrap();
    let audio = schema["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["key"] == "audio.device")
        .unwrap();
    assert_eq!(audio["type"], "enum");
    assert_eq!(audio["restart_required"], true);
    assert!(
        audio["choices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["value"] == "pipewire:test-device")
    );
    f.ok(&["config", "set", "audio.device", "pipewire:offline"]);
    let before = fs::read(f.root.join("config.toml")).unwrap();
    assert!(
        !f.run(&["config", "set", "audio.device", "pipewire:123"])
            .status
            .success()
    );
    assert_eq!(fs::read(f.root.join("config.toml")).unwrap(), before);
    f.ok(&["setup", "audio", "--device", "default"]);
    assert_eq!(fs::read(f.root.join("config.toml")).unwrap(), before);
    f.ok(&["setup", "audio", "--device", "default", "--apply"]);
    assert_eq!(f.ok(&["config", "get", "audio.device"]).trim(), "default");
    f.ok(&["config", "unset", "audio.device"]);
}

#[test]
fn disconnected_and_broken_inventory_remain_readable() {
    let f = Fixture::new();
    f.ok(&["config", "set", "audio.device", "pipewire:offline"]);
    f.script("pw-dump", "#!/bin/sh\nexit 1\n");
    let report: serde_json::Value =
        serde_json::from_str(&f.ok(&["audio-devices", "--detailed", "--json"])).unwrap();
    assert!(
        report["devices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["selector"] == "pipewire:offline" && d["available"] == false)
    );
    f.ok(&["config", "schema", "--json"]);
    assert!(!f.run(&["setup", "audio", "--test"]).status.success());
    f.script("pw-dump", "#!/bin/sh\nprintf invalid\n");
    f.ok(&["audio-devices", "--detailed", "--json"]);
}

#[test]
fn pinned_capture_checks_levels_and_closes_the_recorder() {
    let f = Fixture::new();
    f.script("pw-record", "#!/usr/bin/python3\nimport os,sys,struct,time\nroot=os.environ['AUDIO_FIXTURE']\nopen(root+'/args','w').write('\\n'.join(sys.argv[1:]))\nopen(root+'/pid','w').write(str(os.getpid()))\nwhile True:\n sys.stdout.buffer.write(struct.pack('f',0.25)*256);sys.stdout.buffer.flush();time.sleep(0.016)\n");
    let out = f.ok(&[
        "setup",
        "audio",
        "--device",
        "pipewire:test-device",
        "--test",
    ]);
    assert!(out.contains("RMS 0.2500"));
    let args = fs::read_to_string(f.root.join("args")).unwrap();
    assert!(args.starts_with("--target\ntest-device\n--properties\n"));
    assert!(args.contains("node.dont-fallback = true"));
    let pid: i32 = fs::read_to_string(f.root.join("pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert!(!f.root.join("config.toml").exists());
}

#[test]
fn recorder_exit_is_an_audio_error() {
    let f = Fixture::new();
    f.script("pw-record", "#!/bin/sh\nexit 1\n");
    let out = f.run(&[
        "setup",
        "audio",
        "--device",
        "pipewire:test-device",
        "--test",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("capture"));
}

#[test]
fn interactive_audio_test_apply_and_cancel_preserve_the_selection() {
    let f = Fixture::new();
    f.ok(&["config", "set", "audio.device", "pipewire:test-device"]);
    f.script("pw-record", "#!/usr/bin/python3\nimport sys,struct,time\nwhile True:\n sys.stdout.buffer.write(struct.pack('f',0.25)*256);sys.stdout.buffer.flush();time.sleep(0.016)\n");
    let harness = r#"
import os,pty,select,sys,time,signal
pid,fd=pty.fork()
if pid==0: os.execv(sys.argv[1],[sys.argv[1],'--config',sys.argv[2],'setup','audio'])
pending=b''
def until(needle):
 global pending
 deadline=time.monotonic()+10
 while needle not in pending:
  if time.monotonic()>deadline: raise RuntimeError('missing '+repr(needle)+' '+repr(pending))
  if select.select([fd],[],[],0.1)[0]: pending+=os.read(fd,65536)
 pending=pending.split(needle,1)[1]
try:
 until(b'Audio device')
 os.write(fd,b'\r')
 until(b'Apply audio device')
 os.write(fd,b'\x1b[B\r')
 until(b'RMS 0.2500')
 until(b'Apply audio device')
 os.write(fd,b'\r' if sys.argv[3]=='apply' else b'\x1b')
 until(b'Audio device saved' if sys.argv[3]=='apply' else b'Setup cancelled')
 _,status=os.waitpid(pid,0)
 assert os.waitstatus_to_exitcode(status)==0
finally:
 try: os.kill(pid,signal.SIGKILL)
 except ProcessLookupError: pass
 os.close(fd)
"#;
    for action in ["apply", "cancel"] {
        let before = fs::read(f.root.join("config.toml")).unwrap();
        let out = Command::new("python3")
            .args(["-c", harness, env!("CARGO_BIN_EXE_omawake")])
            .arg(f.root.join("config.toml"))
            .arg(action)
            .env("TERM", "xterm-256color")
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", f.root.join("bin").display()),
            )
            .env("AUDIO_FIXTURE", &f.root)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(fs::read(f.root.join("config.toml")).unwrap(), before);
    }
}
