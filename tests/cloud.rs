#![cfg(target_os = "linux")]
use omawake::config::Config;
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};
fn root(name: &str, kind: &str, port: u16) -> PathBuf {
    let root = std::env::temp_dir().join(format!("omawake-cloud-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let mut c = Config::default();
    c.backend.kind = kind.into();
    c.backend.device = "remote".into();
    c.backend.cloud.base_url = format!("http://127.0.0.1:{port}/v1");
    c.backend.cloud.api_key_env = "OMAWAKE_CLOUD_TEST_KEY".into();
    c.backend.cloud.endpoint_milliseconds = 100;
    c.wake_words = vec![omawake::config::WakeWord {
        id: "test".into(),
        phrase: "hey computer".into(),
        command: vec!["true".into()],
        enrollment: None,
        aliases: vec![],
        enabled: true,
        engine: None,
    }];
    c.save(&root.join("config/omawake/config.toml")).unwrap();
    let mut wav = hound::WavWriter::create(
        root.join("audio.wav"),
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..3200 {
        wav.write_sample(8000i16).unwrap();
    }
    for _ in 0..3200 {
        wav.write_sample(0i16).unwrap();
    }
    wav.finalize().unwrap();
    root
}
fn command(root: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_omawake"));
    c.env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMAWAKE_CLOUD_TEST_KEY", "test-secret")
        .stdin(Stdio::null());
    c
}
fn server(
    listener: TcpListener,
    body: &'static str,
    status: &'static str,
) -> thread::JoinHandle<(String, Vec<u8>)> {
    thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let (mut s, _) = loop {
            match listener.accept() {
                Ok(pair) => break pair,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "cloud request never reached mock server"
                    );
                    thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(error) => panic!("mock accept: {error}"),
            }
        };
        s.set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut r = BufReader::new(&mut s);
        let mut headers = String::new();
        let mut len = 0;
        loop {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if line.to_ascii_lowercase().starts_with("content-length:") {
                len = line.split(':').nth(1).unwrap().trim().parse().unwrap();
            }
            headers.push_str(&line);
        }
        let mut request = vec![0; len];
        r.read_exact(&mut request).unwrap();
        write!(s,"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
        (headers, request)
    })
}
#[test]
fn both_cloud_transcribers_upload_bounded_audio_and_match_complete_local_phrases() {
    for (kind, body) in [
        (
            "deepgram",
            r#"{"results":{"channels":[{"alternatives":[{"transcript":"Hey computer, hello."}]}]}}"#,
        ),
        ("openai-compatible", r#"{"text":"Hey computer, hello."}"#),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let root = root(kind, kind, listener.local_addr().unwrap().port());
        let server = server(listener, body, "200 OK");
        let out = command(&root)
            .args(["test", "--audio"])
            .arg(root.join("audio.wav"))
            .arg("--json")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{kind}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let data: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert!(data.to_string().contains("test"));
        assert!(data.to_string().contains("computer"));
        let (headers, request) = server.join().unwrap();
        assert!(headers.contains("test-secret"));
        assert!(request.len() < 16000);
        if kind == "deepgram" {
            assert!(headers.starts_with("POST /v1/listen?"));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("authorization: token test-secret")
            );
            assert!(request.starts_with(b"RIFF"));
        } else {
            assert!(headers.starts_with("POST /v1/audio/transcriptions "));
            assert!(headers.contains("multipart/form-data"));
            assert!(String::from_utf8_lossy(&request).contains("gpt-4o-mini-transcribe"));
            assert!(request.windows(4).any(|w| w == b"RIFF"));
        }
        fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn silence_is_not_uploaded_and_partial_phrase_does_not_trigger() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let root = root(
        "partial",
        "openai-compatible",
        listener.local_addr().unwrap().port(),
    );
    let server = server(listener, r#"{"text":"computerized"}"#, "200 OK");
    let out = command(&root)
        .args(["test", "--audio"])
        .arg(root.join("audio.wav"))
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    server.join().unwrap();
    let data: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(data["detections"], serde_json::json!([]));
    let mut w = hound::WavWriter::create(
        root.join("silence.wav"),
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..16000 {
        w.write_sample(0i16).unwrap();
    }
    w.finalize().unwrap();
    let out = command(&root)
        .args(["test", "--audio"])
        .arg(root.join("silence.wav"))
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "silence should finish without connecting: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn provider_failure_emits_no_detection_and_redacts_response_body() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let root = root("error", "deepgram", listener.local_addr().unwrap().port());
    let server = server(
        listener,
        "test-secret echoed audio text",
        "401 Unauthorized",
    );
    let out = command(&root)
        .args(["test", "--audio"])
        .arg(root.join("audio.wav"))
        .arg("--json")
        .output()
        .unwrap();
    server.join().unwrap();
    assert!(!out.status.success());
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!msg.contains("test-secret"));
    assert!(!msg.contains("echoed audio text"));
    fs::remove_dir_all(root).unwrap();
}
