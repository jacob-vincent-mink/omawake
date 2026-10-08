use omawake::config::{Config, WakeWord};
use serde_json::json;
use std::{
    fs,
    io::Write,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};
use tungstenite::Message;
fn setup(name: &str, port: u16) -> PathBuf {
    let root = std::env::temp_dir().join(format!("omawake-ws-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let mut c = Config::default();
    c.backend.kind = "deepgram".into();
    c.backend.device = "remote".into();
    c.backend.cloud.base_url = format!("http://127.0.0.1:{port}/v1");
    c.backend.cloud.api_key_env = "OMAWAKE_REALTIME_FIXTURE_KEY".into();
    c.backend.cloud.realtime = true;
    c.backend.cloud.timeout_seconds = 2;
    c.wake_words = vec![WakeWord {
        id: "wake".into(),
        phrase: "hey computer".into(),
        command: vec![
            "touch".into(),
            root.join("must-not-run").to_string_lossy().into_owned(),
        ],
        enrollment: None,
        aliases: vec![],
        enabled: true,
        engine: None,
    }];
    c.save(&root.join("config.toml")).unwrap();
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
        wav.write_sample(1000i16).unwrap();
    }
    wav.finalize().unwrap();
    root
}
fn smoke(root: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_omawake"))
        .arg("--config")
        .arg(root.join("config.toml"))
        .args(["cloud", "smoke", "--audio"])
        .arg(root.join("audio.wav"))
        .env("OMAWAKE_REALTIME_FIXTURE_KEY", "fixture-secret")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}
fn final_msg(start: f64, text: &str, endpoint: bool) -> Message {
    Message::Text(json!({"type":"Results","start":start,"duration":0.1,"is_final":true,"speech_final":endpoint,"channel":{"alternatives":[{"transcript":text}]}}).to_string().into())
}
// tungstenite's callback fixes this error type in its public API.
#[allow(clippy::result_large_err)]
#[test]
fn websocket_smoke_sends_pcm_accumulates_finals_ignores_interims_and_never_executes_actions() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let root = setup("finals", listener.local_addr().unwrap().port());
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut ws = tungstenite::accept_hdr(
            stream,
            |request: &tungstenite::handshake::server::Request, response| {
                assert_eq!(request.headers()["Authorization"], "Token fixture-secret");
                let uri = request.uri().to_string();
                assert!(uri.starts_with("/v1/listen?"));
                assert!(uri.contains("encoding=linear16"));
                assert!(uri.contains("endpointing=500"));
                assert!(!uri.contains("fixture-secret"));
                Ok(response)
            },
        )
        .unwrap();
        let mut count = 0;
        loop {
            match ws.read().unwrap() {
                Message::Binary(bytes) => {
                    assert_eq!(bytes.len() % 2, 0);
                    assert!(bytes.len() <= 3200);
                    count += bytes.len() / 2;
                    if count == 1600 {
                        ws.send(Message::Text(json!({"type":"Results","is_final":false,"speech_final":true,"channel":{"alternatives":[{"transcript":"hey computer"}]}}).to_string().into())).unwrap();
                        ws.send(final_msg(0.0, "hey", false)).unwrap();
                        ws.send(final_msg(0.0, "hey", false)).unwrap();
                        ws.send(final_msg(0.1, "computer", true)).unwrap();
                        ws.send(final_msg(0.1, "computer", true)).unwrap();
                    }
                }
                Message::Text(text) if text.contains("CloseStream") => {
                    ws.send(Message::Text(
                        json!({"type":"Metadata","duration":count as f64/16000.0})
                            .to_string()
                            .into(),
                    ))
                    .unwrap();
                    break;
                }
                _ => (),
            }
        }
        count
    });
    let output = smoke(&root);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(data["actions_executed"], false);
    assert_eq!(data["detections"].as_array().unwrap().len(), 1);
    assert_eq!(data["detections"][0]["id"], "wake");
    assert!(server.join().unwrap() >= 3200);
    assert!(!root.join("must-not-run").exists());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn disconnect_provider_error_and_timeout_fail_without_retry_or_leaking_error_bodies() {
    for mode in ["disconnect", "error", "timeout", "bad-json"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let root = setup(mode, listener.local_addr().unwrap().port());
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut ws = tungstenite::accept(stream).unwrap();
            if mode == "disconnect" {
                ws.close(None).unwrap();
                return;
            }
            let _ = ws.read();
            if mode == "error" {
                ws.send(Message::Text(
                    json!({"type":"Error","description":"fixture-secret"})
                        .to_string()
                        .into(),
                ))
                .unwrap();
            } else if mode == "bad-json" {
                ws.send(Message::Text("fixture-secret-not-json".into()))
                    .unwrap();
            } else {
                thread::sleep(Duration::from_secs(3));
            }
        });
        let output = smoke(&root);
        assert!(!output.status.success(), "mode {mode}");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-secret"));
        assert!(!root.join("must-not-run").exists());
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn worker_rejects_oversized_input_and_remote_insecure_urls() {
    let config = json!({"kind":"deepgram","cloud":{"realtime":true,"base_url":"http://example.com"},"language":"en"});
    let output = Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(["__cloud-stream-worker", &config.to_string()])
        .env("DEEPGRAM_API_KEY", "fixture-secret")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-secret"));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut ws = tungstenite::accept(stream).unwrap();
        let _ = ws.read();
    });
    let config = json!({"kind":"deepgram","cloud":{"realtime":true,"base_url":format!("http://127.0.0.1:{port}/v1"),"timeout_seconds":1},"language":"en"});
    let mut child = Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(["__cloud-stream-worker", &config.to_string()])
        .env("DEEPGRAM_API_KEY", "fixture-secret")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = child
        .stdin
        .take()
        .unwrap()
        .write_all(&vec![b'x'; 1_048_577]);
    assert!(!child.wait_with_output().unwrap().status.success());
    server.join().unwrap();
}
#[test]
fn idle_session_heartbeats_keep_the_connection_alive_until_explicit_finish() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(6)))
            .unwrap();
        let mut ws = tungstenite::accept(stream).unwrap();
        let mut pings = 0;
        let mut keepalives = 0;
        loop {
            match ws.read().unwrap() {
                Message::Ping(_) => {
                    pings += 1;
                    ws.flush().unwrap();
                }
                Message::Text(text) if text.contains("KeepAlive") => {
                    keepalives += 1;
                }
                Message::Text(text) if text.contains("CloseStream") => {
                    ws.send(Message::Text(json!({"type":"Metadata"}).to_string().into()))
                        .unwrap();
                    break;
                }
                _ => (),
            }
        }
        (pings, keepalives)
    });
    let config = json!({"kind":"deepgram","cloud":{"realtime":true,"base_url":format!("http://127.0.0.1:{port}/v1"),"timeout_seconds":2},"language":"en"});
    let mut child = Command::new(env!("CARGO_BIN_EXE_omawake"))
        .args(["__cloud-stream-worker", &config.to_string()])
        .env("DEEPGRAM_API_KEY", "fixture-secret")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    thread::sleep(Duration::from_millis(3500));
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (pings, keepalives) = server.join().unwrap();
    assert!(pings >= 2);
    assert!(keepalives >= 1);
}
