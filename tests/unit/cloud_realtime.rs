use super::*;
use serde_json::json;
fn result(
    start: f64,
    duration: f64,
    text: &str,
    finalized: bool,
    endpoint: bool,
) -> serde_json::Value {
    json!({"type":"Results","start":start,"duration":duration,"is_final":finalized,"speech_final":endpoint,"channel":{"alternatives":[{"transcript":text}]}})
}
#[test]
fn finals_accumulate_only_final_segments_and_emit_once_at_endpoint() {
    let mut finals = Finals::default();
    assert!(
        finals
            .accept(&result(0.0, 1.0, "wrong interim", false, true), 30)
            .unwrap()
            .is_none()
    );
    assert!(
        finals
            .accept(&result(0.0, 1.0, "hey", true, false), 30)
            .unwrap()
            .is_none()
    );
    assert!(
        finals
            .accept(&result(0.0, 1.0, "duplicate", true, false), 30)
            .unwrap()
            .is_none()
    );
    let reply = finals
        .accept(&result(1.0, 1.0, "computer", true, true), 30)
        .unwrap()
        .unwrap();
    assert_eq!(reply.text.as_deref(), Some("hey computer"));
    assert_eq!(reply.start_sample, 0);
    assert!(
        finals
            .accept(&result(1.0, 1.0, "computer", true, true), 30)
            .unwrap()
            .is_none()
    );
    assert!(
        finals
            .accept(&json!({"type":"UtteranceEnd"}), 30)
            .unwrap()
            .is_none()
    );
    let reply = finals
        .accept(&result(2.0, 1.0, "hello again", true, true), 30)
        .unwrap()
        .unwrap();
    assert_eq!(reply.start_sample, 32000);
}
#[test]
fn finals_reject_errors_oversize_malformed_and_unbounded_utterances() {
    let mut f = Finals::default();
    assert!(
        f.accept(&json!({"type":"Error","description":"secret"}), 30)
            .unwrap_err()
            .to_string()
            .find("secret")
            .is_none()
    );
    assert!(
        f.accept(&json!({"type":"SpeechStarted"}), 30)
            .unwrap()
            .is_none()
    );
    assert!(
        f.accept(&json!({"type":"Results","is_final":true}), 30)
            .is_err()
    );
    assert!(f.accept(&result(-1.0, 1.0, "hey", true, true), 30).is_err());
    assert!(f.accept(&result(0.0, -1.0, "hey", true, true), 30).is_err());
    assert!(
        f.accept(&result(0.0, 32.0, "hey", true, false), 30)
            .is_err()
    );
    let mut f = Finals::default();
    assert!(
        f.accept(&result(0.0, 1.0, &"x".repeat(65537), true, true), 30)
            .is_err()
    );
    let mut f = Finals::default();
    assert!(
        f.accept(&result(0.0, 1.0, "", true, true), 30)
            .unwrap()
            .is_none()
    );
    assert!(
        f.accept(
            &json!({"type":"Results","is_final":true,"start":1,"duration":1}),
            30
        )
        .is_err()
    );
}
#[test]
fn realtime_configuration_and_stream_lifecycle_fail_closed() {
    let mut config = Config::default();
    config.backend.kind = "openai-compatible".into();
    config.backend.cloud.realtime = true;
    assert!(CloudBackend::load(&config).is_err());
    config.backend.kind = "deepgram".into();
    config.backend.cloud.api_key_file = std::env::temp_dir()
        .join(format!("omawake-rt-key-{}", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let key = Path::new(&config.backend.cloud.api_key_file);
    std::fs::write(key, b"test").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let backend = CloudBackend::load(&config).unwrap();
    let stream = backend.stream();
    assert!(stream.accept(16000, &[]).unwrap().is_empty());
    assert!(stream.finish().unwrap().is_empty());
    assert!(stream.finish().unwrap().is_empty());
    assert!(stream.accept(16000, &[]).is_err());
    let stream = backend.live_stream();
    assert!(stream.accept(1, &[]).is_err());
    assert!(stream.finish().is_err());
    assert!(stream.accept(16000, &[]).is_err());
    let stream = backend.live_stream();
    assert!(stream.accept(16000, &[f32::NAN]).is_err());
    std::fs::remove_file(key).unwrap();
}
#[test]
fn stale_endpoints_cannot_flush_a_new_utterance() {
    let mut f = Finals::default();
    assert!(
        f.accept(&result(0.0, 1.0, "first", true, true), 30)
            .unwrap()
            .is_some()
    );
    assert!(
        f.accept(&result(1.0, 1.0, "hey", true, false), 30)
            .unwrap()
            .is_none()
    );
    assert!(
        f.accept(&result(0.0, 1.0, "first", true, true), 30)
            .unwrap()
            .is_none()
    );
    assert!(
        f.accept(&json!({"type":"UtteranceEnd","last_word_end":1.0}), 30)
            .unwrap()
            .is_none()
    );
    let reply = f
        .accept(&result(2.0, 1.0, "computer", true, true), 30)
        .unwrap()
        .unwrap();
    assert_eq!(reply.text.as_deref(), Some("hey computer"));
    f.accept(&result(3.0, 1.0, "again", true, false), 30)
        .unwrap();
    let reply = f
        .accept(&json!({"type":"UtteranceEnd","last_word_end":4.0}), 30)
        .unwrap()
        .unwrap();
    assert_eq!(reply.text.as_deref(), Some("again"));
}
#[test]
fn stalled_worker_queue_is_bounded_and_disposal_reaps_it_without_waiting_for_network() {
    use std::os::unix::fs::PermissionsExt;
    let root = crate::test_support::unique_directory("realtime", "stalled-worker");
    let script = root.join("worker");
    std::fs::write(&script, b"#!/bin/sh\nexec sleep 20\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
    let config = Config::default();
    let backend = CloudBackend {
        settings: Settings {
            kind: "deepgram".into(),
            cloud: CloudConfig::default(),
            language: "en".into(),
        },
        matcher: PhraseMatcher::compile(&config.wake_words).unwrap(),
        threshold: 0.01,
        silence_samples: 8000,
        max_samples: 480000,
        worker_executable: script,
    };
    let worker = RealtimeWorker::start(&backend).unwrap();
    let pid = worker.child.id();
    let mut failed = false;
    for _ in 0..1024 {
        if worker.send(Input::Audio(vec![0.2; 1600]), true).is_err() {
            failed = true;
            break;
        }
    }
    assert!(
        failed,
        "stalled transport must stop at bounded queue capacity"
    );
    let started = Instant::now();
    drop(worker);
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn realtime_live_session_errors_close_the_session_and_worker() {
    use std::os::unix::fs::PermissionsExt;
    let root = crate::test_support::unique_directory("realtime", "failed-worker");
    let script = root.join("worker");
    std::fs::write(&script, b"#!/bin/sh\nexit 1\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
    let config = Config::default();
    let backend = CloudBackend {
        settings: Settings {
            kind: "deepgram".into(),
            cloud: CloudConfig::default(),
            language: "en".into(),
        },
        matcher: PhraseMatcher::compile(&config.wake_words).unwrap(),
        threshold: 0.01,
        silence_samples: 8000,
        max_samples: 480000,
        worker_executable: script,
    };
    let stream = RealtimeStream::new(&backend, true);
    let mut failed = false;
    for _ in 0..100 {
        if stream.accept(16000, &[0.1; 1600]).is_err() {
            failed = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(failed);
    assert!(stream.accept(16000, &[]).is_err());
    assert!(stream.finish().is_err());
    assert!(stream.state.borrow().worker.is_none());
    std::fs::remove_dir_all(root).unwrap();
}
