use super::*;
#[test]
fn wav_upload_contract_is_mono_linear16_at_16khz() {
    let wav = encode_wav(&[-1.0, 0.0, 1.0]).unwrap();
    let mut r = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
    assert_eq!(r.spec().sample_rate, 16000);
    assert_eq!(r.spec().channels, 1);
    assert_eq!(
        r.samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        vec![-32767, 0, 32767]
    );
}
#[test]
fn configuration_rejects_accelerators_and_invalid_endpointing_before_network() {
    let mut config = Config::default();
    config.backend.kind = "openai-compatible".into();
    config.backend.cloud.base_url = "http://127.0.0.1:1234/v1".into();
    config.backend.cloud.api_key_env = "".into();
    config.backend.runtime = crate::backend::Runtime::Cuda;
    assert!(CloudBackend::load(&config).is_err());
    config.backend.runtime = crate::backend::Runtime::Default;
    config.backend.cloud.vad_threshold = f32::NAN;
    assert!(CloudBackend::load(&config).is_err());
    config.backend.cloud.vad_threshold = 0.01;
    config.backend.cloud.endpoint_milliseconds = 0;
    assert!(CloudBackend::load(&config).is_err());
}
#[test]
fn unsafe_urls_and_credentials_are_rejected_without_echoing_secrets() {
    for url in [
        "https://secret@example.com",
        "http://example.com",
        "https://example.com/?secret=key",
    ] {
        let c = CloudConfig {
            base_url: url.into(),
            ..Default::default()
        };
        let err = cloud_http::base_url(&c, "").unwrap_err().to_string();
        assert!(!err.contains("secret"));
    }
    let c = CloudConfig {
        api_key_env: "secret\nvalue".into(),
        ..Default::default()
    };
    assert!(
        !cloud_http::credential(&c, "", false)
            .unwrap_err()
            .to_string()
            .contains("value")
    );
}
#[cfg(target_os = "linux")]
fn script(name: &str, source: &str) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!(
        "omawake-cloud-worker-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("worker");
    std::fs::write(&path, format!("#!/usr/bin/python3\n{source}")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}
#[cfg(target_os = "linux")]
fn test_backend(executable: std::path::PathBuf) -> CloudBackend {
    let words: Vec<crate::config::WakeWord> = serde_json::from_value(
        serde_json::json!([{"id":"computer","phrase":"hey computer","command":["true"]}]),
    )
    .unwrap();
    CloudBackend {
        settings: Settings {
            kind: "openai-compatible".into(),
            cloud: CloudConfig {
                base_url: "http://127.0.0.1:1234/v1".into(),
                timeout_seconds: 1,
                ..Default::default()
            },
            language: "en".into(),
        },
        matcher: PhraseMatcher::compile(&words).unwrap(),
        threshold: 0.01,
        silence_samples: 1600,
        max_samples: 480000,
        worker_executable: executable,
    }
}
#[cfg(target_os = "linux")]
#[test]
fn live_accept_returns_during_http_wait_and_polls_final_phrase_once() {
    let exe = script(
        "async",
        "import sys,json,time\nfor line in sys.stdin:\n r=json.loads(line)\n time.sleep(.2)\n print(json.dumps({'start_sample':r['start_sample'],'text':'Hey computer','error':None}),flush=True)\n",
    );
    let backend = test_backend(exe.clone());
    let stream = backend.live_stream();
    let started = std::time::Instant::now();
    let mut audio = vec![0.2; 3200];
    audio.extend(vec![0.0; 3200]);
    assert!(stream.accept(16000, &audio).unwrap().is_empty());
    assert!(started.elapsed() < Duration::from_millis(150));
    thread::sleep(Duration::from_millis(350));
    let detections = stream.accept(16000, &[0.0; 320]).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].id, "computer");
    assert!(stream.accept(16000, &[0.0; 320]).unwrap().is_empty());
    assert!(stream.finish().unwrap().is_empty());
    drop(stream);
    std::fs::remove_dir_all(exe.parent().unwrap()).unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn dropping_live_session_kills_and_reaps_stalled_worker_without_waiting_for_network() {
    let exe = script(
        "cancel",
        "import sys,time\nfor line in sys.stdin:\n time.sleep(30)\n",
    );
    let backend = test_backend(exe.clone());
    let stream = backend.live_stream();
    let mut audio = vec![0.2; 3200];
    audio.extend(vec![0.0; 3200]);
    stream.accept(16000, &audio).unwrap();
    let started = std::time::Instant::now();
    drop(stream);
    assert!(started.elapsed() < Duration::from_secs(1));
    std::fs::remove_dir_all(exe.parent().unwrap()).unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn bounded_cloud_queue_rejects_overflow_without_replay() {
    let exe = script(
        "queue",
        "import sys,time\nfor line in sys.stdin:\n time.sleep(30)\n",
    );
    let backend = test_backend(exe.clone());
    let mut worker = Worker::start(&backend.settings, &exe).unwrap();
    for i in 0..3 {
        worker
            .submit(Utterance {
                start_sample: i,
                samples: vec![0.1; 320],
            })
            .unwrap();
        thread::sleep(Duration::from_millis(20));
    }
    assert!(
        worker
            .submit(Utterance {
                start_sample: 3,
                samples: vec![0.1; 320]
            })
            .is_err()
    );
    drop(worker);
    std::fs::remove_dir_all(exe.parent().unwrap()).unwrap();
}
