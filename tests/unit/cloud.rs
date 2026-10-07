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

#[test]
fn finite_sessions_flush_short_utterances_and_reject_reuse() {
    let exe = script(
        "finite",
        "import sys,json\nfor line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'start_sample':r['start_sample'],'text':'Hey computer','error':None}),flush=True)\n",
    );
    let mut backend = test_backend(exe.clone());
    backend.max_samples = 640;
    assert_eq!(backend.kind(), "openai-compatible");
    backend.settings.kind = "deepgram".into();
    assert_eq!(backend.kind(), "deepgram");
    let stream = backend.stream();
    for (rate, audio) in [(0, vec![0.0]), (16000, vec![f32::NAN]), (16000, vec![2.0])] {
        assert!(stream.accept(rate, &audio).is_err());
    }
    assert_eq!(stream.accept(16000, &[0.2; 640]).unwrap().len(), 1);
    stream.accept(16000, &[0.2; 420]).unwrap();
    assert_eq!(stream.finish().unwrap().len(), 1);
    assert!(stream.finish().unwrap().is_empty());
    assert!(stream.accept(16000, &[0.0]).is_err());
    let path = exe.parent().unwrap().join("audio.wav");
    std::fs::write(&path, encode_wav(&[0.2; 640]).unwrap()).unwrap();
    assert_eq!(backend.detect_file(&path).unwrap().len(), 1);
    std::fs::remove_dir_all(exe.parent().unwrap()).unwrap();
}

#[test]
fn malformed_failed_and_out_of_order_workers_close_sessions() {
    for (name, reply) in [
        ("malformed", "not json"),
        (
            "wrong-order",
            "{\"start_sample\":99,\"text\":\"Hey computer\",\"error\":null}",
        ),
        (
            "provider-error",
            "{\"start_sample\":0,\"text\":null,\"error\":\"provider unavailable\"}",
        ),
    ] {
        let exe = script(
            name,
            &format!("import sys\nfor line in sys.stdin:\n print({reply:?},flush=True)\n"),
        );
        let backend = test_backend(exe.clone());
        let stream = backend.stream();
        let mut audio = vec![0.2; 320];
        audio.extend([0.0; 1600]);
        assert!(stream.accept(16000, &audio).is_err(), "{name}");
        assert!(stream.accept(16000, &[0.0]).is_err());
        assert!(stream.finish().is_err());
        std::fs::remove_dir_all(exe.parent().unwrap()).unwrap();
    }
    let exe = script("closed", "import sys\nsys.exit(0)\n");
    let backend = test_backend(exe.clone());
    let mut worker = Worker::start(&backend.settings, &exe).unwrap();
    worker
        .submit(Utterance {
            start_sample: 0,
            samples: vec![0.2; 320],
        })
        .unwrap();
    assert!(worker.poll(true).is_err());
    drop(worker);
    std::fs::remove_dir_all(exe.parent().unwrap()).unwrap();
}

#[test]
fn cloud_model_and_configuration_policy_are_independent_of_native_assets() {
    let mut config = Config::default();
    assert_eq!(model_name(&config), config.model.name);
    config.backend.kind = "deepgram".into();
    assert_eq!(model_name(&config), "nova-3");
    config.backend.kind = "openai-compatible".into();
    assert_eq!(model_name(&config), "gpt-4o-mini-transcribe");
    config.backend.cloud.base_url = "http://localhost:1234/v1".into();
    config.backend.cloud.model = "custom".into();
    assert_eq!(model_name(&config), "custom");
    assert!(CloudBackend::load(&config).is_ok());
    config.model.language = "bad language".into();
    assert!(CloudBackend::load(&config).is_err());
    config.model.language = "".into();
    config.backend.cloud.model = "bad\nmodel".into();
    assert!(CloudBackend::load(&config).is_err());
    config.backend.cloud.model = "".into();
    config.backend.device = "npu".into();
    assert!(CloudBackend::load(&config).is_err());
    config.backend.device = "remote".into();
    config.backend.options.insert("native".into(), "yes".into());
    assert!(CloudBackend::load(&config).is_err());
    let settings = Settings {
        kind: "unknown".into(),
        cloud: CloudConfig::default(),
        language: "en".into(),
    };
    assert!(settings.validate().is_err());
    for samples in [vec![], vec![f32::NAN], vec![2.0], vec![0.0; 480001]] {
        assert!(settings.transcribe(&samples).is_err());
    }
}

#[test]
fn direct_transcription_validates_final_vendor_payloads_and_multipart_boundary() {
    use std::net::TcpListener;
    for (kind, language, payload, expected) in [
        (
            "openai-compatible",
            "",
            "{\"text\":\" hey computer \"}",
            Some("hey computer"),
        ),
        (
            "deepgram",
            "en",
            "{\"results\":{\"channels\":[{\"alternatives\":[{\"transcript\":\"hey computer\"}]}]}}",
            Some("hey computer"),
        ),
        ("openai-compatible", "en", "{\"partial\":\"secret\"}", None),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let name = format!("OMAWAKE_TEST_DIRECT_KEY_{}", std::process::id());
        unsafe {
            std::env::set_var(&name, "fake-key");
        }
        let settings = Settings {
            kind: kind.into(),
            language: language.into(),
            cloud: CloudConfig {
                base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
                api_key_env: name.clone(),
                ..Default::default()
            },
        };
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut headers = String::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
                headers.push_str(&line);
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        payload.len(),
                        payload
                    )
                    .as_bytes(),
                )
                .unwrap();
            (headers, body)
        });
        let result = settings.transcribe(&[0.2; 320]);
        let (headers, body) = server.join().unwrap();
        assert!(headers.to_lowercase().contains("authorization:"));
        if kind == "deepgram" {
            assert!(headers.contains("/v1/listen?model=nova-3&language=en"));
            assert_eq!(&body[..4], b"RIFF");
        } else {
            assert!(headers.contains("/v1/audio/transcriptions"));
            assert!(body.windows(4).any(|w| w == b"RIFF"));
        }
        if let Some(text) = expected {
            assert_eq!(result.unwrap(), text);
        } else {
            assert!(result.is_err());
        }
        unsafe {
            std::env::remove_var(name);
        }
    }
    let settings = Settings {
        kind: "openai-compatible".into(),
        language: "en".into(),
        cloud: CloudConfig {
            base_url: "http://localhost:1234".into(),
            model: "omawake-cloud-utterance-9f4a91e2".into(),
            ..Default::default()
        },
    };
    assert!(
        settings
            .transcribe(&[0.2; 320])
            .unwrap_err()
            .to_string()
            .contains("multipart")
    );
}
