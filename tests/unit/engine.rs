use super::*;
use anyhow::anyhow;
use std::fs;

struct FakeBackend {
    detections: Vec<Detection>,
}

struct FakeStream {
    detections: Vec<Detection>,
}

struct FailingStream {
    fail_accept: bool,
}

struct FailingBackend;

impl WakeWordBackend for FakeBackend {
    fn kind(&self) -> &'static str {
        "fake"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(FakeStream {
            detections: self.detections.clone(),
        })
    }

    fn detect_file(&self, _: &Path) -> Result<Vec<Detection>> {
        Ok(self.detections.clone())
    }
}

impl WakeWordStream for FakeStream {
    fn accept(&self, _: i32, _: &[f32]) -> Result<Vec<Detection>> {
        Ok(self.detections.clone())
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        Ok(self.detections.clone())
    }
}

impl WakeWordStream for FailingStream {
    fn accept(&self, _: i32, _: &[f32]) -> Result<Vec<Detection>> {
        if self.fail_accept {
            Err(anyhow!("accept failed"))
        } else {
            Ok(vec![])
        }
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        Err(anyhow!("finish failed"))
    }
}

impl WakeWordBackend for FailingBackend {
    fn kind(&self) -> &'static str {
        "failing"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(FailingStream { fail_accept: true })
    }

    fn detect_file(&self, _: &Path) -> Result<Vec<Detection>> {
        Err(anyhow!("file failed"))
    }
}

fn detection(id: &str) -> Detection {
    Detection {
        id: id.into(),
        tokens: vec!["TOKEN".into()],
        timestamps: vec![0.1],
        start_time: 0.0,
    }
}

fn detector(id: &str, command: Vec<String>, detected: &str) -> Detector {
    let action = WakeWord {
        id: id.into(),
        phrase: "Test".into(),
        enabled: true,
        command,
    };
    Detector {
        backend: Box::new(FakeBackend {
            detections: vec![detection(detected)],
        }),
        actions: [(id.into(), action)].into_iter().collect(),
        backend_kind: "fake",
        keywords_buffer: String::new(),
        load_time: Duration::default(),
        effective_runtime: Runtime::Default,
        fallback_used: false,
    }
}

fn temp(name: &str) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("omawake-engine-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn pcm16_wav(sample_rate: u32, sample_count: u32) -> Vec<u8> {
    let data_size = sample_count * 2;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes.resize(bytes.len() + data_size as usize, 0);
    bytes
}

#[test]
fn fake_backend_validates_file_and_stream_detections() {
    let known = detector("known", vec!["true".into()], "known");
    assert_eq!(
        known.detect_file(Path::new("unused")).unwrap()[0].id,
        "known"
    );
    let session = known.session();
    assert_eq!(session.accept(16_000, &[0.0]).unwrap().len(), 1);
    assert_eq!(session.finish().unwrap().len(), 1);

    let unknown = detector("known", vec!["true".into()], "unknown");
    assert!(unknown.detect_file(Path::new("unused")).is_err());
    assert!(unknown.session().accept(16_000, &[]).is_err());
    assert!(unknown.session().finish().is_err());
}

#[test]
fn reads_wav_audio_duration() {
    let root = temp("wav-duration");
    let path = root.join("one-tenth-second.wav");
    fs::write(&path, pcm16_wav(16_000, 1_600)).unwrap();
    assert_eq!(wav_duration(&path).unwrap(), Duration::from_millis(100));
    assert!(wav_duration(&root.join("missing.wav")).is_err());

    let invalid_rate = root.join("zero-rate.wav");
    fs::write(&invalid_rate, pcm16_wav(0, 1)).unwrap();
    assert!(wav_duration(&invalid_rate).is_err());
}

#[test]
fn empty_detection_batches_are_successful_no_match_results() {
    let mut quiet = detector("known", vec!["true".into()], "known");
    quiet.backend = Box::new(FakeBackend { detections: vec![] });
    assert!(
        quiet
            .detect_file(Path::new("silence.wav"))
            .unwrap()
            .is_empty()
    );
    let session = quiet.session();
    assert!(session.accept(16_000, &[0.0; 32]).unwrap().is_empty());
    assert!(session.finish().unwrap().is_empty());
}

#[test]
fn actions_report_success_exit_status_and_errors() {
    let success = detector("known", vec!["true".into()], "known")
        .run_action("known")
        .unwrap();
    assert_eq!(success.status, 0);
    assert_eq!(success.program, "true");

    let failure = detector(
        "known",
        vec!["sh".into(), "-c".into(), "exit 7".into()],
        "known",
    )
    .run_action("known")
    .unwrap();
    assert_eq!(failure.status, 7);
    assert!(
        detector("known", vec![], "known")
            .run_action("known")
            .is_err()
    );
    assert!(
        detector("known", vec!["definitely-missing-command".into()], "known")
            .run_action("known")
            .is_err()
    );
    assert!(
        detector("known", vec!["true".into()], "known")
            .run_action("other")
            .is_err()
    );
}

#[cfg(unix)]
#[test]
fn action_status_code_reports_a_signal_without_spawning_a_crashing_process() {
    use std::os::unix::process::ExitStatusExt;

    let signaled = std::process::ExitStatus::from_raw(15);
    assert_eq!(super::action_status_code(&signaled), -1);
}

#[test]
fn backend_assembly_filters_disabled_actions_and_records_metadata() {
    let mut config = Config::default();
    config.wake_words.push(WakeWord {
        id: "disabled".into(),
        phrase: "Disabled".into(),
        enabled: false,
        command: vec!["false".into()],
    });
    let detector = Detector::from_backend(
        &config,
        Box::new(FakeBackend { detections: vec![] }),
        "TOKENS @computer".into(),
        Runtime::Cuda,
        true,
        Duration::from_millis(12),
    )
    .unwrap();
    assert_eq!(detector.backend_kind, "fake");
    assert_eq!(detector.keywords_buffer, "TOKENS @computer");
    assert_eq!(detector.effective_runtime, Runtime::Cuda);
    assert!(detector.fallback_used);
    assert_eq!(detector.load_time, Duration::from_millis(12));
    assert!(detector.actions.contains_key("computer"));
    assert!(!detector.actions.contains_key("disabled"));
}

#[test]
fn injected_loader_exercises_complete_model_preparation_and_cpu_fallback() {
    let root = temp("load-with");
    let paths = AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut config = Config::default();
    config.model.directory = fixtures.to_string_lossy().into_owned();
    let detector = Detector::load_with(&config, &paths, |_, directory, runtime, keywords| {
        assert_eq!(directory, fixtures);
        assert_eq!(runtime, Runtime::Default);
        assert!(keywords.ends_with("@computer"));
        Ok(Box::new(FakeBackend { detections: vec![] }))
    })
    .unwrap();
    assert_eq!(detector.effective_runtime, Runtime::Default);
    assert!(!detector.fallback_used);

    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.fallback = Fallback::Cpu;
    let mut attempts = Vec::new();
    let detector = Detector::load_with(&config, &paths, |_, _, runtime, _| {
        attempts.push(runtime);
        if runtime == Runtime::Cuda {
            Err(anyhow!("CUDA provider unavailable"))
        } else {
            Ok(Box::new(FakeBackend { detections: vec![] }))
        }
    })
    .unwrap();
    assert_eq!(attempts, [Runtime::Cuda, Runtime::Default]);
    assert_eq!(detector.effective_runtime, Runtime::Default);
    assert!(detector.fallback_used);

    config.backend.fallback = Fallback::Error;
    let error = Detector::load_with(&config, &paths, |_, _, _, _| {
        Err(anyhow!("CUDA provider unavailable"))
    })
    .err()
    .unwrap();
    assert_eq!(error.to_string(), "CUDA provider unavailable");

    config.backend.fallback = Fallback::Cpu;
    let mut attempts = 0;
    let error = Detector::load_with(&config, &paths, |_, _, runtime, _| {
        attempts += 1;
        Err(anyhow!("{runtime:?} initialization failed"))
    })
    .err()
    .unwrap();
    assert_eq!(attempts, 2);
    assert!(error.to_string().contains("CPU fallback also failed"));

    config.backend.kind = "future-local-backend".into();
    let detector = Detector::load_with(&config, &paths, |_, _, _, _| {
        Ok(Box::new(FakeBackend { detections: vec![] }))
    })
    .unwrap();
    assert_eq!(detector.backend_kind, "fake");
    assert!(Detector::load(&config, &paths).is_err());
}

#[test]
fn backend_and_stream_failures_are_returned_unchanged() {
    let mut failed = detector("known", vec!["true".into()], "known");
    failed.backend = Box::new(FailingBackend);
    assert_eq!(failed.backend.kind(), "failing");
    assert_eq!(
        failed
            .detect_file(Path::new("unused"))
            .unwrap_err()
            .to_string(),
        "file failed"
    );
    assert_eq!(
        failed
            .session()
            .accept(16_000, &[])
            .unwrap_err()
            .to_string(),
        "accept failed"
    );
    assert_eq!(
        failed.session().finish().unwrap_err().to_string(),
        "finish failed"
    );
}

#[test]
fn sample_chunking_pads_finishes_and_propagates_stream_errors() {
    let stream = FakeStream {
        detections: vec![detection("known")],
    };
    // Six irregular chunks, followed by padding and finish.
    let detections = detect_samples(&stream, 16_000, &vec![0.0; 6_217]).unwrap();
    assert_eq!(detections.len(), 8);
    // Empty files still flush with half a second of padding and finish.
    assert_eq!(detect_samples(&stream, 0, &[]).unwrap().len(), 2);

    assert!(detect_samples(&FailingStream { fail_accept: true }, 16_000, &[0.0]).is_err());
    assert!(detect_samples(&FailingStream { fail_accept: false }, 16_000, &[]).is_err());
}
