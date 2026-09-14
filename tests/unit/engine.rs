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

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
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

    let signaled = detector(
        "known",
        vec!["sh".into(), "-c".into(), "kill -TERM $$".into()],
        "known",
    )
    .run_action("known")
    .unwrap();
    assert_eq!(signaled.status, -1);
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
        "TOKENS @hey-atreyu".into(),
        Runtime::Cuda,
        true,
        Duration::from_millis(12),
    )
    .unwrap();
    assert_eq!(detector.backend_kind, "fake");
    assert_eq!(detector.keywords_buffer, "TOKENS @hey-atreyu");
    assert_eq!(detector.effective_runtime, Runtime::Cuda);
    assert!(detector.fallback_used);
    assert_eq!(detector.load_time, Duration::from_millis(12));
    assert!(detector.actions.contains_key("hey-atreyu"));
    assert!(!detector.actions.contains_key("disabled"));
}

#[test]
fn injected_loader_exercises_complete_model_preparation_and_cpu_fallback() {
    let root = temp("load-with");
    let paths = AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut config = Config::default();
    config.model.directory = fixtures.to_string_lossy().into_owned();
    let detector = Detector::load_with(&config, &paths, |_, directory, runtime, keywords| {
        assert_eq!(directory, fixtures);
        assert_eq!(runtime, Runtime::Default);
        assert!(keywords.ends_with("@hey-atreyu"));
        Ok(Box::new(FakeBackend { detections: vec![] }))
    })
    .unwrap();
    assert_eq!(detector.effective_runtime, Runtime::Default);
    assert!(!detector.fallback_used);

    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.fallback = Fallback::Cpu;
    let detector = Detector::load_with(&config, &paths, |_, _, runtime, _| {
        assert_eq!(runtime, Runtime::Default);
        Ok(Box::new(FakeBackend { detections: vec![] }))
    })
    .unwrap();
    assert!(detector.fallback_used);

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

#[test]
fn builds_safe_sherpa_configuration_for_cpu_and_cuda() {
    let root = temp("sherpa-config");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.threads = 3;
    for name in [
        &config.model.encoder,
        &config.model.decoder,
        &config.model.joiner,
        &config.model.tokens,
    ] {
        fs::write(root.join(name), b"fixture").unwrap();
    }

    let cpu =
        build_sherpa_config(&config, &paths, &root, Runtime::Default, "HELLO @hello").unwrap();
    assert_eq!(cpu.model_config.provider.as_deref(), Some("cpu"));
    assert_eq!(cpu.model_config.num_threads, 3);
    assert_eq!(cpu.feat_config.sample_rate, config.model.sample_rate);
    assert_eq!(cpu.max_active_paths, config.model.max_active_paths);
    assert_eq!(cpu.num_trailing_blanks, config.model.num_trailing_blanks);
    assert_eq!(cpu.keywords_score, config.model.keywords_score);
    assert_eq!(cpu.keywords_threshold, config.model.keywords_threshold);
    assert_eq!(cpu.keywords_buf.as_deref(), Some("HELLO @hello"));
    assert_eq!(
        cpu.model_config.transducer.encoder.as_deref(),
        Some(root.join(&config.model.encoder).to_string_lossy().as_ref())
    );

    let cuda = build_sherpa_config(&config, &paths, &root, Runtime::Cuda, "X @x").unwrap();
    assert_eq!(cuda.model_config.provider.as_deref(), Some("cuda"));

    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "gpu".into();
    let openvino = build_sherpa_config(&config, &paths, &root, Runtime::Openvino, "X @x").unwrap();
    assert!(
        openvino
            .model_config
            .provider
            .as_deref()
            .unwrap()
            .starts_with("openvino:/")
    );
}

#[test]
fn generates_private_openvino_config_with_npu_defaults_and_overrides() {
    let root = temp("openvino-generated");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    config
        .backend
        .options
        .insert("ProfilingFilePrefix".into(), "/tmp/omawake-profile".into());
    config
        .backend
        .options
        .insert("enable_qdq_optimizer".into(), "False".into());

    let provider = openvino_provider(&config, &paths).unwrap();
    let provider_path = Path::new(provider.strip_prefix("openvino:").unwrap());
    assert!(provider_path.is_absolute());
    assert_eq!(
        provider_path,
        paths.state_dir.join("cache/openvino/npu/provider.config")
    );
    let contents = fs::read_to_string(provider_path).unwrap();
    assert!(contents.contains("device_type=NPU\n"));
    assert!(contents.contains("disable_dynamic_shapes=True\n"));
    assert!(contents.contains("enable_qdq_optimizer=False\n"));
    assert!(contents.contains("ProfilingFilePrefix=/tmp/omawake-profile\n"));
    assert!(contents.contains(&format!(
        "cache_dir={}\n",
        paths.state_dir.join("cache/openvino/npu/compiled").display()
    )));

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(provider_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    config
        .backend
        .options
        .insert("disable_dynamic_shapes".into(), "False".into());
    openvino_provider(&config, &paths).unwrap();
    let replaced = fs::read_to_string(provider_path).unwrap();
    assert!(replaced.contains("disable_dynamic_shapes=False\n"));
}

#[test]
fn separates_openvino_caches_by_canonical_device() {
    let root = temp("openvino-devices");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "hetero:gpu, cpu".into();
    let provider = openvino_provider(&config, &paths).unwrap();
    let provider_path = Path::new(provider.strip_prefix("openvino:").unwrap());
    assert!(provider_path.ends_with("openvino/hetero-gpu-cpu/provider.config"));
    let contents = fs::read_to_string(provider_path).unwrap();
    assert!(contents.contains("device_type=HETERO:GPU,CPU\n"));
    assert!(!contents.contains("enable_qdq_optimizer"));
}

#[test]
fn honors_existing_relative_openvino_provider_config() {
    let root = temp("openvino-supplied");
    let paths = paths(&root);
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    let supplied = paths.config_file.parent().unwrap().join("provider.config");
    fs::write(&supplied, "device_type=GPU\n").unwrap();
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "gpu".into();
    config.backend.provider_config = "provider.config".into();

    assert_eq!(
        openvino_provider(&config, &paths).unwrap(),
        format!("openvino:{}", supplied.canonicalize().unwrap().display())
    );
    config.backend.provider_config = "missing.config".into();
    assert!(openvino_provider(&config, &paths).is_err());
    config.backend.provider_config = "..".into();
    assert!(openvino_provider(&config, &paths).is_err());
}

#[test]
fn rejects_unsafe_or_conflicting_openvino_options() {
    let root = temp("openvino-options");
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "gpu".into();

    for (key, value) in [
        ("bad key", "value"),
        ("bad=key", "value"),
        ("good_key", "nul\0value"),
        ("good_key", "first\nsecond"),
        ("cache_dir", "/tmp/shared"),
        ("device_type", "CPU"),
    ] {
        config.backend.options.clear();
        config.backend.options.insert(key.into(), value.into());
        assert!(openvino_provider(&config, &paths).is_err(), "{key}");
        assert!(!paths.state_dir.exists(), "{key}");
    }

    config.backend.options.clear();
    config
        .backend
        .options
        .insert("device_type".into(), "GPU".into());
    assert!(openvino_provider(&config, &paths).is_ok());
}

#[test]
fn result_payloads_serialize_with_diagnostic_fields() {
    let original = detection("known");
    let cloned = original.clone();
    assert_eq!(cloned.id, original.id);
    assert!(format!("{original:?}").contains("known"));
    let encoded = serde_json::to_value(original).unwrap();
    assert_eq!(encoded["id"], "known");
    assert_eq!(encoded["tokens"][0], "TOKEN");
    let action = ActionResult {
        id: "known".into(),
        program: "true".into(),
        arguments: vec!["--help".into()],
        status: 0,
    };
    let cloned = action.clone();
    assert_eq!(cloned.status, 0);
    assert!(format!("{action:?}").contains("true"));
    let encoded = serde_json::to_value(action).unwrap();
    assert_eq!(encoded["arguments"][0], "--help");

    assert!(detection_from_parts(String::new(), vec![], vec![], 0.0).is_none());
    let detection =
        detection_from_parts("hello".into(), vec!["HE".into()], vec![0.2], 0.1).unwrap();
    assert_eq!(detection.id, "hello");
    assert_eq!(detection.start_time, 0.1);
}

#[test]
fn ready_result_drain_ignores_empty_results_and_preserves_order() {
    let mut results = vec![Some(detection("first")), None, Some(detection("second"))].into_iter();
    let detections = drain_ready(|| results.next());
    assert_eq!(
        detections
            .iter()
            .map(|detection| detection.id.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    assert!(drain_ready(|| None).is_empty());
}

#[test]
fn load_and_sherpa_backend_report_configuration_errors() {
    let root = temp("load");
    let paths = AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    config.backend.threads = 0;
    assert!(Detector::load(&config, &paths).is_err());
    config.backend.threads = 2;
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    assert!(Detector::load(&config, &paths).is_err());
    config.backend.fallback = Fallback::Cpu;
    config.backend.kind = "other".into();
    assert!(Detector::load(&config, &paths).is_err());

    config.backend.runtime = Runtime::Default;
    config.backend.device = "auto".into();
    config.backend.kind = "sherpa-onnx".into();
    let directory = config.model_directory(&paths);
    assert!(
        SherpaOnnxBackend::load(&config, &paths, &directory, Runtime::Default, "X @x").is_err()
    );
    fs::create_dir_all(&directory).unwrap();
    for name in [
        &config.model.encoder,
        &config.model.decoder,
        &config.model.joiner,
        &config.model.tokens,
    ] {
        fs::write(directory.join(name), b"bad").unwrap();
    }
    config.model.directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .to_string_lossy()
        .into_owned();
    assert!(Detector::load(&config, &paths).is_err());
}
