use super::*;
use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

use omawake::engine::{WakeWordBackend, WakeWordStream};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct FakeControl;

struct InMemoryBackend;
struct InMemoryWakeWordStream;

impl WakeWordBackend for InMemoryBackend {
    fn kind(&self) -> &'static str {
        "in-memory"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(InMemoryWakeWordStream)
    }

    fn detect_file(&self, _: &Path) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
}

impl WakeWordStream for InMemoryWakeWordStream {
    fn accept(&self, _: i32, _: &[f32]) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
}

struct ScriptedStream {
    response: Cursor<Vec<u8>>,
    request: Vec<u8>,
    fail_writes: bool,
}

impl ScriptedStream {
    fn responding_with(response: impl Into<Vec<u8>>) -> Self {
        Self {
            response: Cursor::new(response.into()),
            request: Vec::new(),
            fail_writes: false,
        }
    }
}

impl Read for ScriptedStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.response.read(buffer)
    }
}

impl Write for ScriptedStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        if self.fail_writes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "client disconnected",
            ));
        }
        self.request.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl DetectorControl for FakeControl {
    fn backend_kind(&self) -> &str {
        "fake"
    }

    fn effective_runtime(&self) -> Runtime {
        Runtime::Default
    }

    fn fallback_used(&self) -> bool {
        false
    }

    fn load_time(&self) -> Duration {
        Duration::from_millis(2)
    }

    fn keywords_buffer(&self) -> &str {
        "HELLO @hello"
    }

    fn run(&self, id: &str) -> Result<ActionResult> {
        Ok(ActionResult {
            id: id.into(),
            program: "true".into(),
            arguments: Vec::new(),
            status: 0,
        })
    }
}

fn test_paths(name: &str) -> AppPaths {
    let root = std::env::temp_dir().join(format!(
        "omawake-main-{}-{name}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&root);
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

fn send_raw(path: &Path, bytes: Vec<u8>) -> UnixStream {
    let mut stream = UnixStream::connect(path).unwrap();
    stream.write_all(&bytes).unwrap();
    stream
}

fn encoded_request(protocol: u32, id: &str, command: Command) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(&Request {
        protocol,
        id: id.into(),
        command,
    })
    .unwrap();
    bytes.push(b'\n');
    bytes
}

fn sandbox_denied(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|error| error.kind() == std::io::ErrorKind::PermissionDenied)
}

fn anyhow_or_skip<T>(result: Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) if sandbox_denied(&error) => None,
        Err(error) => panic!("{error:#}"),
    }
}

fn io_or_skip<T>(result: std::io::Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => None,
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn word_alias_parses_remove() {
    let cli = Cli::try_parse_from(["omawake", "word", "remove", "hey-atreyu"]).unwrap();
    assert!(matches!(
        cli.command,
        TopCommand::WakeWord {
            command: WakeWordCommand::Remove { ref id }
        } if id == "hey-atreyu"
    ));
}

#[test]
fn last_wake_word_can_be_removed() {
    let mut config = Config::default();
    remove_wake_word(&mut config, "hey-atreyu").unwrap();
    assert!(config.wake_words.is_empty());
    assert!(validate_wake_words(&config.wake_words).is_ok());
}

#[test]
fn command_line_surface_parses_representative_forms() {
    let cases = [
        vec!["omawake", "test", "--seconds", "1", "--execute", "--json"],
        vec!["omawake", "test", "--audio", "fixture.wav"],
        vec!["omawake", "status", "--json"],
        vec!["omawake", "config", "get", "backend.kind", "--json"],
        vec!["omawake", "config", "schema"],
        vec!["omawake", "config", "set", "backend.device", "cpu"],
        vec!["omawake", "config", "unset", "backend.device"],
        vec![
            "omawake",
            "wake-word",
            "add",
            "--id",
            "hello",
            "--phrase",
            "Hello",
            "--",
            "true",
        ],
        vec!["omawake", "wake-word", "list", "--json"],
        vec!["omawake", "audio-devices", "--json"],
        vec!["omawake", "daemon"],
        vec!["omawake", "pause"],
        vec!["omawake", "resume"],
        vec!["omawake", "stop"],
        vec!["omawake", "setup", "check", "--json"],
        vec!["omawake", "setup", "runtime", "--json"],
        vec!["omawake", "setup", "systemd", "--no-start"],
        vec!["omawake", "setup", "menu", "--status"],
        vec![
            "omawake",
            "setup",
            "model",
            "--download",
            "model",
            "--no-activate",
            "--progress-format",
            "json",
        ],
        vec![
            "omawake",
            "setup",
            "all",
            "--model",
            "model",
            "--no-start",
            "--progress-format",
            "human",
        ],
    ];
    for args in cases {
        Cli::try_parse_from(args).unwrap();
    }
    assert!(Cli::try_parse_from(["omawake", "test", "--audio", "a", "--seconds", "1"]).is_err());
    assert!(
        Cli::try_parse_from(["omawake", "setup", "systemd", "--status", "--uninstall"]).is_err()
    );
    let denied = || std::io::Error::from(std::io::ErrorKind::PermissionDenied);
    assert!(io_or_skip::<()>(Err(denied())).is_none());
    assert!(anyhow_or_skip::<()>(Err(denied().into())).is_none());
    assert!(
        std::panic::catch_unwind(|| { io_or_skip::<()>(Err(std::io::Error::other("unexpected"))) })
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| { anyhow_or_skip::<()>(Err(anyhow::anyhow!("unexpected"))) })
            .is_err()
    );
}

#[test]
fn config_helpers_cover_every_supported_key_and_validation() {
    let mut config = Config::default();
    for (key, value) in [
        ("backend.kind", "custom"),
        ("backend.runtime", "OPENVINO"),
        ("backend.device", "npu"),
        ("backend.threads", "3"),
        ("backend.fallback", "CPU"),
        ("backend.device_id", "2"),
        ("backend.provider_config", "provider.json"),
        ("model.name", "custom-model"),
        ("model.directory", "/models/custom"),
        ("model.sample_rate", "8000"),
        ("model.keywords_score", "2.5"),
        ("model.keywords_threshold", "0.75"),
        ("audio.device", "microphone"),
        ("audio.channels", "stereo"),
        ("audio.buffer_milliseconds", "80"),
        ("daemon.cooldown_milliseconds", "90"),
        ("daemon.queue_capacity", "5"),
    ] {
        set_config(&mut config, key, value).unwrap();
        unset_config(&mut config, key).unwrap();
    }
    assert!(set_config(&mut config, "backend.threads", "many").is_err());
    assert!(set_config(&mut config, "missing", "value").is_err());
    assert!(unset_config(&mut config, "missing").is_err());
    assert_eq!(parse_runtime("DEFAULT").unwrap(), Runtime::Default);
    assert_eq!(parse_runtime("cuda").unwrap(), Runtime::Cuda);
    assert!(parse_runtime("rocm").is_err());
    assert_eq!(parse_fallback("error").unwrap(), Fallback::Error);
    assert!(parse_fallback("maybe").is_err());

    let value = json!({"one":{"two":3}});
    assert_eq!(dotted_get(&value, "one.two"), Some(&json!(3)));
    assert!(dotted_get(&value, "one.missing").is_none());
}

#[test]
fn config_mutation_saves_and_wake_words_compile_when_model_exists() {
    let paths = test_paths("config");
    let config = Config::default();
    config_mutation(
        ConfigCommand::Set {
            key: "backend.threads".into(),
            value: "4".into(),
        },
        config.clone(),
        &paths.config_file,
    )
    .unwrap();
    assert_eq!(Config::load(&paths.config_file).unwrap().backend.threads, 4);
    config_mutation(
        ConfigCommand::Unset {
            key: "backend.threads".into(),
        },
        Config::load(&paths.config_file).unwrap(),
        &paths.config_file,
    )
    .unwrap();

    let model_dir = config.model_directory(&paths);
    fs::create_dir_all(&model_dir).unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bpe.model"),
        model_dir.join(&config.model.bpe_model),
    )
    .unwrap();
    wake_word_command(
        WakeWordCommand::Add {
            id: "lovely-child".into(),
            phrase: "Lovely Child".into(),
            command: vec!["true".into()],
        },
        config.clone(),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().wake_words.len(),
        2
    );
    assert!(
        wake_word_command(
            WakeWordCommand::Add {
                id: "hey-atreyu".into(),
                phrase: "Duplicate".into(),
                command: vec!["true".into()],
            },
            config,
            &paths.config_file,
            &paths,
        )
        .is_err()
    );
}

#[test]
fn metadata_helpers_return_stable_shapes() {
    let paths = test_paths("metadata");
    let config = Config::default();
    let status = stopped_status(&config, &paths);
    assert_eq!(status["app"], "omawake");
    assert_eq!(status["daemon"]["state"], "stopped");
    let value = schema(&config, &paths.config_file, &paths);
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["collections"][0]["key"], "wake_words");
    assert!(model_spec(&config.model.name).is_ok());
    assert!(model_spec("missing").is_err());
    let spec = model_spec(&config.model.name).unwrap();
    verify_selected_model(spec.id, &paths, |_, found| {
        assert_eq!(found.id, spec.id);
        Ok(())
    })
    .unwrap();
    assert!(verify_selected_model("missing", &paths, |_, _| Ok(())).is_err());
    assert!(verify_selected_model(spec.id, &paths, |_, _| bail!("invalid model")).is_err());
    set_selected_model(spec.id, &paths.config_file, &paths, |_, _| Ok(())).unwrap();
    assert!(
        set_selected_model(spec.id, &paths.config_file, &paths, |_, _| {
            bail!("invalid model")
        })
        .is_err()
    );
    activate_model(&paths.config_file, spec).unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.name,
        spec.id
    );
    print_model_ready(&paths.data_dir, spec, true, ProgressFormat::Human).unwrap();
    print_model_ready(&paths.data_dir, spec, false, ProgressFormat::Json).unwrap();
    install_selected_model(
        spec.id,
        &paths.config_file,
        &paths,
        Some(Path::new("archive.tar.bz2")),
        false,
        ProgressFormat::Human,
        |_, found, archive, progress| {
            assert_eq!(found.id, spec.id);
            assert_eq!(archive, Some(Path::new("archive.tar.bz2")));
            assert_eq!(progress, ProgressFormat::Human);
            Ok(paths.data_dir.clone())
        },
    )
    .unwrap();
    install_selected_model(
        spec.id,
        &paths.config_file,
        &paths,
        None,
        true,
        ProgressFormat::Json,
        |_, _, _, _| Ok(paths.data_dir.clone()),
    )
    .unwrap();
    assert!(
        install_selected_model(
            spec.id,
            &paths.config_file,
            &paths,
            None,
            true,
            ProgressFormat::Human,
            |_, _, _, _| bail!("install failed"),
        )
        .is_err()
    );
    for progress in [ProgressFormat::Human, ProgressFormat::Json] {
        print_setup_complete(
            &paths.data_dir,
            &paths.config_file,
            Path::new("launcher.desktop"),
            Path::new("omawake.service"),
            progress,
        )
        .unwrap();
        install_everything(
            spec,
            &paths.config_file,
            &paths,
            None,
            true,
            progress,
            |_, _, _, _| Ok(paths.data_dir.join("model")),
            |_| Ok(paths.data_dir.join("launcher.desktop")),
            |_, _, start| {
                assert!(!start);
                Ok(paths.data_dir.join("omawake.service"))
            },
            |_, _| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap();
    }
    assert!(
        install_everything(
            spec,
            &paths.config_file,
            &paths,
            None,
            false,
            ProgressFormat::Human,
            |_, _, _, _| bail!("model failed"),
            |_| unreachable!(),
            |_, _, _| unreachable!(),
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .is_err()
    );
    assert_eq!(socket_path(&paths), paths.runtime_dir.join("control.sock"));
    let first = request_id();
    let second = request_id();
    assert!(first.starts_with(&format!("{}-", std::process::id())));
    assert_ne!(first, second);
    let details = daemon_details(
        "fake",
        Runtime::Cuda,
        true,
        Duration::from_millis(12),
        Some(json!({"device":"test"})),
    );
    assert_eq!(details["backend"]["kind"], "fake");
    assert_eq!(details["model_load_milliseconds"], 12);
}

#[test]
fn detection_output_supports_human_and_diagnostic_json_forms() {
    let detection = || Detection {
        id: "hello".into(),
        tokens: vec!["HELLO".into()],
        timestamps: vec![0.1],
        start_time: 0.0,
    };
    print_detections(
        vec![detection()],
        Vec::new(),
        false,
        Duration::from_millis(3),
        "HELLO @hello",
        "fake",
        Runtime::Default,
        false,
    )
    .unwrap();
    print_detections(
        vec![detection()],
        vec![ActionResult {
            id: "hello".into(),
            program: "true".into(),
            arguments: Vec::new(),
            status: 0,
        }],
        true,
        Duration::from_millis(3),
        "HELLO @hello",
        "fake",
        Runtime::Cuda,
        true,
    )
    .unwrap();

    let detections = vec![detection()];
    assert!(
        collect_detection_actions(&detections, false, |_| unreachable!())
            .unwrap()
            .is_empty()
    );
    let actions = collect_detection_actions(&detections, true, |id| {
        Ok(ActionResult {
            id: id.into(),
            program: "true".into(),
            arguments: Vec::new(),
            status: 0,
        })
    })
    .unwrap();
    assert_eq!(actions[0].id, "hello");
    assert!(collect_detection_actions(&detections, true, |_| bail!("action failed")).is_err());
    present_detections(&FakeControl, vec![detection()], true, true).unwrap();
}

#[test]
fn live_collection_handles_samples_timeouts_and_capture_failures() {
    fn no_detections(_: i32, _: &[f32]) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
    let detection = || Detection {
        id: "hello".into(),
        tokens: Vec::new(),
        timestamps: Vec::new(),
        start_time: 0.0,
    };
    let base = Instant::now();
    let mut times = [base, base, base + Duration::from_millis(3)].into_iter();
    let found = collect_live_detections_with_clock(
        Duration::from_millis(2),
        |_| {
            Ok(AudioEvent::Samples {
                sample_rate: 16_000,
                samples: vec![0.0],
            })
        },
        |sample_rate, samples| {
            assert_eq!(sample_rate, 16_000);
            assert_eq!(samples, &[0.0]);
            Ok(vec![detection()])
        },
        || Ok(vec![detection()]),
        || times.next().unwrap(),
    )
    .unwrap();
    assert_eq!(found.len(), 2);

    assert!(
        collect_live_detections(
            Duration::from_secs(1),
            |_| Ok(AudioEvent::Error("lost microphone".into())),
            no_detections,
            || Ok(Vec::new()),
        )
        .is_err()
    );
    assert!(
        collect_live_detections(
            Duration::from_secs(1),
            |_| Err(RecvTimeoutError::Disconnected),
            no_detections,
            || Ok(Vec::new()),
        )
        .is_err()
    );
    assert!(
        collect_live_detections(
            Duration::ZERO,
            |_| Err(RecvTimeoutError::Timeout),
            no_detections,
            || bail!("finish failed"),
        )
        .is_err()
    );
    assert!(
        collect_live_detections(
            Duration::from_millis(1),
            |_| {
                Ok(AudioEvent::Samples {
                    sample_rate: 16_000,
                    samples: Vec::new(),
                })
            },
            |_, _| bail!("decode failed"),
            || Ok(Vec::new()),
        )
        .is_err()
    );

    let base = Instant::now();
    let mut times = [base, base, base + Duration::from_millis(2)].into_iter();
    assert!(
        collect_live_detections_with_clock(
            Duration::from_millis(1),
            |_| Err(RecvTimeoutError::Timeout),
            no_detections,
            || Ok(Vec::new()),
            || times.next().unwrap(),
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn daemon_event_helpers_cover_controls_audio_and_action_results() {
    let mut paused = false;
    let mut shutdown = false;
    apply_daemon_command(Command::Pause, &mut paused, &mut shutdown);
    assert!(paused);
    apply_daemon_command(Command::Resume, &mut paused, &mut shutdown);
    assert!(!paused);
    apply_daemon_command(Command::Status, &mut paused, &mut shutdown);
    assert!(!paused && !shutdown);
    apply_daemon_command(Command::Shutdown, &mut paused, &mut shutdown);
    assert!(shutdown);

    let samples = handle_audio_event(
        Ok(AudioEvent::Samples {
            sample_rate: 8_000,
            samples: vec![0.25],
        }),
        |rate, samples| {
            assert_eq!(rate, 8_000);
            assert_eq!(samples, &[0.25]);
            Ok(vec![Detection {
                id: "hello".into(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                start_time: 0.0,
            }])
        },
    )
    .unwrap();
    assert_eq!(samples.len(), 1);
    assert!(
        handle_audio_event(Err(RecvTimeoutError::Timeout), |_, _| unreachable!())
            .unwrap()
            .is_empty()
    );
    assert!(
        handle_audio_event(Ok(AudioEvent::Error("capture".into())), |_, _| Ok(
            Vec::new()
        ))
        .is_err()
    );
    assert!(
        handle_audio_event(Err(RecvTimeoutError::Disconnected), |_, _| Ok(Vec::new())).is_err()
    );

    execute_detected_actions(samples.clone(), |id| {
        Ok(ActionResult {
            id: id.into(),
            program: "true".into(),
            arguments: Vec::new(),
            status: 0,
        })
    });
    execute_detected_actions(samples, |_| bail!("action failure"));
    execute_detected_actions(Vec::new(), |_| unreachable!());
}

#[test]
fn armed_cycle_reports_control_commands_and_detections_without_hardware() {
    let mut receives = 0;
    let (detections, command) = collect_armed_detections(
        "test microphone",
        16_000,
        1,
        |audio| {
            assert_eq!(audio["device"], "test microphone");
            assert_eq!(audio["sample_rate"], 16_000);
            Ok(None)
        },
        |_| {
            receives += 1;
            if receives == 1 {
                Err(RecvTimeoutError::Timeout)
            } else {
                Ok(AudioEvent::Samples {
                    sample_rate: 16_000,
                    samples: vec![0.5],
                })
            }
        },
        |rate, samples| {
            assert_eq!(rate, 16_000);
            assert_eq!(samples, &[0.5]);
            Ok(vec![Detection {
                id: "hello".into(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                start_time: 0.0,
            }])
        },
    )
    .unwrap();
    assert_eq!(detections.len(), 1);
    assert!(command.is_none());

    let (detections, command) = collect_armed_detections(
        "test microphone",
        8_000,
        2,
        |_| Ok(Some(Command::Pause)),
        |_| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap();
    assert!(detections.is_empty());
    assert!(matches!(command, Some(Command::Pause)));

    let mut polls = 0;
    let (detections, command) = collect_armed_detections(
        "test microphone",
        16_000,
        1,
        |_| {
            polls += 1;
            Ok((polls == 1).then_some(Command::Resume))
        },
        |_| {
            Ok(AudioEvent::Samples {
                sample_rate: 16_000,
                samples: vec![0.25],
            })
        },
        |_, _| {
            Ok(vec![Detection {
                id: "hello".into(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                start_time: 0.0,
            }])
        },
    )
    .unwrap();
    assert_eq!(detections.len(), 1);
    assert!(command.is_none());

    assert!(
        collect_armed_detections(
            "test microphone",
            16_000,
            1,
            |_| bail!("control failed"),
            |_| unreachable!(),
            |_, _| unreachable!(),
        )
        .is_err()
    );
}

#[test]
fn socket_binding_replaces_stale_socket_and_rejects_live_daemon() {
    let paths = test_paths("bind");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    fs::write(socket_path(&paths), "stale").unwrap();
    let Some(listener) = anyhow_or_skip(bind_socket(&paths)) else {
        return;
    };
    assert_eq!(
        fs::metadata(&paths.runtime_dir)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(socket_path(&paths))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(bind_socket(&paths).is_err());
    drop(listener);
}

#[test]
fn socket_preparation_is_testable_without_creating_a_unix_socket() {
    let paths = test_paths("bind-memory");
    let fake_bind = |path: &Path| {
        fs::write(path, "fake socket")?;
        Ok(())
    };
    bind_socket_with(&paths, |_| false, fake_bind, |_| Ok(())).unwrap();
    assert_eq!(
        fs::metadata(&paths.runtime_dir)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(socket_path(&paths))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );

    bind_socket_with(
        &paths,
        |_| false,
        |path| {
            assert!(!path.exists());
            fs::write(path, "replacement")?;
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap();
    assert!(
        bind_socket_with(
            &paths,
            |_| true,
            |_| unreachable!(),
            |_: &()| unreachable!(),
        )
        .is_err()
    );

    let failing_paths = test_paths("bind-configure-failure");
    assert!(
        bind_socket_with(
            &failing_paths,
            |_| false,
            |path| {
                fs::write(path, "temporary socket")?;
                Ok(())
            },
            |_| Err(std::io::Error::other("nonblocking failed")),
        )
        .is_err()
    );
    assert!(!socket_path(&failing_paths).exists());
}

#[test]
fn daemon_socket_cleanup_runs_after_success_and_failure() {
    for (name, serve_result) in [
        ("success", Ok(())),
        ("failure", Err(anyhow::anyhow!("serve failed"))),
    ] {
        let paths = test_paths(name);
        fs::create_dir_all(&paths.runtime_dir).unwrap();
        fs::write(socket_path(&paths), "socket placeholder").unwrap();
        let result = finish_daemon(&paths, serve_result);
        assert!(!socket_path(&paths).exists());
        assert_eq!(result.is_err(), name == "failure");
    }

    let absent = test_paths("already-absent");
    finish_daemon(&absent, Ok(())).unwrap();

    let cleanup_failure = test_paths("cleanup-failure");
    fs::create_dir_all(socket_path(&cleanup_failure)).unwrap();
    assert!(finish_daemon(&cleanup_failure, Ok(())).is_err());

    let original_error = test_paths("cleanup-and-serve-failure");
    fs::create_dir_all(socket_path(&original_error)).unwrap();
    assert_eq!(
        finish_daemon(&original_error, Err(anyhow::anyhow!("original failure")))
            .unwrap_err()
            .to_string(),
        "original failure"
    );
}

#[test]
fn request_framing_accepts_valid_and_rejects_invalid_messages() {
    let Some((mut sender, receiver)) = io_or_skip(UnixStream::pair()) else {
        return;
    };
    if io_or_skip(sender.write_all(&encoded_request(1, "valid", Command::Pause))).is_none() {
        return;
    }
    let request = read_request(&receiver).unwrap();
    assert_eq!(request.id, "valid");
    assert!(matches!(request.command, Command::Pause));

    let (mut sender, receiver) = UnixStream::pair().unwrap();
    sender.write_all(b"not-json\n").unwrap();
    assert!(read_request(&receiver).is_err());

    let (mut sender, receiver) = UnixStream::pair().unwrap();
    let writer = thread::spawn(move || sender.write_all(&vec![b'x'; 65_537]).unwrap());
    assert!(read_request(&receiver).is_err());
    writer.join().unwrap();
}

#[test]
fn control_socket_handles_status_commands_and_bad_clients() {
    let paths = test_paths("control");
    let Some(listener) = anyhow_or_skip(bind_socket(&paths)) else {
        return;
    };
    let details = json!({"backend":{"kind":"fake"}});

    let mut invalid = send_raw(&socket_path(&paths), b"bad\n".to_vec());
    let mut mismatch = send_raw(
        &socket_path(&paths),
        encoded_request(2, "old", Command::Status),
    );
    let mut status = send_raw(
        &socket_path(&paths),
        encoded_request(1, "status", Command::Status),
    );
    assert!(
        poll_control(&FakeControl, "armed", None, || accept_control(&listener))
            .unwrap()
            .is_none()
    );
    for (stream, code) in [
        (&mut invalid, "invalid_request"),
        (&mut mismatch, "protocol_mismatch"),
    ] {
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        assert!(line.contains(code));
    }
    let mut line = String::new();
    BufReader::new(&mut status).read_line(&mut line).unwrap();
    let response: Response = serde_json::from_str(&line).unwrap();
    assert!(matches!(response.result, ResultPayload::State { ref state, .. } if state == "armed"));

    for (command, expected_state) in [
        (Command::Pause, "paused"),
        (Command::Resume, "armed"),
        (Command::Shutdown, "stopping"),
    ] {
        let mut client = send_raw(
            &socket_path(&paths),
            encoded_request(1, expected_state, command),
        );
        let returned = poll_control_connections(|| accept_control(&listener), "current", &details)
            .unwrap()
            .unwrap();
        assert!(matches!(
            (returned, expected_state),
            (Command::Pause, "paused")
                | (Command::Resume, "armed")
                | (Command::Shutdown, "stopping")
        ));
        let mut line = String::new();
        BufReader::new(&mut client).read_line(&mut line).unwrap();
        assert!(line.contains(expected_state));
    }
}

#[test]
fn request_round_trips_responses_and_reports_malformed_server_data() {
    for valid in [true, false] {
        let paths = test_paths(if valid {
            "request-valid"
        } else {
            "request-invalid"
        });
        fs::create_dir_all(&paths.runtime_dir).unwrap();
        let Some(listener) = io_or_skip(UnixListener::bind(socket_path(&paths))) else {
            return;
        };
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&stream).unwrap();
            assert!(matches!(request.command, Command::Status));
            if valid {
                write_response(
                    &mut stream,
                    &Response {
                        protocol: 1,
                        id: request.id,
                        result: ResultPayload::State {
                            state: "armed".into(),
                            details: json!({}),
                        },
                    },
                );
            } else {
                stream.write_all(b"invalid\n").unwrap();
            }
        });
        let result = request(&paths, Command::Status);
        if valid {
            assert!(
                matches!(result.unwrap().result, ResultPayload::State { state, .. } if state == "armed")
            );
        } else {
            assert!(result.is_err());
        }
        server.join().unwrap();
    }
    assert!(request(&test_paths("not-running"), Command::Status).is_err());
}

#[test]
fn response_output_handles_state_json_and_daemon_errors() {
    let state = || Response {
        protocol: 1,
        id: "id".into(),
        result: ResultPayload::State {
            state: "armed".into(),
            details: json!({}),
        },
    };
    print_response(state(), false).unwrap();
    print_response(state(), true).unwrap();
    assert!(print_response(Response::error("id", "broken", "failure"), false).is_err());

    let Some((mut left, right)) = io_or_skip(UnixStream::pair()) else {
        return;
    };
    drop(right);
    write_response(&mut left, &state());
}

#[test]
fn top_level_dispatch_runs_file_only_commands_with_injected_paths() {
    let paths = test_paths("dispatch");
    Config::default().save(&paths.config_file).unwrap();
    let invoke = |command| {
        run_with_paths(
            Cli {
                config: Some(paths.config_file.clone()),
                command,
            },
            paths.clone(),
        )
    };

    invoke(TopCommand::Status { json: false }).unwrap();
    invoke(TopCommand::Status { json: true }).unwrap();
    invoke(TopCommand::Config {
        command: ConfigCommand::Get {
            key: Some("backend.kind".into()),
            json: false,
        },
    })
    .unwrap();
    invoke(TopCommand::Config {
        command: ConfigCommand::Get {
            key: None,
            json: true,
        },
    })
    .unwrap();
    assert!(
        invoke(TopCommand::Config {
            command: ConfigCommand::Get {
                key: Some("missing.key".into()),
                json: false,
            },
        })
        .is_err()
    );
    for json in [false, true] {
        invoke(TopCommand::Config {
            command: ConfigCommand::Schema { json },
        })
        .unwrap();
        invoke(TopCommand::WakeWord {
            command: WakeWordCommand::List { json },
        })
        .unwrap();
    }
    invoke(TopCommand::Config {
        command: ConfigCommand::Set {
            key: "backend.threads".into(),
            value: "2".into(),
        },
    })
    .unwrap();
    invoke(TopCommand::Config {
        command: ConfigCommand::Unset {
            key: "backend.threads".into(),
        },
    })
    .unwrap();
    assert!(
        invoke(TopCommand::Test {
            audio: None,
            seconds: None,
            execute: false,
            json: false,
        })
        .is_err()
    );
    assert!(
        invoke(TopCommand::Test {
            audio: Some(paths.data_dir.join("missing.wav")),
            seconds: None,
            execute: false,
            json: false,
        })
        .is_err()
    );
    assert!(
        invoke(TopCommand::Test {
            audio: None,
            seconds: Some(0),
            execute: false,
            json: false,
        })
        .is_err()
    );
}

#[test]
fn top_level_control_dispatch_handles_successful_injected_daemon_responses() {
    let paths = test_paths("control-dispatch");
    Config::default().save(&paths.config_file).unwrap();
    for (command, expected_command, state) in [
        (TopCommand::Status { json: false }, Command::Status, "armed"),
        (TopCommand::Status { json: true }, Command::Status, "armed"),
        (TopCommand::Pause, Command::Pause, "paused"),
        (TopCommand::Resume, Command::Resume, "armed"),
        (TopCommand::Stop, Command::Shutdown, "stopping"),
    ] {
        run_with_paths_and_request(
            Cli {
                config: Some(paths.config_file.clone()),
                command,
            },
            paths.clone(),
            |received_paths, received_command| {
                assert_eq!(received_paths.socket(), paths.socket());
                assert_eq!(
                    std::mem::discriminant(&received_command),
                    std::mem::discriminant(&expected_command)
                );
                Ok(Response {
                    protocol: 1,
                    id: "in-memory".into(),
                    result: ResultPayload::State {
                        state: state.into(),
                        details: json!({}),
                    },
                })
            },
        )
        .unwrap();
    }
}

#[test]
fn top_level_audio_device_dispatch_uses_injected_enumerator() {
    let paths = test_paths("audio-device-dispatch");
    Config::default().save(&paths.config_file).unwrap();
    for json in [false, true] {
        run_with_paths_and_services(
            Cli {
                config: Some(paths.config_file.clone()),
                command: TopCommand::AudioDevices { json },
            },
            paths.clone(),
            |_, _| unreachable!(),
            || Ok(vec!["Microphone One".into(), "Microphone Two".into()]),
        )
        .unwrap();
    }
    assert!(
        run_with_paths_and_services(
            Cli {
                config: Some(paths.config_file.clone()),
                command: TopCommand::AudioDevices { json: false },
            },
            paths,
            |_, _| unreachable!(),
            || bail!("enumeration failed"),
        )
        .is_err()
    );
}

#[test]
fn setup_dispatch_covers_checks_catalog_and_safe_failure_paths() {
    let paths = test_paths("setup-dispatch");
    let config = &paths.config_file;

    for json in [false, true] {
        assert!(setup(Some(SetupCommand::Check { json }), config, &paths).is_err());
        setup(Some(SetupCommand::Runtime { json }), config, &paths).unwrap();
        setup(
            Some(SetupCommand::Model {
                list: true,
                json,
                download: None,
                set: None,
                verify: None,
                archive: None,
                no_activate: false,
                progress_format: ProgressFormat::Human,
            }),
            config,
            &paths,
        )
        .unwrap();
    }

    let model = Config::default().model.name;
    let model_command = |download, set, verify, archive| SetupCommand::Model {
        list: false,
        json: false,
        download,
        set,
        verify,
        archive,
        no_activate: false,
        progress_format: ProgressFormat::Human,
    };
    assert!(
        setup(
            Some(model_command(None, None, Some(model.clone()), None)),
            config,
            &paths,
        )
        .is_err()
    );
    assert!(
        setup(
            Some(model_command(None, Some(model.clone()), None, None)),
            config,
            &paths,
        )
        .is_err()
    );
    assert!(
        setup(
            Some(model_command(
                Some(model.clone()),
                None,
                None,
                Some(paths.data_dir.join("missing.tar.bz2")),
            )),
            config,
            &paths,
        )
        .is_err()
    );
    assert!(
        setup(
            Some(SetupCommand::All {
                model,
                archive: Some(paths.data_dir.join("missing.tar.bz2")),
                no_start: true,
                progress_format: ProgressFormat::Json,
            }),
            config,
            &paths,
        )
        .is_err()
    );

    setup(
        Some(SetupCommand::Menu {
            uninstall: false,
            status: false,
        }),
        config,
        &paths,
    )
    .unwrap();
    setup(
        Some(SetupCommand::Menu {
            uninstall: false,
            status: true,
        }),
        config,
        &paths,
    )
    .unwrap();
    setup(
        Some(SetupCommand::Menu {
            uninstall: true,
            status: false,
        }),
        config,
        &paths,
    )
    .unwrap();
}

#[test]
fn request_reader_is_testable_without_a_unix_socket() {
    let request = read_request(std::io::Cursor::new(encoded_request(
        1,
        "cursor",
        Command::Resume,
    )))
    .unwrap();
    assert_eq!(request.id, "cursor");
    assert!(matches!(request.command, Command::Resume));
    assert!(read_request(std::io::Cursor::new(b"invalid\n")).is_err());
    assert!(read_request(std::io::Cursor::new(vec![b'x'; 65_537])).is_err());
}

#[test]
fn control_protocol_maps_every_request_to_a_response_and_transition() {
    let details = json!({"backend": {"kind": "fake"}});
    let request = |protocol: u32, id: &str, command: Command| {
        Ok(Request {
            protocol,
            id: id.into(),
            command,
        })
    };

    for (command, expected_state, transition) in [
        (Command::Status, "current", false),
        (Command::Pause, "paused", true),
        (Command::Resume, "armed", true),
        (Command::Shutdown, "stopping", true),
    ] {
        let (response, next) =
            control_response(request(1, expected_state, command), "current", &details);
        assert_eq!(response.id, expected_state);
        assert_eq!(next.is_some(), transition);
        assert!(matches!(
            response.result,
            ResultPayload::State { ref state, ref details }
                if state == expected_state && details["backend"]["kind"] == "fake"
        ));
    }

    let (response, next) =
        control_response(request(9, "old", Command::Status), "current", &details);
    assert!(next.is_none());
    assert!(
        matches!(response.result, ResultPayload::Error { ref code, .. } if code == "protocol_mismatch")
    );

    let (response, next) =
        control_response(Err(anyhow::anyhow!("broken JSON")), "current", &details);
    assert!(next.is_none());
    assert!(
        matches!(response.result, ResultPayload::Error { ref code, .. } if code == "invalid_request")
    );
}

#[test]
fn control_stream_processes_messages_entirely_in_memory() {
    let details = json!({"backend": {"kind": "fake"}});
    for (bytes, expected_code, expected_command) in [
        (b"not-json\n".to_vec(), Some("invalid_request"), None),
        (
            encoded_request(2, "old", Command::Status),
            Some("protocol_mismatch"),
            None,
        ),
        (encoded_request(1, "status", Command::Status), None, None),
        (
            encoded_request(1, "pause", Command::Pause),
            None,
            Some(Command::Pause),
        ),
    ] {
        let mut stream = ScriptedStream::responding_with(bytes);
        let command = handle_control_stream(&mut stream, "armed", &details);
        assert_eq!(command.is_some(), expected_command.is_some());
        let response: Response = serde_json::from_slice(&stream.request).unwrap();
        match expected_code {
            Some(expected) => assert!(
                matches!(response.result, ResultPayload::Error { ref code, .. } if code == expected)
            ),
            None => assert!(matches!(response.result, ResultPayload::State { .. })),
        }
    }

    let mut disconnected =
        ScriptedStream::responding_with(encoded_request(1, "disconnected", Command::Status));
    disconnected.fail_writes = true;
    assert!(handle_control_stream(&mut disconnected, "armed", &details).is_none());
}

#[test]
fn control_polling_skips_non_commands_and_stops_at_transition_in_memory() {
    let details = json!({"backend": {"kind": "fake"}});
    let mut clients = VecDeque::from([
        ScriptedStream::responding_with(b"invalid\n".to_vec()),
        ScriptedStream::responding_with(encoded_request(1, "status", Command::Status)),
        ScriptedStream::responding_with(encoded_request(1, "stop", Command::Shutdown)),
    ]);
    let command = poll_control_connections(|| Ok(clients.pop_front()), "armed", &details).unwrap();
    assert!(matches!(command, Some(Command::Shutdown)));
    assert!(clients.is_empty());

    assert!(
        poll_control_connections(
            || Ok::<Option<ScriptedStream>, anyhow::Error>(None),
            "armed",
            &details,
        )
        .unwrap()
        .is_none()
    );
    assert!(
        poll_control_connections(
            || Err::<Option<ScriptedStream>, _>(anyhow::anyhow!("accept failed")),
            "armed",
            &details,
        )
        .is_err()
    );

    let command = poll_control(
        &FakeControl,
        "armed",
        Some(json!({"device": "test microphone"})),
        || {
            Ok(Some(ScriptedStream::responding_with(encoded_request(
                1,
                "pause",
                Command::Pause,
            ))))
        },
    )
    .unwrap();
    assert!(matches!(command, Some(Command::Pause)));

    assert!(
        prepare_accepted_control(Ok(()), |_| Ok(()))
            .unwrap()
            .is_some()
    );
    assert!(
        prepare_accepted_control::<(), _>(
            Err(std::io::Error::from(std::io::ErrorKind::WouldBlock)),
            |_| unreachable!(),
        )
        .unwrap()
        .is_none()
    );
    assert!(
        prepare_accepted_control::<(), _>(Err(std::io::Error::other("accept failed")), |_| {
            unreachable!()
        })
        .is_err()
    );
    assert!(
        prepare_accepted_control(Ok(()), |_| Err(std::io::Error::other("configure failed")))
            .is_err()
    );
}

#[test]
fn real_detector_forwards_control_metadata_without_native_backend() {
    let mut config = Config::default();
    config.wake_words[0].command = vec!["true".into()];
    let detector = Detector::from_backend(
        &config,
        Box::new(InMemoryBackend),
        "HEY ATREYU @hey-atreyu".into(),
        Runtime::Cuda,
        true,
        Duration::from_millis(17),
    )
    .unwrap();

    assert_eq!(DetectorControl::backend_kind(&detector), "in-memory");
    assert_eq!(DetectorControl::effective_runtime(&detector), Runtime::Cuda);
    assert!(DetectorControl::fallback_used(&detector));
    assert_eq!(
        DetectorControl::load_time(&detector),
        Duration::from_millis(17)
    );
    assert_eq!(
        DetectorControl::keywords_buffer(&detector),
        "HEY ATREYU @hey-atreyu"
    );
    assert_eq!(
        DetectorControl::run(&detector, "hey-atreyu")
            .unwrap()
            .status,
        0
    );

    let mut streams = VecDeque::from([ScriptedStream::responding_with(encoded_request(
        1,
        "pause",
        Command::Pause,
    ))]);
    assert!(matches!(
        poll_control(&detector, "armed", None, || Ok(streams.pop_front())).unwrap(),
        Some(Command::Pause)
    ));
}

#[test]
fn client_request_exchange_is_testable_without_a_socket() {
    let response = Response {
        protocol: 1,
        id: "server-id".into(),
        result: ResultPayload::State {
            state: "armed".into(),
            details: json!({}),
        },
    };
    let mut encoded_response = serde_json::to_vec(&response).unwrap();
    encoded_response.push(b'\n');
    let mut stream = ScriptedStream::responding_with(encoded_response);
    let received = request_over_stream(&mut stream, Command::Status).unwrap();
    assert!(matches!(
        received.result,
        ResultPayload::State { ref state, .. } if state == "armed"
    ));
    let sent: Request = serde_json::from_slice(&stream.request).unwrap();
    assert_eq!(sent.protocol, 1);
    assert!(matches!(sent.command, Command::Status));

    let mut malformed = ScriptedStream::responding_with(b"invalid\n".to_vec());
    assert!(request_over_stream(&mut malformed, Command::Pause).is_err());

    let mut disconnected = ScriptedStream::responding_with(Vec::new());
    disconnected.fail_writes = true;
    assert!(request_over_stream(&mut disconnected, Command::Resume).is_err());

    let paths = test_paths("request-connector");
    let mut encoded_response = serde_json::to_vec(&response).unwrap();
    encoded_response.push(b'\n');
    let received = request_with_connector(&paths, Command::Status, |path| {
        assert_eq!(path, socket_path(&paths));
        Ok(ScriptedStream::responding_with(encoded_response))
    })
    .unwrap();
    assert!(matches!(received.result, ResultPayload::State { .. }));
    assert!(
        request_with_connector::<ScriptedStream, _>(&paths, Command::Status, |_| {
            Err(std::io::Error::other("connect failed"))
        })
        .is_err()
    );
}
