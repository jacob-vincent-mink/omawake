use super::*;

fn fixture(name: &str) -> AppPaths {
    let root = crate::test_support::unique_directory("cache", name);
    AppPaths {
        config_file: root.join("config/omawake/config.toml"),
        config_home: root.join("config"),
        data_dir: root.join("data/omawake"),
        cache_dir: root.join("cache/omawake"),
        state_dir: root.join("state/omawake"),
        runtime_dir: root.join("run/omawake"),
    }
}

fn openvino(device: &str) -> Config {
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = device.into();
    config
}

fn report(paths: &AppPaths, device: &str, prepared: bool) -> CacheReport {
    CacheReport {
        required: true,
        prepared,
        directory: Some(paths.cache_dir.join(format!("openvino/{device}/compiled"))),
        artifacts: usize::from(prepared),
        bytes: if prepared { 7 } else { 0 },
        elapsed_milliseconds: Some(2.5),
    }
}

fn shell_script(path: &Path) -> Command {
    let mut command = Command::new("/bin/sh");
    command.arg(path);
    command
}

fn placement(
    device: &str,
    static_pipeline: bool,
) -> crate::engine::openvino_genai::PlacementEvidence {
    crate::engine::openvino_genai::PlacementEvidence {
        profile_id: "whisper-base.en-int8-ov".into(),
        languages: vec!["en".into()],
        language: String::new(),
        multilingual: false,
        runtime_build: "test".into(),
        runtime_description: "safe injected OpenVINO".into(),
        requested_device: device.into(),
        available_device: device.into(),
        full_device_name: format!("Test {device}"),
        device_architecture: "test".into(),
        driver_version: "test".into(),
        static_pipeline,
        pipeline_load_milliseconds: 1.0,
        cache_directory: "/test/cache".into(),
        cache_files: 1,
        cache_bytes: 8,
        genai_library: "/test/libopenvino_genai_c.so".into(),
        core_library: "/test/libopenvino_c.so".into(),
    }
}

#[test]
fn cache_requirement_and_status_follow_explicit_accelerator_selection() {
    let paths = fixture("status");
    let default = Config::default();
    assert!(!required(&default));
    let skipped = status(&default, &paths).unwrap();
    assert!(!skipped.required && skipped.prepared && skipped.directory.is_none());

    let config = openvino("npu");
    assert!(required(&config));
    let missing = status(&config, &paths).unwrap();
    assert!(missing.required && !missing.prepared && missing.artifacts == 0);

    let directory = crate::engine::openvino_cache_directory(&config, &paths).unwrap();
    fs::create_dir_all(directory.join("nested")).unwrap();
    fs::write(directory.join("provider.config"), b"not a compiled graph").unwrap();
    fs::write(directory.join("model.bin"), b"not a compiled graph").unwrap();
    fs::write(directory.join("empty.blob"), []).unwrap();
    fs::write(directory.join("nested/model.blob"), b"compiled").unwrap();
    let ready = status(&config, &paths).unwrap();
    assert!(ready.prepared);
    assert_eq!(ready.artifacts, 1);
    assert_eq!(ready.bytes, 8);

    let gpu = openvino("gpu");
    assert!(required(&gpu));
    let missing_gpu = status(&gpu, &paths).unwrap();
    assert!(missing_gpu.required && !missing_gpu.prepared);

    let cpu = openvino("cpu");
    assert!(!required(&cpu));
}

#[test]
fn runtime_preparation_defers_until_the_catalog_probe_audio_is_installed() {
    let paths = fixture("runtime-deferred");
    let config = openvino("gpu");
    let deferred = prepare_for_runtime_with(
        &config,
        &paths.config_file,
        &paths,
        ProgressFormat::Human,
        |_, _, _| unreachable!(),
    )
    .unwrap();
    assert!(deferred.is_none());

    assert!(
        crate::catalog::model(&config.model.name)
            .unwrap()
            .probe_audio
            .is_none()
    );
}

#[test]
fn preparation_skips_other_devices_and_requires_a_persisted_artifact() {
    let paths = fixture("prepare");
    let skipped = prepare_with(
        &Config::default(),
        &paths.config_file,
        &paths,
        ProgressFormat::Human,
        |_, _, _| unreachable!(),
    )
    .unwrap();
    assert!(skipped.is_none());

    let config = openvino("npu");
    for progress in [ProgressFormat::Human, ProgressFormat::Json] {
        let prepared = prepare_with(
            &config,
            &paths.config_file,
            &paths,
            progress,
            |_, _, paths| Ok(report(paths, "npu", true)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(prepared.artifacts, 1);
    }
    assert!(
        prepare_with(
            &config,
            &paths.config_file,
            &paths,
            ProgressFormat::Human,
            |_, _, paths| Ok(report(paths, "npu", false)),
        )
        .is_err()
    );
    assert!(
        prepare_with(
            &config,
            &paths.config_file,
            &paths,
            ProgressFormat::Human,
            |_, _, _| bail!("inference failed"),
        )
        .is_err()
    );
}

#[test]
fn child_requires_managed_accelerator_catalog_audio_and_real_cache_output() {
    let paths = fixture("child");
    let mut config = openvino("npu");
    assert!(child_with(&Config::default(), &paths, |_, _, _| Ok(())).is_err());
    config.backend.fallback = Fallback::Cpu;
    assert!(child_with(&config, &paths, |_, _, _| Ok(())).is_err());
    config.backend.fallback = Fallback::Error;
    config.model.name = "unknown".into();
    assert!(child_with(&config, &paths, |_, _, _| Ok(())).is_err());
    config.model.name = crate::catalog::models()[0].id.into();
    assert!(child_with(&config, &paths, |_, _, _| Ok(())).is_err());

    assert!(crate::catalog::models()[0].probe_audio.is_none());
}

#[test]
fn direct_openvino_child_requires_exact_placement_and_a_persisted_compiled_graph() {
    let paths = fixture("direct-openvino-child");
    let mut config = openvino("npu");
    config.backend.kind = "openvino-genai".into();

    let mut cpu = config.clone();
    cpu.backend.device = "cpu".into();
    let mut unexpected = |_: &Config, _: &AppPaths| unreachable!();
    assert!(openvino_child_with(&cpu, &paths, &mut unexpected).is_err());
    config.backend.fallback = Fallback::Cpu;
    assert!(openvino_child_with(&config, &paths, &mut unexpected).is_err());
    config.backend.fallback = Fallback::Error;

    let mut wrong_device = |_: &Config, _: &AppPaths| Ok(placement("GPU", false));
    assert!(openvino_child_with(&config, &paths, &mut wrong_device).is_err());
    let mut no_static_pipeline = |_: &Config, _: &AppPaths| Ok(placement("NPU", false));
    assert!(openvino_child_with(&config, &paths, &mut no_static_pipeline).is_err());
    let mut no_artifact = |_: &Config, _: &AppPaths| Ok(placement("NPU", true));
    assert!(openvino_child_with(&config, &paths, &mut no_artifact).is_err());

    let mut prepare = |config: &Config, paths: &AppPaths| {
        let cache = crate::engine::openvino_cache_directory(config, paths)?;
        fs::create_dir_all(&cache)?;
        fs::write(cache.join("compiled.blob"), b"compiled")?;
        Ok(placement("NPU", true))
    };
    let report = openvino_child_with(&config, &paths, &mut prepare).unwrap();
    assert!(report.prepared);
    assert_eq!(report.artifacts, 1);
    assert_eq!(report.bytes, 8);
    assert!(report.elapsed_milliseconds.is_some());
}

#[test]
fn isolated_retries_up_to_five_times_only_for_signal_termination() {
    let paths = fixture("signal-retry");
    let expected = report(&paths, "gpu", true);
    let attempts = std::cell::Cell::new(0);
    let result = retry_signaled("GPU", || {
        attempts.set(attempts.get() + 1);
        if attempts.get() < 5 {
            Ok(AttemptOutcome::Failed {
                status: "signal".into(),
                signal: Some(11),
                stderr: "vendor crash".into(),
            })
        } else {
            Ok(AttemptOutcome::Complete(expected.clone()))
        }
    })
    .unwrap();
    assert!(result.prepared);
    assert_eq!(attempts.get(), 5);

    let attempts = std::cell::Cell::new(0);
    let error = retry_signaled("GPU", || {
        attempts.set(attempts.get() + 1);
        Ok(AttemptOutcome::Failed {
            status: "exit status: 2".into(),
            signal: None,
            stderr: "ordinary failure".into(),
        })
    })
    .unwrap_err();
    assert_eq!(attempts.get(), 1);
    assert!(error.to_string().contains("ordinary failure"));
    assert!(error.to_string().contains("attempt 1/5"));

    let attempts = std::cell::Cell::new(0);
    let error = retry_signaled("GPU", || {
        attempts.set(attempts.get() + 1);
        Ok(AttemptOutcome::Failed {
            status: "signal".into(),
            signal: Some(11),
            stderr: String::new(),
        })
    })
    .unwrap_err();
    assert_eq!(attempts.get(), 5);
    assert!(error.to_string().contains("signal 11"));
    assert!(error.to_string().contains("attempt 5/5"));
}

#[test]
fn isolated_child_protocol_captures_success_failure_and_malformed_output() {
    let paths = fixture("isolated-protocol");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    let candidate = openvino("npu");
    let library_path = std::ffi::OsStr::new("");

    let success = paths.runtime_dir.join("success.sh");
    fs::write(
        &success,
        "#!/bin/sh\nfor response do :; done\nprintf '%s' '{\"required\":true,\"prepared\":true,\"directory\":null,\"artifacts\":1,\"bytes\":8,\"elapsed_milliseconds\":1.0}' > \"$response\"\n",
    )
    .unwrap();
    match isolated_attempt_with_command(
        &candidate,
        &paths.config_file,
        library_path,
        "NPU",
        &paths.runtime_dir,
        shell_script(&success),
    )
    .unwrap()
    {
        AttemptOutcome::Complete(report) => {
            assert!(report.prepared);
            assert_eq!(report.artifacts, 1);
        }
        AttemptOutcome::Failed { .. } => panic!("successful child was reported as failed"),
    }

    let failure = paths.runtime_dir.join("failure.sh");
    fs::write(
        &failure,
        "#!/bin/sh\necho provider-stdout\necho provider-failed >&2\nexit 7\n",
    )
    .unwrap();
    match isolated_attempt_with_command(
        &candidate,
        &paths.config_file,
        library_path,
        "NPU",
        &paths.runtime_dir,
        shell_script(&failure),
    )
    .unwrap()
    {
        AttemptOutcome::Failed { status, stderr, .. } => {
            assert!(status.contains('7'));
            assert!(stderr.contains("provider-stdout"));
            assert!(stderr.contains("provider-failed"));
        }
        AttemptOutcome::Complete(_) => panic!("failed child was reported as successful"),
    }

    let malformed = paths.runtime_dir.join("malformed.sh");
    fs::write(
        &malformed,
        "#!/bin/sh\nfor response do :; done\nprintf not-json > \"$response\"\n",
    )
    .unwrap();
    assert!(
        isolated_attempt_with_command(
            &candidate,
            &paths.config_file,
            library_path,
            "NPU",
            &paths.runtime_dir,
            shell_script(&malformed),
        )
        .is_err()
    );
}

#[test]
fn isolated_child_protocol_terminates_a_stalled_child() {
    let paths = fixture("isolated-timeout");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    let stalled = paths.runtime_dir.join("stalled.sh");
    fs::write(&stalled, "#!/bin/sh\nsleep 10\n").unwrap();

    let error = match isolated_attempt_with_timeout(
        &openvino("npu"),
        &paths.config_file,
        std::ffi::OsStr::new(""),
        "NPU",
        &paths.runtime_dir,
        shell_script(&stalled),
        Duration::from_millis(1),
    ) {
        Ok(_) => panic!("stalled child unexpectedly completed"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("timed out"));
}

#[test]
fn isolated_preparation_forces_error_fallback_and_resolves_child_environment() {
    let paths = fixture("isolated-plan");
    let libraries = paths.data_dir.join("runtime");
    fs::create_dir_all(&libraries).unwrap();
    let provider = libraries.join("libopenvino_genai_c.so");
    fs::write(&provider, b"fixture").unwrap();
    let mut config = openvino("npu");
    config.backend.fallback = Fallback::Cpu;
    config.backend.library_dirs = vec![libraries.clone()];
    config.backend.library = provider.clone();

    let expected = report(&paths, "npu", true);
    let actual = isolated_with(
        &config,
        &paths.config_file,
        |candidate, path, loader, device| {
            assert_eq!(candidate.backend.fallback, Fallback::Error);
            assert_eq!(candidate.backend.library, provider);
            assert_eq!(path, paths.config_file);
            assert_eq!(device, "NPU");
            assert!(std::env::split_paths(loader).any(|entry| entry == libraries));
            Ok(AttemptOutcome::Complete(expected.clone()))
        },
    )
    .unwrap();
    assert!(actual.prepared);
}

#[test]
fn cache_validation_and_json_deferred_paths_are_explicit() {
    let paths = fixture("validation");
    let mut config = openvino("npu");
    config.model.name = "not-in-catalog".into();
    assert!(
        prepare_for_runtime_with(
            &config,
            &paths.config_file,
            &paths,
            ProgressFormat::Json,
            |_, _, _| unreachable!(),
        )
        .unwrap()
        .is_none()
    );
    assert!(catalog_probe_audio(&config, &paths).is_none());

    for (runtime, device) in [
        (Runtime::Default, "cpu"),
        (Runtime::Openvino, "cpu"),
        (Runtime::Cuda, "gpu"),
    ] {
        config.backend.runtime = runtime;
        config.backend.device = device.into();
        assert!(cache_device(&config).is_err());
    }

    let error = retry_signaled("NPU", || {
        Ok(AttemptOutcome::Failed {
            status: "exit status: 1".into(),
            signal: None,
            stderr: "   ".into(),
        })
    })
    .unwrap_err();
    assert!(!error.to_string().ends_with(':'));
}

#[test]
fn public_prepare_reports_an_isolated_child_failure() {
    let paths = fixture("public-isolated-failure");
    let config = openvino("npu");
    let error = prepare(&config, &paths.config_file, &paths, ProgressFormat::Human).unwrap_err();
    assert!(error.to_string().contains("model-cache preparation failed"));
}
