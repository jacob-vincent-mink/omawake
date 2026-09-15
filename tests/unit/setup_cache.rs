use super::*;

fn fixture(name: &str) -> AppPaths {
    let root =
        std::env::temp_dir().join(format!("omawake-cache-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    AppPaths {
        config_file: root.join("config/omawake/config.toml"),
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

    let mut external = config;
    external.backend.provider_config = "provider.config".into();
    assert!(status(&external, &paths).is_err());
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

    let spec = crate::catalog::model(&config.model.name).unwrap();
    let probe_audio = config.model_directory(&paths).join(spec.probe_audio);
    fs::create_dir_all(probe_audio.parent().unwrap()).unwrap();
    fs::write(probe_audio, b"wav").unwrap();
    let prepared = prepare_for_runtime_with(
        &config,
        &paths.config_file,
        &paths,
        ProgressFormat::Json,
        |_, _, paths| Ok(report(paths, "gpu", true)),
    )
    .unwrap()
    .unwrap();
    assert!(prepared.prepared);

    let mut custom = config;
    custom.backend.provider_config = "custom-provider.config".into();
    assert!(
        prepare_for_runtime_with(
            &custom,
            &paths.config_file,
            &paths,
            ProgressFormat::Human,
            |_, _, _| unreachable!(),
        )
        .is_err()
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
    config.backend.provider_config = "custom".into();
    assert!(child_with(&config, &paths, |_, _, _| Ok(())).is_err());
    config.backend.provider_config.clear();
    config.model.name = "unknown".into();
    assert!(child_with(&config, &paths, |_, _, _| Ok(())).is_err());
    config.model.name = crate::catalog::models()[0].id.into();
    assert!(child_with(&config, &paths, |_, _, _| Ok(())).is_err());

    let spec = crate::catalog::models().first().unwrap();
    let model = config.model_directory(&paths);
    fs::create_dir_all(model.join("test_wavs")).unwrap();
    fs::write(model.join(spec.probe_audio), b"wav").unwrap();
    assert!(
        child_with(&config, &paths, |_, _, audio| {
            assert!(audio.ends_with(spec.probe_audio));
            bail!("NPU rejected model")
        })
        .is_err()
    );
    assert!(child_with(&config, &paths, |_, _, _| Ok(())).is_err());

    let ready = child_with(&config, &paths, |candidate, received_paths, audio| {
        assert_eq!(candidate.backend.fallback, Fallback::Error);
        assert_eq!(received_paths, &paths);
        assert_eq!(audio, model.join(spec.probe_audio));
        let cache = crate::engine::openvino_cache_directory(candidate, received_paths)?;
        fs::create_dir_all(&cache)?;
        fs::write(cache.join("model.blob"), b"compiled")?;
        Ok(())
    })
    .unwrap();
    assert!(ready.prepared);
    assert_eq!(ready.artifacts, 1);
    assert!(ready.elapsed_milliseconds.is_some());

    config.backend.device = "gpu".into();
    let gpu = child_with(&config, &paths, |candidate, received_paths, audio| {
        assert_eq!(audio, model.join(spec.probe_audio));
        let cache = crate::engine::openvino_cache_directory(candidate, received_paths)?;
        fs::create_dir_all(&cache)?;
        fs::write(cache.join("model.blob"), b"gpu-compiled")?;
        Ok(())
    })
    .unwrap();
    assert!(gpu.prepared);
    assert_eq!(gpu.bytes, 12);
}

#[test]
fn isolated_rejects_custom_provider_configs_before_spawning() {
    let paths = fixture("isolated");
    let mut config = openvino("npu");
    config.backend.provider_config = "custom".into();
    let error = isolated(&config, &paths.config_file, &paths).unwrap_err();
    assert!(error.to_string().contains("managed provider config"));
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
