use super::*;

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omawake-runtime-path-test-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn report_separates_config_environment_package_and_loadability() {
    let root = temp("report");
    let configured = root.join("configured");
    let environment = root.join("environment");
    let package = root.join("package/lib");
    for directory in [&configured, &environment, &package] {
        fs::create_dir_all(directory).unwrap();
    }
    fs::write(package.join("libonnxruntime_providers_cuda.so.1"), b"").unwrap();
    fs::write(package.join("libonnxruntime.so.1"), b"").unwrap();
    fs::write(package.join("libsherpa-onnx-c-api.so.1"), b"").unwrap();
    fs::write(
        environment.join("libonnxruntime_providers_openvino.so"),
        b"",
    )
    .unwrap();
    let executable = root.join("package/omawake");
    fs::write(&executable, b"").unwrap();
    let config = BackendConfig {
        library_dirs: vec![PathBuf::from("configured"), environment.clone()],
        ..Default::default()
    };
    let app_path = env::join_paths([&environment]).unwrap();
    let config_path = root.join("config.toml");
    let report = report_with(
        &config,
        &config_path,
        Some(&executable),
        Some(&app_path),
        None,
        None,
        [None, None, None],
        |_, _| true,
        |_, _| true,
        |_, _, _, _, _, _| true,
    );
    assert_eq!(
        report.configured_library_dirs,
        [configured.clone(), environment.clone()]
    );
    assert_eq!(
        report.environment_library_dirs.as_slice(),
        std::slice::from_ref(&environment)
    );
    assert_eq!(
        report.package_library_dirs.as_slice(),
        std::slice::from_ref(&package)
    );
    assert_eq!(
        report.effective_library_dirs,
        [configured.clone(), environment, package]
    );
    assert!(report.missing_library_dirs.is_empty());
    let context = report.tui_context();
    assert!(context.contains(&configured.display().to_string()));
    assert!(context.contains("Effective:"));
    assert!(context.contains("Remediation: none"));
    assert!(report.runtime_loadable["default"]);
    assert!(report.runtime_loadable["openvino"]);
    assert!(report.runtime_loadable["cuda"]);
}

#[test]
fn invalid_paths_are_reported_and_reexec_is_refused() {
    let root = temp("invalid");
    let relative = PathBuf::from("relative/lib");
    let missing = root.join("missing");
    let config = BackendConfig {
        library_dirs: vec![relative.clone(), missing.clone()],
        ..Default::default()
    };
    let config_path = root.join("config/config.toml");
    let report = report_with(
        &config,
        &config_path,
        None,
        None,
        None,
        None,
        [None, None, None],
        |_, _| true,
        |_, _| true,
        |_, _, _, _, _, _| true,
    );
    assert_eq!(
        report.missing_library_dirs,
        [root.join("config").join(relative), missing]
    );
    assert!(!report.remediation.is_empty());
    assert!(reexec_library_path(&report, None, false).is_err());
}

#[test]
fn reexec_plan_prepends_owned_paths_preserves_ambient_and_stops_loops() {
    let root = temp("reexec");
    let owned = root.join("owned");
    let ambient = root.join("ambient");
    fs::create_dir_all(&owned).unwrap();
    fs::create_dir_all(&ambient).unwrap();
    let config = BackendConfig {
        library_dirs: vec![owned.clone()],
        ..Default::default()
    };
    let report = report_with(
        &config,
        &root.join("config.toml"),
        None,
        None,
        None,
        None,
        [None, None, None],
        |_, _| true,
        |_, _| true,
        |_, _, _, _, _, _| true,
    );
    let ambient_path = env::join_paths([&ambient]).unwrap();
    let augmented = reexec_library_path(&report, Some(&ambient_path), false)
        .unwrap()
        .unwrap();
    assert_eq!(split_paths(Some(&augmented)), [owned.clone(), ambient]);
    assert!(
        reexec_library_path(&report, Some(&augmented), false)
            .unwrap()
            .is_none()
    );
    assert!(reexec_library_path(&report, None, true).is_err());
}

#[test]
fn provider_with_unresolved_dependencies_is_not_loadable() {
    let root = temp("unresolved-provider");
    fs::write(
        root.join("libonnxruntime_providers_openvino.so"),
        b"not a real shared object",
    )
    .unwrap();
    assert!(!provider_dependencies_resolve(
        &root.join("libonnxruntime_providers_openvino.so"),
        std::slice::from_ref(&root),
    ));
    let config = BackendConfig {
        library_dirs: vec![root.clone()],
        ..Default::default()
    };
    let report = report_with(
        &config,
        &root.join("config.toml"),
        None,
        None,
        None,
        None,
        [None, None, None],
        |_, _| false,
        |_, _| true,
        |_, _, _, _, _, _| true,
    );
    assert!(!report.runtime_loadable["openvino"]);
}

#[test]
fn exact_library_precedence_and_loader_fallbacks_are_reported() {
    let root = temp("exact-precedence");
    let config_dir = root.join("config");
    let ambient = root.join("ambient");
    fs::create_dir_all(&config_dir).unwrap();
    fs::create_dir_all(&ambient).unwrap();
    let ort_environment = ambient.join("libonnxruntime.so.1.29.0");
    let sherpa_config = config_dir.join("libsherpa-onnx-c-api.so");
    let openvino_config = config_dir.join("libonnxruntime_providers_openvino.so");
    let cuda_environment = ambient.join("libonnxruntime_providers_cuda.so");
    for path in [
        &ort_environment,
        &sherpa_config,
        &openvino_config,
        &cuda_environment,
    ] {
        fs::write(path, b"fixture").unwrap();
    }
    let config = BackendConfig {
        runtime: crate::backend::Runtime::Openvino,
        sherpa_library: PathBuf::from("libsherpa-onnx-c-api.so"),
        provider_library: PathBuf::from("libonnxruntime_providers_openvino.so"),
        ..Default::default()
    };
    let report = report_with(
        &config,
        &config_dir.join("config.toml"),
        Some(Path::new("")),
        None,
        Some(ambient.as_os_str()),
        None,
        [Some(ort_environment.clone()), None, None],
        |_, _| true,
        |_, _| true,
        |_, _, _, _, _, _| true,
    );
    assert_eq!(report.onnxruntime_library, Some(ort_environment));
    assert_eq!(report.sherpa_library, Some(sherpa_config));
    assert_eq!(report.provider_library, Some(openvino_config));
    assert!(report.runtime_loadable["openvino"]);

    let located = runtime_library(
        "libfromcache.so",
        &[],
        Some("libfromcache.so (libc6,x86-64) => /opt/runtime/libfromcache.so"),
    );
    assert_eq!(located, Some(PathBuf::from("/opt/runtime/libfromcache.so")));
    assert!(package_library_dirs(Path::new("")).is_empty());
    assert!(provider_dependencies_resolve(Path::new("/bin/ls"), &[]));

    let owned = root.join("owned");
    fs::create_dir_all(&owned).unwrap();
    let config = BackendConfig {
        library_dirs: vec![owned.clone()],
        ..Default::default()
    };
    assert_eq!(
        split_paths(
            effective_library_path(&config, &root.join("config.toml"))
                .unwrap()
                .as_deref()
        ),
        [owned]
    );
}
