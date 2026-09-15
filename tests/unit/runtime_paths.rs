use super::*;

fn fixture(name: &str) -> PathBuf {
    let root = env::temp_dir().join(format!(
        "omawake-runtime-paths-test-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn discovery_prefers_exact_files_and_reports_provider_remediation() {
    let root = fixture("discover");
    let config_dir = root.join("config");
    let libraries = root.join("libraries");
    fs::create_dir_all(&config_dir).unwrap();
    fs::create_dir_all(&libraries).unwrap();
    for name in [
        "libonnxruntime.so.1.30.0",
        "libonnxruntime_providers_openvino_plugin.so",
        "libonnxruntime_providers_cuda.so.1",
    ] {
        fs::write(libraries.join(name), b"fixture").unwrap();
    }
    let config_path = config_dir.join("omawake.toml");
    let mut config = BackendConfig {
        library_dirs: vec![PathBuf::from("../libraries"), libraries.clone()],
        onnxruntime_library: PathBuf::from("../libraries/libonnxruntime.so.1.30.0"),
        ..Default::default()
    };
    let cpu = discover(&config, &config_path);
    assert_eq!(
        cpu.configured_library_dirs,
        [config_dir.join("../libraries")]
    );
    assert_eq!(
        cpu.onnxruntime_library,
        Some(config_dir.join("../libraries/libonnxruntime.so.1.30.0"))
    );
    assert!(cpu.provider_library.is_none());
    assert!(cpu.remediation.is_empty());
    assert!(cpu.runtime_loadable["default"]);
    assert!(cpu.runtime_loadable["openvino"]);
    assert!(cpu.runtime_loadable["cuda"]);
    assert!(cpu.tui_context().contains("Provider: not selected"));
    assert!(
        effective_library_path(&config, &config_path)
            .unwrap()
            .is_some()
    );

    config.runtime = Runtime::Openvino;
    config.provider_library = PathBuf::from("../libraries/missing-provider.so");
    let missing = discover(&config, &config_path);
    assert_eq!(
        missing.provider_library,
        Some(config_dir.join("../libraries/missing-provider.so"))
    );
    assert!(
        missing
            .remediation
            .join(" ")
            .contains("provider plugin is missing")
    );
    assert!(missing.tui_context().contains("Remediation:"));

    config.runtime = Runtime::Cuda;
    config.provider_library.clear();
    let cuda = discover(&config, &config_path);
    assert_eq!(
        cuda.provider_library,
        Some(config_dir.join("../libraries/libonnxruntime_providers_cuda.so.1"))
    );
}

#[test]
fn directory_scanning_loader_fallbacks_and_validation_are_deterministic() {
    let root = fixture("helpers");
    let binary = root.join("bin/omawake");
    let bin = binary.parent().unwrap();
    fs::create_dir_all(bin.join("lib")).unwrap();
    fs::write(bin.join("lib/libonnxruntime.so.2"), b"new").unwrap();
    fs::write(bin.join("lib/libonnxruntime.so.1"), b"old").unwrap();
    assert_eq!(
        library_path_in_directory("libonnxruntime.so", &bin.join("lib")),
        Some(bin.join("lib/libonnxruntime.so.1"))
    );
    assert!(library_in_directory("libonnxruntime.so", &bin.join("lib")));
    assert!(!library_in_directory("absent.so", &bin.join("lib")));
    assert_eq!(package_library_dirs(&binary), [bin.join("lib")]);

    let system = root.join("system/libonnxruntime.so.1.30.0");
    let listing = format!("libonnxruntime.so (libc6) => {}\n", system.display());
    assert_eq!(
        runtime_library("libonnxruntime.so", &[], Some(&listing)),
        Some(system)
    );
    assert!(runtime_library("absent.so", &[], Some(&listing)).is_none());
    assert_eq!(
        exact_or_discover(Path::new("exact.so"), None, "ignored", &root, &[], None),
        Some(root.join("exact.so"))
    );
    assert_eq!(
        exact_or_discover(
            Path::new(""),
            Some(Path::new("/env/core.so")),
            "ignored",
            &root,
            &[],
            None
        ),
        Some(PathBuf::from("/env/core.so"))
    );

    let duplicate = root.join("duplicate");
    fs::create_dir_all(&duplicate).unwrap();
    let link = root.join("duplicate-link");
    std::os::unix::fs::symlink(&duplicate, &link).unwrap();
    assert_eq!(deduplicate([duplicate.clone(), link]), [duplicate]);
    assert!(split_paths(None).is_empty());
    assert_eq!(display_or_none(&[]), "none");

    let invalid = RuntimeLibraryReport {
        onnxruntime_library: None,
        provider_library: None,
        configured_library_dirs: vec![],
        environment_library_dirs: vec![],
        package_library_dirs: vec![],
        effective_library_dirs: vec![],
        missing_library_dirs: vec![PathBuf::from("relative")],
        runtime_loadable: BTreeMap::new(),
        remediation: vec![],
    };
    assert!(validate(&invalid).is_err());
}

#[test]
fn runtime_report_probes_each_candidate_without_mutating_configuration() {
    let root = fixture("report");
    let missing = root.join("missing/libonnxruntime.so");
    let config = BackendConfig {
        onnxruntime_library: missing,
        ..Default::default()
    };
    let report = report(&config, &root.join("config.toml"));
    assert_eq!(report.runtime_loadable.len(), 3);
    assert!(report.runtime_loadable.values().all(|ready| !ready));
    assert_eq!(report.remediation.len(), 3);
    assert!(report.remediation[0].contains("reinstall Omawake"));
    assert!(report.remediation[1].contains("provider plugin"));
}
