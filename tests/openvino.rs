use std::{env, path::PathBuf};

use omawake::{
    backend::{Fallback, Runtime},
    config::{Config, WakeWord},
    engine::Detector,
    paths::AppPaths,
};

#[test]
fn real_openvino_cpu_plugin_preserves_detection_when_available() {
    let (Some(ort), Some(model), Some(provider)) = (
        env::var_os("OMAWAKE_TEST_ONNXRUNTIME").map(PathBuf::from),
        env::var_os("OMAWAKE_TEST_MODEL").map(PathBuf::from),
        env::var_os("OMAWAKE_TEST_OPENVINO_PROVIDER").map(PathBuf::from),
    ) else {
        return;
    };
    assert!(ort.is_file(), "OMAWAKE_TEST_ONNXRUNTIME is not a file");
    assert!(model.is_dir(), "OMAWAKE_TEST_MODEL is not a directory");
    assert!(
        provider.is_file(),
        "OMAWAKE_TEST_OPENVINO_PROVIDER is not a file"
    );

    let root = env::temp_dir().join(format!("omawake-openvino-test-{}", std::process::id()));
    let paths = AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "cpu".into();
    config.backend.fallback = Fallback::Error;
    config.backend.onnxruntime_library = ort;
    config.backend.provider_library = provider.clone();
    config.backend.library_dirs = vec![provider.parent().unwrap().to_owned()];
    config.model.directory = model.to_string_lossy().into_owned();
    config.wake_words = vec![WakeWord {
        id: "light-up".into(),
        phrase: "Light up".into(),
        enabled: true,
        command: vec!["true".into()],
    }];

    let detector = match Detector::load(&config, &paths) {
        Ok(detector) => detector,
        Err(error)
            if error
                .to_string()
                .contains("plugin exposed no matching device") =>
        {
            return;
        }
        Err(error) => panic!("load OpenVINO CPU detector: {error:#}"),
    };
    assert_eq!(detector.effective_runtime, Runtime::Openvino);
    assert!(!detector.fallback_used);
    assert_eq!(
        detector
            .detect_file(&model.join("test_wavs/0.wav"))
            .unwrap()[0]
            .id,
        "light-up"
    );
}
