use super::*;
use crate::catalog::{LicenseNotice, ModelAsset};

fn temp(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("omawake-model-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn leak(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn digest(bytes: &[u8]) -> &'static str {
    leak(format!("{:x}", Sha256::digest(bytes)))
}

fn fixture_spec(first: &'static [u8], second: &'static [u8]) -> &'static ModelSpec {
    let assets = Box::leak(
        vec![
            ModelAsset {
                role: "verifier",
                path: "model.gguf",
                url: "https://example.invalid/model.gguf",
                size: first.len() as u64,
                sha256: digest(first),
                source_url: "https://example.invalid/original",
                source_revision: "1111111111111111111111111111111111111111",
                license: "MIT",
            },
            ModelAsset {
                role: "vad",
                path: "vad.safetensors",
                url: "https://example.invalid/vad.safetensors",
                size: second.len() as u64,
                sha256: digest(second),
                source_url: "https://example.invalid/vad",
                source_revision: "2222222222222222222222222222222222222222",
                license: "MIT",
            },
        ]
        .into_boxed_slice(),
    );
    let notices = Box::leak(
        vec![LicenseNotice {
            path: "LICENSES/test.txt",
            license: "MIT",
            copyright: "Copyright test",
            source_url: "https://example.invalid/license",
        }]
        .into_boxed_slice(),
    );
    Box::leak(Box::new(ModelSpec {
        id: "tiny",
        backend: "audiocpp",
        family: "test",
        description: "test model",
        license: "MIT",
        license_status: "verified",
        license_url: "https://example.invalid/license",
        source_url: "https://example.invalid/original",
        source_revision: "1111111111111111111111111111111111111111",
        languages: &["en"],
        multilingual: false,
        converted_source_url: "https://example.invalid/converted",
        converted_source_revision: "3333333333333333333333333333333333333333",
        downloadable: true,
        verifier: "model.gguf",
        vad: "vad.safetensors",
        sample_rate: 16_000,
        probe_audio: None,
        assets,
        notices,
    }))
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

#[test]
fn model_path_is_backend_neutral() {
    let app = paths(Path::new("/tmp/root"));
    assert_eq!(
        model_directory(&app, crate::catalog::models().first().unwrap()),
        PathBuf::from("/tmp/root/data/models/moonshine-streaming-tiny-q8_0-silero-v6.2.1")
    );
}

#[test]
fn multi_file_download_is_verified_manifested_atomic_and_idempotent() {
    let root = temp("download");
    let app = paths(&root);
    let spec = fixture_spec(b"model", b"vad");
    let fetches = std::cell::Cell::new(0);
    let installed = install_with_fetch(&app, spec, None, ProgressFormat::Json, |asset| {
        fetches.set(fetches.get() + 1);
        let bytes: &'static [u8] = if asset.path == "model.gguf" {
            b"model"
        } else {
            b"vad"
        };
        Ok(Box::new(std::io::Cursor::new(bytes)))
    })
    .unwrap();
    assert_eq!(fetches.get(), 2);
    assert_eq!(fs::read(installed.join("model.gguf")).unwrap(), b"model");
    assert_eq!(fs::read(installed.join("vad.safetensors")).unwrap(), b"vad");
    assert!(installed.join("PROVENANCE.json").is_file());
    assert!(installed.join("LICENSES/test.txt").is_file());
    let provenance: serde_json::Value =
        serde_json::from_slice(&fs::read(installed.join("PROVENANCE.json")).unwrap()).unwrap();
    assert_eq!(provenance["assets"].as_array().unwrap().len(), 2);
    assert_eq!(provenance["installed_from"], "catalog-download");
    verify(&app, spec).unwrap();

    install_with_fetch(&app, spec, None, ProgressFormat::Human, |_| {
        panic!("verified installation should not download")
    })
    .unwrap();
}

#[test]
fn local_directory_install_verifies_every_asset_and_repairs_corruption() {
    let root = temp("local");
    let app = paths(&root);
    let spec = fixture_spec(b"model", b"vad");
    let source = root.join("source");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("model.gguf"), b"model").unwrap();
    fs::write(source.join("vad.safetensors"), b"vad").unwrap();
    let installed = install(&app, spec, Some(&source), ProgressFormat::Human).unwrap();
    let provenance: serde_json::Value =
        serde_json::from_slice(&fs::read(installed.join("PROVENANCE.json")).unwrap()).unwrap();
    assert_eq!(provenance["installed_from"], "local-directory");

    fs::write(installed.join("PROVENANCE.json"), b"{}").unwrap();
    assert!(verify(&app, spec).is_err());
    install(&app, spec, Some(&source), ProgressFormat::Human).unwrap();
    fs::write(installed.join("LICENSES/test.txt"), b"changed").unwrap();
    assert!(verify(&app, spec).is_err());
    install(&app, spec, Some(&source), ProgressFormat::Human).unwrap();

    fs::write(installed.join("model.gguf"), b"wrong").unwrap();
    assert!(verify(&app, spec).is_err());
    install(&app, spec, Some(&source), ProgressFormat::Human).unwrap();
    verify(&app, spec).unwrap();
}

#[test]
fn failed_second_asset_leaves_previous_model_and_cleans_staging() {
    let root = temp("rollback");
    let app = paths(&root);
    let spec = fixture_spec(b"model", b"vad");
    let target = model_directory(&app, spec);
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("previous"), b"keep").unwrap();
    let error = install_with_fetch(&app, spec, None, ProgressFormat::Human, |asset| {
        if asset.path == "model.gguf" {
            Ok(Box::new(std::io::Cursor::new(b"model")))
        } else {
            bail!("second fetch failed")
        }
    })
    .unwrap_err();
    assert!(error.to_string().contains("prepare model installation"));
    assert_eq!(fs::read(target.join("previous")).unwrap(), b"keep");
    let staging = app
        .data_dir
        .join("models")
        .join(format!(".tiny.install-{}", std::process::id()));
    assert!(!staging.exists());
}

#[test]
fn corrupt_download_and_unsafe_catalog_paths_are_rejected() {
    let root = temp("invalid");
    let app = paths(&root);
    let spec = fixture_spec(b"model", b"vad");
    assert!(
        install_with_fetch(&app, spec, None, ProgressFormat::Human, |_| {
            Ok(Box::new(std::io::Cursor::new(b"wrong")))
        })
        .is_err()
    );
    assert!(!model_directory(&app, spec).exists());
    assert!(validate_relative_file("../escape").is_err());
    assert!(validate_relative_file("/absolute").is_err());
    assert!(validate_relative_file("").is_err());
}

#[test]
fn download_writer_checks_length_checksum_and_progress() {
    let root = temp("writer");
    let spec = fixture_spec(b"model", b"vad");
    let asset = &spec.assets[0];
    let target = root.join("model.gguf");
    let part = root.join("model.gguf.part");
    write_download(
        b"model".as_slice(),
        &part,
        &target,
        spec,
        asset,
        ProgressFormat::Json,
    )
    .unwrap();
    verify_file(&target, asset.size, asset.sha256).unwrap();
    assert!(should_report_progress(1024 * 1024, 0, 2 * 1024 * 1024));
    assert!(!should_report_progress(1, 0, 2));
    assert!(
        write_download(
            b"mod".as_slice(),
            &part,
            &target,
            spec,
            asset,
            ProgressFormat::Human
        )
        .is_err()
    );
    assert!(
        write_download(
            b"models".as_slice(),
            &part,
            &target,
            spec,
            asset,
            ProgressFormat::Human
        )
        .is_err()
    );
}

#[test]
fn unsupported_notice_and_invalid_source_directory_fail_before_activation() {
    let root = temp("notice");
    let app = paths(&root);
    let base = fixture_spec(b"model", b"vad");
    let mut invalid = *base;
    invalid.notices = Box::leak(
        vec![LicenseNotice {
            path: "LICENSES/x",
            license: "Other",
            copyright: "x",
            source_url: "https://example.invalid",
        }]
        .into_boxed_slice(),
    );
    assert!(
        install_with_fetch(
            &app,
            Box::leak(Box::new(invalid)),
            None,
            ProgressFormat::Human,
            |_| unreachable!()
        )
        .is_err()
    );
    assert!(
        install(
            &app,
            base,
            Some(Path::new("relative")),
            ProgressFormat::Human
        )
        .is_err()
    );
    assert!(emit(ProgressFormat::Human, "event", base, None, None, None).is_ok());
}
