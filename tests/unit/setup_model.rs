use super::*;
use bzip2::Compression;
use bzip2::write::BzEncoder;

use crate::catalog::RequiredFile;

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

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn archive(root: &str, path: &str, bytes: &[u8]) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, format!("{root}/{path}"), bytes)
        .unwrap();
    builder.into_inner().unwrap().finish().unwrap()
}

fn archive_with_directory(root: &str, path: &str, bytes: &[u8]) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut directory = tar::Header::new_gnu();
    directory.set_entry_type(tar::EntryType::Directory);
    directory.set_size(0);
    directory.set_mode(0o755);
    directory.set_cksum();
    builder
        .append_data(&mut directory, format!("{root}/assets"), &[][..])
        .unwrap();
    let mut file = tar::Header::new_gnu();
    file.set_size(bytes.len() as u64);
    file.set_mode(0o644);
    file.set_cksum();
    builder
        .append_data(&mut file, format!("{root}/{path}"), bytes)
        .unwrap();
    builder.into_inner().unwrap().finish().unwrap()
}

fn archive_blocking_manifest(root: &str, path: &str, bytes: &[u8]) -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut file = tar::Header::new_gnu();
    file.set_size(bytes.len() as u64);
    file.set_mode(0o644);
    file.set_cksum();
    builder
        .append_data(&mut file, format!("{root}/{path}"), bytes)
        .unwrap();
    let mut directory = tar::Header::new_gnu();
    directory.set_entry_type(tar::EntryType::Directory);
    directory.set_size(0);
    directory.set_mode(0o755);
    directory.set_cksum();
    builder
        .append_data(
            &mut directory,
            format!("{root}/.omawake-model.json"),
            &[][..],
        )
        .unwrap();
    builder.into_inner().unwrap().finish().unwrap()
}

fn archive_with_unsafe_path() -> Vec<u8> {
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_size(1);
    header.set_mode(0o644);
    header.set_path("safe").unwrap();
    header.as_mut_bytes()[..9].copy_from_slice(b"../escape");
    header.set_cksum();
    builder.append(&header, &b"x"[..]).unwrap();
    builder.into_inner().unwrap().finish().unwrap()
}

fn spec(archive_bytes: &[u8], url: &str) -> &'static ModelSpec {
    let asset = b"tiny model";
    let required: &'static [RequiredFile] = Box::leak(
        vec![RequiredFile {
            path: "model.bin",
            size: asset.len() as u64,
            sha256: leak(digest(asset)),
        }]
        .into_boxed_slice(),
    );
    Box::leak(Box::new(ModelSpec {
        id: "tiny",
        backend: "sherpa-onnx",
        family: "zipformer-kws",
        description: "test model",
        license: "MIT",
        license_status: "verified",
        downloadable: true,
        archive_url: leak(url.to_owned()),
        archive_size: archive_bytes.len() as u64,
        archive_sha256: leak(digest(archive_bytes)),
        archive_root: "tiny-root",
        encoder: "model.bin",
        openvino_accelerator_encoder: "model.bin",
        cuda_encoder: "model.bin",
        decoder: "model.bin",
        cuda_decoder: "model.bin",
        joiner: "model.bin",
        cuda_joiner: "model.bin",
        tokens: "model.bin",
        bpe_model: "model.bin",
        required_files: required,
    }))
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
fn model_path_is_backend_neutral() {
    let paths = AppPaths {
        config_file: "/tmp/config".into(),
        data_dir: "/tmp/data".into(),
        state_dir: "/tmp/state".into(),
        runtime_dir: "/tmp/run".into(),
    };
    let spec = crate::catalog::models().first().unwrap();
    assert_eq!(
        model_directory(&paths, spec),
        PathBuf::from("/tmp/data/models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")
    );
}

#[test]
fn progress_is_rate_limited_and_always_reports_completion() {
    assert!(!should_report_progress(128 * 1024, 0, 2 * 1024 * 1024));
    assert!(should_report_progress(1024 * 1024, 0, 2 * 1024 * 1024));
    assert!(should_report_progress(
        2 * 1024 * 1024,
        1024 * 1024,
        2 * 1024 * 1024
    ));
}

#[test]
fn local_archive_install_is_verified_idempotent_and_repairable() {
    let root = temp("install");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let spec = spec(&archive_bytes, "http://unused.invalid/model");
    let paths = paths(&root);
    let installed = install(&paths, spec, Some(&archive_path), ProgressFormat::Json).unwrap();
    assert_eq!(installed, model_directory(&paths, spec));
    assert!(installed.join(".omawake-model.json").is_file());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(installed.join(".omawake-model.json")).unwrap()).unwrap();
    assert_eq!(manifest["provenance"]["source"], "user-supplied-archive");
    assert_eq!(
        manifest["provenance"]["archive_sha256"],
        spec.archive_sha256
    );
    verify(&paths, spec).unwrap();
    install(&paths, spec, Some(&archive_path), ProgressFormat::Human).unwrap();
    fs::write(installed.join("model.bin"), b"bad").unwrap();
    assert!(verify(&paths, spec).is_err());
    let staging =
        paths
            .data_dir
            .join("models")
            .join(format!(".{}.install-{}", spec.id, std::process::id()));
    let old =
        paths
            .data_dir
            .join("models")
            .join(format!(".{}.old-{}", spec.id, std::process::id()));
    fs::create_dir_all(&staging).unwrap();
    fs::create_dir_all(&old).unwrap();
    install(&paths, spec, Some(&archive_path), ProgressFormat::Human).unwrap();
    verify(&paths, spec).unwrap();
    assert!(!staging.exists());
    assert!(!old.exists());
}

#[test]
fn install_without_override_uses_injected_downloader() {
    let root = temp("install-download");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let archive_path = root.join("downloaded.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let spec = spec(&archive_bytes, "https://example.invalid/model");
    let paths = paths(&root);
    let installed = install_with_download(
        &paths,
        spec,
        None,
        ProgressFormat::Json,
        |received_paths, received_spec, received_progress| {
            assert_eq!(received_paths.data_dir, paths.data_dir);
            assert_eq!(received_spec.id, "tiny");
            assert_eq!(received_progress, ProgressFormat::Json);
            Ok(archive_path.clone())
        },
    )
    .unwrap();
    assert_eq!(
        fs::read(installed.join("model.bin")).unwrap(),
        b"tiny model"
    );
}

#[test]
fn unverified_model_cannot_be_downloaded_automatically() {
    let root = temp("unverified-license");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let mut blocked = *spec(&archive_bytes, "https://example.invalid/model");
    blocked.downloadable = false;
    blocked.license = "unknown";
    blocked.license_status = "unverified";
    let blocked = Box::leak(Box::new(blocked));
    let error = install_with_download(
        &paths(&root),
        blocked,
        None,
        ProgressFormat::Human,
        |_, _, _| panic!("an unverified model must not reach the downloader"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("license not verified"));
}

#[test]
fn failed_manifest_activation_restores_the_previous_model() {
    let root = temp("activation-rollback");
    let archive_bytes = archive_blocking_manifest("tiny-root", "model.bin", b"tiny model");
    let archive_path = root.join("tiny.tar.bz2");
    fs::write(&archive_path, &archive_bytes).unwrap();
    let spec = spec(&archive_bytes, "http://unused.invalid/model");
    let paths = paths(&root);
    let target = model_directory(&paths, spec);
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("previous"), b"keep me").unwrap();

    let error = install(&paths, spec, Some(&archive_path), ProgressFormat::Human).unwrap_err();
    assert!(error.to_string().contains("finalize installed model"));
    assert_eq!(fs::read(target.join("previous")).unwrap(), b"keep me");
    assert!(!target.join("model.bin").exists());
}

#[test]
fn archive_and_asset_corruption_are_rejected() {
    let root = temp("corruption");
    let bytes = archive("tiny-root", "model.bin", b"tiny model");
    let path = root.join("archive.tar.bz2");
    fs::write(&path, &bytes).unwrap();
    let spec = spec(&bytes, "http://unused.invalid/model");
    verify_archive(&path, spec).unwrap();
    let mut changed = bytes.clone();
    let middle = changed.len() / 2;
    changed[middle] ^= 1;
    fs::write(&path, &changed).unwrap();
    assert!(verify_archive(&path, spec).is_err());
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(verify_archive(&path, spec).is_err());
    assert!(verify_archive(&root.join("missing"), spec).is_err());
    let directory = root.join("assets");
    fs::create_dir_all(&directory).unwrap();
    assert!(verify_directory(&directory, spec).is_err());
    fs::write(directory.join("model.bin"), b"bad model!").unwrap();
    assert!(verify_directory(&directory, spec).is_err());
    fs::write(directory.join("model.bin"), b"tiny xodel").unwrap();
    assert!(verify_directory(&directory, spec).is_err());
}

#[test]
fn extraction_rejects_wrong_roots_and_non_files() {
    let root = temp("unsafe");
    let wrong = archive("other-root", "model.bin", b"tiny model");
    let wrong_path = root.join("wrong.tar.bz2");
    fs::write(&wrong_path, wrong).unwrap();
    let expected = archive("tiny-root", "model.bin", b"tiny model");
    let spec = spec(&expected, "http://unused.invalid/model");
    assert!(extract_archive(&wrong_path, &root.join("stage"), spec).is_err());
    let encoder = BzEncoder::new(Vec::new(), Compression::best());
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_mode(0o777);
    header.set_link_name("target").unwrap();
    header.set_cksum();
    builder
        .append_data(&mut header, "tiny-root/link", &[][..])
        .unwrap();
    let links = builder.into_inner().unwrap().finish().unwrap();
    let links_path = root.join("links.tar.bz2");
    fs::write(&links_path, links).unwrap();
    assert!(extract_archive(&links_path, &root.join("stage2"), spec).is_err());

    let with_directory = archive_with_directory("tiny-root", "model.bin", b"tiny model");
    let directory_path = root.join("directory.tar.bz2");
    fs::write(&directory_path, with_directory).unwrap();
    let stage3 = root.join("stage3");
    fs::create_dir_all(&stage3).unwrap();
    let extracted = extract_archive(&directory_path, &stage3, spec).unwrap();
    assert_eq!(
        fs::read(extracted.join("model.bin")).unwrap(),
        b"tiny model"
    );

    let malformed = root.join("malformed.tar.bz2");
    fs::write(&malformed, b"not bzip data").unwrap();
    assert!(extract_archive(&malformed, &root.join("stage4"), spec).is_err());

    let unsafe_path = root.join("unsafe.tar.bz2");
    fs::write(&unsafe_path, archive_with_unsafe_path()).unwrap();
    assert!(extract_archive(&unsafe_path, &root.join("stage5"), spec).is_err());
}

#[test]
fn downloaded_bytes_succeed_cache_and_reject_bad_lengths() {
    let root = temp("network");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let model_spec = spec(&archive_bytes, "http://unused.invalid/model");
    let app_paths = paths(&root);
    fs::create_dir_all(app_paths.data_dir.join("downloads")).unwrap();
    let downloaded = app_paths.data_dir.join("downloads/tiny.tar.bz2");
    let part = downloaded.with_extension("tar.bz2.part");
    write_download(
        &archive_bytes[..],
        &part,
        &downloaded,
        model_spec,
        ProgressFormat::Json,
    )
    .unwrap();
    verify_archive(&downloaded, model_spec).unwrap();
    assert_eq!(
        download_archive(&app_paths, model_spec, ProgressFormat::Human).unwrap(),
        downloaded
    );
    let short_root = temp("short");
    let short_spec = spec(&archive_bytes, "http://unused.invalid/model");
    let short_paths = paths(&short_root);
    fs::create_dir_all(short_paths.data_dir.join("downloads")).unwrap();
    let target = short_paths.data_dir.join("downloads/tiny.tar.bz2");
    assert!(
        write_download(
            &archive_bytes[..archive_bytes.len() - 1],
            &target.with_extension("tar.bz2.part"),
            &target,
            short_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );

    let corrupt_root = temp("download-checksum");
    let corrupt_paths = paths(&corrupt_root);
    fs::create_dir_all(corrupt_paths.data_dir.join("downloads")).unwrap();
    let target = corrupt_paths.data_dir.join("downloads/tiny.tar.bz2");
    let mut corrupt = archive_bytes.clone();
    corrupt[0] ^= 1;
    assert!(
        write_download(
            &corrupt[..],
            &target.with_extension("tar.bz2.part"),
            &target,
            model_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );

    let missing_parent = corrupt_root.join("missing/part");
    assert!(
        write_download(
            &archive_bytes[..],
            &missing_parent,
            &target,
            model_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );
    let overflow_root = temp("overflow");
    let expected = &archive_bytes[..archive_bytes.len() - 1];
    let overflow_spec = spec(expected, "http://unused.invalid/model");
    let overflow_paths = paths(&overflow_root);
    fs::create_dir_all(overflow_paths.data_dir.join("downloads")).unwrap();
    let target = overflow_paths.data_dir.join("downloads/tiny.tar.bz2");
    assert!(
        write_download(
            &archive_bytes[..],
            &target.with_extension("tar.bz2.part"),
            &target,
            overflow_spec,
            ProgressFormat::Human,
        )
        .is_err()
    );

    let large_root = temp("large-progress");
    let large_bytes = vec![0x5a; 1024 * 1024 + 17];
    let large_spec = spec(&large_bytes, "http://unused.invalid/large");
    let large_target = large_root.join("large.download");
    write_download(
        &large_bytes[..],
        &large_target.with_extension("part"),
        &large_target,
        large_spec,
        ProgressFormat::Json,
    )
    .unwrap();
    assert_eq!(
        fs::metadata(large_target).unwrap().len(),
        large_bytes.len() as u64
    );
}

#[test]
fn injected_download_replaces_partial_files_and_reports_human_progress() {
    let root = temp("injected-download");
    let archive_bytes = archive("tiny-root", "model.bin", b"tiny model");
    let model_spec = spec(&archive_bytes, "https://example.invalid/tiny.tar.bz2");
    let app_paths = paths(&root);
    let downloads = app_paths.data_dir.join("downloads");
    fs::create_dir_all(&downloads).unwrap();
    let target = downloads.join("tiny.tar.bz2");
    let part = target.with_extension("tar.bz2.part");
    fs::write(&part, b"stale").unwrap();

    let result = download_archive_with(&app_paths, model_spec, ProgressFormat::Human, |url| {
        assert_eq!(url, "https://example.invalid/tiny.tar.bz2");
        Ok(Box::new(std::io::Cursor::new(archive_bytes.clone())))
    })
    .unwrap();
    assert_eq!(result, target);
    assert!(!part.exists());
    verify_archive(&result, model_spec).unwrap();

    fs::remove_file(&result).unwrap();
    let error = download_archive_with(&app_paths, model_spec, ProgressFormat::Json, |_| {
        Err(anyhow::anyhow!("fetch failed"))
    })
    .unwrap_err();
    assert_eq!(error.to_string(), "fetch failed");

    emit(
        ProgressFormat::Human,
        "download-progress",
        model_spec,
        Some(model_spec.archive_size),
        Some(model_spec.archive_size),
    )
    .unwrap();

    let invalid_url_root = temp("invalid-download-url");
    let invalid_url_paths = paths(&invalid_url_root);
    let invalid_url_spec = spec(&archive_bytes, "://invalid-url");
    let error =
        download_archive(&invalid_url_paths, invalid_url_spec, ProgressFormat::Human).unwrap_err();
    assert!(error.to_string().contains("download"));
}
