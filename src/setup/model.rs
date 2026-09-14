use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use bzip2::read::BzDecoder;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::catalog::ModelSpec;
use crate::paths::AppPaths;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
pub enum ProgressFormat {
    #[default]
    Human,
    Json,
}

#[derive(Serialize)]
struct Event<'a> {
    event: &'a str,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    current: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
}

pub fn model_directory(paths: &AppPaths, spec: &ModelSpec) -> PathBuf {
    paths.data_dir.join("models").join(spec.id)
}

pub fn verify(paths: &AppPaths, spec: &ModelSpec) -> Result<()> {
    verify_directory(&model_directory(paths, spec), spec)
}

pub fn install(
    paths: &AppPaths,
    spec: &ModelSpec,
    archive_override: Option<&Path>,
    progress: ProgressFormat,
) -> Result<PathBuf> {
    let target = model_directory(paths, spec);
    if verify_directory(&target, spec).is_ok() {
        emit(progress, "already-installed", spec, None, None)?;
        return Ok(target);
    }

    fs::create_dir_all(paths.data_dir.join("models"))?;
    fs::create_dir_all(paths.data_dir.join("downloads"))?;
    let archive = match archive_override {
        Some(path) => path.to_owned(),
        None => download_archive(paths, spec, progress)?,
    };
    verify_archive(&archive, spec)?;

    let staging =
        paths
            .data_dir
            .join("models")
            .join(format!(".{}.install-{}", spec.id, std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    emit(progress, "extract", spec, None, None)?;
    let extracted = extract_archive(&archive, &staging, spec)
        .with_context(|| format!("extract model archive {}", archive.display()))?;
    verify_directory(&extracted, spec)?;

    let old =
        paths
            .data_dir
            .join("models")
            .join(format!(".{}.old-{}", spec.id, std::process::id()));
    if old.exists() {
        fs::remove_dir_all(&old)?;
    }
    if target.exists() {
        fs::rename(&target, &old)?;
    }
    if let Err(error) = fs::rename(&extracted, &target) {
        if old.exists() {
            let _ = fs::rename(&old, &target);
        }
        return Err(error).context("activate extracted model");
    }
    let activate = || -> Result<()> {
        fs::write(
            target.join(".omawake-model.json"),
            serde_json::to_vec_pretty(spec)?,
        )?;
        verify_directory(&target, spec)
    };
    if let Err(error) = activate() {
        let _ = fs::remove_dir_all(&target);
        if old.exists() {
            let _ = fs::rename(&old, &target);
        }
        let _ = fs::remove_dir_all(&staging);
        return Err(error).context("finalize installed model");
    }
    let _ = fs::remove_dir_all(&old);
    let _ = fs::remove_dir_all(&staging);
    emit(progress, "installed", spec, None, None)?;
    Ok(target)
}

fn download_archive(
    paths: &AppPaths,
    spec: &ModelSpec,
    progress: ProgressFormat,
) -> Result<PathBuf> {
    let target = paths
        .data_dir
        .join("downloads")
        .join(format!("{}.tar.bz2", spec.id));
    if verify_archive(&target, spec).is_ok() {
        emit(
            progress,
            "cached",
            spec,
            Some(spec.archive_size),
            Some(spec.archive_size),
        )?;
        return Ok(target);
    }
    let part = target.with_extension("tar.bz2.part");
    let _ = fs::remove_file(&part);
    emit(
        progress,
        "download-start",
        spec,
        Some(0),
        Some(spec.archive_size),
    )?;
    let response = ureq::get(spec.archive_url)
        .call()
        .with_context(|| format!("download {}", spec.archive_url))?;
    let mut input = response.into_reader();
    let file = File::create(&part)?;
    let mut output = BufWriter::new(file);
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut reported = 0_u64;
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > spec.archive_size {
            bail!("download exceeded expected size for {}", spec.id);
        }
        hasher.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
        if should_report_progress(total, reported, spec.archive_size) {
            emit(
                progress,
                "download-progress",
                spec,
                Some(total),
                Some(spec.archive_size),
            )?;
            reported = total;
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    if total != spec.archive_size {
        bail!(
            "downloaded {} bytes for {}, expected {}",
            total,
            spec.id,
            spec.archive_size
        );
    }
    let digest = format!("{:x}", hasher.finalize());
    if digest != spec.archive_sha256 {
        bail!("download checksum mismatch for {}", spec.id);
    }
    fs::rename(&part, &target)?;
    emit(progress, "downloaded", spec, Some(total), Some(total))?;
    Ok(target)
}

fn should_report_progress(current: u64, last: u64, total: u64) -> bool {
    current.saturating_sub(last) >= 1024 * 1024 || current == total
}

fn verify_archive(path: &Path, spec: &ModelSpec) -> Result<()> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("model archive is missing: {}", path.display()))?;
    if metadata.len() != spec.archive_size {
        bail!("archive size mismatch for {}", path.display());
    }
    let digest = sha256_file(path)?;
    if digest != spec.archive_sha256 {
        bail!("archive checksum mismatch for {}", path.display());
    }
    Ok(())
}

fn extract_archive(archive: &Path, staging: &Path, spec: &ModelSpec) -> Result<PathBuf> {
    let decoder = BzDecoder::new(BufReader::new(File::open(archive)?));
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        {
            bail!("archive contains unsafe path {}", path.display());
        }
        if path.components().next().and_then(|part| match part {
            Component::Normal(value) => value.to_str(),
            _ => None,
        }) != Some(spec.archive_root)
        {
            bail!("archive entry is outside expected root: {}", path.display());
        }
        let kind = entry.header().entry_type();
        if !kind.is_file() && !kind.is_dir() {
            bail!("archive contains unsupported entry {}", path.display());
        }
        if !entry.unpack_in(staging)? {
            bail!("archive entry escaped destination: {}", path.display());
        }
    }
    Ok(staging.join(spec.archive_root))
}

fn verify_directory(directory: &Path, spec: &ModelSpec) -> Result<()> {
    for required in spec.required_files {
        let path = directory.join(required.path);
        let metadata = fs::metadata(&path)
            .with_context(|| format!("required model asset is missing: {}", path.display()))?;
        if metadata.len() != required.size {
            bail!("model asset has wrong size: {}", path.display());
        }
        if sha256_file(&path)? != required.sha256 {
            bail!("model asset checksum mismatch: {}", path.display());
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn emit(
    format: ProgressFormat,
    event: &'static str,
    spec: &ModelSpec,
    current: Option<u64>,
    total: Option<u64>,
) -> Result<()> {
    match format {
        ProgressFormat::Human => match (current, total) {
            (Some(current), Some(total)) if event == "download-progress" => {
                eprint!(
                    "\rDownloading {}: {:>3}%",
                    spec.id,
                    current.saturating_mul(100) / total.max(1)
                );
                std::io::stderr().flush()?;
            }
            _ => {
                if event == "downloaded" {
                    eprintln!();
                }
                eprintln!("{}: {}", event, spec.id);
            }
        },
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&Event {
                event,
                model: spec.id,
                current,
                total
            })?
        ),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
