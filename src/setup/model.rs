use super::install_guard::{self, InstallGuard};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::catalog::{ModelAsset, ModelSpec};
use crate::paths::AppPaths;

const MIT_TERMS: &str = "Permission is hereby granted, free of charge, to any person obtaining a copy\
 of this software and associated documentation files (the \"Software\"), to deal\
 in the Software without restriction, including without limitation the rights\
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\
 copies of the Software, and to permit persons to whom the Software is\
 furnished to do so, subject to the following conditions:\n\n\
 The above copyright notice and this permission notice shall be included in all\
 copies or substantial portions of the Software.\n\n\
 THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\
 SOFTWARE.\n";
const APACHE_2_0_TERMS: &str = include_str!("../../licenses/APACHE-2.0.txt");

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
    asset: Option<&'a str>,
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

/// Atomically install every asset in a catalog profile.
///
/// `source_directory` is an optional local directory containing the catalog's
/// exact filenames. It is useful for offline setup; every file is still checked
/// against its pinned size and SHA-256 before it can replace an installed model.
pub fn install(
    paths: &AppPaths,
    spec: &ModelSpec,
    source_directory: Option<&Path>,
    progress: ProgressFormat,
) -> Result<PathBuf> {
    install_with_fetch(paths, spec, source_directory, progress, |asset| {
        Ok(Box::new(
            ureq::AgentBuilder::new()
                .timeout_connect(std::time::Duration::from_secs(10))
                .timeout_read(std::time::Duration::from_secs(5))
                .build()
                .get(asset.url)
                .call()
                .with_context(|| format!("download {}", asset.url))?
                .into_reader(),
        ))
    })
}

fn install_with_fetch(
    paths: &AppPaths,
    spec: &ModelSpec,
    source_directory: Option<&Path>,
    progress: ProgressFormat,
    mut fetch: impl FnMut(&ModelAsset) -> Result<Box<dyn Read>>,
) -> Result<PathBuf> {
    let mut guard = InstallGuard::acquire(&paths.data_dir, spec.id)?;
    let target = model_directory(paths, spec);
    if verify_directory(&target, spec).is_ok() {
        emit(progress, "already-installed", spec, None, None, None)?;
        return Ok(target);
    }
    if !spec.downloadable && source_directory.is_none() {
        bail!(
            "license not verified for {}; supply an exact local model directory with --source-dir",
            spec.id
        );
    }
    if let Some(directory) = source_directory
        && (!directory.is_absolute() || !directory.is_dir())
    {
        bail!(
            "model source directory must be an absolute existing directory: {}",
            directory.display()
        );
    }
    for asset in spec.assets {
        validate_relative_file(asset.path)?;
    }
    for notice in spec.notices {
        validate_relative_file(notice.path)?;
        if !matches!(notice.license, "MIT" | "Apache-2.0") {
            bail!("unsupported catalog license notice {}", notice.license);
        }
    }

    let models = paths.data_dir.join("models");
    let downloads = paths.data_dir.join("downloads").join(spec.id);
    fs::create_dir_all(&models)?;
    fs::create_dir_all(&downloads)?;
    let missing = if source_directory.is_some() {
        0
    } else {
        spec.assets
            .iter()
            .filter(|asset| {
                verify_file(&downloads.join(asset.path), asset.size, asset.sha256).is_err()
            })
            .map(|asset| asset.size)
            .sum()
    };
    install_guard::preflight(&models, &downloads, spec.total_size(), missing)?;
    let staging = models.join(format!(".{}.install-{}", spec.id, std::process::id()));
    let old = models.join(format!(".{}.old-{}", spec.id, std::process::id()));
    remove_directory_if_present(&staging)?;
    remove_directory_if_present(&old)?;
    fs::create_dir_all(&staging)?;
    guard.staging(&staging);

    let prepare = (|| -> Result<()> {
        for asset in spec.assets {
            let source = match source_directory {
                Some(directory) => directory.join(asset.path),
                None => download_asset(&downloads, spec, asset, progress, &mut fetch)?,
            };
            verify_file(&source, asset.size, asset.sha256)
                .with_context(|| format!("verify {} asset {}", asset.role, source.display()))?;
            let destination = staging.join(asset.path);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            copy_synced(&source, &destination)?;
        }
        write_manifests(&staging, spec, source_directory)?;
        verify_directory(&staging, spec)
    })();
    if let Err(error) = prepare {
        let _ = fs::remove_dir_all(&staging);
        return Err(error).context("prepare model installation");
    }

    install_guard::check_cancelled()?;
    if target.exists() {
        fs::rename(&target, &old).context("retain previous model during activation")?;
    }
    if let Err(error) = fs::rename(&staging, &target) {
        if old.exists() {
            let _ = fs::rename(&old, &target);
        }
        return Err(error).context("activate installed model");
    }
    if let Err(error) = verify_directory(&target, spec) {
        let _ = fs::remove_dir_all(&target);
        if old.exists() {
            let _ = fs::rename(&old, &target);
        }
        return Err(error).context("verify activated model");
    }
    remove_directory_if_present(&old)?;
    emit(progress, "installed", spec, None, None, None)?;
    Ok(target)
}

fn download_asset(
    downloads: &Path,
    spec: &ModelSpec,
    asset: &ModelAsset,
    progress: ProgressFormat,
    fetch: &mut impl FnMut(&ModelAsset) -> Result<Box<dyn Read>>,
) -> Result<PathBuf> {
    let target = downloads.join(asset.path);
    if verify_file(&target, asset.size, asset.sha256).is_ok() {
        emit(
            progress,
            "cached",
            spec,
            Some(asset.path),
            Some(asset.size),
            Some(asset.size),
        )?;
        return Ok(target);
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = target.with_extension(format!(
        "{}.part",
        target
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("download")
    ));
    let _ = fs::remove_file(&part);
    emit(
        progress,
        "download-start",
        spec,
        Some(asset.path),
        Some(0),
        Some(asset.size),
    )?;
    let result = write_download(fetch(asset)?, &part, &target, spec, asset, progress);
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result.map(|()| target)
}

fn write_download(
    mut input: impl Read,
    part: &Path,
    target: &Path,
    spec: &ModelSpec,
    asset: &ModelAsset,
    progress: ProgressFormat,
) -> Result<()> {
    let mut output = BufWriter::new(File::create(part)?);
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut reported = 0_u64;
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        install_guard::check_cancelled()?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total.saturating_add(count as u64);
        if total > asset.size {
            bail!("download exceeded expected size for {}", asset.path);
        }
        hasher.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
        if should_report_progress(total, reported, asset.size) {
            emit(
                progress,
                "download-progress",
                spec,
                Some(asset.path),
                Some(total),
                Some(asset.size),
            )?;
            reported = total;
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    if total != asset.size {
        bail!(
            "downloaded {total} bytes for {}, expected {}",
            asset.path,
            asset.size
        );
    }
    let digest = format!("{:x}", hasher.finalize());
    if digest != asset.sha256 {
        bail!("download checksum mismatch for {}", asset.path);
    }
    fs::rename(part, target)?;
    emit(
        progress,
        "downloaded",
        spec,
        Some(asset.path),
        Some(total),
        Some(total),
    )
}

fn should_report_progress(current: u64, last: u64, total: u64) -> bool {
    current.saturating_sub(last) >= 1024 * 1024 || current == total
}

fn write_manifests(
    directory: &Path,
    spec: &ModelSpec,
    source_directory: Option<&Path>,
) -> Result<()> {
    for notice in spec.notices {
        let path = directory.join(notice.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, notice_text(notice))?;
    }
    let provenance = serde_json::json!({
        "schema_version": 1,
        "profile": spec.id,
        "installed_from": if source_directory.is_some() { "local-directory" } else { "catalog-download" },
        "original_model": {
            "url": spec.source_url,
            "revision": spec.source_revision,
            "license": spec.license,
        },
        "converted_artifacts": {
            "url": spec.converted_source_url,
            "revision": spec.converted_source_revision,
        },
        "assets": spec.assets,
        "license_notices": spec.notices,
    });
    fs::write(
        directory.join("PROVENANCE.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    fs::write(
        directory.join(".omawake-model.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "catalog": spec,
        }))?,
    )?;
    Ok(())
}

fn verify_directory(directory: &Path, spec: &ModelSpec) -> Result<()> {
    for asset in spec.assets {
        verify_file(&directory.join(asset.path), asset.size, asset.sha256)
            .with_context(|| format!("verify installed {}", asset.role))?;
    }
    let provenance_path = directory.join("PROVENANCE.json");
    let provenance: serde_json::Value = serde_json::from_slice(
        &fs::read(&provenance_path)
            .with_context(|| format!("read model provenance {}", provenance_path.display()))?,
    )
    .with_context(|| format!("parse model provenance {}", provenance_path.display()))?;
    if provenance["schema_version"] != 1
        || provenance["profile"] != spec.id
        || provenance["original_model"]["revision"] != spec.source_revision
        || provenance["converted_artifacts"]["revision"] != spec.converted_source_revision
        || provenance["assets"] != serde_json::to_value(spec.assets)?
    {
        bail!(
            "model provenance does not match catalog profile {}",
            spec.id
        );
    }
    let catalog_path = directory.join(".omawake-model.json");
    let catalog: serde_json::Value = serde_json::from_slice(
        &fs::read(&catalog_path)
            .with_context(|| format!("read model manifest {}", catalog_path.display()))?,
    )
    .with_context(|| format!("parse model manifest {}", catalog_path.display()))?;
    if catalog["schema_version"] != 1 || catalog["catalog"] != serde_json::to_value(spec)? {
        bail!("model manifest does not match catalog profile {}", spec.id);
    }
    for notice in spec.notices {
        let path = directory.join(notice.path);
        let actual = fs::read_to_string(&path)
            .with_context(|| format!("read model license notice {}", path.display()))?;
        if actual != notice_text(notice) {
            bail!(
                "model license notice does not match catalog: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn notice_text(notice: &crate::catalog::LicenseNotice) -> String {
    match notice.license {
        "MIT" => format!(
            "MIT License\n\n{}\n\n{}\n\nSource: {}\n",
            notice.copyright, MIT_TERMS, notice.source_url
        ),
        "Apache-2.0" => format!(
            "{}\n\n{}\nSource: {}\n",
            notice.copyright,
            APACHE_2_0_TERMS.trim_end(),
            notice.source_url
        ),
        other => format!(
            "Unsupported license {other}\nSource: {}\n",
            notice.source_url
        ),
    }
}

fn verify_file(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("file is missing: {}", path.display()))?;
    if !metadata.is_file() || metadata.len() != expected_size {
        bail!("file size mismatch for {}", path.display());
    }
    let digest = sha256_file(path)?;
    if digest != expected_sha256 {
        bail!("file checksum mismatch for {}", path.display());
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        install_guard::check_cancelled()?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_relative_file(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("catalog asset must be a safe relative file path: {value:?}");
    }
    Ok(())
}

fn copy_synced(source: &Path, destination: &Path) -> Result<()> {
    install_guard::copy(source, destination)
}

fn remove_directory_if_present(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("remove stale directory {}", path.display()))
        }
    }
}

fn emit(
    format: ProgressFormat,
    event: &'static str,
    spec: &ModelSpec,
    asset: Option<&str>,
    current: Option<u64>,
    total: Option<u64>,
) -> Result<()> {
    match format {
        ProgressFormat::Human => match (asset, current, total) {
            (Some(asset), Some(current), Some(total)) => {
                eprintln!("{event}: {} {asset} {current}/{total}", spec.id)
            }
            (Some(asset), _, _) => eprintln!("{event}: {} {asset}", spec.id),
            _ => eprintln!("{event}: {}", spec.id),
        },
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&Event {
                event,
                model: spec.id,
                asset,
                current,
                total,
            })?
        ),
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_model.rs"]
mod tests;
