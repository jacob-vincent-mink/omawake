//! Immutable, private artifacts. A config change selects a new artifact only
//! after training and validation succeed; previous versions remain reusable.
use super::head::Head;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

const MAX_ARTIFACT_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnrollmentBinding {
    /// Keep artifacts when switching back to transcript recognition.
    #[serde(default = "active_by_default")]
    pub active: bool,
    /// Model/preprocessing contract to an immutable, content-addressed head.
    /// Changing the runtime/device does not discard earlier encoder heads.
    pub heads: BTreeMap<String, PathBuf>,
    /// Explicit operating threshold; None uses the head's calibrated value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(
        default,
        skip_serializing_if = "super::history::HistoryConfig::is_default"
    )]
    pub history: super::history::HistoryConfig,
}

impl EnrollmentBinding {
    pub fn validate(&self) -> Result<()> {
        if let Some(threshold) = self.threshold {
            ensure!(
                threshold.is_finite() && threshold > 0.0 && threshold <= 1.0,
                "trained threshold must be greater than 0 and at most 1"
            );
        }
        self.history.validate()
    }
}

fn active_by_default() -> bool {
    true
}
impl Default for EnrollmentBinding {
    fn default() -> Self {
        Self {
            active: true,
            heads: BTreeMap::new(),
            threshold: None,
            history: Default::default(),
        }
    }
}

pub fn load(path: &Path) -> Result<Head> {
    let input = fs::File::open(path).with_context(|| format!("open head {}", path.display()))?;
    ensure!(
        input.metadata()?.len() <= MAX_ARTIFACT_BYTES,
        "head artifact exceeds 2 MiB"
    );
    let mut bytes = Vec::new();
    input.take(MAX_ARTIFACT_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_ARTIFACT_BYTES,
        "head artifact exceeds 2 MiB"
    );
    let head: Head = serde_json::from_slice(&bytes).context("parse head artifact")?;
    head.validate()?;
    ensure!(
        head.locally_validated(),
        "head has not passed held-out local validation"
    );
    Ok(head)
}

pub fn install(directory: &Path, head: &Head) -> Result<PathBuf> {
    head.validate()?;
    ensure!(
        head.locally_validated(),
        "refusing to install an unvalidated head"
    );
    fs::create_dir_all(directory)?;
    fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    let bytes = serde_json::to_vec_pretty(head)?;
    ensure!(
        bytes.len() as u64 <= MAX_ARTIFACT_BYTES,
        "head artifact exceeds 2 MiB"
    );
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let destination = directory.join(format!("{digest}.json"));
    // create_new never overwrites an earlier artifact. Existing content must
    // match exactly; a interrupted write is an error, never silently accepted.
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&destination)
    {
        Ok(mut file) => {
            let result = file.write_all(&bytes).and_then(|()| file.sync_all());
            if let Err(error) = result {
                let _ = fs::remove_file(&destination);
                return Err(error.into());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(
                !destination.symlink_metadata()?.file_type().is_symlink(),
                "head artifact is a symlink"
            );
            let existing = load(&destination)?;
            ensure!(
                serde_json::to_vec_pretty(&existing)? == bytes,
                "existing head artifact does not match its digest"
            );
        }
        Err(error) => return Err(error.into()),
    }
    Ok(destination)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub training: Vec<LabeledRecording>,
    pub calibration: Vec<LabeledRecording>,
    pub validation: Vec<LabeledRecording>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LabeledRecording {
    pub audio: PathBuf,
    pub positive: bool,
    /// Synthetic provenance is retained; these clips are never validation evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated: Option<GeneratedRecording>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedRecording {
    pub generator: String,
    pub voice: String,
    pub text: String,
    pub speed: f32,
}
impl Dataset {
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() <= 256 * 1024,
            "dataset manifest exceeds 256 KiB"
        );
        let bytes = fs::read(path)?;
        ensure!(
            bytes.len() <= 256 * 1024,
            "dataset manifest exceeds 256 KiB"
        );
        let mut dataset: Self =
            serde_json::from_slice(&bytes).context("parse training dataset manifest")?;
        ensure!(
            dataset
                .calibration
                .iter()
                .chain(&dataset.validation)
                .all(|r| r.generated.is_none()),
            "synthetic recordings may only appear in training, never calibration or validation"
        );
        let base = path.parent().unwrap_or(Path::new("."));
        for split in [
            &mut dataset.training,
            &mut dataset.calibration,
            &mut dataset.validation,
        ] {
            ensure!(
                (4..=128).contains(&split.len()),
                "each split needs 4..128 recordings"
            );
            let positives = split.iter().filter(|item| item.positive).count();
            ensure!(
                positives >= 2 && split.len() - positives >= 2,
                "each split needs at least two positive and two negative recordings"
            );
            for item in split {
                if item.audio.is_relative() {
                    item.audio = base.join(&item.audio);
                }
            }
        }
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::super::head::Example;
    use super::*;
    fn split(prefix: &str) -> Vec<Example> {
        (0..4)
            .map(|i| Example {
                id: format!("{prefix}-{i}"),
                values: vec![if i < 2 { 1.0 } else { -1.0 }, i as f32 * 0.01],
                positive: i < 2,
            })
            .collect()
    }
    #[test]
    fn immutable_private_heads_survive_new_versions_and_reject_corruption() {
        let root = std::env::temp_dir().join(format!("omawake-head-store-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut h = Head::train("encoder-one", &split("train"), &split("cal")).unwrap();
        assert!(install(&root, &h).is_err());
        h.validate_held_out(&split("held")).unwrap();
        let first = install(&root, &h).unwrap();
        assert_eq!(first, install(&root, &h).unwrap());
        assert_eq!(
            fs::metadata(&first).unwrap().permissions().mode() & 0o777,
            0o600
        );
        h.encoder_contract = "encoder-two".into();
        let second = install(&root, &h).unwrap();
        assert_ne!(first, second);
        assert_eq!(load(&first).unwrap().encoder_contract, "encoder-one");
        assert_eq!(load(&second).unwrap().encoder_contract, "encoder-two");
        fs::write(&second, b"{}").unwrap();
        assert!(load(&second).is_err());
        assert!(install(&root, &h).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn malformed_counts_do_not_overflow() {
        let mut h = Head::train("encoder", &split("train"), &split("cal")).unwrap();
        h.training_positives = usize::MAX;
        assert!(h.validate().is_err());
        h.training_positives = 2;
        h.validate_held_out(&split("held")).unwrap();
        h.validation.as_mut().unwrap().positives = usize::MAX;
        assert!(h.validate().is_err());
    }
}
