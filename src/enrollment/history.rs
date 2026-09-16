//! Opt-in, bounded local detection evidence. Labels never change a running head.
use crate::paths::AppPaths;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HistoryConfig {
    pub enabled: bool,
    pub max_events: usize,
}
impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_events: 100,
        }
    }
}
impl HistoryConfig {
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=1000).contains(&self.max_events),
            "history max_events must be 1..1000"
        );
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Label {
    Unreviewed,
    FalsePositive,
    TruePositive,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub id: String,
    pub word_id: String,
    pub created_ms: u64,
    pub score: f32,
    pub threshold: f32,
    pub encoder_contract: String,
    pub head: String,
    pub device: String,
    pub label: Label,
    #[serde(default)]
    pub audio: PathBuf,
}
fn component(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "invalid history identifier"
    );
    Ok(())
}
fn directory(paths: &AppPaths, word: &str) -> Result<PathBuf> {
    component(word)?;
    let root = paths.data_dir.join("history");
    for p in [&root, &root.join(word)] {
        if p.exists() || p.is_symlink() {
            ensure!(
                p.symlink_metadata()?.file_type().is_dir(),
                "history directory must not be a symlink"
            );
        }
    }
    Ok(root.join(word))
}
fn private_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}
fn read_event(path: &Path, word: &str, id: &str) -> Result<Event> {
    let meta = path.join("event.json");
    ensure!(
        meta.symlink_metadata()?.file_type().is_file() && fs::metadata(&meta)?.len() <= 65536,
        "invalid history metadata"
    );
    let mut event: Event = serde_json::from_slice(&fs::read(&meta)?)?;
    ensure!(
        event.id == id
            && event.word_id == word
            && event.score.is_finite()
            && event.threshold.is_finite(),
        "invalid history event identity or score"
    );
    event.audio = path.join("audio.wav");
    ensure!(
        event.audio.symlink_metadata()?.file_type().is_file(),
        "history audio must be a regular file"
    );
    Ok(event)
}
pub fn list(paths: &AppPaths, word: &str) -> Result<Vec<Event>> {
    let dir = directory(paths, word)?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut events = Vec::new();
    for entry in fs::read_dir(dir)?.take(1002) {
        let entry = entry?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if id.starts_with('.') {
            continue;
        }
        component(&id)?;
        ensure!(entry.file_type()?.is_dir(), "unexpected history entry");
        events.push(read_event(&entry.path(), word, &id)?);
        ensure!(events.len() <= 1001, "history exceeds its bounded limit");
    }
    events.sort_by(|a, b| (a.created_ms, &a.id).cmp(&(b.created_ms, &b.id)));
    Ok(events)
}
fn write_json(path: &Path, event: &Event) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(event)?)?;
    file.sync_all()?;
    Ok(())
}
pub fn record(
    paths: &AppPaths,
    options: &HistoryConfig,
    mut event: Event,
    samples: &[i16],
) -> Result<Option<String>> {
    if !options.enabled {
        return Ok(None);
    }
    options.validate()?;
    ensure!(
        !samples.is_empty() && samples.len() <= 480_000,
        "history clip must contain at most 30 seconds"
    );
    let parent = directory(paths, &event.word_id)?;
    private_directory(&paths.data_dir.join("history"))?;
    private_directory(&parent)?;
    let existing = list(paths, &event.word_id)?;
    // Evict before adding so disk usage stays bounded, including labeled clips.
    for old in existing
        .iter()
        .take(existing.len().saturating_sub(options.max_events - 1))
    {
        fs::remove_dir_all(parent.join(&old.id))?;
    }
    event.created_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?;
    event.id = format!(
        "{}-{}-{}",
        event.created_ms,
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let staging = parent.join(format!(".{}", event.id));
    fs::DirBuilder::new().mode(0o700).create(&staging)?;
    let result = (|| -> Result<()> {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(staging.join("audio.wav"))?;
        let mut wav = hound::WavWriter::new(
            file,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for sample in samples {
            wav.write_sample(*sample)?;
        }
        wav.finalize()?;
        event.audio = parent.join(&event.id).join("audio.wav");
        write_json(&staging.join("event.json"), &event)?;
        fs::rename(&staging, parent.join(&event.id))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result?;
    Ok(Some(event.id))
}
pub fn label(paths: &AppPaths, word: &str, id: &str, label: Label) -> Result<()> {
    component(id)?;
    let dir = directory(paths, word)?.join(id);
    ensure!(
        dir.symlink_metadata()?.file_type().is_dir(),
        "history event must not be a symlink"
    );
    let mut event = read_event(&dir, word, id)?;
    event.label = label;
    let temp = dir.join(format!(
        ".label-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = write_json(&temp, &event)
        .and_then(|()| fs::rename(&temp, dir.join("event.json")).context("save history label"));
    if temp.exists() {
        let _ = fs::remove_file(temp);
    }
    result
}
pub fn clear(paths: &AppPaths, word: &str) -> Result<()> {
    let dir = directory(paths, word)?;
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    Ok(())
}

fn fingerprint(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let (rate, samples) = crate::engine::audio::read_wave(path)?;
    let mut resampler = crate::engine::audio::AudioResampler::new();
    let mut audio = resampler.accept(rate, &samples)?;
    audio.extend(resampler.finish()?);
    let mut hash = Sha256::new();
    for sample in audio {
        hash.update(sample.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}
/// Only explicit labels become training examples. Evaluation clips are never reused.
pub fn merge_feedback(
    paths: &AppPaths,
    word: &str,
    dataset: &mut super::artifact::Dataset,
) -> Result<usize> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut candidate = dataset.clone();
    let held: BTreeSet<_> = candidate
        .calibration
        .iter()
        .chain(&candidate.validation)
        .map(|r| fingerprint(&r.audio))
        .collect::<Result<_>>()?;
    let mut known: BTreeMap<_, _> = candidate
        .training
        .iter()
        .enumerate()
        .map(|(i, r)| Ok((fingerprint(&r.audio)?, i)))
        .collect::<Result<_>>()?;
    let mut count = 0;
    for event in list(paths, word)?
        .into_iter()
        .filter(|e| e.label != Label::Unreviewed)
    {
        let hash = fingerprint(&event.audio)?;
        ensure!(
            !held.contains(&hash),
            "labeled history overlaps evaluation audio; use guided onboarding for fresh evaluation clips"
        );
        let positive = event.label == Label::TruePositive;
        if let Some(&index) = known.get(&hash) {
            if candidate.training[index].positive != positive {
                candidate.training[index].positive = positive;
                count += 1;
            }
        } else {
            ensure!(
                candidate.training.len() < 128,
                "feedback exceeds 128 training clips; existing dataset unchanged"
            );
            known.insert(hash, candidate.training.len());
            candidate.training.push(super::artifact::LabeledRecording {
                audio: event.audio,
                positive,
                generated: None,
            });
            count += 1;
        }
    }
    *dataset = candidate;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enrollment::{
        SampleSet,
        artifact::{Dataset, LabeledRecording},
    };
    use crate::test_support::{isolated_paths, unique_directory};
    fn event(word: &str) -> Event {
        Event {
            id: String::new(),
            word_id: word.into(),
            created_ms: 0,
            score: 0.8,
            threshold: 0.6,
            encoder_contract: "test".into(),
            head: "head.json".into(),
            device: "CPU".into(),
            label: Label::Unreviewed,
            audio: PathBuf::new(),
        }
    }
    #[test]
    fn opt_in_history_is_private_bounded_labelable_and_clearable() {
        let root = unique_directory("history", "privacy");
        let paths = isolated_paths(&root);
        assert!(list(&paths, "computer").unwrap().is_empty());
        assert!(
            record(&paths, &HistoryConfig::default(), event("computer"), &[])
                .unwrap()
                .is_none()
        );
        assert!(!paths.data_dir.join("history").exists());
        let options = HistoryConfig {
            enabled: true,
            max_events: 2,
        };
        let first = record(&paths, &options, event("computer"), &[2000; 1600])
            .unwrap()
            .unwrap();
        label(&paths, "computer", &first, Label::FalsePositive).unwrap();
        let rows = list(&paths, "computer").unwrap();
        assert_eq!(rows[0].label, Label::FalsePositive);
        assert_eq!(
            fs::metadata(&rows[0].audio).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(rows[0].audio.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        let (rate, audio) = crate::engine::audio::read_wave(&rows[0].audio).unwrap();
        assert_eq!(rate, 16000);
        assert_eq!(audio.len(), 1600);
        record(&paths, &options, event("computer"), &[3000; 1600]).unwrap();
        record(&paths, &options, event("computer"), &[4000; 1600]).unwrap();
        let rows = list(&paths, "computer").unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.id != first));
        assert!(label(&paths, "computer", "../outside", Label::TruePositive).is_err());
        assert!(list(&paths, "../outside").is_err());
        assert!(
            record(
                &paths,
                &HistoryConfig {
                    enabled: true,
                    max_events: 0
                },
                event("computer"),
                &[1]
            )
            .is_err()
        );
        assert!(record(&paths, &options, event("computer"), &[]).is_err());
        let external = root.join("external");
        fs::create_dir(&external).unwrap();
        clear(&paths, "computer").unwrap();
        clear(&paths, "computer").unwrap();
        std::os::unix::fs::symlink(&external, paths.data_dir.join("history/computer")).unwrap();
        assert!(clear(&paths, "computer").is_err());
        assert!(list(&paths, "computer").is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn feedback_is_training_only_deduplicated_and_rejects_evaluation_overlap() {
        let root = unique_directory("history", "feedback");
        let paths = isolated_paths(&root);
        let mut clips = SampleSet::create(&paths).unwrap();
        for i in 0..12 {
            clips.push(&vec![0.1 + i as f32 * 0.01; 1600]).unwrap();
        }
        let rows: Vec<_> = clips
            .files
            .iter()
            .enumerate()
            .map(|(i, p)| LabeledRecording {
                audio: p.clone(),
                positive: i % 4 < 2,
                generated: None,
            })
            .collect();
        let mut dataset = Dataset {
            training: rows[..4].to_vec(),
            calibration: rows[4..8].to_vec(),
            validation: rows[8..].to_vec(),
        };
        let held = serde_json::to_vec(&dataset.validation).unwrap();
        let options = HistoryConfig {
            enabled: true,
            max_events: 10,
        };
        let id = record(&paths, &options, event("computer"), &[-1234; 1600])
            .unwrap()
            .unwrap();
        assert_eq!(merge_feedback(&paths, "computer", &mut dataset).unwrap(), 0);
        label(&paths, "computer", &id, Label::FalsePositive).unwrap();
        assert_eq!(merge_feedback(&paths, "computer", &mut dataset).unwrap(), 1);
        assert!(!dataset.training.last().unwrap().positive);
        assert_eq!(merge_feedback(&paths, "computer", &mut dataset).unwrap(), 0);
        label(&paths, "computer", &id, Label::TruePositive).unwrap();
        assert_eq!(merge_feedback(&paths, "computer", &mut dataset).unwrap(), 1);
        assert!(dataset.training.last().unwrap().positive);
        assert_eq!(serde_json::to_vec(&dataset.validation).unwrap(), held);
        let recorded = list(&paths, "computer").unwrap();
        dataset.validation[0].audio = recorded[0].audio.clone();
        let before = serde_json::to_vec(&dataset).unwrap();
        assert!(
            merge_feedback(&paths, "computer", &mut dataset)
                .unwrap_err()
                .to_string()
                .contains("overlaps evaluation")
        );
        assert_eq!(serde_json::to_vec(&dataset).unwrap(), before);
        fs::remove_dir_all(root).unwrap();
    }
}
