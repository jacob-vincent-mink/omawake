//! Shared, local-only onboarding samples and transcript alias review.
pub mod artifact;
pub mod head;
pub mod history;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::audio::{AudioEvent, Capture};
use crate::config::{Config, WakeWord};
use crate::engine::Detector;
use crate::engine::audio::{AudioResampler, read_wave};
use crate::paths::AppPaths;
use crate::phrase::{capture_transcripts, normalize_tokens};

const MAX_SAMPLES: usize = 384;
const MAX_SECONDS: usize = 30;
static NEXT_SESSION: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub sample: usize,
    pub transcripts: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AliasProposal {
    pub text: String,
    pub occurrences: usize,
    pub samples: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AliasReview {
    pub wake_word: String,
    pub phrase: String,
    pub observations: Vec<Observation>,
    pub proposals: Vec<AliasProposal>,
}

/// Every session owns exactly one fresh private directory. Cancellation and
/// errors remove only that directory, never a glob of other sessions.
pub struct SampleSet {
    directory: PathBuf,
    pub files: Vec<PathBuf>,
}

impl SampleSet {
    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn create(paths: &AppPaths) -> Result<Self> {
        let parent = paths.cache_dir.join("onboarding");
        fs::create_dir_all(&parent)?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = parent.join(format!(
            "{}-{nonce}-{}",
            std::process::id(),
            NEXT_SESSION.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&directory)?;
        Ok(Self {
            directory,
            files: Vec::new(),
        })
    }

    pub fn import(&mut self, path: &Path) -> Result<()> {
        ensure!(
            fs::metadata(path)?.len() <= 32 * 1024 * 1024,
            "enrollment WAV exceeds 32 MiB"
        );
        let (rate, samples) = read_wave(path)?;
        ensure!(
            rate > 0 && samples.len() <= MAX_SECONDS * rate as usize,
            "enrollment clips must be at most {MAX_SECONDS} seconds"
        );
        let mut resampler = AudioResampler::new();
        let mut converted = resampler.accept(rate, &samples)?;
        converted.extend(resampler.finish()?);
        self.push(&converted)
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        ensure!(
            self.files.len() < MAX_SAMPLES,
            "too many enrollment clips (maximum {MAX_SAMPLES})"
        );
        validate_audio(samples)?;
        let file = self
            .directory
            .join(format!("sample-{:03}.wav", self.files.len() + 1));
        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&file)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut wav = hound::WavWriter::new(output, spec)?;
        for sample in samples {
            wav.write_sample(*sample)?;
        }
        wav.finalize()?;
        self.files.push(file);
        Ok(())
    }

    pub fn observe(&self, detector: &Detector) -> Result<Vec<Observation>> {
        self.files
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let (_, transcripts) = capture_transcripts(|| detector.detect_file(path))?;
                Ok(Observation {
                    sample: index + 1,
                    transcripts,
                })
            })
            .collect()
    }

    /// Retention is called only after explicit user choice. Caller owns the
    /// returned directory until config activation succeeds and removes it on
    /// rollback. Old recordings and heads are never overwritten.
    pub fn retain(&self, paths: &AppPaths, id: &str, manifest: &impl Serialize) -> Result<PathBuf> {
        ensure!(
            !id.is_empty()
                && id
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-'),
            "invalid enrollment ID"
        );
        let parent = paths.data_dir.join("enrollments").join(id);
        fs::create_dir_all(&parent)?;
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o700))?;
        let destination = parent.join(self.directory.file_name().context("missing session name")?);
        fs::DirBuilder::new().mode(0o700).create(&destination)?;
        let result = (|| -> Result<()> {
            for (index, source) in self.files.iter().enumerate() {
                let mut output = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(destination.join(format!("sample-{:03}.wav", index + 1)))?;
                std::io::copy(&mut fs::File::open(source)?, &mut output)?;
                output.sync_all()?;
            }
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(destination.join("manifest.json"))?;
            output.write_all(&serde_json::to_vec_pretty(manifest)?)?;
            output.sync_all()?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_dir_all(&destination);
            return Err(error);
        }
        Ok(destination)
    }
}

impl Drop for SampleSet {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

pub fn validate_audio(samples: &[f32]) -> Result<()> {
    ensure!(
        (1600..=MAX_SECONDS * 16000).contains(&samples.len()),
        "record between 0.1 and {MAX_SECONDS} seconds per clip"
    );
    ensure!(
        samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
        "recording contains invalid audio samples"
    );
    let energy = samples.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / samples.len() as f64;
    ensure!(
        energy.sqrt() >= 0.0001,
        "recording is silent or too quiet; record it again"
    );
    let clipped = samples.iter().filter(|v| v.abs() >= 0.999).count();
    ensure!(
        clipped * 100 < samples.len(),
        "recording is clipped; lower microphone gain and try again"
    );
    Ok(())
}

pub fn review(word: &WakeWord, observations: Vec<Observation>) -> AliasReview {
    let known: BTreeSet<_> = std::iter::once(&word.phrase)
        .chain(&word.aliases)
        .map(|s| normalize_tokens(s).concat())
        .collect();
    let mut proposals: BTreeMap<String, AliasProposal> = BTreeMap::new();
    for observation in &observations {
        for transcript in &observation.transcripts {
            let normalized = normalize_tokens(transcript).concat();
            if normalized.is_empty() || known.contains(&normalized) {
                continue;
            }
            let proposal = proposals
                .entry(normalized)
                .or_insert_with(|| AliasProposal {
                    text: transcript.trim().into(),
                    occurrences: 0,
                    samples: Vec::new(),
                });
            proposal.occurrences += 1;
            if !proposal.samples.contains(&observation.sample) {
                proposal.samples.push(observation.sample);
            }
        }
    }
    AliasReview {
        wake_word: word.id.clone(),
        phrase: word.phrase.clone(),
        observations,
        proposals: proposals.into_values().collect(),
    }
}

pub fn apply_aliases(word: &mut WakeWord, review: &AliasReview, accepted: &[String]) -> Result<()> {
    ensure!(
        word.id == review.wake_word && word.phrase == review.phrase,
        "wake word changed during onboarding; review again"
    );
    let available: BTreeMap<_, _> = review
        .proposals
        .iter()
        .map(|p| (normalize_tokens(&p.text).concat(), &p.text))
        .collect();
    let mut candidate = word.clone();
    let mut seen: BTreeSet<_> = std::iter::once(&word.phrase)
        .chain(&word.aliases)
        .map(|s| normalize_tokens(s).concat())
        .collect();
    for text in accepted {
        let normalized = normalize_tokens(text).concat();
        let original = available
            .get(&normalized)
            .with_context(|| format!("alias {text:?} was not observed in these recordings"))?;
        if seen.insert(normalized) {
            candidate.aliases.push((*original).clone());
        }
    }
    *word = candidate;
    Ok(())
}

pub struct Cancellation {
    pub flag: Arc<AtomicBool>,
    registrations: Vec<signal_hook::SigId>,
}
impl Cancellation {
    pub fn new() -> Result<Self> {
        let flag = Arc::new(AtomicBool::new(false));
        let mut registration = Self {
            flag,
            registrations: Vec::new(),
        };
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            registration.registrations.push(signal_hook::flag::register(
                signal,
                Arc::clone(&registration.flag),
            )?);
        }
        Ok(registration)
    }
    pub fn check(&self) -> Result<()> {
        ensure!(
            !self.flag.load(Ordering::Relaxed),
            "onboarding cancelled; configuration unchanged"
        );
        Ok(())
    }
}
impl Drop for Cancellation {
    fn drop(&mut self) {
        for id in self.registrations.drain(..) {
            signal_hook::low_level::unregister(id);
        }
    }
}

pub fn record(config: &Config, seconds: u64, cancellation: &Cancellation) -> Result<Vec<f32>> {
    ensure!(
        (1..=15).contains(&seconds),
        "recording duration must be 1..15 seconds"
    );
    cancellation.check()?;
    let capture = Capture::start(&config.audio.device, config.daemon.queue_capacity)?;
    record_events(
        seconds,
        capture.sample_rate as i32,
        cancellation,
        |timeout| capture.receiver().recv_timeout(timeout),
        Instant::now,
    )
}
fn record_events(
    seconds: u64,
    expected_rate: i32,
    cancellation: &Cancellation,
    mut receive: impl FnMut(Duration) -> std::result::Result<AudioEvent, RecvTimeoutError>,
    mut now: impl FnMut() -> Instant,
) -> Result<Vec<f32>> {
    let mut resampler = AudioResampler::new();
    let deadline = now() + Duration::from_secs(seconds);
    let mut samples = Vec::new();
    while now() < deadline {
        cancellation.check()?;
        match receive(Duration::from_millis(100)) {
            Ok(AudioEvent::Samples {
                sample_rate,
                samples: chunk,
            }) => {
                ensure!(
                    sample_rate == expected_rate,
                    "microphone sample rate changed during recording"
                );
                samples.extend(resampler.accept(sample_rate, &chunk)?);
                ensure!(
                    samples.len() <= MAX_SECONDS * 16000,
                    "recording exceeded sample budget"
                );
            }
            Ok(AudioEvent::Error(error)) => bail!("recording failed: {error}"),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => bail!("microphone disconnected"),
        }
    }
    samples.extend(resampler.finish()?);
    validate_audio(&samples)?;
    Ok(samples)
}

#[derive(Debug, Serialize)]
pub struct RecordingSession {
    pub session: String,
    pub directory: PathBuf,
    pub clips: usize,
}

pub fn recordings(paths: &AppPaths, id: &str) -> Result<Vec<RecordingSession>> {
    validate_component(id)?;
    let parent = paths.data_dir.join("enrollments").join(id);
    if !parent.exists() {
        return Ok(Vec::new());
    }
    ensure!(
        parent.symlink_metadata()?.is_dir(),
        "recording directory must not be a symlink"
    );
    let mut sessions = Vec::new();
    for entry in fs::read_dir(parent)?.take(4097) {
        ensure!(
            sessions.len() < 4096,
            "too many retained enrollment sessions"
        );
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let session = entry.file_name().to_string_lossy().into_owned();
        validate_component(&session)?;
        let clips = fs::read_dir(entry.path())?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "wav"))
            .count();
        sessions.push(RecordingSession {
            session,
            directory: entry.path(),
            clips,
        });
    }
    sessions.sort_by(|a, b| a.session.cmp(&b.session));
    Ok(sessions)
}

pub fn remove_recordings(paths: &AppPaths, id: &str, session: &str) -> Result<()> {
    validate_component(session)?;
    let sessions = recordings(paths, id)?;
    let selected = sessions
        .into_iter()
        .find(|item| item.session == session)
        .context("unknown retained recording session")?;
    ensure!(
        selected.directory.symlink_metadata()?.is_dir(),
        "recording session must not be a symlink"
    );
    fs::remove_dir_all(selected.directory).context("remove the selected recording session")
}

fn validate_component(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-'),
        "invalid recording identifier"
    );
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/enrollment.rs"]
mod tests;
