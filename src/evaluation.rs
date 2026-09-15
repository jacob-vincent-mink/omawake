use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::engine::Detection;

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationManifest {
    pub schema_version: u32,
    pub corpus: CorpusMetadata,
    #[serde(default)]
    pub matching: MatchingPolicy,
    pub clips: Vec<ManifestClip>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusMetadata {
    pub id: String,
    pub version: String,
    pub license_spdx: String,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MatchingPolicy {
    pub early_tolerance_ms: u64,
    pub late_tolerance_ms: u64,
}

impl Default for MatchingPolicy {
    fn default() -> Self {
        Self {
            early_tolerance_ms: 250,
            late_tolerance_ms: 750,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestClip {
    pub id: String,
    pub path: PathBuf,
    pub sha256: String,
    pub split: String,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    #[serde(default)]
    pub expected: Vec<ExpectedEvent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedEvent {
    pub keyword_id: String,
    #[serde(default)]
    pub start_ms: Option<u64>,
    #[serde(default)]
    pub end_ms: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PreparedManifest {
    manifest: EvaluationManifest,
    manifest_sha256: String,
    clips: Vec<PreparedClip>,
}

#[derive(Clone, Debug)]
struct PreparedClip {
    manifest: ManifestClip,
    absolute_path: PathBuf,
    duration: Duration,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeIdentity {
    pub backend_kind: String,
    pub requested_runtime: String,
    pub requested_device: String,
    pub effective_runtime: String,
    pub fallback_used: bool,
    pub placement_verified: bool,
    pub placement_evidence: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvaluationInputs {
    pub manifest_sha256: String,
    pub config_sha256: String,
    pub corpus: CorpusMetadata,
    pub matching: MatchingPolicy,
    pub enabled_keyword_ids: Vec<String>,
    pub model_name: String,
    pub model_directory: String,
    pub keyword_score: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub evaluation: String,
    pub application_version: String,
    pub inputs: EvaluationInputs,
    pub thresholds: Vec<ThresholdReport>,
    pub prediction_fingerprint: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ThresholdReport {
    pub threshold: f32,
    pub model_load_milliseconds: f64,
    pub backend: RuntimeIdentity,
    pub files: Vec<FileReport>,
    pub summary: AccuracyMetrics,
    pub per_keyword: BTreeMap<String, AccuracyMetrics>,
    pub timing: TimingSummary,
    pub prediction_fingerprint: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FileReport {
    pub id: String,
    pub path: PathBuf,
    pub split: String,
    pub tags: BTreeMap<String, String>,
    pub audio_duration_milliseconds: f64,
    pub elapsed_milliseconds: f64,
    pub real_time_factor: Option<f64>,
    pub expected: Vec<ExpectedEvent>,
    pub predictions: Vec<PredictedEvent>,
    pub matches: Vec<EventMatch>,
    pub wrong_id_prediction_indices: Vec<usize>,
    pub duplicate_prediction_indices: Vec<usize>,
    pub counts: FileCounts,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PredictedEvent {
    pub keyword_id: String,
    pub observed_start_ms: f64,
    pub observed_end_ms: f64,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EventMatch {
    pub expected_index: usize,
    pub prediction_index: usize,
    pub keyword_id: String,
    pub end_error_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FileCounts {
    pub true_positives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub wrong_id_predictions: u64,
    pub duplicate_predictions: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AccuracyMetrics {
    pub expected_events: u64,
    pub predictions: u64,
    pub true_positives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub wrong_id_predictions: u64,
    pub duplicate_predictions: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub negative_audio_hours: f64,
    pub false_activations: u64,
    pub false_activations_per_hour: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TimingSummary {
    pub files: usize,
    pub total_audio_milliseconds: f64,
    pub total_elapsed_milliseconds: f64,
    pub aggregate_real_time_factor: Option<f64>,
    pub p50_file_milliseconds: Option<f64>,
    pub p95_file_milliseconds: Option<f64>,
}

pub fn normalize_thresholds(mut thresholds: Vec<f32>, default: f32) -> Result<Vec<f32>> {
    if thresholds.is_empty() {
        thresholds.push(default);
    }
    for threshold in &thresholds {
        if !threshold.is_finite() || !(0.0..=1.0).contains(threshold) {
            bail!("evaluation thresholds must be finite values between 0 and 1");
        }
    }
    thresholds.sort_by(f32::total_cmp);
    thresholds.dedup_by(|left, right| *left == *right);
    Ok(thresholds)
}

pub fn load_manifest(path: &Path, enabled_keywords: &BTreeSet<String>) -> Result<PreparedManifest> {
    load_manifest_with(path, enabled_keywords, crate::engine::wav_duration)
}

fn load_manifest_with<D>(
    path: &Path,
    enabled_keywords: &BTreeSet<String>,
    mut duration: D,
) -> Result<PreparedManifest>
where
    D: FnMut(&Path) -> Result<Duration>,
{
    let encoded =
        fs::read(path).with_context(|| format!("read evaluation manifest {}", path.display()))?;
    let manifest: EvaluationManifest = serde_json::from_slice(&encoded)
        .with_context(|| format!("parse evaluation manifest {}", path.display()))?;
    validate_manifest(&manifest, enabled_keywords)?;
    let parent = manifest_directory(path)?;
    let mut clips = Vec::with_capacity(manifest.clips.len());
    for clip in &manifest.clips {
        let absolute_path = parent.join(&clip.path).canonicalize().with_context(|| {
            format!(
                "resolve evaluation audio {}",
                parent.join(&clip.path).display()
            )
        })?;
        if !absolute_path.starts_with(&parent) {
            bail!(
                "evaluation audio {} escapes the manifest directory",
                clip.path.display()
            );
        }
        let actual_sha256 = file_sha256(&absolute_path)?;
        if actual_sha256 != clip.sha256.to_ascii_lowercase() {
            bail!(
                "evaluation audio checksum mismatch for {}: expected {}, got {}",
                clip.path.display(),
                clip.sha256,
                actual_sha256
            );
        }
        let clip_duration = duration(&absolute_path)
            .with_context(|| format!("inspect evaluation audio {}", clip.path.display()))?;
        let duration_ms = duration_milliseconds(clip_duration);
        if let Some(event) = clip.expected.iter().find(|event| {
            event
                .end_ms
                .is_some_and(|end| end as f64 > duration_ms + 0.5)
        }) {
            bail!(
                "expected event {} in clip {} ends after the audio",
                event.keyword_id,
                clip.id
            );
        }
        clips.push(PreparedClip {
            manifest: clip.clone(),
            absolute_path,
            duration: clip_duration,
        });
    }
    Ok(PreparedManifest {
        manifest,
        manifest_sha256: sha256_bytes(&encoded),
        clips,
    })
}

fn manifest_directory(path: &Path) -> Result<PathBuf> {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .with_context(|| format!("resolve evaluation manifest directory {}", path.display()))
}

fn validate_manifest(
    manifest: &EvaluationManifest,
    enabled_keywords: &BTreeSet<String>,
) -> Result<()> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        bail!(
            "unsupported evaluation manifest schema version {}; expected {}",
            manifest.schema_version,
            MANIFEST_SCHEMA_VERSION
        );
    }
    for (name, value) in [
        ("corpus.id", &manifest.corpus.id),
        ("corpus.version", &manifest.corpus.version),
        ("corpus.license_spdx", &manifest.corpus.license_spdx),
        ("corpus.source", &manifest.corpus.source),
    ] {
        if value.trim().is_empty() {
            bail!("evaluation manifest {name} must not be empty");
        }
    }
    if manifest.clips.is_empty() {
        bail!("evaluation manifest must contain at least one clip");
    }
    let mut ids = BTreeSet::new();
    for clip in &manifest.clips {
        if clip.id.trim().is_empty() || !ids.insert(clip.id.as_str()) {
            bail!(
                "evaluation clip IDs must be non-empty and unique: {:?}",
                clip.id
            );
        }
        if clip.split.trim().is_empty() {
            bail!("evaluation clip {} has an empty split", clip.id);
        }
        if clip.path.is_absolute()
            || clip.path.as_os_str().is_empty()
            || clip.path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            bail!(
                "evaluation clip {} path must stay under the manifest directory",
                clip.id
            );
        }
        if clip.sha256.len() != 64 || !clip.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("evaluation clip {} has an invalid SHA-256", clip.id);
        }
        let mut previous_start = None;
        for event in &clip.expected {
            if !enabled_keywords.contains(&event.keyword_id) {
                bail!(
                    "evaluation clip {} references unknown or disabled keyword {}",
                    clip.id,
                    event.keyword_id
                );
            }
            if event.start_ms.is_some() != event.end_ms.is_some() {
                bail!(
                    "evaluation clip {} event {} must provide both start_ms and end_ms or neither",
                    clip.id,
                    event.keyword_id
                );
            }
            if event
                .start_ms
                .zip(event.end_ms)
                .is_some_and(|(start, end)| start > end)
            {
                bail!(
                    "evaluation clip {} has an event with start after end",
                    clip.id
                );
            }
            if previous_start
                .zip(event.start_ms)
                .is_some_and(|(previous, start)| start < previous)
            {
                bail!(
                    "evaluation clip {} expected events are not time ordered",
                    clip.id
                );
            }
            if event.start_ms.is_some() {
                previous_start = event.start_ms;
            }
        }
    }
    Ok(())
}

pub struct EvaluationContext {
    pub config_sha256: String,
    pub enabled_keyword_ids: Vec<String>,
    pub model_name: String,
    pub model_directory: String,
    pub keyword_score: f32,
    pub application_version: String,
}

pub fn evaluate_with<D, L, F, I>(
    prepared: &PreparedManifest,
    thresholds: &[f32],
    context: EvaluationContext,
    mut load: L,
    mut detect: F,
    mut identity: I,
) -> Result<EvaluationReport>
where
    L: FnMut(f32) -> Result<D>,
    F: FnMut(&D, &Path) -> Result<Vec<Detection>>,
    I: FnMut(&D) -> Result<RuntimeIdentity>,
{
    if thresholds.is_empty() {
        bail!("evaluation requires at least one threshold");
    }
    let mut reports = Vec::with_capacity(thresholds.len());
    for &threshold in thresholds {
        let load_started = Instant::now();
        let detector = load(threshold)
            .with_context(|| format!("load detector for evaluation threshold {threshold}"))?;
        let load_time = load_started.elapsed();
        let backend = identity(&detector)?;
        let mut files = Vec::with_capacity(prepared.clips.len());
        for clip in &prepared.clips {
            let started = Instant::now();
            let detections = detect(&detector, &clip.absolute_path).with_context(|| {
                format!(
                    "evaluate audio {} at threshold {threshold}",
                    clip.manifest.path.display()
                )
            })?;
            files.push(score_file(
                clip,
                detections,
                started.elapsed(),
                &prepared.manifest.matching,
            )?);
        }
        reports.push(build_threshold_report(
            threshold,
            load_time,
            backend,
            files,
            &context.enabled_keyword_ids,
        ));
    }
    let prediction_fingerprint = report_fingerprint(&reports);
    Ok(EvaluationReport {
        schema_version: REPORT_SCHEMA_VERSION,
        evaluation: "omawake-kws-accuracy".into(),
        application_version: context.application_version,
        inputs: EvaluationInputs {
            manifest_sha256: prepared.manifest_sha256.clone(),
            config_sha256: context.config_sha256,
            corpus: prepared.manifest.corpus.clone(),
            matching: prepared.manifest.matching.clone(),
            enabled_keyword_ids: context.enabled_keyword_ids,
            model_name: context.model_name,
            model_directory: context.model_directory,
            keyword_score: context.keyword_score,
        },
        thresholds: reports,
        prediction_fingerprint,
    })
}

fn score_file(
    clip: &PreparedClip,
    detections: Vec<Detection>,
    elapsed: Duration,
    policy: &MatchingPolicy,
) -> Result<FileReport> {
    let predictions = detections
        .iter()
        .map(predicted_event)
        .collect::<Result<Vec<_>>>()?;
    let matches = align_events(&clip.manifest.expected, &predictions, policy);
    let matched_expected = matches
        .iter()
        .map(|item| item.expected_index)
        .collect::<BTreeSet<_>>();
    let matched_predictions = matches
        .iter()
        .map(|item| item.prediction_index)
        .collect::<BTreeSet<_>>();
    let mut counts = FileCounts {
        true_positives: matches.len() as u64,
        false_positives: (predictions.len() - matched_predictions.len()) as u64,
        false_negatives: (clip.manifest.expected.len() - matched_expected.len()) as u64,
        ..FileCounts::default()
    };
    let mut wrong_id_prediction_indices = Vec::new();
    let mut duplicate_prediction_indices = Vec::new();
    for (prediction_index, prediction) in predictions.iter().enumerate() {
        if matched_predictions.contains(&prediction_index) {
            continue;
        }
        let duplicate = matches.iter().any(|matched| {
            clip.manifest.expected[matched.expected_index].keyword_id == prediction.keyword_id
                && in_window(
                    prediction.observed_end_ms,
                    &clip.manifest.expected[matched.expected_index],
                    policy,
                )
        });
        if duplicate {
            counts.duplicate_predictions += 1;
            duplicate_prediction_indices.push(prediction_index);
            continue;
        }
        if clip.manifest.expected.iter().any(|expected| {
            expected.keyword_id != prediction.keyword_id
                && in_window(prediction.observed_end_ms, expected, policy)
        }) {
            counts.wrong_id_predictions += 1;
            wrong_id_prediction_indices.push(prediction_index);
        }
    }
    let audio_ms = duration_milliseconds(clip.duration);
    let elapsed_ms = duration_milliseconds(elapsed);
    Ok(FileReport {
        id: clip.manifest.id.clone(),
        path: clip.manifest.path.clone(),
        split: clip.manifest.split.clone(),
        tags: clip.manifest.tags.clone(),
        audio_duration_milliseconds: audio_ms,
        elapsed_milliseconds: elapsed_ms,
        real_time_factor: (audio_ms > 0.0).then_some(elapsed_ms / audio_ms),
        expected: clip.manifest.expected.clone(),
        predictions,
        matches,
        wrong_id_prediction_indices,
        duplicate_prediction_indices,
        counts,
    })
}

fn predicted_event(detection: &Detection) -> Result<PredictedEvent> {
    if detection.id.trim().is_empty() {
        bail!("detector returned an empty wake-word id during evaluation");
    }
    if detection
        .timestamps
        .iter()
        .any(|time| !time.is_finite() || *time < 0.0)
    {
        bail!(
            "detector returned a negative or non-finite timestamp for {}",
            detection.id
        );
    }
    if detection
        .timestamps
        .windows(2)
        .any(|pair| pair[0] > pair[1])
    {
        bail!(
            "detector returned non-monotonic timestamps for {}",
            detection.id
        );
    }
    if detection.timestamps.is_empty()
        && (!detection.start_time.is_finite() || detection.start_time < 0.0)
    {
        bail!(
            "detector returned a negative or non-finite start time for {}",
            detection.id
        );
    }
    let observed_start_ms = detection
        .timestamps
        .first()
        .map_or(detection.start_time as f64 * 1_000.0, |time| {
            *time as f64 * 1_000.0
        });
    let observed_end_ms = detection
        .timestamps
        .last()
        .map_or(detection.start_time as f64 * 1_000.0, |time| {
            *time as f64 * 1_000.0
        });
    Ok(PredictedEvent {
        keyword_id: detection.id.clone(),
        observed_start_ms,
        observed_end_ms,
        tokens: detection.tokens.clone(),
        timestamps: detection.timestamps.clone(),
    })
}

#[derive(Clone, Copy)]
enum AlignmentAction {
    Start,
    SkipExpected,
    SkipPrediction,
    Match,
}

#[derive(Clone, Copy)]
struct AlignmentCell {
    matches: usize,
    cost_millis: u64,
    action: AlignmentAction,
}

fn align_events(
    expected: &[ExpectedEvent],
    predictions: &[PredictedEvent],
    policy: &MatchingPolicy,
) -> Vec<EventMatch> {
    let keywords = expected
        .iter()
        .map(|event| event.keyword_id.as_str())
        .chain(predictions.iter().map(|event| event.keyword_id.as_str()))
        .collect::<BTreeSet<_>>();
    let mut output = Vec::new();
    for keyword in keywords {
        let expected_indices = expected
            .iter()
            .enumerate()
            .filter_map(|(index, event)| (event.keyword_id == keyword).then_some(index))
            .collect::<Vec<_>>();
        let mut prediction_indices = predictions
            .iter()
            .enumerate()
            .filter_map(|(index, event)| (event.keyword_id == keyword).then_some(index))
            .collect::<Vec<_>>();
        prediction_indices.sort_by(|left, right| {
            predictions[*left]
                .observed_end_ms
                .total_cmp(&predictions[*right].observed_end_ms)
                .then_with(|| left.cmp(right))
        });
        output.extend(align_keyword(
            expected,
            predictions,
            &expected_indices,
            &prediction_indices,
            policy,
        ));
    }
    output.sort_by_key(|item| (item.expected_index, item.prediction_index));
    output
}

fn align_keyword(
    expected: &[ExpectedEvent],
    predictions: &[PredictedEvent],
    expected_indices: &[usize],
    prediction_indices: &[usize],
    policy: &MatchingPolicy,
) -> Vec<EventMatch> {
    let width = prediction_indices.len() + 1;
    let mut cells = vec![
        AlignmentCell {
            matches: 0,
            cost_millis: 0,
            action: AlignmentAction::Start,
        };
        (expected_indices.len() + 1) * width
    ];
    for i in 1..=expected_indices.len() {
        cells[i * width].action = AlignmentAction::SkipExpected;
    }
    for cell in cells.iter_mut().take(width).skip(1) {
        cell.action = AlignmentAction::SkipPrediction;
    }
    for i in 1..=expected_indices.len() {
        for j in 1..=prediction_indices.len() {
            let mut best = AlignmentCell {
                action: AlignmentAction::SkipExpected,
                ..cells[(i - 1) * width + j]
            };
            let skip_prediction = AlignmentCell {
                action: AlignmentAction::SkipPrediction,
                ..cells[i * width + j - 1]
            };
            if better_alignment(skip_prediction, best) {
                best = skip_prediction;
            }
            let expected_event = &expected[expected_indices[i - 1]];
            let prediction = &predictions[prediction_indices[j - 1]];
            if in_window(prediction.observed_end_ms, expected_event, policy) {
                let previous = cells[(i - 1) * width + j - 1];
                let matched = AlignmentCell {
                    matches: previous.matches + 1,
                    cost_millis: previous.cost_millis.saturating_add(
                        expected_event.end_ms.map_or(0, |end| {
                            (prediction.observed_end_ms - end as f64).abs().round() as u64
                        }),
                    ),
                    action: AlignmentAction::Match,
                };
                if better_alignment(matched, best) {
                    best = matched;
                }
            }
            cells[i * width + j] = best;
        }
    }
    let mut i = expected_indices.len();
    let mut j = prediction_indices.len();
    let mut matches = Vec::new();
    while i > 0 || j > 0 {
        match cells[i * width + j].action {
            AlignmentAction::Match => {
                let expected_index = expected_indices[i - 1];
                let prediction_index = prediction_indices[j - 1];
                matches.push(EventMatch {
                    expected_index,
                    prediction_index,
                    keyword_id: expected[expected_index].keyword_id.clone(),
                    end_error_ms: expected[expected_index]
                        .end_ms
                        .map(|end| predictions[prediction_index].observed_end_ms - end as f64),
                });
                i -= 1;
                j -= 1;
            }
            AlignmentAction::SkipExpected => i -= 1,
            AlignmentAction::SkipPrediction => j -= 1,
            AlignmentAction::Start => break,
        }
    }
    matches.reverse();
    matches
}

fn better_alignment(candidate: AlignmentCell, current: AlignmentCell) -> bool {
    candidate.matches > current.matches
        || (candidate.matches == current.matches
            && (candidate.cost_millis < current.cost_millis
                || (candidate.cost_millis == current.cost_millis
                    && matches!(candidate.action, AlignmentAction::Match)
                    && !matches!(current.action, AlignmentAction::Match))))
}

fn in_window(observed_end_ms: f64, expected: &ExpectedEvent, policy: &MatchingPolicy) -> bool {
    match (expected.start_ms, expected.end_ms) {
        (Some(start), Some(end)) => {
            let lower = start.saturating_sub(policy.early_tolerance_ms) as f64;
            let upper = end.saturating_add(policy.late_tolerance_ms) as f64;
            observed_end_ms >= lower && observed_end_ms <= upper
        }
        (None, None) => true,
        _ => false,
    }
}

fn build_threshold_report(
    threshold: f32,
    load_time: Duration,
    backend: RuntimeIdentity,
    files: Vec<FileReport>,
    enabled_keyword_ids: &[String],
) -> ThresholdReport {
    let mut summary = AccuracyMetrics::default();
    let mut per_keyword = enabled_keyword_ids
        .iter()
        .cloned()
        .map(|id| (id, AccuracyMetrics::default()))
        .collect::<BTreeMap<_, _>>();
    let mut negative_audio_hours = 0.0;
    for file in &files {
        summary.expected_events += file.expected.len() as u64;
        summary.predictions += file.predictions.len() as u64;
        summary.true_positives += file.counts.true_positives;
        summary.false_positives += file.counts.false_positives;
        summary.false_negatives += file.counts.false_negatives;
        summary.wrong_id_predictions += file.counts.wrong_id_predictions;
        summary.duplicate_predictions += file.counts.duplicate_predictions;
        if file.expected.is_empty() {
            let file_hours = file.audio_duration_milliseconds / 3_600_000.0;
            summary.negative_audio_hours += file_hours;
            negative_audio_hours += file_hours;
            summary.false_activations += file.predictions.len() as u64;
            for prediction in &file.predictions {
                per_keyword
                    .entry(prediction.keyword_id.clone())
                    .or_default()
                    .false_activations += 1;
            }
        }
        for expected in &file.expected {
            per_keyword
                .entry(expected.keyword_id.clone())
                .or_default()
                .expected_events += 1;
        }
        for prediction in &file.predictions {
            per_keyword
                .entry(prediction.keyword_id.clone())
                .or_default()
                .predictions += 1;
        }
        for matched in &file.matches {
            per_keyword
                .entry(matched.keyword_id.clone())
                .or_default()
                .true_positives += 1;
        }
        for &index in &file.wrong_id_prediction_indices {
            per_keyword
                .entry(file.predictions[index].keyword_id.clone())
                .or_default()
                .wrong_id_predictions += 1;
        }
        for &index in &file.duplicate_prediction_indices {
            per_keyword
                .entry(file.predictions[index].keyword_id.clone())
                .or_default()
                .duplicate_predictions += 1;
        }
    }
    finalize_metrics(&mut summary);
    for metrics in per_keyword.values_mut() {
        metrics.false_positives = metrics.predictions - metrics.true_positives;
        metrics.false_negatives = metrics.expected_events - metrics.true_positives;
        metrics.negative_audio_hours = negative_audio_hours;
        finalize_metrics(metrics);
    }
    let timing = timing_summary(&files);
    let prediction_fingerprint = threshold_fingerprint(threshold, &files);
    ThresholdReport {
        threshold,
        model_load_milliseconds: duration_milliseconds(load_time),
        backend,
        files,
        summary,
        per_keyword,
        timing,
        prediction_fingerprint,
    }
}

fn finalize_metrics(metrics: &mut AccuracyMetrics) {
    metrics.precision = ratio(
        metrics.true_positives,
        metrics.true_positives + metrics.false_positives,
    );
    metrics.recall = ratio(
        metrics.true_positives,
        metrics.true_positives + metrics.false_negatives,
    );
    metrics.f1 = match (metrics.precision, metrics.recall) {
        (Some(precision), Some(recall)) if precision + recall > 0.0 => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    metrics.false_activations_per_hour = (metrics.negative_audio_hours > 0.0)
        .then_some(metrics.false_activations as f64 / metrics.negative_audio_hours);
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn timing_summary(files: &[FileReport]) -> TimingSummary {
    let total_audio = files
        .iter()
        .map(|file| file.audio_duration_milliseconds)
        .sum::<f64>();
    let total_elapsed = files
        .iter()
        .map(|file| file.elapsed_milliseconds)
        .sum::<f64>();
    let samples = files
        .iter()
        .map(|file| file.elapsed_milliseconds)
        .collect::<Vec<_>>();
    TimingSummary {
        files: files.len(),
        total_audio_milliseconds: total_audio,
        total_elapsed_milliseconds: total_elapsed,
        aggregate_real_time_factor: (total_audio > 0.0).then_some(total_elapsed / total_audio),
        p50_file_milliseconds: percentile(&samples, 0.50),
        p95_file_milliseconds: percentile(&samples, 0.95),
    }
}

fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[rank])
}

fn threshold_fingerprint(threshold: f32, files: &[FileReport]) -> String {
    let mut digest = Sha256::new();
    digest.update(threshold.to_bits().to_le_bytes());
    for file in files {
        digest.update((file.id.len() as u64).to_le_bytes());
        digest.update(file.id.as_bytes());
        for prediction in &file.predictions {
            digest.update((prediction.keyword_id.len() as u64).to_le_bytes());
            digest.update(prediction.keyword_id.as_bytes());
            digest.update(prediction.observed_start_ms.round().to_le_bytes());
            digest.update(prediction.observed_end_ms.round().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn report_fingerprint(reports: &[ThresholdReport]) -> String {
    let mut digest = Sha256::new();
    for report in reports {
        digest.update(report.threshold.to_bits().to_le_bytes());
        digest.update(report.prediction_fingerprint.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> Result<String> {
    let file = fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn duration_milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
#[path = "../tests/unit/evaluation.rs"]
mod tests;
