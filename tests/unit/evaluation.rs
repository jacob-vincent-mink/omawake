use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

fn corpus() -> CorpusMetadata {
    CorpusMetadata {
        id: "fixture".into(),
        version: "1".into(),
        license_spdx: "CC0-1.0".into(),
        source: "generated test data".into(),
    }
}

fn event(keyword: &str, start_ms: u64, end_ms: u64) -> ExpectedEvent {
    ExpectedEvent {
        keyword_id: keyword.into(),
        start_ms: Some(start_ms),
        end_ms: Some(end_ms),
    }
}

fn prediction(keyword: &str, end_ms: f64) -> PredictedEvent {
    PredictedEvent {
        keyword_id: keyword.into(),
        observed_start_ms: end_ms - 100.0,
        observed_end_ms: end_ms,
        tokens: Vec::new(),
        timestamps: Vec::new(),
    }
}

fn detection(keyword: &str, end_ms: f32) -> Detection {
    Detection {
        id: keyword.into(),
        tokens: vec![keyword.into()],
        timestamps: vec![(end_ms - 100.0) / 1_000.0, end_ms / 1_000.0],
        start_time: 0.0,
    }
}

fn clip(id: &str, expected: Vec<ExpectedEvent>, duration_ms: u64) -> PreparedClip {
    PreparedClip {
        manifest: ManifestClip {
            id: id.into(),
            path: PathBuf::from(format!("{id}.wav")),
            sha256: "0".repeat(64),
            split: "test".into(),
            tags: BTreeMap::new(),
            expected,
        },
        absolute_path: PathBuf::from(format!("/{id}.wav")),
        duration: Duration::from_millis(duration_ms),
    }
}

fn identity() -> RuntimeIdentity {
    RuntimeIdentity {
        backend_kind: "fake".into(),
        requested_runtime: "default".into(),
        requested_device: "cpu".into(),
        effective_runtime: "default".into(),
        fallback_used: false,
        placement_verified: true,
        placement_evidence: "test".into(),
    }
}

fn assert_schema_keys(value: &serde_json::Value, schema: &serde_json::Value) {
    let actual = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let required = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|key| key.as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, required);
    assert_eq!(schema["additionalProperties"], false);
}

#[test]
fn thresholds_are_defaulted_sorted_deduplicated_and_validated() {
    assert_eq!(normalize_thresholds(Vec::new(), 0.25).unwrap(), [0.25]);
    assert_eq!(
        normalize_thresholds(vec![0.5, 0.25, 0.5, -0.0, 0.0], 0.1).unwrap(),
        [-0.0, 0.25, 0.5]
    );
    for invalid in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(normalize_thresholds(vec![invalid], 0.25).is_err());
    }
}

#[test]
fn manifest_validation_rejects_bad_versions_paths_ids_hashes_and_events() {
    let known = BTreeSet::from(["wake".to_owned()]);
    let valid_clip = ManifestClip {
        id: "one".into(),
        path: "audio/one.wav".into(),
        sha256: "a".repeat(64),
        split: "test".into(),
        tags: BTreeMap::new(),
        expected: vec![event("wake", 10, 20)],
    };
    let manifest = EvaluationManifest {
        schema_version: 1,
        corpus: corpus(),
        matching: MatchingPolicy::default(),
        clips: vec![valid_clip.clone()],
    };
    validate_manifest(&manifest, &known).unwrap();

    let mut broken = manifest.clone();
    broken.schema_version = 2;
    assert!(validate_manifest(&broken, &known).is_err());
    broken = manifest.clone();
    broken.clips.push(valid_clip.clone());
    assert!(validate_manifest(&broken, &known).is_err());
    broken = manifest.clone();
    broken.clips[0].path = "../escape.wav".into();
    assert!(validate_manifest(&broken, &known).is_err());
    broken = manifest.clone();
    broken.clips[0].sha256 = "nope".into();
    assert!(validate_manifest(&broken, &known).is_err());
    broken = manifest.clone();
    broken.clips[0].expected[0].keyword_id = "disabled".into();
    assert!(validate_manifest(&broken, &known).is_err());
    broken = manifest;
    broken.clips[0].expected = vec![event("wake", 20, 10)];
    assert!(validate_manifest(&broken, &known).is_err());
    broken.clips[0].expected[0].end_ms = None;
    assert!(validate_manifest(&broken, &known).is_err());
}

#[test]
fn manifest_loading_checks_audio_hash_and_duration() {
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "omawake-evaluation-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir_all(root.join("audio")).unwrap();
    let audio = b"deterministic fake audio bytes";
    fs::write(root.join("audio/one.wav"), audio).unwrap();
    let manifest = EvaluationManifest {
        schema_version: 1,
        corpus: corpus(),
        matching: MatchingPolicy::default(),
        clips: vec![ManifestClip {
            id: "one".into(),
            path: "audio/one.wav".into(),
            sha256: sha256_bytes(audio),
            split: "test".into(),
            tags: BTreeMap::new(),
            expected: vec![event("wake", 10, 20)],
        }],
    };
    let path = root.join("manifest.json");
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let known = BTreeSet::from(["wake".to_owned()]);
    let loaded = load_manifest_with(&path, &known, |_| Ok(Duration::from_secs(1))).unwrap();
    assert_eq!(loaded.clips.len(), 1);
    assert_eq!(loaded.manifest.corpus.id, "fixture");

    let mut bad = manifest;
    bad.clips[0].sha256 = "f".repeat(64);
    fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(load_manifest_with(&path, &known, |_| Ok(Duration::from_secs(1))).is_err());

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let contained = root.join("contained");
        fs::create_dir_all(contained.join("audio")).unwrap();
        fs::write(root.join("outside.wav"), audio).unwrap();
        symlink("../../outside.wav", contained.join("audio/escape.wav")).unwrap();
        let escaping = EvaluationManifest {
            schema_version: 1,
            corpus: corpus(),
            matching: MatchingPolicy::default(),
            clips: vec![ManifestClip {
                id: "escape".into(),
                path: "audio/escape.wav".into(),
                sha256: sha256_bytes(audio),
                split: "test".into(),
                tags: BTreeMap::new(),
                expected: Vec::new(),
            }],
        };
        let escaping_path = contained.join("manifest.json");
        fs::write(&escaping_path, serde_json::to_vec(&escaping).unwrap()).unwrap();
        let error =
            load_manifest_with(&escaping_path, &known, |_| Ok(Duration::from_secs(1))).unwrap_err();
        assert!(error.to_string().contains("escapes the manifest directory"));
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_bare_relative_manifest_path_resolves_against_the_current_directory() {
    let parent = manifest_directory(Path::new("manifest.json")).unwrap();
    assert_eq!(
        parent,
        std::env::current_dir().unwrap().canonicalize().unwrap()
    );
}

#[test]
fn alignment_maximizes_matches_before_minimizing_timing_error() {
    let expected = vec![event("wake", 900, 1_000), event("wake", 1_050, 1_150)];
    let predictions = vec![prediction("wake", 950.0), prediction("wake", 1_100.0)];
    let matches = align_events(
        &expected,
        &predictions,
        &MatchingPolicy {
            early_tolerance_ms: 200,
            late_tolerance_ms: 200,
        },
    );
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].expected_index, 0);
    assert_eq!(matches[0].prediction_index, 0);
    assert_eq!(matches[1].expected_index, 1);
    assert_eq!(matches[1].prediction_index, 1);

    let clip_level = ExpectedEvent {
        keyword_id: "wake".into(),
        start_ms: None,
        end_ms: None,
    };
    let matches = align_events(
        &[clip_level],
        &[prediction("wake", 9_999.0)],
        &MatchingPolicy::default(),
    );
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].end_error_ms, None);
}

#[test]
fn scoring_distinguishes_duplicates_wrong_ids_and_negative_false_activations() {
    let policy = MatchingPolicy::default();
    let positive = clip(
        "positive",
        vec![event("wake", 900, 1_000), event("hello", 1_900, 2_000)],
        3_000,
    );
    let report = score_file(
        &positive,
        vec![
            detection("wake", 1_000.0),
            detection("wake", 1_020.0),
            detection("wrong", 2_000.0),
        ],
        Duration::from_millis(30),
        &policy,
    )
    .unwrap();
    assert_eq!(report.counts.true_positives, 1);
    assert_eq!(report.counts.false_positives, 2);
    assert_eq!(report.counts.false_negatives, 1);
    assert_eq!(report.counts.duplicate_predictions, 1);
    assert_eq!(report.counts.wrong_id_predictions, 1);
    assert_eq!(report.real_time_factor, Some(0.01));

    let negative = score_file(
        &clip("negative", Vec::new(), 3_600_000),
        vec![detection("wake", 100.0), detection("wake", 200.0)],
        Duration::from_secs(1),
        &policy,
    )
    .unwrap();
    let threshold = build_threshold_report(
        0.25,
        Duration::from_millis(5),
        identity(),
        vec![report, negative],
    );
    assert_eq!(threshold.summary.true_positives, 1);
    assert_eq!(threshold.summary.false_activations, 2);
    assert_eq!(threshold.summary.false_activations_per_hour, Some(2.0));
    assert_eq!(threshold.summary.precision, Some(0.2));
    assert_eq!(threshold.summary.recall, Some(0.5));
    assert_eq!(threshold.per_keyword["wake"].duplicate_predictions, 1);
    assert_eq!(threshold.per_keyword["wrong"].wrong_id_predictions, 1);
    assert_eq!(threshold.per_keyword["wake"].false_activations, 2);
    assert_eq!(
        threshold.per_keyword["wake"].false_activations_per_hour,
        Some(2.0)
    );
}

#[test]
fn invalid_detection_times_cannot_escape_into_a_report() {
    let mut invalid = detection("wake", 100.0);
    invalid.timestamps = vec![0.2, 0.1];
    assert!(predicted_event(&invalid).is_err());

    invalid.timestamps = vec![f32::NAN];
    assert!(predicted_event(&invalid).is_err());

    invalid.timestamps.clear();
    invalid.start_time = -0.1;
    assert!(predicted_event(&invalid).is_err());

    invalid.start_time = 0.0;
    invalid.id.clear();
    assert!(predicted_event(&invalid).is_err());
}

#[test]
fn evaluation_reloads_and_reinfers_for_each_threshold_and_is_fingerprinted() {
    let prepared = PreparedManifest {
        manifest: EvaluationManifest {
            schema_version: 1,
            corpus: corpus(),
            matching: MatchingPolicy::default(),
            clips: vec![ManifestClip {
                id: "positive".into(),
                path: "positive.wav".into(),
                sha256: "0".repeat(64),
                split: "test".into(),
                tags: BTreeMap::new(),
                expected: vec![event("wake", 900, 1_000)],
            }],
        },
        manifest_sha256: "1".repeat(64),
        clips: vec![clip("positive", vec![event("wake", 900, 1_000)], 2_000)],
    };
    let mut loads = Vec::new();
    let mut detections = 0;
    let report = evaluate_with(
        &prepared,
        &[0.25, 0.5],
        EvaluationContext {
            config_sha256: "2".repeat(64),
            model_name: "fake-model".into(),
            model_directory: "/fake".into(),
            keyword_score: 1.5,
            application_version: "test".into(),
        },
        |threshold| {
            loads.push(threshold);
            Ok(threshold)
        },
        |threshold, _| {
            detections += 1;
            Ok(if *threshold < 0.5 {
                vec![detection("wake", 1_000.0)]
            } else {
                Vec::new()
            })
        },
        |_| Ok(identity()),
    )
    .unwrap();
    assert_eq!(loads, [0.25, 0.5]);
    assert_eq!(detections, 2);
    assert_eq!(report.schema_version, 1);
    assert_eq!(report.thresholds[0].summary.recall, Some(1.0));
    assert_eq!(report.thresholds[1].summary.recall, Some(0.0));
    assert_ne!(
        report.thresholds[0].prediction_fingerprint,
        report.thresholds[1].prediction_fingerprint
    );
    assert_eq!(report.prediction_fingerprint.len(), 64);
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["evaluation"], "omawake-kws-accuracy");
    assert_eq!(json["thresholds"][0]["backend"]["fallback_used"], false);
    assert_eq!(json["inputs"]["matching"]["early_tolerance_ms"], 250);

    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../schemas/evaluation-report-v1.schema.json"
    ))
    .unwrap();
    let definitions = &schema["$defs"];
    assert_schema_keys(&json, &schema);
    assert_schema_keys(&json["inputs"], &definitions["inputs"]);
    assert_schema_keys(&json["inputs"]["corpus"], &definitions["corpus"]);
    assert_schema_keys(&json["thresholds"][0], &definitions["threshold"]);
    assert_schema_keys(&json["thresholds"][0]["backend"], &definitions["backend"]);
    assert_schema_keys(&json["thresholds"][0]["files"][0], &definitions["file"]);
    assert_schema_keys(
        &json["thresholds"][0]["files"][0]["expected"][0],
        &definitions["expected"],
    );
    assert_schema_keys(
        &json["thresholds"][0]["files"][0]["predictions"][0],
        &definitions["prediction"],
    );
    assert_schema_keys(
        &json["thresholds"][0]["files"][0]["matches"][0],
        &definitions["match"],
    );
    assert_schema_keys(
        &json["thresholds"][0]["files"][0]["counts"],
        &definitions["counts"],
    );
    assert_schema_keys(&json["thresholds"][0]["summary"], &definitions["metrics"]);
    assert_schema_keys(&json["thresholds"][0]["timing"], &definitions["timing"]);
}

#[test]
fn zero_denominators_are_null_instead_of_misleading_zero_rates() {
    let mut metrics = AccuracyMetrics::default();
    finalize_metrics(&mut metrics);
    assert_eq!(metrics.precision, None);
    assert_eq!(metrics.recall, None);
    assert_eq!(metrics.f1, None);
    assert_eq!(metrics.false_activations_per_hour, None);
}

#[test]
fn published_json_schemas_are_valid_and_pin_version_one() {
    let manifest_schema = include_str!("../../schemas/evaluation-manifest-v1.schema.json");
    for encoded in [
        manifest_schema,
        include_str!("../../schemas/evaluation-report-v1.schema.json"),
    ] {
        let schema: serde_json::Value = serde_json::from_str(encoded).unwrap();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert!(schema["$id"].as_str().unwrap().contains("v1.schema.json"));
    }

    let manifest: EvaluationManifest = serde_json::from_value(serde_json::json!({
        "schema_version": 1,
        "corpus": {
            "id": "fixture",
            "version": "1",
            "license_spdx": "CC0-1.0",
            "source": "generated"
        },
        "clips": [{
            "id": "clip",
            "path": "clip.wav",
            "sha256": "0".repeat(64),
            "split": "test",
            "expected": [{
                "keyword_id": "wake",
                "start_ms": null,
                "end_ms": null
            }]
        }]
    }))
    .unwrap();
    assert_eq!(manifest.clips[0].expected[0].start_ms, None);
    let encoded = serde_json::to_value(manifest).unwrap();
    assert!(encoded["clips"][0]["expected"][0]["start_ms"].is_null());
    let schema: serde_json::Value = serde_json::from_str(manifest_schema).unwrap();
    assert!(
        schema["$defs"]["event"]["properties"]["start_ms"]["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|kind| kind["type"] == "null")
    );
}
