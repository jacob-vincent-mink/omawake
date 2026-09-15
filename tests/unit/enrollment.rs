use super::*;

fn word() -> WakeWord {
    Config::default().wake_words.remove(0)
}
fn paths() -> AppPaths {
    let mut paths = AppPaths::discover();
    paths.cache_dir = std::env::temp_dir().join(format!(
        "ow-onboarding-unit-{}-{}",
        std::process::id(),
        NEXT_SESSION.fetch_add(1, Ordering::Relaxed)
    ));
    paths.data_dir = paths.cache_dir.join("data");
    paths
}

#[test]
fn observations_require_explicit_alias_approval_and_normalize_duplicates() {
    let mut word = word();
    let review = review(
        &word,
        vec![
            Observation {
                sample: 1,
                transcripts: vec!["Computer!".into(), "come pewter".into(), "!!!".into()],
            },
            Observation {
                sample: 2,
                transcripts: vec!["Come, pewter.".into(), "Thanks for watching".into()],
            },
        ],
    );
    assert_eq!(review.proposals.len(), 2);
    assert_eq!(review.proposals[0].occurrences, 2);
    assert_eq!(review.proposals[0].samples, [1, 2]);
    let original = word.aliases.clone();
    assert!(apply_aliases(&mut word, &review, &["not observed".into()]).is_err());
    assert_eq!(word.aliases, original);
    apply_aliases(&mut word, &review, &["come pewter".into()]).unwrap();
    assert_eq!(word.aliases, ["come pewter"]);
    assert!(!word.aliases.iter().any(|a| a == "Thanks for watching"));
}

#[test]
fn recording_quality_rejects_silence_clipping_nonfinite_and_excessive_length() {
    for audio in [
        vec![0.0; 16000],
        vec![1.0; 16000],
        vec![f32::NAN; 16000],
        vec![0.1; 100],
        vec![0.1; 480001],
    ] {
        assert!(validate_audio(&audio).is_err());
    }
    validate_audio(&vec![0.1; 16000]).unwrap();
}

#[test]
fn retained_samples_are_private_and_session_cleanup_is_narrow() {
    let paths = paths();
    let mut samples = SampleSet::create(&paths).unwrap();
    samples.push(&vec![0.1; 16000]).unwrap();
    let ephemeral = samples.directory.clone();
    let unrelated = paths.cache_dir.join("keep-me");
    fs::write(&unrelated, b"keep").unwrap();
    let retained = samples
        .retain(&paths, "computer", &serde_json::json!({"version":1}))
        .unwrap();
    assert_eq!(
        fs::metadata(&retained).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(retained.join("sample-001.wav"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(samples.retain(&paths, "../escape", &()).is_err());
    drop(samples);
    assert!(!ephemeral.exists());
    assert!(retained.is_dir());
    assert!(unrelated.exists());
    fs::remove_dir_all(&paths.cache_dir).unwrap();
}

#[test]
fn unretained_import_never_removes_the_source() {
    let paths = paths();
    let mut source = SampleSet::create(&paths).unwrap();
    source.push(&vec![0.2; 3200]).unwrap();
    let original = source.files[0].clone();
    let mut imported = SampleSet::create(&paths).unwrap();
    imported.import(&original).unwrap();
    assert_eq!(imported.files.len(), 1);
    drop(imported);
    assert!(original.exists());
    drop(source);
    fs::remove_dir_all(paths.cache_dir).unwrap();
}

#[test]
fn transcript_capture_is_scoped_and_restored_on_errors() {
    use crate::phrase::record_transcript;
    let (_, observed) = capture_transcripts(|| {
        record_transcript("outer");
        let (_, inner) = capture_transcripts(|| {
            record_transcript("inner");
            Ok(())
        })?;
        assert_eq!(inner, ["inner"]);
        record_transcript("outer again");
        Ok(())
    })
    .unwrap();
    assert_eq!(observed, ["outer", "outer again"]);
    assert!(
        capture_transcripts::<()>(|| {
            record_transcript("discard");
            bail!("cancelled")
        })
        .is_err()
    );
    assert!(capture_transcripts(|| Ok(())).unwrap().1.is_empty());
}

#[test]
fn recording_removal_is_explicit_and_preserves_other_sessions() {
    let paths = paths();
    let mut first = SampleSet::create(&paths).unwrap();
    first.push(&vec![0.1; 3200]).unwrap();
    let a = first.retain(&paths, "computer", &()).unwrap();
    let mut second = SampleSet::create(&paths).unwrap();
    second.push(&vec![0.1; 3200]).unwrap();
    let b = second.retain(&paths, "computer", &()).unwrap();
    assert_eq!(recordings(&paths, "computer").unwrap().len(), 2);
    assert!(remove_recordings(&paths, "computer", "../other").is_err());
    remove_recordings(&paths, "computer", a.file_name().unwrap().to_str().unwrap()).unwrap();
    assert!(!a.exists());
    assert!(b.exists());
    assert!(remove_recordings(&paths, "computer", "unknown").is_err());
    drop(first);
    drop(second);
    fs::remove_dir_all(paths.cache_dir).unwrap();
}

#[test]
fn recording_event_pipeline_handles_disconnect_rate_changes_and_cancel() {
    fn collect(
        events: Vec<std::result::Result<AudioEvent, RecvTimeoutError>>,
        cancel: &Cancellation,
    ) -> Result<Vec<f32>> {
        let mut events = events.into_iter();
        let base = Instant::now();
        let mut tick = 0;
        record_events(
            1,
            16000,
            cancel,
            |_| events.next().unwrap_or(Err(RecvTimeoutError::Timeout)),
            || {
                tick += 1;
                base + Duration::from_millis(tick * 100)
            },
        )
    }
    let cancel = Cancellation::new().unwrap();
    let audio = collect(
        vec![
            Err(RecvTimeoutError::Timeout),
            Ok(AudioEvent::Samples {
                sample_rate: 16000,
                samples: vec![0.1; 4000],
            }),
        ],
        &cancel,
    )
    .unwrap();
    assert_eq!(audio.len(), 4000);
    assert!(
        collect(vec![Err(RecvTimeoutError::Disconnected)], &cancel)
            .unwrap_err()
            .to_string()
            .contains("disconnected")
    );
    assert!(collect(vec![Ok(AudioEvent::Error("input failed".into()))], &cancel).is_err());
    assert!(
        collect(
            vec![Ok(AudioEvent::Samples {
                sample_rate: 48000,
                samples: vec![0.1; 4000]
            })],
            &cancel
        )
        .unwrap_err()
        .to_string()
        .contains("sample rate changed")
    );
    cancel.flag.store(true, Ordering::Relaxed);
    assert!(collect(vec![], &cancel).is_err());
    assert!(record(&Config::default(), 0, &cancel).is_err());
}
