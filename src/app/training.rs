use super::*;
use crate::engine::{
    audio::read_wave,
    embedding_worker::{EmbeddingSession, load_session},
};
use crate::enrollment::{
    Cancellation, SampleSet,
    artifact::{self, Dataset, LabeledRecording},
    head::{Example, Head},
};
use sha2::{Digest, Sha256};

#[derive(clap::Args)]
pub(super) struct TrainArgs {
    pub id: String,
    /// JSON manifest with separate training, calibration and validation WAVs.
    #[arg(
        long,
        required_unless_present = "reuse_recordings",
        conflicts_with = "reuse_recordings"
    )]
    pub dataset: Option<PathBuf>,
    /// Re-enroll from the newest retained labeled session for this word.
    #[arg(long)]
    pub reuse_recordings: bool,
    /// Configured encoder profile; default is the word's current profile.
    #[arg(long)]
    pub engine: Option<String>,
    /// Activate only after all held-out local examples pass. No actions run here.
    #[arg(long)]
    pub apply: bool,
    /// Retain labeled private recording copies for future encoder changes.
    #[arg(long)]
    pub keep_recordings: bool,
    #[arg(long)]
    pub json: bool,
}

pub(super) fn run(args: TrainArgs, config: Config, path: &Path, paths: &AppPaths) -> Result<()> {
    run_with(args, config, path, paths, load_session)
}
fn run_with(
    args: TrainArgs,
    mut config: Config,
    path: &Path,
    paths: &AppPaths,
    load: impl FnOnce(&Config, &AppPaths, Arc<AtomicBool>) -> Result<Box<dyn EmbeddingSession>>,
) -> Result<()> {
    validate_wake_words(&config.wake_words)?;
    let original = config_snapshot(path)?;
    let index = config
        .wake_words
        .iter()
        .position(|w| w.id == args.id)
        .context("unknown wake-word ID; add the word or onboard its spellings first")?;
    let selected_engine = args
        .engine
        .as_deref()
        .or(config.wake_words[index].engine.as_deref());
    let selected_engine = selected_engine.filter(|s| *s != "default");
    let selected = config.for_engine(selected_engine)?;
    anyhow::ensure!(
        selected.backend.runtime == Runtime::Openvino,
        "trainable heads currently require a configured raw OpenVINO Whisper encoder profile"
    );
    let dataset_path = dataset_path(&args, paths)?;
    let dataset = Dataset::load(&dataset_path)?;
    let cancellation = Cancellation::new()?;
    let mut samples = SampleSet::create(paths)?;
    let mut worker = load(&selected, paths, Arc::clone(&cancellation.flag))?;
    let contract = worker.contract().to_owned();
    let execution_devices = worker.execution_devices().to_owned();
    let mut examples = Vec::new();
    let mut retained_splits = Vec::new();
    for (label, split) in [
        ("training", &dataset.training),
        ("calibration", &dataset.calibration),
        ("validation", &dataset.validation),
    ] {
        let mut encoded = Vec::new();
        let mut retained = Vec::new();
        for (i, item) in split.iter().enumerate() {
            cancellation.check()?;
            if !args.json {
                eprintln!("Encoding {label} {}/{}…", i + 1, split.len());
            }
            samples.import(&item.audio)?;
            let (_, audio) =
                read_wave(samples.files.last().context("missing imported recording")?)?;
            let mut hash = Sha256::new();
            for value in &audio {
                hash.update(value.to_le_bytes());
            }
            let fingerprint = format!("{:x}", hash.finalize());
            // Enrollment and live detection use the identical Silero endpoint
            // and pre/post-roll. Reject ambiguous multi-phrase recordings.
            worker.start()?;
            let mut utterances = Vec::new();
            for chunk in audio.chunks(16_000) {
                utterances.extend(worker.audio(chunk)?);
            }
            utterances.extend(worker.finish()?);
            anyhow::ensure!(
                utterances.len() == 1,
                "{} produced {} speech segments; each enrollment clip must contain one natural utterance",
                item.audio.display(),
                utterances.len()
            );
            let embedding = utterances
                .pop()
                .context("missing speech embedding")?
                .embedding;
            anyhow::ensure!(
                embedding.encoder_contract == contract,
                "encoder changed during enrollment"
            );
            encoded.push(Example {
                id: fingerprint,
                values: embedding.values,
                positive: item.positive,
            });
            retained.push(LabeledRecording {
                audio: PathBuf::from(format!("sample-{:03}.wav", samples.files.len())),
                positive: item.positive,
            });
        }
        examples.push(encoded);
        retained_splits.push(retained);
    }
    cancellation.check()?;
    let mut head = Head::train(&contract, &examples[0], &examples[1])?;
    head.validate_held_out(&examples[2])?;
    cancellation.check()?;
    let mut retained = None;
    let mut artifact_path = None;
    if args.apply {
        anyhow::ensure!(
            config_snapshot(path)? == original,
            "configuration changed during training; rerun with the updated configuration"
        );
        let directory = paths.data_dir.join("heads").join(&args.id);
        let installed = artifact::install(&directory, &head)?;
        if args.keep_recordings {
            let [training, calibration, validation]: [Vec<LabeledRecording>; 3] = retained_splits
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid dataset splits"))?;
            retained = Some(samples.retain(
                paths,
                &args.id,
                &Dataset {
                    training,
                    calibration,
                    validation,
                },
            )?);
        }
        let word = &mut config.wake_words[index];
        let enrollment = word.enrollment.get_or_insert_with(Default::default);
        enrollment.active = true;
        enrollment.heads.insert(contract.clone(), installed.clone());
        // Pin the exact backend+model as a named profile. Changing the default
        // backend later cannot silently reinterpret or deactivate this word.
        let profile_id = format!(
            "enrolled-{}-{}",
            args.id,
            &format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&(
                    &contract,
                    &selected.backend,
                    &selected.model
                ))?)
            )[..12]
        );
        word.engine = Some(profile_id.clone());
        config.engines.insert(
            profile_id,
            crate::config::EngineProfile {
                backend: selected.backend.clone(),
                model: selected.model.clone(),
            },
        );
        if let Err(error) = save_and_reload_active(config, path, paths) {
            if let Some(directory) = &retained {
                let _ = fs::remove_dir_all(directory);
            }
            // Immutable head remains reusable; no active config points at it.
            return Err(error);
        }
        artifact_path = Some(installed);
    }
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"applied":args.apply,"word":args.id,"encoder_contract":contract,"execution_devices":execution_devices,"local_validation":head.validation,"head":artifact_path,"recordings":retained,"actions_executed":false,"qualification":"local held-out clips only; not a measured ambient false-activation rate"})
            )?
        );
    } else {
        println!(
            "{} trained head for {}",
            if args.apply {
                "Activated"
            } else {
                "Validated preview of"
            },
            args.id
        );
        println!("Held-out local clips: {:?}", head.validation);
        println!(
            "This checks your clips; it does not establish an ambient false-activation rate. Test with representative background speech before relying on it."
        );
        if !args.apply {
            println!(
                "Rerun with --apply to activate; use --keep-recordings to retain a retraining dataset."
            );
        }
    }
    Ok(())
}

fn dataset_path(args: &TrainArgs, paths: &AppPaths) -> Result<PathBuf> {
    if let Some(path) = &args.dataset {
        return Ok(path.clone());
    }
    anyhow::ensure!(
        args.reuse_recordings,
        "provide --dataset or --reuse-recordings"
    );
    let mut candidates = Vec::new();
    for session in crate::enrollment::recordings(paths, &args.id)? {
        let manifest = session.directory.join("manifest.json");
        if let Ok(dataset) = Dataset::load(&manifest)
            && dataset
                .training
                .iter()
                .chain(&dataset.calibration)
                .chain(&dataset.validation)
                .all(|r| r.audio.is_file())
        {
            candidates.push((fs::metadata(&manifest)?.modified()?, manifest));
        }
    }
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    candidates.pop().map(|(_,path)|path).context("no retained labeled recordings for this word; provide a new --dataset. The current word and heads are unchanged")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{FakeEmbeddingSession, isolated_paths, unique_directory};
    fn fixture() -> (PathBuf, AppPaths, Config, PathBuf) {
        let root = unique_directory("training", "transaction");
        let paths = isolated_paths(&root);
        let mut samples = SampleSet::create(&paths).unwrap();
        let mut splits = Vec::new();
        for n in 0..3 {
            let mut split = Vec::new();
            for i in 0..4 {
                let positive = i < 2;
                let amplitude =
                    (0.1 + (n * 4 + i) as f32 * 0.001) * if positive { 1.0 } else { -1.0 };
                samples.push(&vec![amplitude; 4000]).unwrap();
                split.push(LabeledRecording {
                    audio: samples.files.last().unwrap().clone(),
                    positive,
                });
            }
            splits.push(split);
        }
        let manifest = Dataset {
            training: splits.remove(0),
            calibration: splits.remove(0),
            validation: splits.remove(0),
        };
        let directory = samples.retain(&paths, "computer", &manifest).unwrap();
        // Input manifest points at copied, durable source clips in this fixture.
        let mut manifest = manifest;
        for (i, item) in manifest
            .training
            .iter_mut()
            .chain(&mut manifest.calibration)
            .chain(&mut manifest.validation)
            .enumerate()
        {
            item.audio = directory.join(format!("sample-{:03}.wav", i + 1));
        }
        let dataset = root.join("dataset.json");
        fs::write(&dataset, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let mut config = Config::default();
        config.backend.runtime = Runtime::Openvino;
        config.backend.kind = "openvino-genai".into();
        config.backend.device = "cpu".into();
        config.wake_words[0].aliases.push("old spelling".into());
        config.save(&root.join("candidate.toml")).unwrap();
        (root, paths, config, dataset)
    }
    fn args(dataset: PathBuf, apply: bool) -> TrainArgs {
        TrainArgs {
            id: "computer".into(),
            dataset: Some(dataset),
            reuse_recordings: false,
            engine: None,
            apply,
            keep_recordings: true,
            json: false,
        }
    }
    fn load(_: &Config, _: &AppPaths, _: Arc<AtomicBool>) -> Result<Box<dyn EmbeddingSession>> {
        Ok(Box::new(FakeEmbeddingSession::new("test-encoder")))
    }
    #[test]
    fn preview_then_activation_preserves_word_and_reusable_dataset() {
        let (root, paths, config, dataset) = fixture();
        let file = root.join("candidate.toml");
        let before = fs::read(&file).unwrap();
        run_with(
            args(dataset.clone(), false),
            config.clone(),
            &file,
            &paths,
            load,
        )
        .unwrap();
        assert_eq!(fs::read(&file).unwrap(), before);
        run_with(args(dataset, true), config.clone(), &file, &paths, load).unwrap();
        let saved = Config::load(&file).unwrap();
        let word = &saved.wake_words[0];
        assert_eq!(word.command, config.wake_words[0].command);
        assert_eq!(word.aliases, ["old spelling"]);
        assert!(word.uses_trained_head());
        let head =
            artifact::load(&word.enrollment.as_ref().unwrap().heads["test-encoder"]).unwrap();
        assert!(head.locally_validated());
        assert!(saved.engines.contains_key(word.engine.as_ref().unwrap()));
        let mut reuse = args(root.join("unused"), false);
        reuse.dataset = None;
        reuse.reuse_recordings = true;
        let mut missing = args(root.join("unused"), false);
        missing.dataset = None;
        missing.reuse_recordings = true;
        missing.id = "no-recordings".into();
        assert!(
            dataset_path(&missing, &paths)
                .unwrap_err()
                .to_string()
                .contains("no retained labeled recordings")
        );
        let selected = dataset_path(&reuse, &paths).unwrap();
        assert!(selected.is_file());
        run_with(reuse, saved.clone(), &file, &paths, load).unwrap();
        let sessions = crate::enrollment::recordings(&paths, "computer").unwrap();
        assert_eq!(sessions.len(), 2);
        let retained = sessions
            .iter()
            .find_map(|s| {
                Dataset::load(&s.directory.join("manifest.json"))
                    .ok()
                    .filter(|d| d.training.iter().all(|r| r.audio.is_file()))
            })
            .unwrap();
        assert!(retained.training.iter().all(|s| s.audio.is_file()));
        let mut changed = saved.clone();
        changed.backend.kind = "different-default".into();
        assert_eq!(
            changed
                .for_engine(word.engine.as_deref())
                .unwrap()
                .backend
                .kind,
            "openvino-genai"
        );
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn contaminated_splits_and_concurrent_edits_never_activate() {
        let (root, paths, config, dataset) = fixture();
        let file = root.join("candidate.toml");
        let before = fs::read(&file).unwrap();
        let mut data = Dataset::load(&dataset).unwrap();
        data.calibration[0].audio = data.training[0].audio.clone();
        fs::write(&dataset, serde_json::to_vec(&data).unwrap()).unwrap();
        assert!(
            run_with(
                args(dataset.clone(), true),
                config.clone(),
                &file,
                &paths,
                load
            )
            .unwrap_err()
            .to_string()
            .contains("disjoint")
        );
        assert_eq!(fs::read(&file).unwrap(), before);
        fs::remove_dir_all(root).unwrap();
        let (root, paths, config, dataset) = fixture();
        let file = root.join("candidate.toml");
        let changed = config.clone();
        let concurrent = file.clone();
        let result = run_with(
            args(dataset, true),
            config,
            &file,
            &paths,
            move |c, p, f| {
                let mut c2 = changed.clone();
                c2.wake_words[0].phrase = "Changed elsewhere".into();
                c2.save(&concurrent)?;
                load(c, p, f)
            },
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("configuration changed")
        );
        assert_eq!(
            Config::load(&file).unwrap().wake_words[0].phrase,
            "Changed elsewhere"
        );
        assert!(!paths.data_dir.join("heads").exists());
        assert_eq!(
            fs::read_dir(paths.cache_dir.join("onboarding"))
                .unwrap()
                .count(),
            0
        );
        fs::remove_dir_all(root).unwrap();
    }
}
