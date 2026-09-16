use super::*;
use crate::engine::{
    audio::read_wave,
    embedding_worker::{EmbeddingSession, load_session},
};
use crate::enrollment::{
    Cancellation, SampleSet,
    artifact::{self, Dataset},
    head::{Example, Head},
};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(super) enum EncoderDevice {
    Cpu,
    #[value(alias = "igpu")]
    Gpu,
    Npu,
}
impl EncoderDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Npu => "npu",
        }
    }
    pub fn from_config(config: &Config) -> Result<Self> {
        match config.backend.device.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" | "igpu" => Ok(Self::Gpu),
            "npu" => Ok(Self::Npu),
            other => bail!("unsupported trained encoder device {other}"),
        }
    }
}

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
    /// Source model/runtime profile; defaults to the word's current profile.
    #[arg(long)]
    pub engine: Option<String>,
    /// Device for extracting training features; classifier fitting stays on CPU.
    #[arg(long, value_enum, default_value = "cpu")]
    pub training_device: EncoderDevice,
    /// Where the finished detector runs; defaults to the source profile's device.
    #[arg(long, value_enum)]
    pub run_device: Option<EncoderDevice>,
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
    config: Config,
    path: &Path,
    paths: &AppPaths,
    load: impl FnMut(&Config, &AppPaths, Arc<AtomicBool>) -> Result<Box<dyn EmbeddingSession>>,
) -> Result<()> {
    let original = config_snapshot(path)?;
    run_reviewed(args, config, path, paths, original, load, |_| Ok(()))
}

/// Review the validated candidate before any artifact, recording, or config is saved.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_reviewed(
    args: TrainArgs,
    mut config: Config,
    path: &Path,
    paths: &AppPaths,
    original: Option<Vec<u8>>,
    mut load: impl FnMut(&Config, &AppPaths, Arc<AtomicBool>) -> Result<Box<dyn EmbeddingSession>>,
    review: impl FnOnce(&Head) -> Result<()>,
) -> Result<()> {
    validate_wake_words(&config.wake_words)?;
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
    let mut deployment = config.for_engine(selected_engine)?;
    if let Some(device) = args.run_device {
        deployment.backend.device = device.as_str().into();
    }
    EncoderDevice::from_config(&deployment)?;
    let mut selected = deployment.clone();
    selected.backend.device = args.training_device.as_str().into();
    anyhow::ensure!(
        selected.backend.runtime == Runtime::Openvino,
        "trainable heads currently require a configured raw OpenVINO Whisper encoder profile"
    );
    let dataset_path = dataset_path(&args, paths)?;
    let dataset = Dataset::load(&dataset_path)?;
    let cancellation = Cancellation::new()?;
    let mut samples = SampleSet::create(paths)?;
    let mut prepared = dataset.clone();
    let mut retained_manifest = dataset;
    for (item, retained_item) in prepared
        .training
        .iter_mut()
        .chain(&mut prepared.calibration)
        .chain(&mut prepared.validation)
        .zip(
            retained_manifest
                .training
                .iter_mut()
                .chain(&mut retained_manifest.calibration)
                .chain(&mut retained_manifest.validation),
        )
    {
        cancellation.check()?;
        samples.import(&item.audio)?;
        item.audio = samples
            .files
            .last()
            .context("missing imported recording")?
            .clone();
        retained_item.audio = PathBuf::from(item.audio.file_name().context("missing sample name")?);
    }
    // Explicit retention is independent of model loading, segmentation, fitting,
    // validation, and config activation. Never roll back user-owned recordings.
    let retained = if args.keep_recordings {
        Some(samples.retain(paths, &args.id, &retained_manifest)?)
    } else {
        None
    };
    if !args.json
        && let Some(directory) = &retained
    {
        eprintln!(
            "Saved labeled recordings: {}",
            directory.join("manifest.json").display()
        );
    }
    let result = (|| -> Result<()> {
        let mut worker = load(&selected, paths, Arc::clone(&cancellation.flag))?;
        let contract = worker.contract().to_owned();
        let training_execution_devices = worker.execution_devices().to_owned();
        let mut execution_devices = training_execution_devices.clone();
        let mut examples = Vec::new();
        for (label, split) in [
            ("training", &prepared.training),
            ("calibration", &prepared.calibration),
            ("validation", &prepared.validation),
        ] {
            if label == "calibration" && selected.backend.device != deployment.backend.device {
                drop(worker);
                worker = load(&deployment, paths, Arc::clone(&cancellation.flag))
                    .context("initialize deployment device for calibration and held-out validation; current detector unchanged")?;
                anyhow::ensure!(
                    worker.contract() == contract,
                    "deployment encoder differs from training encoder; current detector unchanged"
                );
                execution_devices = worker.execution_devices().to_owned();
            }
            let mut encoded = Vec::new();
            for (i, item) in split.iter().enumerate() {
                cancellation.check()?;
                if !args.json {
                    eprintln!("Encoding {label} {}/{}…", i + 1, split.len());
                }
                let (_, audio) = read_wave(&item.audio)?;
                let mut hash = Sha256::new();
                for value in &audio {
                    hash.update(value.to_le_bytes());
                }
                let fingerprint = format!("{:x}", hash.finalize());
                // Enrollment and live detection use the identical Silero endpoint
                // and pre/post-roll. Reject ambiguous multi-phrase recordings.
                let embedding = single_utterance(worker.as_mut(), &audio).with_context(|| {
                    format!("{} example {} ({})", label, i + 1, item.audio.display())
                })?;
                anyhow::ensure!(
                    embedding.encoder_contract == contract,
                    "encoder changed during enrollment"
                );
                encoded.push(Example {
                    id: fingerprint,
                    values: embedding.values,
                    positive: item.positive,
                });
            }
            examples.push(encoded);
        }
        cancellation.check()?;
        let mut head = Head::train(&contract, &examples[0], &examples[1])?;
        head.validate_held_out(&examples[2]).with_context(|| {
            let scores = examples[2]
                .iter()
                .zip(&prepared.validation)
                .map(|(sample, recording)| {
                    let score = head.score(&contract, &sample.values).unwrap_or(f32::NAN);
                    format!(
                        "{}: expected {}, score {:.4}, threshold {:.4}",
                        recording
                            .audio
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        if sample.positive {
                            "wake phrase"
                        } else {
                            "other speech"
                        },
                        score,
                        head.threshold
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            format!("Held-out clip scores: {scores}")
        })?;
        cancellation.check()?;
        let mut artifact_path = None;
        if args.apply {
            review(&head)?;
            cancellation.check()?;
            anyhow::ensure!(
                config_snapshot(path)? == original,
                "configuration changed during training; rerun with the updated configuration"
            );
            let directory = paths.data_dir.join("heads").join(&args.id);
            let installed = artifact::install(&directory, &head)?;
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
                        &deployment.backend,
                        &deployment.model
                    ))?)
                )[..12]
            );
            word.engine = Some(profile_id.clone());
            config.engines.insert(
                profile_id,
                crate::config::EngineProfile {
                    backend: deployment.backend.clone(),
                    model: deployment.model.clone(),
                },
            );
            // On failure, immutable heads and explicitly retained recordings remain reusable.
            save_and_reload_active(config, path, paths)?;
            artifact_path = Some(installed);
        }
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"applied":args.apply,"word":args.id,"encoder_contract":contract,"execution_devices":execution_devices,"training_execution_devices":training_execution_devices,"training_device":selected.backend.device,"run_device":deployment.backend.device,"local_validation":head.validation,"head":artifact_path,"recordings":retained,"actions_executed":false,"qualification":"local held-out clips only; not a measured ambient false-activation rate"})
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
            println!(
                "Training encoder: {} ({}); finished detector encoder: {} ({})",
                selected.backend.device,
                training_execution_devices,
                deployment.backend.device,
                execution_devices
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
    })();
    result.map_err(|error| match retained {
        Some(directory) => error.context(format!("Training did not complete; kept your labeled recordings at {} as requested. Configuration activation did not complete", directory.join("manifest.json").display())),
        None => error.context("Training did not complete; additional recording copies were not retained. Supplied input files are unchanged. Choose Keep recordings locally to save a new dataset copy before inference"),
    })
}

pub(super) fn single_utterance(
    worker: &mut dyn EmbeddingSession,
    audio: &[f32],
) -> Result<crate::engine::embedding::Embedding> {
    worker.start()?;
    let mut utterances = Vec::new();
    for chunk in audio.chunks(16_000) {
        utterances.extend(worker.audio(chunk)?);
    }
    utterances.extend(worker.finish()?);
    anyhow::ensure!(
        utterances.len() == 1,
        "VAD found {} speech segments; expected one. A pause, repeated phrase, or background voice can split a recording. Retry only this clip with one continuous phrase",
        utterances.len()
    );
    Ok(utterances
        .pop()
        .context("missing speech embedding")?
        .embedding)
}

/// Durable human checkpoint before optional external augmentation begins.
pub(super) fn retain_dataset(paths: &AppPaths, id: &str, dataset: &Dataset) -> Result<PathBuf> {
    let mut owned = SampleSet::create(paths)?;
    let mut manifest = dataset.clone();
    for item in manifest
        .training
        .iter_mut()
        .chain(&mut manifest.calibration)
        .chain(&mut manifest.validation)
    {
        owned.import(&item.audio)?;
        item.audio = PathBuf::from(format!("sample-{:03}.wav", owned.files.len()));
    }
    owned.retain(paths, id, &manifest)
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
    use crate::enrollment::artifact::LabeledRecording;
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
                    generated: None,
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
            training_device: EncoderDevice::Cpu,
            run_device: None,
            apply,
            keep_recordings: true,
            json: false,
        }
    }
    fn load(_: &Config, _: &AppPaths, _: Arc<AtomicBool>) -> Result<Box<dyn EmbeddingSession>> {
        Ok(Box::new(FakeEmbeddingSession::new("test-encoder")))
    }
    #[test]
    fn training_and_deployment_are_independent_and_target_failures_never_activate() {
        for (training_device, run_device, failure) in [
            (EncoderDevice::Cpu, EncoderDevice::Npu, 0),
            (EncoderDevice::Npu, EncoderDevice::Cpu, 0),
            (EncoderDevice::Gpu, EncoderDevice::Npu, 0),
            (EncoderDevice::Cpu, EncoderDevice::Npu, 1),
            (EncoderDevice::Cpu, EncoderDevice::Npu, 2),
            (EncoderDevice::Cpu, EncoderDevice::Npu, 3),
        ] {
            let (root, paths, config, dataset) = fixture();
            let file = root.join("candidate.toml");
            let before = fs::read(&file).unwrap();
            let mut options = args(dataset, true);
            options.training_device = training_device;
            options.run_device = Some(run_device);
            let mut loaded = Vec::new();
            let result = run_with(options, config.clone(), &file, &paths, |candidate, _, _| {
                loaded.push(candidate.backend.device.clone());
                let deployment = loaded.len() == 2;
                anyhow::ensure!(!(deployment && failure == 1), "target unavailable");
                let mut worker = FakeEmbeddingSession::new(if deployment && failure == 2 {
                    "different-encoder"
                } else {
                    "test-encoder"
                });
                worker.inverted = deployment && failure == 3;
                Ok(Box::new(worker))
            });
            assert_eq!(loaded, [training_device.as_str(), run_device.as_str()]);
            if failure == 0 {
                result.unwrap();
                let saved = Config::load(&file).unwrap();
                let profile = saved
                    .engine_profile(saved.wake_words[0].engine.as_deref())
                    .unwrap();
                assert_eq!(profile.backend.device, run_device.as_str());
                assert_eq!(saved.backend, config.backend);
                assert!(saved.wake_words[0].uses_trained_head());
            } else {
                assert!(result.is_err());
                assert_eq!(fs::read(&file).unwrap(), before);
                assert!(!paths.data_dir.join("heads").exists());
            }
            fs::remove_dir_all(root).unwrap();
        }
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
        assert_eq!(sessions.len(), 4);
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
    struct BadSegmentation {
        inner: FakeEmbeddingSession,
        count: usize,
    }
    impl EmbeddingSession for BadSegmentation {
        fn contract(&self) -> &str {
            self.inner.contract()
        }
        fn execution_devices(&self) -> &str {
            "TEST"
        }
        fn start(&mut self) -> Result<()> {
            self.inner.start()
        }
        fn audio(
            &mut self,
            samples: &[f32],
        ) -> Result<Vec<crate::engine::embedding_worker::EncodedUtterance>> {
            self.inner.audio(samples)
        }
        fn finish(&mut self) -> Result<Vec<crate::engine::embedding_worker::EncodedUtterance>> {
            let encoded = self.inner.finish()?;
            let bytes = serde_json::to_vec(&encoded[0])?;
            (0..self.count)
                .map(|_| serde_json::from_slice(&bytes).map_err(Into::into))
                .collect()
        }
    }
    #[test]
    fn complete_dataset_is_saved_before_model_loading_or_segmentation_can_fail() {
        for count in [0, 2, 99] {
            let (root, paths, config, dataset) = fixture();
            let file = root.join("candidate.toml");
            let original = config_snapshot(&file).unwrap();
            let result = run_with(args(dataset, true), config, &file, &paths, |_, paths, _| {
                let sessions = crate::enrollment::recordings(paths, "computer")?;
                assert_eq!(sessions.len(), 2); // Durable snapshot already exists at native load.
                if count == 99 {
                    anyhow::bail!("native load failed");
                }
                Ok(Box::new(BadSegmentation {
                    inner: FakeEmbeddingSession::new("test-encoder"),
                    count,
                }))
            });
            let error = format!("{:#}", result.unwrap_err());
            assert!(error.contains("kept your labeled recordings"), "{error}");
            if count != 99 {
                assert!(error.contains(&format!("VAD found {count} speech segments")));
            }
            assert_eq!(config_snapshot(&file).unwrap(), original);
            assert!(!paths.data_dir.join("heads").exists());
            let mut reuse = args(root.join("unused"), false);
            reuse.dataset = None;
            reuse.reuse_recordings = true;
            let recovered = Dataset::load(&dataset_path(&reuse, &paths).unwrap()).unwrap();
            let clips: Vec<_> = recovered
                .training
                .iter()
                .chain(&recovered.calibration)
                .chain(&recovered.validation)
                .collect();
            assert_eq!(clips.len(), 12);
            assert!(clips.iter().all(|item| item.audio.is_file()));
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn rejected_held_out_clips_are_retained_only_when_requested_and_can_be_reloaded() {
        for keep in [false, true] {
            let (root, paths, config, dataset) = fixture();
            let file = root.join("candidate.toml");
            let before = config_snapshot(&file).unwrap();
            let mut data = Dataset::load(&dataset).unwrap();
            for sample in &mut data.validation {
                sample.positive = !sample.positive;
            }
            fs::write(&dataset, serde_json::to_vec(&data).unwrap()).unwrap();
            let mut options = args(dataset, true);
            options.keep_recordings = keep;
            let error = run_with(options, config.clone(), &file, &paths, load).unwrap_err();
            let diagnostic = format!("{error:#}");
            assert!(diagnostic.contains("missed 2/2"), "{diagnostic}");
            assert!(
                diagnostic.contains("sample-009.wav: expected other speech, score"),
                "{diagnostic}"
            );
            assert!(
                diagnostic.contains("sample-011.wav: expected wake phrase, score"),
                "{diagnostic}"
            );
            assert!(diagnostic.contains("activated on 2/2"), "{diagnostic}");
            assert_eq!(config_snapshot(&file).unwrap(), before);
            assert!(!paths.data_dir.join("heads").exists());
            let sessions = crate::enrollment::recordings(&paths, "computer").unwrap();
            assert_eq!(sessions.len(), 1 + usize::from(keep));
            if keep {
                assert!(diagnostic.contains("kept your labeled recordings"));
                let mut reuse = args(root.join("unused"), false);
                reuse.dataset = None;
                reuse.reuse_recordings = true;
                let manifest = Dataset::load(&dataset_path(&reuse, &paths).unwrap()).unwrap();
                assert!(manifest.validation.iter().all(|r| r.audio.is_file()));
                assert!(
                    format!(
                        "{:#}",
                        run_with(reuse, config, &file, &paths, load).unwrap_err()
                    )
                    .contains("missed 2/2")
                );
            } else {
                assert!(diagnostic.contains("not retained"));
            }
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn review_cannot_overwrite_an_edit_made_during_onboarding_or_final_confirmation() {
        for during_review in [false, true] {
            let (root, paths, config, dataset) = fixture();
            let file = root.join("candidate.toml");
            let original = config_snapshot(&file).unwrap();
            let mut edited = config.clone();
            edited.wake_words[0].phrase = "Updated in another terminal".into();
            if !during_review {
                edited.save(&file).unwrap();
            }
            let result = run_reviewed(
                args(dataset, true),
                config,
                &file,
                &paths,
                original,
                load,
                |head| {
                    assert!(head.locally_validated());
                    assert!(!paths.data_dir.join("heads").exists());
                    if during_review {
                        edited.save(&file)?;
                    }
                    Ok(())
                },
            );
            assert!(
                result
                    .unwrap_err()
                    .root_cause()
                    .to_string()
                    .contains("configuration changed")
            );
            assert_eq!(
                Config::load(&file).unwrap().wake_words[0].phrase,
                edited.wake_words[0].phrase
            );
            assert!(!paths.data_dir.join("heads").exists());
            fs::remove_dir_all(root).unwrap();
        }
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
            .root_cause()
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
                .root_cause()
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
