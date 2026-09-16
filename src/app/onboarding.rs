use super::*;
use crate::enrollment::{self, AliasReview, Cancellation, SampleSet};
use crate::setup::wizard::{MenuItem, select};

#[derive(clap::Args)]
pub(super) struct OnboardArgs {
    /// Existing wake-word ID, or a new ID with --phrase.
    pub id: Option<String>,
    #[arg(long)]
    pub phrase: Option<String>,
    /// Named engine profile; omit to use the word's current engine.
    #[arg(long)]
    pub engine: Option<String>,
    /// Import a WAV example instead of opening the microphone (repeatable).
    #[arg(long = "audio")]
    pub audio: Vec<PathBuf>,
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u16).range(1..=64))]
    pub samples: u16,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u64).range(1..=15))]
    pub seconds: u64,
    /// Retain private local recording copies for future retraining.
    #[arg(long)]
    pub keep_recordings: bool,
    /// Approve an observed transcript variant in noninteractive mode (repeatable).
    #[arg(long = "accept-alias", requires = "apply")]
    pub accepted: Vec<String>,
    /// Save the reviewed word and explicitly accepted aliases.
    #[arg(long)]
    pub apply: bool,
    /// Preview/result JSON; requires file input and never opens a microphone.
    #[arg(long)]
    pub json: bool,
    /// Action argument vector for a new word; never executed during onboarding.
    #[arg(last = true)]
    pub command: Vec<String>,
}

pub(super) fn run(
    mut args: OnboardArgs,
    mut config: Config,
    path: &Path,
    paths: &AppPaths,
) -> Result<()> {
    let interactive = setup_is_interactive() && !args.json && !args.apply;
    anyhow::ensure!(
        !args.audio.is_empty() || interactive,
        "microphone onboarding needs an interactive terminal; use --audio FILE for file-only onboarding"
    );
    anyhow::ensure!(
        args.audio.len() <= 64,
        "at most 64 onboarding examples may be imported"
    );
    let original = config_snapshot(path)?;
    let id = if let Some(id) = args.id {
        id
    } else {
        anyhow::ensure!(interactive, "specify a wake-word ID");
        let mut items: Vec<_> = config
            .wake_words
            .iter()
            .map(|word| MenuItem::available(&word.id, &word.phrase))
            .collect();
        items.push(MenuItem::available(
            "Add a new wake word",
            "Enter a phrase and action, then record its spellings.",
        ));
        let selected = select(
            "Wake-word onboarding",
            "Choose a word to teach. No actions run during onboarding.",
            &items,
            0,
        )?
        .context("onboarding cancelled; configuration unchanged")?;
        if selected == config.wake_words.len() {
            let id = text_input("New word ID (lowercase letters, numbers and hyphens)")?;
            args.phrase = Some(text_input("Wake phrase (as you intend to say it)")?);
            let action =
                text_input("Action arguments as a JSON array, e.g. [\"notify-send\",\"Ready\"]")?;
            args.command = serde_json::from_str(&action)
                .context("action must be a JSON array of argument strings")?;
            anyhow::ensure!(
                !args.command.is_empty(),
                "action must contain an executable"
            );
            id
        } else {
            config.wake_words[selected].id.clone()
        }
    };
    let index = match config.wake_words.iter().position(|word| word.id == id) {
        Some(index) => {
            anyhow::ensure!(
                args.phrase.is_none() && args.command.is_empty(),
                "existing word: change its phrase/action separately before onboarding"
            );
            index
        }
        None => {
            let phrase = args
                .phrase
                .context("new wake words require --phrase; supply action arguments after --")?;
            let command = if args.command.is_empty() {
                vec!["notify-send".into(), format!("Wake word heard: {phrase}")]
            } else {
                args.command
            };
            config.wake_words.push(WakeWord {
                engine: None,
                enrollment: None,
                id: id.clone(),
                phrase,
                aliases: Vec::new(),
                enabled: true,
                command,
            });
            config.wake_words.len() - 1
        }
    };
    validate_wake_words(&config.wake_words)?;
    let mut selected = select_engine(&mut config, index, args.engine.as_deref(), interactive)?;
    // A sample is observed through exactly one chosen profile. Engine routing
    // must not let another word's transcriber contribute an alias here.
    selected.wake_words = vec![config.wake_words[index].clone()];
    selected.wake_words[0].enabled = true;
    selected.wake_words[0].engine = None;
    if let Some(enrollment) = &mut selected.wake_words[0].enrollment {
        enrollment.active = false;
    }
    let detector = Detector::load(&selected, paths)
        .context("initialize onboarding engine; run setup for its model/runtime first")?;
    let cancellation = Cancellation::new()?;
    let mut samples = SampleSet::create(paths)?;
    if args.audio.is_empty() {
        collect_recordings(
            &mut samples,
            &config.wake_words[index].phrase,
            args.samples,
            args.seconds,
            &cancellation,
            |title, help| confirm(title, help, "Record", "Cancel", 0),
            || enrollment::record(&selected, args.seconds, &cancellation),
        )?;
    } else {
        for audio in &args.audio {
            cancellation.check()?;
            samples
                .import(audio)
                .with_context(|| format!("import {}", audio.display()))?;
        }
    }
    cancellation.check()?;
    let review = enrollment::review(&config.wake_words[index], samples.observe(&detector)?);
    cancellation.check()?;
    let mut accepted = args.accepted;
    if interactive {
        for proposal in &review.proposals {
            let items = [
                MenuItem::available(
                    "Accept this spelling",
                    "Treat this exact transcript as the selected wake phrase",
                ),
                MenuItem::available("Skip", "Do not accept this transcript"),
            ];
            let selection = select(&format!("Heard: {:?}", proposal.text), &format!("Observed in {} recording(s). Approve only a spelling of {:?}; skip unrelated or hallucinated text.", proposal.samples.len(), review.phrase), &items, 1)?.context("onboarding cancelled; configuration unchanged")?;
            if selection == 0 {
                accepted.push(proposal.text.clone());
            }
        }
    }
    enrollment::apply_aliases(&mut config.wake_words[index], &review, &accepted)?;
    validate_wake_words(&config.wake_words)?;
    if interactive {
        let choices = [
            MenuItem::available(
                "Use reviewed Whisper spellings",
                "Recognize this phrase using transcript aliases",
            ),
            MenuItem::available(
                "Train this wake phrase",
                "Experimental: collect 10 wake-phrase and 10 other-speech examples; review before activation",
            ),
        ];
        let mode = select(
            "Recognition method",
            "Use spellings first. Train a head if transcription is not reliable enough.",
            &choices,
            0,
        )?
        .context("onboarding cancelled; configuration unchanged")?;
        if mode == 1 {
            drop(detector);
            return train_guided(
                config,
                index,
                path,
                paths,
                original,
                selected,
                samples,
                args.seconds,
                args.keep_recordings,
                &cancellation,
                &NativeTrainingInteraction,
            );
        }
    }
    let keep = choose_retention(interactive, args.keep_recordings)?;
    if interactive {
        confirm(
            "Apply wake-word onboarding",
            &format!(
                "Word: {:?}\nRecognition: transcript aliases (previous trained heads retained)\nApproved new spellings: {}\nAction: {:?}\nRetain recordings: {}",
                review.phrase,
                accepted.len(),
                config.wake_words[index].command,
                keep
            ),
            "Apply",
            "Cancel",
            0,
        )?;
    }
    cancellation.check()?;
    let apply = args.apply || interactive;
    let mut retained = None;
    if apply {
        anyhow::ensure!(
            config_snapshot(path)? == original,
            "configuration changed during onboarding; rerun to review the new configuration"
        );
        if keep {
            retained = Some(samples.retain(paths, &id, &serde_json::json!({"format_version":1,"mode":"transcript","review":review,"accepted_aliases":accepted,"backend":selected.backend,"model":selected.model}))?);
        }
        if let Err(error) = save_and_reload_active(config, path, paths) {
            if let Some(directory) = &retained {
                let _ = fs::remove_dir_all(directory);
            }
            return Err(error);
        }
    }
    print_result(&review, &accepted, apply, retained.as_deref(), args.json)
}

trait TrainingInteraction {
    fn confirm(&self, title: &str, help: &str, accept: &str, cancel: &str) -> Result<()>;
    fn record(&self, config: &Config, seconds: u64, cancel: &Cancellation) -> Result<Vec<f32>>;
    fn retention(&self, requested: bool) -> Result<bool>;
    fn load(
        &self,
        config: &Config,
        paths: &AppPaths,
        cancel: Arc<AtomicBool>,
    ) -> Result<Box<dyn crate::engine::embedding_worker::EmbeddingSession>>;
}
struct NativeTrainingInteraction;
impl TrainingInteraction for NativeTrainingInteraction {
    fn confirm(&self, title: &str, help: &str, accept: &str, cancel: &str) -> Result<()> {
        confirm(title, help, accept, cancel, 0)
    }
    fn record(&self, config: &Config, seconds: u64, cancel: &Cancellation) -> Result<Vec<f32>> {
        enrollment::record(config, seconds, cancel)
    }
    fn retention(&self, requested: bool) -> Result<bool> {
        choose_retention(true, requested)
    }
    fn load(
        &self,
        config: &Config,
        paths: &AppPaths,
        cancel: Arc<AtomicBool>,
    ) -> Result<Box<dyn crate::engine::embedding_worker::EmbeddingSession>> {
        crate::engine::embedding_worker::load_session(config, paths, cancel)
    }
}

#[allow(clippy::too_many_arguments)]
fn train_guided(
    config: Config,
    index: usize,
    path: &Path,
    paths: &AppPaths,
    original: Option<Vec<u8>>,
    selected: Config,
    mut positives: SampleSet,
    seconds: u64,
    keep_recordings: bool,
    cancellation: &Cancellation,
    ui: &impl TrainingInteraction,
) -> Result<()> {
    use crate::enrollment::artifact::Dataset;
    use std::os::unix::fs::OpenOptionsExt;
    // Check the actual encoder before asking for more microphone recordings.
    // Onboarding aliases are only a draft until the complete operation applies.
    drop(ui.load(
        &selected, paths, Arc::clone(&cancellation.flag),
    ).context("initialize training encoder; select a configured OpenVINO Whisper base.en profile for trained onboarding")?);
    let phrase = &config.wake_words[index].phrase;
    let total = positives.files.len().max(10);
    ui.confirm(
        "Collect training examples",
        &format!(
            "Phrase: {phrase:?}\nReuse {} wake-phrase recording(s), then collect {} more and {total} other-speech recordings.\nOther speech must NOT contain the wake phrase. Include similar-sounding phrases and everyday speech.\nSeparate recordings are reserved for calibration and validation. No actions will run.",
            positives.files.len(),
            total - positives.files.len()
        ),
        "Continue",
        "Cancel",
    )?;
    let remaining = (total - positives.files.len()) as u16;
    collect_recordings(
        &mut positives,
        phrase,
        remaining,
        seconds,
        cancellation,
        |title, help| ui.confirm(title, help, "Record", "Cancel"),
        || ui.record(&selected, seconds, cancellation),
    )?;
    let mut negatives = SampleSet::create(paths)?;
    collect_speech(
        &mut negatives,
        phrase,
        total as u16,
        seconds,
        false,
        cancellation,
        |title, help| ui.confirm(title, help, "Record", "Cancel"),
        || ui.record(&selected, seconds, cancellation),
    )?;
    let dataset: Dataset = split_recordings(&positives.files, &negatives.files)?;
    // The manifest lives alongside the owned temporary samples and is removed
    // with them, including on cancellation and native/training errors.
    let manifest = positives.files[0]
        .parent()
        .context("missing sample directory")?
        .join("dataset.json");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&manifest)?;
    serde_json::to_writer(&mut file, &dataset)?;
    drop(file);
    let keep = ui.retention(keep_recordings)?;
    let id = config.wake_words[index].id.clone();
    let action = config.wake_words[index].command.clone();
    let phrase = phrase.clone();
    // Keep samples alive until the shared training transaction has finished.
    let result = training::run_reviewed(
        training::TrainArgs {
            id,
            dataset: Some(manifest),
            reuse_recordings: false,
            engine: None,
            apply: true,
            keep_recordings: keep,
            json: false,
        },
        config,
        path,
        paths,
        original,
        |config, paths, cancel| ui.load(config, paths, cancel),
        |head| {
            let v = head
                .validation
                .as_ref()
                .context("missing held-out validation")?;
            ui.confirm(
                "Apply trained wake word",
                &format!(
                    "Phrase: {phrase:?}\nHeld-out positives: {}; misses: {}\nHeld-out negatives: {}; false activations: {}\nAction: {action:?}\nRetain recordings: {keep}\nThis is a local check, not a measured background false-activation rate.",
                    v.positives, v.misses, v.negatives, v.false_activations
                ),
                "Apply",
                "Cancel",
            )
        },
    );
    result.context("trained onboarding did not complete; existing configuration unchanged. Review the error before collecting a fresh dataset")
}

/// Allocate recordings by capture order before fitting: first 60% for training,
/// next 20% for calibration, final 20% held out. Never copy a clip between roles.
fn split_recordings(
    positives: &[PathBuf],
    negatives: &[PathBuf],
) -> Result<crate::enrollment::artifact::Dataset> {
    use crate::enrollment::artifact::{Dataset, LabeledRecording};
    anyhow::ensure!(
        positives.len() >= 10 && negatives.len() >= 10,
        "guided training needs at least 10 wake-phrase and 10 other-speech recordings"
    );
    let mut splits = [Vec::new(), Vec::new(), Vec::new()];
    for (files, positive) in [(positives, true), (negatives, false)] {
        let reserved = files.len() / 5;
        let training = files.len() - 2 * reserved;
        for (i, audio) in files.iter().enumerate() {
            let split = if i < training {
                0
            } else if i < training + reserved {
                1
            } else {
                2
            };
            splits[split].push(LabeledRecording {
                audio: audio.clone(),
                positive,
            });
        }
    }
    let [training, calibration, validation] = splits;
    Ok(Dataset {
        training,
        calibration,
        validation,
    })
}

fn choose_retention(interactive: bool, requested: bool) -> Result<bool> {
    if interactive && !requested {
        let items = [
            MenuItem::available(
                "Discard recordings after onboarding",
                "Keep the word and approved aliases; future retraining needs new samples",
            ),
            MenuItem::available(
                "Keep recordings locally",
                "Keep the labeled samples even if training fails, for diagnosis or retraining",
            ),
        ];
        Ok(select(
            "Enrollment recordings",
            "Recordings never leave this machine. Retention is optional.",
            &items,
            0,
        )?
        .context("onboarding cancelled; configuration unchanged")?
            == 1)
    } else {
        Ok(requested)
    }
}

fn select_engine(
    config: &mut Config,
    index: usize,
    engine: Option<&str>,
    interactive: bool,
) -> Result<Config> {
    let mut chosen = config.wake_words[index].engine.clone();
    if let Some(name) = engine {
        chosen = (name != "default").then(|| name.to_owned());
    } else if interactive && !config.engines.is_empty() {
        let mut names = vec![None];
        names.extend(config.engines.keys().cloned().map(Some));
        let items: Vec<_> = names
            .iter()
            .map(|name| {
                let (backend, model) = match name {
                    None => (&config.backend, &config.model),
                    Some(name) => (&config.engines[name].backend, &config.engines[name].model),
                };
                MenuItem::available(
                    name.as_deref().unwrap_or("default"),
                    format!("{} / {} / {}", backend.kind, backend.device, model.name),
                )
            })
            .collect();
        let preferred = names.iter().position(|name| *name == chosen).unwrap_or(0);
        let selection = select(
            "Recognition engine",
            "Choose the transcriber that will recognize this word's spellings.",
            &items,
            preferred,
        )?
        .context("onboarding cancelled; configuration unchanged")?;
        chosen = names[selection].clone();
    }
    config.wake_words[index].engine = chosen.clone();
    if let Some(enrollment) = &mut config.wake_words[index].enrollment {
        enrollment.active = false;
    }
    config.for_engine(chosen.as_deref())
}

fn confirm(title: &str, help: &str, accept: &str, cancel: &str, preferred: usize) -> Result<()> {
    let items = [
        MenuItem::available(accept, ""),
        MenuItem::available(cancel, "Leave the existing configuration unchanged"),
    ];
    anyhow::ensure!(
        select(title, help, &items, preferred)? == Some(0),
        "onboarding cancelled; configuration unchanged"
    );
    Ok(())
}

fn print_result(
    review: &AliasReview,
    accepted: &[String],
    applied: bool,
    retained: Option<&Path>,
    json: bool,
) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"applied":applied,"review":review,"accepted_aliases":accepted,"recordings":retained,"actions_executed":false})
            )?
        );
    } else {
        println!(
            "{}: {}",
            if applied {
                "Applied onboarding"
            } else {
                "Onboarding preview; config unchanged"
            },
            review.wake_word
        );
        for observation in &review.observations {
            println!(
                "  example {}: {:?}",
                observation.sample, observation.transcripts
            );
        }
        for proposal in &review.proposals {
            println!(
                "  spelling {:?}: {} recording(s)",
                proposal.text,
                proposal.samples.len()
            );
        }
        if let Some(path) = retained {
            println!("Recordings: {}", path.display());
        }
        if !applied {
            println!("Rerun with --apply and --accept-alias TEXT for each spelling you approve.");
        }
    }
    Ok(())
}

pub(super) fn guided(config: Config, path: &Path, paths: &AppPaths) -> Result<()> {
    run(
        OnboardArgs {
            id: None,
            phrase: None,
            engine: None,
            audio: Vec::new(),
            samples: 5,
            seconds: 3,
            keep_recordings: false,
            accepted: Vec::new(),
            apply: false,
            json: false,
            command: Vec::new(),
        },
        config,
        path,
        paths,
    )
}

fn text_input(label: &str) -> Result<String> {
    eprintln!("{label} (blank cancels):");
    let mut input = String::new();
    std::io::stdin().lock().take(4097).read_line(&mut input)?;
    anyhow::ensure!(input.len() <= 4096, "onboarding input is too long");
    let input = input.trim().to_owned();
    anyhow::ensure!(
        !input.is_empty(),
        "onboarding cancelled; configuration unchanged"
    );
    Ok(input)
}

#[allow(clippy::too_many_arguments)]
fn collect_recordings(
    samples: &mut SampleSet,
    phrase: &str,
    count: u16,
    seconds: u64,
    cancellation: &Cancellation,
    ask: impl FnMut(&str, &str) -> Result<()>,
    record: impl FnMut() -> Result<Vec<f32>>,
) -> Result<()> {
    collect_speech(
        samples,
        phrase,
        count,
        seconds,
        true,
        cancellation,
        ask,
        record,
    )
}

#[allow(clippy::too_many_arguments)]
fn collect_speech(
    samples: &mut SampleSet,
    phrase: &str,
    count: u16,
    seconds: u64,
    positive: bool,
    cancellation: &Cancellation,
    mut ask: impl FnMut(&str, &str) -> Result<()>,
    mut record: impl FnMut() -> Result<Vec<f32>>,
) -> Result<()> {
    for sample in 0..count {
        loop {
            cancellation.check()?;
            ask(
                &format!(
                    "Record {}example {} of {}",
                    if positive { "" } else { "other-speech " },
                    sample + 1,
                    count
                ),
                &if positive {
                    format!(
                        "Say {phrase:?} naturally. Recording lasts {seconds} seconds. Vary pace and distance between examples."
                    )
                } else {
                    format!(
                        "Say one {} WITHOUT {phrase:?}. Speak naturally for one utterance; recording lasts {seconds} seconds. Use a different phrase each time.",
                        if sample % 2 == 0 {
                            "similar-sounding phrase"
                        } else {
                            "ordinary everyday phrase"
                        }
                    )
                },
            )?;
            eprintln!("Recording example {}…", sample + 1);
            match record().and_then(|audio| samples.push(&audio)) {
                Ok(()) => break,
                Err(error) => {
                    cancellation.check()?;
                    ask("Recording needs another try", &error.to_string())?;
                }
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    struct TestInteraction {
        recordings: Cell<usize>,
        screens: RefCell<Vec<String>>,
        cancel_at: Option<&'static str>,
        keep: bool,
        encoder_error: bool,
        invalid_negatives: bool,
    }
    impl TrainingInteraction for TestInteraction {
        fn confirm(&self, title: &str, help: &str, _: &str, _: &str) -> Result<()> {
            self.screens.borrow_mut().push(format!("{title} {help}"));
            anyhow::ensure!(self.cancel_at != Some(title), "cancelled by user");
            Ok(())
        }
        fn record(&self, _: &Config, _: u64, _: &Cancellation) -> Result<Vec<f32>> {
            let n = self.recordings.get();
            self.recordings.set(n + 1);
            let positive = n < 5 || self.invalid_negatives;
            Ok(vec![
                (0.11 + n as f32 * 0.001)
                    * if positive { 1.0 } else { -1.0 };
                4000
            ])
        }
        fn retention(&self, requested: bool) -> Result<bool> {
            Ok(requested || self.keep)
        }
        fn load(
            &self,
            _: &Config,
            _: &AppPaths,
            _: Arc<AtomicBool>,
        ) -> Result<Box<dyn crate::engine::embedding_worker::EmbeddingSession>> {
            anyhow::ensure!(!self.encoder_error, "encoder unavailable");
            Ok(Box::new(crate::test_support::FakeEmbeddingSession::new(
                "guided-encoder",
            )))
        }
    }
    fn test_interaction() -> TestInteraction {
        TestInteraction {
            recordings: Cell::new(0),
            screens: RefCell::new(Vec::new()),
            cancel_at: None,
            keep: false,
            encoder_error: false,
            invalid_negatives: false,
        }
    }
    fn training_fixture() -> (PathBuf, AppPaths, Config, SampleSet) {
        let root = crate::test_support::unique_directory("guided-training", "capture");
        let paths = crate::test_support::isolated_paths(&root);
        let mut config = Config::default();
        config.backend.runtime = Runtime::Openvino;
        config.backend.kind = "openvino-genai".into();
        config.backend.device = "cpu".into();
        config.save(&root.join("config.toml")).unwrap();
        let mut samples = SampleSet::create(&paths).unwrap();
        for i in 0..5 {
            samples.push(&vec![0.10 + i as f32 * 0.001; 4000]).unwrap();
        }
        (root, paths, config, samples)
    }
    #[test]
    fn guided_training_reuses_samples_and_applies_only_after_review() {
        for keep in [false, true] {
            let (root, paths, mut config, samples) = training_fixture();
            let file = root.join("config.toml");
            let original = config_snapshot(&file).unwrap();
            let mut ui = test_interaction();
            ui.keep = keep;
            config.wake_words[0]
                .aliases
                .push("reviewed spelling".into());
            let action = config.wake_words[0].command.clone();
            train_guided(
                config.clone(),
                0,
                &file,
                &paths,
                original,
                config,
                samples,
                3,
                false,
                &Cancellation::new().unwrap(),
                &ui,
            )
            .unwrap();
            assert_eq!(ui.recordings.get(), 15); // Reuse five positives, collect five more and ten negatives.
            let saved = Config::load(&file).unwrap();
            assert!(saved.wake_words[0].uses_trained_head());
            assert_eq!(saved.wake_words[0].command, action);
            assert_eq!(saved.wake_words[0].aliases, ["reviewed spelling"]);
            let screens = ui.screens.borrow();
            assert!(
                screens
                    .iter()
                    .any(|s| s.contains("similar-sounding phrase WITHOUT"))
            );
            assert!(
                screens
                    .iter()
                    .any(|s| s.contains("ordinary everyday phrase WITHOUT"))
            );
            assert!(
                screens
                    .last()
                    .unwrap()
                    .contains("Held-out positives: 2; misses: 0")
            );
            let sessions = enrollment::recordings(&paths, "computer").unwrap();
            assert_eq!(sessions.len(), usize::from(keep));
            if keep {
                let data = crate::enrollment::artifact::Dataset::load(
                    &sessions[0].directory.join("manifest.json"),
                )
                .unwrap();
                assert_eq!(
                    (
                        data.training.len(),
                        data.calibration.len(),
                        data.validation.len()
                    ),
                    (12, 4, 4)
                );
                assert!(
                    data.training
                        .iter()
                        .chain(&data.calibration)
                        .chain(&data.validation)
                        .all(|r| r.audio.is_file())
                );
            }
            assert_eq!(
                fs::read_dir(paths.cache_dir.join("onboarding"))
                    .unwrap()
                    .count(),
                0
            );
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn guided_training_cancellation_or_failure_never_saves_drafts() {
        for case in 0..5 {
            let (root, paths, mut config, samples) = training_fixture();
            let file = root.join("config.toml");
            let original = config_snapshot(&file).unwrap();
            config.wake_words[0].aliases.push("unsaved alias".into());
            let mut ui = test_interaction();
            ui.keep = true;
            match case {
                0 => ui.cancel_at = Some("Collect training examples"),
                1 => ui.cancel_at = Some("Record other-speech example 2 of 10"),
                2 => ui.cancel_at = Some("Apply trained wake word"),
                3 => ui.encoder_error = true,
                _ => ui.invalid_negatives = true,
            }
            assert!(
                train_guided(
                    config.clone(),
                    0,
                    &file,
                    &paths,
                    original.clone(),
                    config,
                    samples,
                    3,
                    false,
                    &Cancellation::new().unwrap(),
                    &ui
                )
                .is_err()
            );
            assert_eq!(config_snapshot(&file).unwrap(), original);
            assert!(!paths.data_dir.join("heads").exists());
            let sessions = enrollment::recordings(&paths, "computer").unwrap();
            assert_eq!(sessions.len(), usize::from(case == 4));
            if case == 4 {
                let dataset = crate::enrollment::artifact::Dataset::load(
                    &sessions[0].directory.join("manifest.json"),
                )
                .unwrap();
                assert!(dataset.validation.iter().all(|r| r.audio.is_file()));
            }
            assert_eq!(
                fs::read_dir(paths.cache_dir.join("onboarding"))
                    .unwrap()
                    .count(),
                0
            );
            if case == 0 || case == 3 {
                assert_eq!(ui.recordings.get(), 0);
            }
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn generated_splits_are_balanced_disjoint_and_preserve_all_examples() {
        for count in [10, 11, 20, 64] {
            let pos: Vec<_> = (0..count)
                .map(|i| PathBuf::from(format!("positive-{i}")))
                .collect();
            let neg: Vec<_> = (0..count)
                .map(|i| PathBuf::from(format!("negative-{i}")))
                .collect();
            let d = split_recordings(&pos, &neg).unwrap();
            let mut seen = BTreeSet::new();
            for split in [&d.training, &d.calibration, &d.validation] {
                assert_eq!(split.iter().filter(|r| r.positive).count(), split.len() / 2);
                assert!(split.len() >= 4);
                for item in split {
                    assert!(seen.insert(&item.audio));
                }
            }
            assert_eq!(seen.len(), count * 2);
        }
        assert!(split_recordings(&[], &[]).is_err());
    }
    #[test]
    fn guided_recording_requires_a_terminal_before_loading_a_model() {
        if setup_is_interactive() {
            return;
        } // Never open a microphone when tests run in a terminal.
        let root = crate::test_support::unique_directory("onboarding", "no-terminal");
        let paths = crate::test_support::isolated_paths(&root);
        let result = guided(Config::default(), &root.join("config.toml"), &paths);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("interactive terminal")
        );
        assert!(!paths.cache_dir.exists());
        assert!(!root.join("config.toml").exists());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn low_quality_clips_are_retried_and_cancel_does_not_collect_more() {
        let root = crate::test_support::unique_directory("onboarding-record", "retry");
        let paths = crate::test_support::isolated_paths(&root);
        let cancel = Cancellation::new().unwrap();
        let mut samples = SampleSet::create(&paths).unwrap();
        let mut calls = 0;
        let mut screens = Vec::new();
        collect_recordings(
            &mut samples,
            "unusual",
            2,
            3,
            &cancel,
            |t, _| {
                screens.push(t.to_owned());
                Ok(())
            },
            || {
                calls += 1;
                Ok(vec![if calls == 1 { 0.0 } else { 0.1 }; 4000])
            },
        )
        .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(samples.files.len(), 2);
        assert!(screens.iter().any(|t| t.contains("another try")));
        let result = collect_recordings(
            &mut samples,
            "unusual",
            2,
            3,
            &cancel,
            |_, _| anyhow::bail!("cancel"),
            || panic!("cancel must not record"),
        );
        assert!(result.is_err());
        assert_eq!(samples.files.len(), 2);
        drop(samples);
        fs::remove_dir_all(root).unwrap();
    }
}
