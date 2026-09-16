use super::*;
use crate::enrollment::{self, AliasReview, Cancellation, SampleSet};
use crate::setup::wizard::{MenuItem, select};

#[derive(clap::Args)]
pub(super) struct OnboardArgs {
    /// Existing wake-word ID, or a new ID with --phrase.
    pub id: Option<String>,
    #[arg(long)]
    pub phrase: Option<String>,
    /// Optional engine profile override; guided onboarding offers the recognition method.
    #[arg(long)]
    pub engine: Option<String>,
    /// Import a WAV example instead of opening the microphone (repeatable).
    #[arg(long = "audio")]
    pub audio: Vec<PathBuf>,
    /// Resume a labeled session; preserve training and record fresh human calibration/validation.
    #[arg(long, conflicts_with_all = ["audio", "json", "apply"])]
    pub dataset: Option<PathBuf>,
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
    // Offer training before microphone capture or transcription, not only after
    // a full alias-review session. File-based alias onboarding stays unchanged.
    let method = if interactive && args.audio.is_empty() && args.dataset.is_none() {
        select(
            "Recognition method",
            "Choose how to recognize this wake phrase. No engine flag is needed.",
            &recognition_methods(),
            0,
        )?
        .context("onboarding cancelled; configuration unchanged")?
    } else {
        0
    };
    let training = method != 0 || args.dataset.is_some();
    let mut selected = if training && args.engine.is_none() {
        select_training_engine(&mut config, index)?
    } else {
        select_engine(&mut config, index, args.engine.as_deref(), interactive)?
    };
    if method != 0 {
        let assistance = method == 2 && assisted::prepare()?;
        return train_guided(
            config,
            index,
            path,
            paths,
            original,
            selected,
            SampleSet::create(paths)?,
            args.seconds,
            args.keep_recordings,
            &Cancellation::new()?,
            &NativeTrainingInteraction {
                assistance,
                resume: None,
            },
        );
    }
    if let Some(dataset_path) = args.dataset {
        anyhow::ensure!(
            interactive,
            "resuming enrollment needs an interactive terminal for fresh validation recordings"
        );
        let dataset = crate::enrollment::artifact::Dataset::load(&dataset_path)?;
        let choice = select("Resume enrollment", "Keep the saved training examples. Collect four fresh wake-phrase and four fresh other-speech clips for calibration and validation.", &[
            MenuItem::available("Use saved training examples", "Train without synthetic augmentation"),
            MenuItem::available("Add Omaspeak examples", "Install Omaspeak if missing; review pronunciations before generation"),
        ], 0)?.context("onboarding cancelled; configuration unchanged")?;
        let assistance = choice == 1 && assisted::prepare()?;
        return train_guided(
            config,
            index,
            path,
            paths,
            original,
            selected,
            SampleSet::create(paths)?,
            args.seconds,
            args.keep_recordings,
            &Cancellation::new()?,
            &NativeTrainingInteraction {
                assistance,
                resume: Some(dataset),
            },
        );
    }
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
            if assisted::executable("omaspeak").is_some() {
                MenuItem::available(
                    "Train with Omaspeak examples",
                    "Optional: approved synthetic voices supplement human training clips",
                )
            } else {
                MenuItem::available(
                    "Install Omaspeak for assisted training",
                    "Optional installation; human-only training remains available",
                )
            },
        ];
        let mode = select(
            "Recognition method",
            "Use spellings first. Train a head if transcription is not reliable enough.",
            &choices,
            0,
        )?
        .context("onboarding cancelled; configuration unchanged")?;
        if mode == 1 || mode == 2 {
            if args.engine.is_none() {
                selected = select_training_engine(&mut config, index)?;
            }
            let assistance = mode == 2 && assisted::prepare()?;
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
                &NativeTrainingInteraction {
                    assistance,
                    resume: None,
                },
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
    fn augment(
        &self,
        _dataset: &mut crate::enrollment::artifact::Dataset,
        _phrase: &str,
        _paths: &AppPaths,
        _cancel: &Cancellation,
        _validate: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<SampleSet> {
        bail!("synthetic assistance unavailable")
    }
    fn resume(&self) -> Option<crate::enrollment::artifact::Dataset> {
        None
    }
    fn assistance(&self) -> bool {
        false
    }
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
struct NativeTrainingInteraction {
    assistance: bool,
    resume: Option<crate::enrollment::artifact::Dataset>,
}
impl TrainingInteraction for NativeTrainingInteraction {
    fn augment(
        &self,
        dataset: &mut crate::enrollment::artifact::Dataset,
        phrase: &str,
        paths: &AppPaths,
        cancel: &Cancellation,
        validate: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<SampleSet> {
        assisted::augment(dataset, phrase, paths, cancel, validate)
    }
    fn resume(&self) -> Option<crate::enrollment::artifact::Dataset> {
        self.resume.clone()
    }
    fn assistance(&self) -> bool {
        self.assistance
    }
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
    let mut worker = ui.load(
        &selected, paths, Arc::clone(&cancellation.flag),
    ).context("initialize training encoder; select a configured OpenVINO Whisper base.en profile for trained onboarding")?;
    let phrase = &config.wake_words[index].phrase;
    let mut negatives = SampleSet::create(paths)?;
    let mut dataset: Dataset = if let Some(mut saved) = ui.resume() {
        ui.confirm("Fresh evaluation recordings", "The old held-out clips have already been examined. Keep training examples, but record four new wake-phrase and four new other-speech clips. Two of each calibrate; two of each remain held out.", "Record fresh examples", "Cancel")?;
        for (positive, samples) in [(true, &mut positives), (false, &mut negatives)] {
            collect_speech(
                samples,
                phrase,
                4,
                seconds,
                positive,
                cancellation,
                |title, help| ui.confirm(title, help, "Record", "Cancel"),
                || {
                    let audio = ui.record(&selected, seconds, cancellation)?;
                    training::single_utterance(worker.as_mut(), &audio)?;
                    Ok(audio)
                },
            )?;
        }
        replace_evaluation(&mut saved, &positives.files, &negatives.files)?;
        saved
    } else {
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
        check_recordings(
            &mut positives,
            phrase,
            cancellation,
            |title, help| ui.confirm(title, help, "Record again", "Cancel"),
            || ui.record(&selected, seconds, cancellation),
            |audio| training::single_utterance(worker.as_mut(), audio).map(|_| ()),
        )?;
        let remaining = (total - positives.files.len()) as u16;
        collect_recordings(
            &mut positives,
            phrase,
            remaining,
            seconds,
            cancellation,
            |title, help| ui.confirm(title, help, "Record", "Cancel"),
            || {
                let audio = ui.record(&selected, seconds, cancellation)?;
                training::single_utterance(worker.as_mut(), &audio)?;
                Ok(audio)
            },
        )?;
        collect_speech(
            &mut negatives,
            phrase,
            total as u16,
            seconds,
            false,
            cancellation,
            |title, help| ui.confirm(title, help, "Record", "Cancel"),
            || {
                let audio = ui.record(&selected, seconds, cancellation)?;
                training::single_utterance(worker.as_mut(), &audio)?;
                Ok(audio)
            },
        )?;
        split_recordings(&positives.files, &negatives.files)?
    };
    drop(worker);
    let keep = ui.retention(keep_recordings)?;
    let _synthetic = if ui.assistance() {
        if keep {
            let checkpoint =
                training::retain_dataset(paths, &config.wake_words[index].id, &dataset)?;
            eprintln!(
                "Enrollment checkpoint before synthesis: {}",
                checkpoint.join("manifest.json").display()
            );
        }
        let mut worker = ui.load(&selected, paths, Arc::clone(&cancellation.flag))?;
        match ui.augment(&mut dataset, phrase, paths, cancellation, &mut |audio| {
            training::single_utterance(worker.as_mut(), audio).map(|_| ())
        }) {
            Ok(samples) => Some(samples),
            Err(error) => {
                cancellation.check()?;
                ui.confirm("Omaspeak assistance did not complete", &format!("{error:#}\nContinue with the original human dataset? Explicitly saved recordings remain available."), "Use human recordings", "Cancel")?;
                None
            }
        }
    } else {
        None
    };
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

fn replace_evaluation(
    dataset: &mut crate::enrollment::artifact::Dataset,
    positives: &[PathBuf],
    negatives: &[PathBuf],
) -> Result<()> {
    use crate::enrollment::artifact::LabeledRecording;
    anyhow::ensure!(
        positives.len() == 4 && negatives.len() == 4,
        "resumed enrollment needs four new recordings of each class"
    );
    dataset.calibration.clear();
    dataset.validation.clear();
    for (files, positive) in [(positives, true), (negatives, false)] {
        for (index, audio) in files.iter().enumerate() {
            let split = if index < 2 {
                &mut dataset.calibration
            } else {
                &mut dataset.validation
            };
            split.push(LabeledRecording {
                audio: audio.clone(),
                positive,
                generated: None,
            });
        }
    }
    Ok(())
}

/// Reused transcript examples need the same speech-boundary check as new clips.
fn check_recordings(
    samples: &mut SampleSet,
    phrase: &str,
    cancellation: &Cancellation,
    mut ask: impl FnMut(&str, &str) -> Result<()>,
    mut record: impl FnMut() -> Result<Vec<f32>>,
    mut validate: impl FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    for index in 0..samples.files.len() {
        cancellation.check()?;
        let (_, mut audio) = crate::engine::audio::read_wave(&samples.files[index])?;
        while let Err(error) = enrollment::validate_audio(&audio).and_then(|()| validate(&audio)) {
            cancellation.check()?;
            ask(
                &format!("Retry wake-phrase example {}", index + 1),
                &format!(
                    "{error:#}\nSay {phrase:?} once, continuously. Other recordings are kept."
                ),
            )?;
            audio = record()?;
        }
        // Replace only after both waveform quality and segmentation pass.
        samples.push(&audio)?;
        let replacement = samples.files.pop().context("missing retry recording")?;
        fs::rename(replacement, &samples.files[index])?;
    }
    Ok(())
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
                generated: None,
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
                "Save before training; keep even if inference fails or activation is cancelled",
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

fn recognition_methods() -> [MenuItem; 3] {
    [
        MenuItem::available(
            "Whisper spellings (recommended)",
            "Record examples and accept the spellings Whisper hears",
        ),
        MenuItem::available(
            "Trainable KWS — learn from my voice",
            "Experimental: teach unusual names with wake-phrase and other-speech recordings",
        ),
        MenuItem::available(
            "Trainable KWS with Omaspeak assistance",
            "Review synthetic voices alongside your recordings; offers installation if needed",
        ),
    ]
}

fn training_engine_choices(config: &Config) -> (Vec<Option<String>>, Vec<MenuItem>) {
    let mut names = vec![None];
    names.extend(config.engines.keys().cloned().map(Some));
    let items = names
        .iter()
        .map(|name| {
            let profile = config
                .engine_profile(name.as_deref())
                .expect("configured engine");
            let supported = profile.backend.runtime == Runtime::Openvino
                && profile.backend.device.eq_ignore_ascii_case("cpu");
            let label = name.as_deref().unwrap_or("default");
            let detail = format!(
                "{} / {} / {}; requires Whisper base.en encoder assets",
                profile.backend.kind, profile.backend.device, profile.model.name
            );
            if supported {
                MenuItem::available(label, detail)
            } else {
                MenuItem::unavailable(label, "Training currently requires an OpenVINO CPU profile")
            }
        })
        .collect();
    (names, items)
}

fn select_training_engine(config: &mut Config, index: usize) -> Result<Config> {
    let (names, items) = training_engine_choices(config);
    let preferred = names.iter().zip(&items).position(|(name, item)|
        item.enabled && *name == config.wake_words[index].engine)
        .or_else(|| items.iter().position(|item| item.enabled))
        .context("Trainable KWS needs an OpenVINO CPU engine with Whisper base.en encoder assets. Run `omawake setup` to configure the model/runtime first. No recordings were collected; configuration unchanged.")?;
    let choice = select("Trainable KWS engine",
        "Choose an OpenVINO CPU profile. Encoder assets are checked before recording; the profile name can be anything.",
        &items, preferred)?.context("onboarding cancelled; configuration unchanged")?;
    select_engine(
        config,
        index,
        Some(names[choice].as_deref().unwrap_or("default")),
        false,
    )
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
            dataset: None,
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

pub(super) fn text_input(label: &str) -> Result<String> {
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
        assistance: bool,
        generation_succeeds: bool,
        resume: Option<crate::enrollment::artifact::Dataset>,
        recordings: Cell<usize>,
        screens: RefCell<Vec<String>>,
        cancel_at: Option<&'static str>,
        keep: bool,
        encoder_error: bool,
        invalid_negatives: bool,
    }
    impl TrainingInteraction for TestInteraction {
        fn assistance(&self) -> bool {
            self.assistance
        }
        fn augment(
            &self,
            dataset: &mut crate::enrollment::artifact::Dataset,
            _: &str,
            paths: &AppPaths,
            _: &Cancellation,
            validate: &mut dyn FnMut(&[f32]) -> Result<()>,
        ) -> Result<SampleSet> {
            anyhow::ensure!(self.generation_succeeds, "test synthesis failed");
            let mut samples = SampleSet::create(paths)?;
            for positive in [true, false] {
                let audio = vec![if positive { 0.23 } else { -0.23 }; 4000];
                validate(&audio)?;
                samples.push(&audio)?;
                dataset
                    .training
                    .push(crate::enrollment::artifact::LabeledRecording {
                        audio: samples.files.last().unwrap().clone(),
                        positive,
                        generated: Some(crate::enrollment::artifact::GeneratedRecording {
                            generator: "omaspeak".into(),
                            voice: "test".into(),
                            text: "test".into(),
                            speed: 1.0,
                        }),
                    });
            }
            Ok(samples)
        }
        fn resume(&self) -> Option<crate::enrollment::artifact::Dataset> {
            self.resume.clone()
        }
        fn confirm(&self, title: &str, help: &str, _: &str, _: &str) -> Result<()> {
            self.screens.borrow_mut().push(format!("{title} {help}"));
            anyhow::ensure!(self.cancel_at != Some(title), "cancelled by user");
            Ok(())
        }
        fn record(&self, _: &Config, _: u64, _: &Cancellation) -> Result<Vec<f32>> {
            let n = self.recordings.get();
            self.recordings.set(n + 1);
            let positive = n < if self.resume.is_some() { 4 } else { 5 } || self.invalid_negatives;
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
            assistance: false,
            generation_succeeds: false,
            resume: None,
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
    fn training_choices_explain_method_and_only_offer_cpu_openvino() {
        let methods = recognition_methods();
        assert!(methods[1].label.contains("Trainable KWS"));
        assert!(methods[2].label.contains("Omaspeak"));
        let mut config = Config::default();
        let mut profile = crate::config::EngineProfile::default();
        profile.backend.runtime = Runtime::Openvino;
        profile.backend.device = "CPU".into();
        config.engines.insert("my-encoder".into(), profile.clone());
        profile.backend.device = "npu".into();
        config.engines.insert("accelerator".into(), profile);
        let (names, items) = training_engine_choices(&config);
        let usable: Vec<_> = names
            .iter()
            .zip(&items)
            .filter(|(_, item)| item.enabled)
            .map(|(name, _)| name.as_deref())
            .collect();
        assert_eq!(usable, vec![Some("my-encoder")]);
        config.engines.clear();
        let before = toml::to_string(&config).unwrap();
        let error = select_training_engine(&mut config, 0).unwrap_err();
        assert!(error.to_string().contains("No recordings were collected"));
        assert_eq!(before, toml::to_string(&config).unwrap());
    }

    #[test]
    fn assistance_checkpoints_humans_and_can_fall_back_without_losing_recordings() {
        for succeeds in [false, true] {
            let (root, paths, config, samples) = training_fixture();
            let file = root.join("config.toml");
            let mut ui = test_interaction();
            ui.assistance = true;
            ui.generation_succeeds = succeeds;
            ui.keep = true;
            train_guided(
                config.clone(),
                0,
                &file,
                &paths,
                config_snapshot(&file).unwrap(),
                config,
                samples,
                3,
                false,
                &Cancellation::new().unwrap(),
                &ui,
            )
            .unwrap();
            let sessions = enrollment::recordings(&paths, "computer").unwrap();
            assert_eq!(sessions.len(), 2);
            let datasets: Vec<_> = sessions
                .iter()
                .map(|s| {
                    crate::enrollment::artifact::Dataset::load(&s.directory.join("manifest.json"))
                        .unwrap()
                })
                .collect();
            assert!(datasets.iter().all(|d| {
                d.validation
                    .iter()
                    .chain(&d.calibration)
                    .all(|r| r.generated.is_none())
            }));
            assert!(datasets.iter().any(|d| d.training.len() == 12));
            assert!(
                datasets
                    .iter()
                    .any(|d| d.training.len() == if succeeds { 14 } else { 12 })
            );
            assert_eq!(ui.recordings.get(), 15);
            assert_eq!(
                ui.screens
                    .borrow()
                    .iter()
                    .any(|s| s.contains("assistance did not complete")),
                !succeeds
            );
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn resumed_dataset_preserves_training_and_collects_fresh_human_evaluation() {
        let (root, paths, config, mut positives) = training_fixture();
        for i in 5..10 {
            positives
                .push(&vec![0.10 + i as f32 * 0.001; 4000])
                .unwrap();
        }
        let mut negatives = SampleSet::create(&paths).unwrap();
        for i in 0..10 {
            negatives
                .push(&vec![-0.13 - i as f32 * 0.001; 4000])
                .unwrap();
        }
        let dataset = split_recordings(&positives.files, &negatives.files).unwrap();
        let old_validation: BTreeSet<_> =
            dataset.validation.iter().map(|r| r.audio.clone()).collect();
        let source_fingerprints: Vec<_> = dataset
            .training
            .iter()
            .map(|r| fs::read(&r.audio).unwrap())
            .collect();
        let mut ui = test_interaction();
        ui.resume = Some(dataset.clone());
        ui.keep = true;
        let file = root.join("config.toml");
        train_guided(
            config.clone(),
            0,
            &file,
            &paths,
            config_snapshot(&file).unwrap(),
            config,
            SampleSet::create(&paths).unwrap(),
            3,
            false,
            &Cancellation::new().unwrap(),
            &ui,
        )
        .unwrap();
        assert_eq!(ui.recordings.get(), 8);
        assert!(old_validation.iter().all(|p| p.is_file()));
        assert_eq!(
            dataset
                .training
                .iter()
                .map(|r| fs::read(&r.audio).unwrap())
                .collect::<Vec<_>>(),
            source_fingerprints
        );
        let sessions = enrollment::recordings(&paths, "computer").unwrap();
        assert_eq!(sessions.len(), 1);
        let saved = crate::enrollment::artifact::Dataset::load(
            &sessions[0].directory.join("manifest.json"),
        )
        .unwrap();
        assert_eq!(saved.training.len(), 12);
        assert_eq!(saved.calibration.len(), 4);
        assert_eq!(saved.validation.len(), 4);
        assert!(saved.validation.iter().all(|r| r.generated.is_none()));
        assert!(replace_evaluation(&mut saved.clone(), &[], &[]).is_err());
        drop(positives);
        drop(negatives);
        fs::remove_dir_all(root).unwrap();
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
            assert_eq!(sessions.len(), usize::from(case == 2 || case == 4));
            if case == 2 || case == 4 {
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
    fn reused_sample_segmentation_retries_only_the_bad_clip_and_checks_quality() {
        let (root, paths, _, mut samples) = training_fixture();
        let untouched = fs::read(&samples.files[0]).unwrap();
        let mut retries = 0;
        let mut prompts = Vec::new();
        check_recordings(
            &mut samples,
            "Hey name",
            &Cancellation::new().unwrap(),
            |title, _| {
                prompts.push(title.to_owned());
                Ok(())
            },
            || {
                retries += 1;
                Ok(vec![if retries == 1 { 0.0 } else { 0.2 }; 4000])
            },
            |audio| {
                anyhow::ensure!(
                    (audio[0] - 0.101).abs() > 0.0001,
                    "VAD found 2 speech segments"
                );
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(retries, 2);
        assert_eq!(
            prompts,
            ["Retry wake-phrase example 2", "Retry wake-phrase example 2"]
        );
        assert_eq!(samples.files.len(), 5);
        assert_eq!(fs::read(&samples.files[0]).unwrap(), untouched);
        assert_eq!(
            crate::engine::audio::read_wave(&samples.files[1])
                .unwrap()
                .1[0],
            0.2
        );
        drop(samples);
        assert_eq!(
            fs::read_dir(paths.cache_dir.join("onboarding"))
                .unwrap()
                .count(),
            0
        );
        fs::remove_dir_all(root).unwrap();
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
