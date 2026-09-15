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
    let keep = if interactive && !args.keep_recordings {
        let items = [
            MenuItem::available(
                "Discard recordings after onboarding",
                "Keep the word and approved aliases; future retraining needs new samples",
            ),
            MenuItem::available(
                "Keep recordings locally",
                "Store private audio copies for adapting this word to another model",
            ),
        ];
        select(
            "Enrollment recordings",
            "Recordings never leave this machine. Retention is optional.",
            &items,
            0,
        )?
        .context("onboarding cancelled; configuration unchanged")?
            == 1
    } else {
        args.keep_recordings
    };
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
    mut ask: impl FnMut(&str, &str) -> Result<()>,
    mut record: impl FnMut() -> Result<Vec<f32>>,
) -> Result<()> {
    for sample in 0..count {
        loop {
            cancellation.check()?;
            ask(
                &format!("Record example {} of {}", sample + 1, count),
                &format!(
                    "Say {phrase:?} naturally. Recording lasts {seconds} seconds. Vary pace and distance between examples."
                ),
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
