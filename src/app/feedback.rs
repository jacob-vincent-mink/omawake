use super::*;
use crate::enrollment::{artifact, history};

#[derive(clap::Args)]
pub(super) struct HistoryArgs {
    pub id: String,
    #[command(subcommand)]
    pub command: HistoryCommand,
}
#[derive(clap::Subcommand)]
pub(super) enum HistoryCommand {
    /// Opt in to private audio clips for live trained detections (oldest clips expire).
    Enable {
        #[arg(long, default_value_t=100, value_parser=clap::value_parser!(u16).range(1..=1000))]
        max_events: u16,
    },
    /// Stop saving new detections; retain existing clips and labels.
    Disable,
    /// List captured clips, scores, thresholds, and labels.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Play one captured clip explicitly. Ctrl+C stops playback.
    Play { event: String },
    /// Mark a detection for later retraining; does not update the running model.
    Label {
        event: String,
        #[arg(value_enum)]
        label: history::Label,
    },
    /// Delete this word's captured history and labels (not retained training datasets).
    Clear,
}
fn word_index(config: &Config, id: &str) -> Result<usize> {
    config
        .wake_words
        .iter()
        .position(|w| w.id == id)
        .context("unknown wake-word ID")
}
pub(super) fn threshold(
    id: String,
    value: Option<String>,
    mut config: Config,
    path: &Path,
    paths: &AppPaths,
) -> Result<()> {
    let index = word_index(&config, &id)?;
    let binding = config.wake_words[index]
        .enrollment
        .as_mut()
        .context("word has no trained head; run word onboard first")?;
    if let Some(value) = value {
        binding.threshold = if value == "auto" {
            None
        } else {
            Some(
                value
                    .parse()
                    .context("threshold must be auto or a number greater than 0 and at most 1")?,
            )
        };
        validate_wake_words(&config.wake_words)?;
        save_and_reload_active(config.clone(), path, paths)?;
    }
    let binding = config.wake_words[index].enrollment.as_ref().unwrap();
    println!(
        "{} threshold: {}",
        id,
        binding
            .threshold
            .map(|v| v.to_string())
            .unwrap_or_else(|| "auto (calibrated)".into())
    );
    for (contract, file) in &binding.heads {
        let file = if file.is_absolute() {
            file.clone()
        } else {
            path.parent().unwrap_or(Path::new(".")).join(file)
        };
        let head = artifact::load(&file)?;
        println!(
            "{contract}: calibrated={}, effective={}",
            head.threshold,
            binding.threshold.unwrap_or(head.threshold)
        );
    }
    Ok(())
}
pub(super) fn run(
    args: HistoryArgs,
    mut config: Config,
    path: &Path,
    paths: &AppPaths,
) -> Result<()> {
    match args.command {
        HistoryCommand::Enable { .. } | HistoryCommand::Disable => {
            let index = word_index(&config, &args.id)?;
            let binding = config.wake_words[index]
                .enrollment
                .as_mut()
                .context("history currently requires a trained word")?;
            match args.command {
                HistoryCommand::Enable { max_events } => {
                    binding.history.enabled = true;
                    binding.history.max_events = usize::from(max_events);
                }
                _ => binding.history.enabled = false,
            }
            let enabled = binding.history.enabled;
            validate_wake_words(&config.wake_words)?;
            save_and_reload_active(config, path, paths)?;
            println!(
                "History {} for {}",
                if enabled { "enabled" } else { "disabled" },
                args.id
            );
        }
        HistoryCommand::List { json } => {
            let events = history::list(paths, &args.id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&events)?);
            } else {
                for e in events {
                    println!(
                        "{} score={:.6} threshold={:.6} label={:?} device={} audio={}",
                        e.id,
                        e.score,
                        e.threshold,
                        e.label,
                        e.device,
                        e.audio.display()
                    );
                }
            }
        }
        HistoryCommand::Play { event } => {
            let clip = history::list(paths, &args.id)?
                .into_iter()
                .find(|e| e.id == event)
                .context("unknown history event")?;
            let player = assisted::executable("pw-play")
                .or_else(|| assisted::executable("aplay"))
                .context("history playback requires pw-play or aplay")?;
            anyhow::ensure!(
                ProcessCommand::new(player)
                    .arg(&clip.audio)
                    .status()?
                    .success(),
                "history playback failed or was cancelled"
            );
        }
        HistoryCommand::Label { event, label } => {
            history::label(paths, &args.id, &event, label)?;
            println!("Labeled {event}: {label:?}; retrain to use this correction");
        }
        HistoryCommand::Clear => {
            history::clear(paths, &args.id)?;
            println!("Cleared history for {}", args.id);
        }
    }
    Ok(())
}
