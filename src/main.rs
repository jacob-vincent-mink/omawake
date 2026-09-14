use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use omawake::audio::{AudioEvent, Capture, input_devices};
use omawake::backend::{Fallback, Runtime, compiled_capabilities};
use omawake::config::{Config, WakeWord};
use omawake::engine::{ActionResult, Detection, Detector, wav_duration};
use omawake::keyword::{KeywordCompiler, validate_wake_words};
use omawake::paths::AppPaths;
use omawake::protocol::{Command, Request, Response, ResultPayload};
use omawake::setup as app_setup;
use omawake::setup::model::ProgressFormat;
use omawake::setup::wizard::{self, RuntimeSelection, SetupMode};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(
    name = "omawake",
    version,
    about = "Local-first configurable wake-word daemon"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: TopCommand,
}

#[derive(Subcommand)]
enum TopCommand {
    Test {
        #[arg(long, conflicts_with = "seconds")]
        audio: Option<PathBuf>,
        #[arg(long, conflicts_with = "audio")]
        seconds: Option<u64>,
        #[arg(long)]
        execute: bool,
        #[arg(long)]
        json: bool,
    },
    /// Benchmark one loaded detector against WAV files and print JSON.
    Benchmark {
        #[arg(required = true, value_name = "WAV")]
        audio: Vec<PathBuf>,
        #[arg(long, default_value_t = 1)]
        warmup: u32,
        #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..))]
        iterations: u32,
    },
    Status {
        #[arg(long)]
        json: bool,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    #[command(visible_alias = "word")]
    WakeWord {
        #[command(subcommand)]
        command: WakeWordCommand,
    },
    AudioDevices {
        #[arg(long)]
        json: bool,
    },
    Daemon,
    Pause,
    Resume,
    Stop,
    Setup {
        #[command(subcommand)]
        command: Option<SetupCommand>,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    Schema {
        #[arg(long)]
        json: bool,
    },
    Get {
        key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Set {
        key: String,
        value: String,
    },
    Unset {
        key: String,
    },
}

#[derive(Subcommand)]
enum WakeWordCommand {
    List {
        #[arg(long)]
        json: bool,
    },
    Add {
        #[arg(long)]
        id: String,
        #[arg(long)]
        phrase: String,
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    Remove {
        id: String,
    },
}

#[derive(Subcommand)]
enum SetupCommand {
    Check {
        #[arg(long)]
        json: bool,
    },
    All {
        #[arg(
            long,
            default_value = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
        )]
        model: String,
        #[arg(long)]
        archive: Option<PathBuf>,
        #[arg(long)]
        no_start: bool,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    Model {
        #[arg(long)]
        list: bool,
        #[arg(long, conflicts_with_all = ["download", "set", "verify"])]
        json: bool,
        #[arg(long, value_name = "MODEL", conflicts_with_all = ["set", "verify"])]
        download: Option<String>,
        #[arg(long, value_name = "MODEL", conflicts_with_all = ["download", "verify"])]
        set: Option<String>,
        #[arg(long, value_name = "MODEL", conflicts_with_all = ["download", "set"])]
        verify: Option<String>,
        #[arg(long, requires = "download")]
        archive: Option<PathBuf>,
        #[arg(long, requires = "download")]
        no_activate: bool,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    Runtime {
        #[arg(long)]
        json: bool,
    },
    Systemd {
        #[arg(long, conflicts_with = "status")]
        uninstall: bool,
        #[arg(long)]
        status: bool,
        #[arg(long, conflicts_with_all = ["uninstall", "status"])]
        no_start: bool,
    },
    Menu {
        #[arg(long, conflicts_with = "status")]
        uninstall: bool,
        #[arg(long)]
        status: bool,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("omawake: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    run_with_paths(cli, AppPaths::discover())
}

fn run_with_paths(cli: Cli, paths: AppPaths) -> Result<()> {
    run_with_paths_and_request(cli, paths, request)
}

fn run_with_paths_and_request<F>(cli: Cli, paths: AppPaths, mut send_request: F) -> Result<()>
where
    F: FnMut(&AppPaths, Command) -> Result<Response>,
{
    run_with_paths_and_services(cli, paths, &mut send_request, input_devices)
}

fn run_with_paths_and_services<F, D>(
    cli: Cli,
    mut paths: AppPaths,
    mut send_request: F,
    list_devices: D,
) -> Result<()>
where
    F: FnMut(&AppPaths, Command) -> Result<Response>,
    D: FnOnce() -> Result<Vec<String>>,
{
    let config_path = cli.config.unwrap_or_else(|| paths.config_file.clone());
    paths.config_file = config_path.clone();
    let command = match cli.command {
        TopCommand::Setup { command } => return setup(command, &config_path, &paths),
        command => command,
    };
    let config = Config::load(&config_path)?;
    match command {
        TopCommand::Benchmark {
            audio,
            warmup,
            iterations,
        } => run_file_benchmark(&config, &paths, &audio, warmup, iterations),
        TopCommand::Test {
            audio: Some(audio),
            execute,
            json: as_json,
            ..
        } => {
            let detector = Detector::load(&config, &paths)?;
            present_detections(&detector, detector.detect_file(&audio)?, execute, as_json)
        }
        TopCommand::Test {
            seconds: Some(seconds),
            execute,
            json: as_json,
            ..
        } => {
            let detector = Detector::load(&config, &paths)?;
            let detections = detect_live(&detector, &config, Duration::from_secs(seconds))?;
            present_detections(&detector, detections, execute, as_json)
        }
        TopCommand::Test { .. } => bail!("test requires --audio FILE or --seconds N"),
        TopCommand::Daemon => run_daemon(&config, &paths),
        TopCommand::Status { json: as_json } => match send_request(&paths, Command::Status) {
            Ok(response) => print_response(response, as_json),
            Err(_) => {
                let stopped = stopped_status(&config, &paths);
                if as_json {
                    println!("{}", serde_json::to_string_pretty(&stopped)?);
                } else {
                    println!("stopped");
                }
                Ok(())
            }
        },
        TopCommand::Pause => print_response(send_request(&paths, Command::Pause)?, false),
        TopCommand::Resume => print_response(send_request(&paths, Command::Resume)?, false),
        TopCommand::Stop => print_response(send_request(&paths, Command::Shutdown)?, false),
        TopCommand::AudioDevices { json: as_json } => {
            let devices = list_devices()?;
            if as_json {
                println!("{}", serde_json::to_string_pretty(&devices)?);
            } else {
                for device in devices {
                    println!("{device}");
                }
            }
            Ok(())
        }
        TopCommand::WakeWord { command } => {
            wake_word_command(command, config, &config_path, &paths)
        }
        TopCommand::Config {
            command: ConfigCommand::Get { key, json: as_json },
        } => {
            let value = serde_json::to_value(config)?;
            let selected = key
                .as_deref()
                .map_or(Some(&value), |key| dotted_get(&value, key))
                .with_context(|| format!("unknown config key {}", key.unwrap_or_default()))?;
            if as_json || !selected.is_string() {
                println!("{}", serde_json::to_string_pretty(selected)?);
            } else {
                println!("{}", selected.as_str().unwrap_or_default());
            }
            Ok(())
        }
        TopCommand::Config {
            command: ConfigCommand::Schema { json: as_json },
        } => {
            let value = schema(&config, &config_path, &paths);
            if as_json {
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                println!(
                    "backend.runtime\tdefault|openvino|cuda\nbackend.device\truntime-dependent\nwake_words\tcollection"
                );
            }
            Ok(())
        }
        TopCommand::Setup { .. } => unreachable!(),
        TopCommand::Config { command } => config_mutation(command, config, &config_path),
    }
}

fn config_mutation(command: ConfigCommand, mut config: Config, path: &Path) -> Result<()> {
    match command {
        ConfigCommand::Set { key, value } => set_config(&mut config, &key, &value)?,
        ConfigCommand::Unset { key } => unset_config(&mut config, &key)?,
        _ => unreachable!(),
    }
    config.backend.validate_shape()?;
    save_config(path, &config)
}

fn set_config(config: &mut Config, key: &str, value: &str) -> Result<()> {
    match key {
        "backend.kind" => config.backend.kind = value.into(),
        "backend.runtime" => config.backend.runtime = parse_runtime(value)?,
        "backend.device" => config.backend.device = value.into(),
        "backend.threads" => config.backend.threads = value.parse()?,
        "backend.fallback" => config.backend.fallback = parse_fallback(value)?,
        "backend.device_id" => config.backend.device_id = value.parse()?,
        "backend.provider_config" => config.backend.provider_config = value.into(),
        "model.name" => config.model.name = value.into(),
        "model.directory" => config.model.directory = value.into(),
        "model.sample_rate" => config.model.sample_rate = value.parse()?,
        "model.keywords_score" => config.model.keywords_score = value.parse()?,
        "model.keywords_threshold" => config.model.keywords_threshold = value.parse()?,
        "audio.device" => config.audio.device = value.into(),
        "audio.channels" => config.audio.channels = value.into(),
        "audio.buffer_milliseconds" => config.audio.buffer_milliseconds = value.parse()?,
        "daemon.cooldown_milliseconds" => config.daemon.cooldown_milliseconds = value.parse()?,
        "daemon.queue_capacity" => config.daemon.queue_capacity = value.parse()?,
        _ => bail!("unknown or unsupported config key {key}"),
    }
    Ok(())
}

fn unset_config(config: &mut Config, key: &str) -> Result<()> {
    let defaults = Config::default();
    match key {
        "backend.kind" => config.backend.kind = defaults.backend.kind,
        "backend.runtime" => config.backend.runtime = defaults.backend.runtime,
        "backend.device" => config.backend.device = defaults.backend.device,
        "backend.threads" => config.backend.threads = defaults.backend.threads,
        "backend.fallback" => config.backend.fallback = defaults.backend.fallback,
        "backend.device_id" => config.backend.device_id = defaults.backend.device_id,
        "backend.provider_config" => {
            config.backend.provider_config = defaults.backend.provider_config
        }
        "model.name" => config.model.name = defaults.model.name,
        "model.directory" => config.model.directory = defaults.model.directory,
        "model.sample_rate" => config.model.sample_rate = defaults.model.sample_rate,
        "model.keywords_score" => config.model.keywords_score = defaults.model.keywords_score,
        "model.keywords_threshold" => {
            config.model.keywords_threshold = defaults.model.keywords_threshold
        }
        "audio.device" => config.audio.device = defaults.audio.device,
        "audio.channels" => config.audio.channels = defaults.audio.channels,
        "audio.buffer_milliseconds" => {
            config.audio.buffer_milliseconds = defaults.audio.buffer_milliseconds
        }
        "daemon.cooldown_milliseconds" => {
            config.daemon.cooldown_milliseconds = defaults.daemon.cooldown_milliseconds
        }
        "daemon.queue_capacity" => config.daemon.queue_capacity = defaults.daemon.queue_capacity,
        _ => bail!("unknown or unsupported config key {key}"),
    }
    Ok(())
}

fn wake_word_command(
    command: WakeWordCommand,
    mut config: Config,
    path: &Path,
    paths: &AppPaths,
) -> Result<()> {
    let message = match command {
        WakeWordCommand::List { json: as_json } => {
            if as_json {
                println!("{}", serde_json::to_string_pretty(&config.wake_words)?);
            } else {
                for item in config.wake_words {
                    println!(
                        "{}\t{}\t{}",
                        item.id,
                        if item.enabled { "enabled" } else { "disabled" },
                        item.phrase
                    );
                }
            }
            return Ok(());
        }
        WakeWordCommand::Add {
            id,
            phrase,
            command,
        } => {
            let message = format!("added wake word: {id}");
            config.wake_words.push(WakeWord {
                id,
                phrase,
                enabled: true,
                command,
            });
            message
        }
        WakeWordCommand::Remove { id } => {
            remove_wake_word(&mut config, &id)?;
            format!("removed wake word: {id}")
        }
    };
    validate_wake_words(&config.wake_words)?;
    let bpe = config.model_directory(paths).join(&config.model.bpe_model);
    if bpe.exists() && config.wake_words.iter().any(|word| word.enabled) {
        KeywordCompiler::open(&bpe)?.compile(&config.wake_words)?;
    }
    save_config(path, &config)?;
    println!("{message}");
    Ok(())
}

fn remove_wake_word(config: &mut Config, id: &str) -> Result<()> {
    let before = config.wake_words.len();
    config.wake_words.retain(|item| item.id != id);
    if config.wake_words.len() == before {
        bail!("unknown wake-word id {id}");
    }
    Ok(())
}

fn save_config(path: &Path, config: &Config) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, toml::to_string_pretty(config)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn setup(command: Option<SetupCommand>, config_path: &Path, paths: &AppPaths) -> Result<()> {
    if command.is_none() && setup_is_interactive() {
        return guided_setup(config_path, paths);
    }
    if command.is_none() {
        println!(
            "Interactive setup needs a terminal. Run `omawake setup all` for the default install, or `omawake setup runtime` / `omawake setup model --list` to inspect choices."
        );
    }
    match command.unwrap_or(SetupCommand::Check { json: false }) {
        SetupCommand::Check { json } => app_setup::print_checks(config_path, paths, json),
        SetupCommand::Runtime { json } if !json && setup_is_interactive() => {
            guided_runtime(config_path)
        }
        SetupCommand::Runtime { json } => app_setup::print_runtime(json),
        SetupCommand::Model {
            list,
            json,
            download,
            set,
            verify,
            archive,
            no_activate,
            progress_format,
        } => {
            if list || json {
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(omawake::catalog::models())?
                    );
                } else {
                    print_models(paths);
                }
                if download.is_none() && set.is_none() && verify.is_none() {
                    return Ok(());
                }
            }
            if let Some(id) = verify {
                verify_selected_model(&id, paths, app_setup::model::verify)?;
                return Ok(());
            }
            if let Some(id) = set {
                set_selected_model(&id, config_path, paths, app_setup::model::verify)?;
                return Ok(());
            }
            let default_id = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01";
            let selected = match download {
                Some(id) => Some(id),
                None if !list && !json && setup_is_interactive() => {
                    return guided_model(config_path, paths);
                }
                None => {
                    print_models(paths);
                    println!(
                        "Run `omawake setup model --download {default_id}` to install the default model."
                    );
                    None
                }
            };
            if let Some(id) = selected {
                install_selected_model(
                    &id,
                    config_path,
                    paths,
                    archive.as_deref(),
                    no_activate,
                    progress_format,
                    app_setup::model::install,
                )?;
            }
            Ok(())
        }
        SetupCommand::Systemd {
            uninstall,
            status,
            no_start,
        } => {
            if status {
                app_setup::systemd::status(paths)
            } else if uninstall {
                app_setup::systemd::uninstall(paths)
            } else {
                app_setup::ensure_config(config_path)?;
                let path = app_setup::systemd::install(paths, config_path, !no_start)?;
                println!("installed: {}", path.display());
                Ok(())
            }
        }
        SetupCommand::Menu { uninstall, status } => {
            if status {
                app_setup::menu::status(paths)
            } else if uninstall {
                app_setup::menu::uninstall(paths)
            } else {
                let path = app_setup::menu::install(paths)?;
                println!("installed: {}", path.display());
                Ok(())
            }
        }
        SetupCommand::All {
            model,
            archive,
            no_start,
            progress_format,
        } => {
            let spec = model_spec(&model)?;
            install_everything(
                spec,
                config_path,
                paths,
                archive.as_deref(),
                no_start,
                progress_format,
                app_setup::model::install,
                app_setup::menu::install,
                app_setup::systemd::install,
                |config, paths| app_setup::print_checks(config, paths, false),
                app_setup::print_checks_event,
            )
        }
    }
}

fn setup_is_interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

trait GuidedPrompts {
    fn setup_mode(&mut self) -> Result<Option<SetupMode>>;
    fn runtime(&mut self, current: &Config) -> Result<Option<RuntimeSelection>>;
    fn model(
        &mut self,
        paths: &AppPaths,
        current: &Config,
    ) -> Result<Option<&'static omawake::catalog::ModelSpec>>;
    fn confirm(&mut self, selection: &RuntimeSelection, model: &str) -> Result<bool>;
}

struct TerminalGuidedPrompts;

impl GuidedPrompts for TerminalGuidedPrompts {
    fn setup_mode(&mut self) -> Result<Option<SetupMode>> {
        wizard::choose_setup_mode()
    }

    fn runtime(&mut self, current: &Config) -> Result<Option<RuntimeSelection>> {
        wizard::choose_runtime(
            compiled_capabilities(),
            current.backend.runtime,
            &current.backend.device,
        )
    }

    fn model(
        &mut self,
        paths: &AppPaths,
        current: &Config,
    ) -> Result<Option<&'static omawake::catalog::ModelSpec>> {
        choose_model(paths, &current.model.name)
    }

    fn confirm(&mut self, selection: &RuntimeSelection, model: &str) -> Result<bool> {
        wizard::confirm_apply(selection.runtime, &selection.device, model)
    }
}

fn guided_setup(config_path: &Path, paths: &AppPaths) -> Result<()> {
    guided_setup_with(config_path, paths, &mut TerminalGuidedPrompts)
}

fn guided_setup_with(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
) -> Result<()> {
    match prompts.setup_mode()? {
        Some(SetupMode::Full) => guided_all_with(config_path, paths, prompts),
        Some(SetupMode::Runtime) => guided_runtime_with(config_path, prompts),
        Some(SetupMode::Model) => guided_model_with(config_path, paths, prompts),
        Some(SetupMode::Check) => app_setup::print_checks(config_path, paths, false),
        None => {
            println!("Setup cancelled.");
            Ok(())
        }
    }
}

fn guided_runtime(config_path: &Path) -> Result<()> {
    guided_runtime_with(config_path, &mut TerminalGuidedPrompts)
}

fn guided_runtime_with(config_path: &Path, prompts: &mut impl GuidedPrompts) -> Result<()> {
    let current = Config::load(config_path)?;
    let Some(selection) = prompts.runtime(&current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    save_runtime_selection(config_path, &selection)?;
    println!(
        "runtime configured: {} / {}",
        runtime_name(selection.runtime),
        selection.device
    );
    Ok(())
}

fn guided_model(config_path: &Path, paths: &AppPaths) -> Result<()> {
    guided_model_with(config_path, paths, &mut TerminalGuidedPrompts)
}

fn guided_model_with(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
) -> Result<()> {
    guided_model_with_services(
        config_path,
        paths,
        prompts,
        app_setup::model::verify,
        app_setup::model::install,
    )
}

fn guided_model_with_services<FV, FI>(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
    verify: FV,
    install: FI,
) -> Result<()>
where
    FV: FnOnce(&AppPaths, &omawake::catalog::ModelSpec) -> Result<()>,
    FI: FnOnce(
        &AppPaths,
        &omawake::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
{
    let current = Config::load(config_path)?;
    let Some(spec) = prompts.model(paths, &current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    if verify(paths, spec).is_ok() {
        activate_model(config_path, spec)?;
        println!("active model: {}", spec.id);
        return Ok(());
    }
    install_selected_model(
        spec.id,
        config_path,
        paths,
        None,
        false,
        ProgressFormat::Human,
        install,
    )
}

fn guided_all_with(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
) -> Result<()> {
    guided_all_with_services(
        config_path,
        paths,
        prompts,
        app_setup::model::install,
        app_setup::menu::install,
        app_setup::systemd::install,
        |config, paths| app_setup::print_checks(config, paths, false),
        app_setup::print_checks_event,
    )
}

#[allow(clippy::too_many_arguments)]
fn guided_all_with_services<FI, FM, FS, CH, CJ>(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
    install_model: FI,
    install_menu: FM,
    install_systemd: FS,
    check_human: CH,
    check_json: CJ,
) -> Result<()>
where
    FI: FnOnce(
        &AppPaths,
        &omawake::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FS: FnOnce(&AppPaths, &Path, bool) -> Result<PathBuf>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    let current = Config::load(config_path)?;
    let Some(selection) = prompts.runtime(&current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let Some(spec) = prompts.model(paths, &current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    if !prompts.confirm(&selection, spec.id)? {
        println!("Setup cancelled.");
        return Ok(());
    }
    save_runtime_selection(config_path, &selection)?;
    install_everything(
        spec,
        config_path,
        paths,
        None,
        false,
        ProgressFormat::Human,
        install_model,
        install_menu,
        install_systemd,
        check_human,
        check_json,
    )
}

fn choose_model(
    paths: &AppPaths,
    active_model: &str,
) -> Result<Option<&'static omawake::catalog::ModelSpec>> {
    choose_model_with(paths, active_model, |items, preferred| {
        wizard::select(
            "Wake-word model",
            "● active · ○ installed · · available to download",
            items,
            preferred,
        )
    })
}

fn choose_model_with<S>(
    paths: &AppPaths,
    active_model: &str,
    select: S,
) -> Result<Option<&'static omawake::catalog::ModelSpec>>
where
    S: FnOnce(&[wizard::MenuItem], usize) -> Result<Option<usize>>,
{
    let items: Vec<_> = omawake::catalog::models()
        .iter()
        .map(|model| {
            let installed = app_setup::model::verify(paths, model).is_ok();
            let status = if model.id == active_model && installed {
                "● active"
            } else if model.id == active_model {
                "● active · download required"
            } else if installed {
                "○ installed"
            } else {
                "· download"
            };
            wizard::MenuItem::available(
                format!("{status}  {}", model.id),
                format!(
                    "{} · backend: {} · family: {} · {:.1} MiB",
                    model.description,
                    model.backend,
                    model.family,
                    model.archive_size as f64 / 1_048_576.0
                ),
            )
        })
        .collect();
    let preferred = omawake::catalog::models()
        .iter()
        .position(|model| model.id == active_model)
        .unwrap_or(0);
    Ok(select(&items, preferred)?.map(|index| &omawake::catalog::models()[index]))
}

fn save_runtime_selection(config_path: &Path, selection: &RuntimeSelection) -> Result<()> {
    let mut config = app_setup::ensure_config(config_path)?;
    config.backend.runtime = selection.runtime;
    config.backend.device = selection.device.clone();
    config
        .backend
        .validate_capabilities(compiled_capabilities())?;
    config.save(config_path)
}

fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
    }
}

#[allow(clippy::too_many_arguments)]
fn install_everything<FI, FM, FS, CH, CJ>(
    spec: &omawake::catalog::ModelSpec,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    no_start: bool,
    progress_format: ProgressFormat,
    install_model: FI,
    install_menu: FM,
    install_systemd: FS,
    check_human: CH,
    check_json: CJ,
) -> Result<()>
where
    FI: FnOnce(
        &AppPaths,
        &omawake::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FS: FnOnce(&AppPaths, &Path, bool) -> Result<PathBuf>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    let mut config = app_setup::ensure_config(config_path)?;
    let directory = install_model(paths, spec, archive, progress_format)?;
    spec.activate(&mut config);
    config.save(config_path)?;
    let launcher = install_menu(paths)?;
    let service = install_systemd(paths, config_path, !no_start)?;
    print_setup_complete(
        &directory,
        config_path,
        &launcher,
        &service,
        progress_format,
    )?;
    match progress_format {
        ProgressFormat::Human => check_human(config_path, paths),
        ProgressFormat::Json => check_json(config_path, paths),
    }
}

fn verify_selected_model<F>(id: &str, paths: &AppPaths, verify: F) -> Result<()>
where
    F: FnOnce(&AppPaths, &omawake::catalog::ModelSpec) -> Result<()>,
{
    let spec = model_spec(id)?;
    verify(paths, spec)?;
    println!(
        "verified: {}",
        app_setup::model::model_directory(paths, spec).display()
    );
    Ok(())
}

fn set_selected_model<F>(id: &str, config_path: &Path, paths: &AppPaths, verify: F) -> Result<()>
where
    F: FnOnce(&AppPaths, &omawake::catalog::ModelSpec) -> Result<()>,
{
    let spec = model_spec(id)?;
    verify(paths, spec)?;
    activate_model(config_path, spec)?;
    println!("active model: {}", spec.id);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn install_selected_model<F>(
    id: &str,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    no_activate: bool,
    progress_format: ProgressFormat,
    install: F,
) -> Result<()>
where
    F: FnOnce(
        &AppPaths,
        &omawake::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
{
    let spec = model_spec(id)?;
    let directory = install(paths, spec, archive, progress_format)?;
    if !no_activate {
        activate_model(config_path, spec)?;
    }
    print_model_ready(&directory, spec, !no_activate, progress_format)
}

fn print_setup_complete(
    directory: &Path,
    config_path: &Path,
    launcher: &Path,
    service: &Path,
    progress_format: ProgressFormat,
) -> Result<()> {
    match progress_format {
        ProgressFormat::Human => println!(
            "model: {}\nconfig: {}\nlauncher: {}\nservice: {}",
            directory.display(),
            config_path.display(),
            launcher.display(),
            service.display()
        ),
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "setup-complete",
                "model": directory,
                "config": config_path,
                "launcher": launcher,
                "service": service,
            }))?
        ),
    }
    Ok(())
}

fn activate_model(config_path: &Path, spec: &omawake::catalog::ModelSpec) -> Result<()> {
    let mut config = app_setup::ensure_config(config_path)?;
    spec.activate(&mut config);
    config.save(config_path)
}

fn print_model_ready(
    directory: &Path,
    spec: &omawake::catalog::ModelSpec,
    active: bool,
    progress_format: ProgressFormat,
) -> Result<()> {
    match progress_format {
        ProgressFormat::Human => println!("model ready: {}", directory.display()),
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "model-ready",
                "model": spec.id,
                "path": directory,
                "active": active,
            }))?
        ),
    }
    Ok(())
}

fn model_spec(id: &str) -> Result<&'static omawake::catalog::ModelSpec> {
    omawake::catalog::model(id)
        .ok_or_else(|| anyhow::anyhow!("unknown model {id}; run `omawake setup model --list`"))
}

fn print_models(paths: &AppPaths) {
    for model in omawake::catalog::models() {
        let status = if app_setup::model::verify(paths, model).is_ok() {
            "installed"
        } else {
            "available"
        };
        println!(
            "{}\t{}\t{}\t{}",
            model.id, model.backend, status, model.description
        );
    }
}

fn dotted_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    key.split('.')
        .try_fold(value, |value, part| value.get(part))
}

fn parse_runtime(value: &str) -> Result<Runtime> {
    match value.to_ascii_lowercase().as_str() {
        "default" => Ok(Runtime::Default),
        "openvino" => Ok(Runtime::Openvino),
        "cuda" => Ok(Runtime::Cuda),
        _ => bail!("backend runtime must be default, openvino, or cuda"),
    }
}

fn parse_fallback(value: &str) -> Result<Fallback> {
    match value.to_ascii_lowercase().as_str() {
        "error" => Ok(Fallback::Error),
        "cpu" => Ok(Fallback::Cpu),
        _ => bail!("backend fallback must be error or cpu"),
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkIteration {
    iteration: u32,
    elapsed_milliseconds: f64,
    real_time_factor: Option<f64>,
    detections: Vec<Detection>,
}

#[derive(Debug, Serialize)]
struct BenchmarkFile {
    path: PathBuf,
    audio_duration_milliseconds: f64,
    iterations: Vec<BenchmarkIteration>,
    summary: BenchmarkSummary,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkSummary {
    samples: usize,
    p50_milliseconds: Option<f64>,
    p95_milliseconds: Option<f64>,
    p50_real_time_factor: Option<f64>,
    p95_real_time_factor: Option<f64>,
}

fn run_file_benchmark(
    config: &Config,
    paths: &AppPaths,
    audio: &[PathBuf],
    warmup: u32,
    iterations: u32,
) -> Result<()> {
    let detector = Detector::load(config, paths)?;
    let files = benchmark_files(
        audio,
        warmup,
        iterations,
        wav_duration,
        |path| detector.detect_file(path),
        Instant::now,
    )?;
    let report = benchmark_report(config, &detector, files, warmup, iterations)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn benchmark_report(
    config: &Config,
    detector: &impl DetectorControl,
    files: Vec<BenchmarkFile>,
    warmup: u32,
    iterations: u32,
) -> Result<Value> {
    let summary = benchmark_summary(
        files
            .iter()
            .flat_map(|file| file.iterations.iter())
            .map(|iteration| (iteration.elapsed_milliseconds, iteration.real_time_factor)),
    );
    Ok(json!({
        "schema_version": 1,
        "benchmark": "omawake-file-detection",
        "model_load_milliseconds": milliseconds(detector.load_time()),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "backend": {
            "kind": detector.backend_kind(),
            "requested_runtime": config.backend.runtime,
            "requested_device": config.backend.canonical_device()?,
            "effective_runtime": detector.effective_runtime(),
            "fallback_used": detector.fallback_used(),
            "placement_verified": false,
        },
        "files": files,
        "summary": summary,
    }))
}

fn benchmark_files<D, F, N>(
    paths: &[PathBuf],
    warmup: u32,
    iterations: u32,
    mut duration: D,
    mut detect: F,
    mut now: N,
) -> Result<Vec<BenchmarkFile>>
where
    D: FnMut(&Path) -> Result<Duration>,
    F: FnMut(&Path) -> Result<Vec<Detection>>,
    N: FnMut() -> Instant,
{
    if paths.is_empty() {
        bail!("benchmark requires at least one WAV path");
    }
    if iterations == 0 {
        bail!("benchmark iterations must be at least one");
    }
    let inputs = paths
        .iter()
        .map(|path| {
            duration(path)
                .with_context(|| format!("inspect benchmark audio {}", path.display()))
                .map(|duration| (path, duration))
        })
        .collect::<Result<Vec<_>>>()?;

    for _ in 0..warmup {
        for (path, _) in &inputs {
            detect(path).with_context(|| format!("warm up benchmark audio {}", path.display()))?;
        }
    }

    inputs
        .into_iter()
        .map(|(path, audio_duration)| {
            let mut measurements = Vec::with_capacity(iterations as usize);
            for iteration in 1..=iterations {
                let started = now();
                let detections =
                    detect(path).with_context(|| format!("benchmark audio {}", path.display()))?;
                let elapsed = now().saturating_duration_since(started);
                let real_time_factor = (audio_duration > Duration::ZERO)
                    .then(|| elapsed.as_secs_f64() / audio_duration.as_secs_f64());
                measurements.push(BenchmarkIteration {
                    iteration,
                    elapsed_milliseconds: milliseconds(elapsed),
                    real_time_factor,
                    detections,
                });
            }
            let summary = benchmark_summary(measurements.iter().map(|measurement| {
                (
                    measurement.elapsed_milliseconds,
                    measurement.real_time_factor,
                )
            }));
            Ok(BenchmarkFile {
                path: path.clone(),
                audio_duration_milliseconds: milliseconds(audio_duration),
                iterations: measurements,
                summary,
            })
        })
        .collect()
}

fn benchmark_summary(values: impl IntoIterator<Item = (f64, Option<f64>)>) -> BenchmarkSummary {
    let (elapsed, real_time_factors): (Vec<_>, Vec<_>) = values.into_iter().unzip();
    let real_time_factors = real_time_factors.into_iter().flatten().collect::<Vec<_>>();
    BenchmarkSummary {
        samples: elapsed.len(),
        p50_milliseconds: percentile(&elapsed, 0.50),
        p95_milliseconds: percentile(&elapsed, 0.95),
        p50_real_time_factor: percentile(&real_time_factors, 0.50),
        p95_real_time_factor: percentile(&real_time_factors, 0.95),
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

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn detect_live(detector: &Detector, config: &Config, duration: Duration) -> Result<Vec<Detection>> {
    let capture = Capture::start(&config.audio.device, config.daemon.queue_capacity)?;
    eprintln!(
        "capturing {} Hz, {} channel(s) from {}",
        capture.sample_rate, capture.channels, capture.device_name
    );
    let session = detector.session();
    collect_live_detections(
        duration,
        |timeout| capture.receiver().recv_timeout(timeout),
        |sample_rate, samples| session.accept(sample_rate, samples),
        || session.finish(),
    )
}

fn collect_live_detections<R, A, F>(
    duration: Duration,
    receive: R,
    accept: A,
    finish: F,
) -> Result<Vec<Detection>>
where
    R: FnMut(Duration) -> std::result::Result<AudioEvent, RecvTimeoutError>,
    A: FnMut(i32, &[f32]) -> Result<Vec<Detection>>,
    F: FnOnce() -> Result<Vec<Detection>>,
{
    collect_live_detections_with_clock(duration, receive, accept, finish, Instant::now)
}

fn collect_live_detections_with_clock<R, A, F, N>(
    duration: Duration,
    mut receive: R,
    mut accept: A,
    finish: F,
    mut now: N,
) -> Result<Vec<Detection>>
where
    R: FnMut(Duration) -> std::result::Result<AudioEvent, RecvTimeoutError>,
    A: FnMut(i32, &[f32]) -> Result<Vec<Detection>>,
    F: FnOnce() -> Result<Vec<Detection>>,
    N: FnMut() -> Instant,
{
    let deadline = now() + duration;
    let mut detections = Vec::new();
    while now() < deadline {
        match receive(Duration::from_millis(100)) {
            Ok(AudioEvent::Samples {
                sample_rate,
                samples,
            }) => detections.extend(accept(sample_rate, &samples)?),
            Ok(AudioEvent::Error(error)) => bail!("audio capture failed: {error}"),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => bail!("audio capture disconnected"),
        }
    }
    detections.extend(finish()?);
    Ok(detections)
}

fn present_detections(
    detector: &impl DetectorControl,
    detections: Vec<Detection>,
    execute: bool,
    as_json: bool,
) -> Result<()> {
    let actions = collect_detection_actions(&detections, execute, |id| detector.run(id))?;
    print_detections(
        detections,
        actions,
        as_json,
        detector.load_time(),
        detector.keywords_buffer(),
        detector.backend_kind(),
        detector.effective_runtime(),
        detector.fallback_used(),
    )
}

fn collect_detection_actions<F>(
    detections: &[Detection],
    execute: bool,
    run_action: F,
) -> Result<Vec<ActionResult>>
where
    F: FnMut(&str) -> Result<ActionResult>,
{
    if execute {
        detections
            .iter()
            .map(|item| item.id.as_str())
            .map(run_action)
            .collect()
    } else {
        Ok(Vec::new())
    }
}

#[allow(clippy::too_many_arguments)]
fn print_detections(
    detections: Vec<Detection>,
    actions: Vec<ActionResult>,
    as_json: bool,
    load_time: Duration,
    keywords_buffer: &str,
    backend_kind: &str,
    effective_runtime: Runtime,
    fallback_used: bool,
) -> Result<()> {
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "model_load_milliseconds": load_time.as_millis() as u64,
                "keywords_buffer": keywords_buffer,
                "detections": detections,
                "actions": actions,
                "backend": {"kind": backend_kind, "effective_runtime": effective_runtime, "fallback_used": fallback_used}
            }))?
        );
    } else {
        for detection in detections {
            println!("{}", detection.id);
        }
    }
    Ok(())
}

fn run_daemon(config: &Config, paths: &AppPaths) -> Result<()> {
    let detector = Detector::load(config, paths)?;
    let listener = bind_socket(paths)?;
    eprintln!(
        "loaded wake-word model in {} ms",
        detector.load_time.as_millis()
    );
    let serve_result = (|| -> Result<()> {
        let mut paused = false;
        let mut shutdown = false;
        while !shutdown {
            if paused {
                if let Some(command) =
                    poll_control(&detector, "paused", None, || accept_control(&listener))?
                {
                    apply_daemon_command(command, &mut paused, &mut shutdown);
                }
                thread::sleep(Duration::from_millis(50));
                continue;
            }

            let capture = Capture::start(&config.audio.device, config.daemon.queue_capacity)?;
            eprintln!(
                "armed on {} ({} Hz, {} channel(s))",
                capture.device_name, capture.sample_rate, capture.channels
            );
            let session = detector.session();
            let (triggered, command) = collect_armed_detections(
                &capture.device_name,
                capture.sample_rate,
                capture.channels,
                |audio| {
                    poll_control(&detector, "armed", Some(audio), || {
                        accept_control(&listener)
                    })
                },
                |timeout| capture.receiver().recv_timeout(timeout),
                |sample_rate, samples| session.accept(sample_rate, samples),
            )?;
            if let Some(command) = command {
                apply_daemon_command(command, &mut paused, &mut shutdown);
            }
            drop(session);
            drop(capture);
            execute_detected_actions(triggered, |id| detector.run_action(id));
            if !paused && !shutdown {
                thread::sleep(Duration::from_millis(config.daemon.cooldown_milliseconds));
            }
        }
        Ok(())
    })();
    drop(listener);
    finish_daemon(paths, serve_result)
}

fn finish_daemon(paths: &AppPaths, serve_result: Result<()>) -> Result<()> {
    let path = socket_path(paths);
    let cleanup_result = fs::remove_file(&path)
        .or_else(|error| {
            (error.kind() == std::io::ErrorKind::NotFound)
                .then_some(())
                .ok_or(error)
        })
        .with_context(|| format!("remove daemon socket {}", path.display()));
    if let Err(error) = serve_result {
        if let Err(cleanup_error) = cleanup_result {
            eprintln!("omawake: {cleanup_error:#}");
        }
        return Err(error);
    }
    cleanup_result?;
    eprintln!("stopped");
    Ok(())
}

fn collect_armed_detections<P, R, A>(
    device_name: &str,
    sample_rate: u32,
    channels: u16,
    mut poll: P,
    mut receive: R,
    mut accept: A,
) -> Result<(Vec<Detection>, Option<Command>)>
where
    P: FnMut(Value) -> Result<Option<Command>>,
    R: FnMut(Duration) -> std::result::Result<AudioEvent, RecvTimeoutError>,
    A: FnMut(i32, &[f32]) -> Result<Vec<Detection>>,
{
    let mut triggered = Vec::new();
    loop {
        let audio = json!({"device":device_name,"sample_rate":sample_rate,"channels":channels});
        if let Some(command) = poll(audio)? {
            match command {
                Command::Pause | Command::Shutdown => return Ok((triggered, Some(command))),
                Command::Resume | Command::Status => continue,
            }
        }
        triggered.extend(handle_audio_event(
            receive(Duration::from_millis(50)),
            &mut accept,
        )?);
        if !triggered.is_empty() {
            return Ok((triggered, None));
        }
    }
}

fn apply_daemon_command(command: Command, paused: &mut bool, shutdown: &mut bool) {
    match command {
        Command::Pause => *paused = true,
        Command::Resume => *paused = false,
        Command::Shutdown => *shutdown = true,
        Command::Status => {}
    }
}

fn handle_audio_event<F>(
    event: std::result::Result<AudioEvent, RecvTimeoutError>,
    mut accept: F,
) -> Result<Vec<Detection>>
where
    F: FnMut(i32, &[f32]) -> Result<Vec<Detection>>,
{
    match event {
        Ok(AudioEvent::Samples {
            sample_rate,
            samples,
        }) => accept(sample_rate, &samples),
        Ok(AudioEvent::Error(error)) => bail!("audio capture failed: {error}"),
        Err(RecvTimeoutError::Timeout) => Ok(Vec::new()),
        Err(RecvTimeoutError::Disconnected) => bail!("audio capture disconnected"),
    }
}

fn execute_detected_actions<F>(detections: Vec<Detection>, mut run: F)
where
    F: FnMut(&str) -> Result<ActionResult>,
{
    for detection in detections {
        match run(&detection.id) {
            Ok(result) => eprintln!("detected {}; action exited {}", detection.id, result.status),
            Err(error) => eprintln!("action for {} failed: {error:#}", detection.id),
        }
    }
}

fn bind_socket(paths: &AppPaths) -> Result<UnixListener> {
    bind_socket_with(
        paths,
        |path| UnixStream::connect(path).is_ok(),
        |path| UnixListener::bind(path),
        |listener| listener.set_nonblocking(true),
    )
}

fn bind_socket_with<L, C, B, N>(
    paths: &AppPaths,
    is_live: C,
    bind: B,
    set_nonblocking: N,
) -> Result<L>
where
    C: FnOnce(&Path) -> bool,
    B: FnOnce(&Path) -> std::io::Result<L>,
    N: FnOnce(&L) -> std::io::Result<()>,
{
    fs::create_dir_all(&paths.runtime_dir)
        .with_context(|| format!("create {}", paths.runtime_dir.display()))?;
    fs::set_permissions(&paths.runtime_dir, fs::Permissions::from_mode(0o700))?;
    let path = socket_path(paths);
    if path.exists() {
        if is_live(&path) {
            bail!("daemon is already running at {}", path.display());
        }
        fs::remove_file(&path)
            .with_context(|| format!("remove stale socket {}", path.display()))?;
    }
    let listener = bind(&path).with_context(|| format!("bind {}", path.display()))?;
    if let Err(error) = fs::set_permissions(&path, fs::Permissions::from_mode(0o600)) {
        drop(listener);
        let _ = fs::remove_file(&path);
        return Err(error.into());
    }
    if let Err(error) = set_nonblocking(&listener) {
        drop(listener);
        let _ = fs::remove_file(&path);
        return Err(error.into());
    }
    Ok(listener)
}

fn poll_control<S, A>(
    detector: &impl DetectorControl,
    state: &str,
    audio: Option<serde_json::Value>,
    accept: A,
) -> Result<Option<Command>>
where
    S: Read + Write,
    A: FnMut() -> Result<Option<S>>,
{
    let details = daemon_details(
        detector.backend_kind(),
        detector.effective_runtime(),
        detector.fallback_used(),
        detector.load_time(),
        audio,
    );
    poll_control_connections(accept, state, &details)
}

trait DetectorControl {
    fn backend_kind(&self) -> &str;
    fn effective_runtime(&self) -> Runtime;
    fn fallback_used(&self) -> bool;
    fn load_time(&self) -> Duration;
    fn keywords_buffer(&self) -> &str;
    fn run(&self, id: &str) -> Result<ActionResult>;
}

impl DetectorControl for Detector {
    fn backend_kind(&self) -> &str {
        self.backend_kind
    }

    fn effective_runtime(&self) -> Runtime {
        self.effective_runtime
    }

    fn fallback_used(&self) -> bool {
        self.fallback_used
    }

    fn load_time(&self) -> Duration {
        self.load_time
    }

    fn keywords_buffer(&self) -> &str {
        &self.keywords_buffer
    }

    fn run(&self, id: &str) -> Result<ActionResult> {
        self.run_action(id)
    }
}

fn daemon_details(
    backend_kind: &str,
    effective_runtime: Runtime,
    fallback_used: bool,
    load_time: Duration,
    audio: Option<Value>,
) -> Value {
    json!({
        "backend": {
            "kind": backend_kind,
            "effective_runtime": effective_runtime,
            "fallback_used": fallback_used,
        },
        "model_load_milliseconds": load_time.as_millis() as u64,
        "audio": audio,
    })
}

fn accept_control(listener: &UnixListener) -> Result<Option<UnixStream>> {
    prepare_accepted_control(listener.accept().map(|(stream, _)| stream), |stream| {
        stream.set_read_timeout(Some(Duration::from_millis(500)))
    })
}

fn prepare_accepted_control<S, F>(accepted: std::io::Result<S>, configure: F) -> Result<Option<S>>
where
    F: FnOnce(&S) -> std::io::Result<()>,
{
    match accepted {
        Ok(stream) => {
            configure(&stream)?;
            Ok(Some(stream))
        }
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn poll_control_connections<S, A>(
    mut accept: A,
    state: &str,
    details: &Value,
) -> Result<Option<Command>>
where
    S: Read + Write,
    A: FnMut() -> Result<Option<S>>,
{
    loop {
        let Some(mut stream) = accept()? else {
            return Ok(None);
        };
        let command = handle_control_stream(&mut stream, state, details);
        if command.is_some() {
            return Ok(command);
        }
    }
}

fn handle_control_stream(
    stream: &mut (impl Read + Write),
    state: &str,
    details: &Value,
) -> Option<Command> {
    let (response, command) = control_response(read_request(&mut *stream), state, details);
    write_response(stream, &response);
    command
}

fn control_response(
    request: Result<Request>,
    state: &str,
    details: &Value,
) -> (Response, Option<Command>) {
    let request = match request {
        Ok(request) => request,
        Err(error) => {
            return (Response::error("unknown", "invalid_request", error), None);
        }
    };
    if request.protocol != 1 {
        return (
            Response::error(
                request.id,
                "protocol_mismatch",
                format!("unsupported protocol version {}", request.protocol),
            ),
            None,
        );
    }
    let next_state = match &request.command {
        Command::Pause => "paused",
        Command::Resume => "armed",
        Command::Shutdown => "stopping",
        Command::Status => state,
    };
    let command = (!matches!(&request.command, Command::Status)).then_some(request.command);
    (
        Response {
            protocol: 1,
            id: request.id,
            result: ResultPayload::State {
                state: next_state.into(),
                details: details.clone(),
            },
        },
        command,
    )
}

fn read_request(reader: impl Read) -> Result<Request> {
    let mut bytes = Vec::new();
    BufReader::new(reader)
        .take(65_537)
        .read_until(b'\n', &mut bytes)
        .context("read daemon request")?;
    if bytes.len() > 65_536 {
        bail!("message exceeds 65536 bytes");
    }
    serde_json::from_slice(&bytes).context("parse daemon request")
}

fn request(paths: &AppPaths, command: Command) -> Result<Response> {
    request_with_connector(paths, command, |path| UnixStream::connect(path))
}

fn request_with_connector<S, C>(paths: &AppPaths, command: Command, connect: C) -> Result<Response>
where
    S: Read + Write,
    C: FnOnce(&Path) -> std::io::Result<S>,
{
    let path = socket_path(paths);
    let mut stream =
        connect(&path).with_context(|| format!("daemon is not running at {}", path.display()))?;
    request_over_stream(&mut stream, command)
}

fn request_over_stream(stream: &mut (impl Read + Write), command: Command) -> Result<Response> {
    serde_json::to_writer(
        &mut *stream,
        &Request {
            protocol: 1,
            id: request_id(),
            command,
        },
    )?;
    stream.write_all(b"\n")?;
    let mut line = String::new();
    BufReader::new(&mut *stream).read_line(&mut line)?;
    serde_json::from_str(&line).context("parse daemon response")
}

fn print_response(response: Response, as_json: bool) -> Result<()> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(&response)?);
    } else {
        match response.result {
            ResultPayload::State { state, .. } => println!("{state}"),
            ResultPayload::Error { code, message } => bail!("{code}: {message}"),
        }
    }
    Ok(())
}

fn socket_path(paths: &AppPaths) -> PathBuf {
    paths.socket()
}

fn write_response(stream: &mut impl Write, response: &Response) {
    match serde_json::to_vec(response) {
        Ok(mut bytes) => {
            bytes.push(b'\n');
            if let Err(error) = stream.write_all(&bytes) {
                eprintln!("omawake: client disconnected before response: {error}");
            }
        }
        Err(error) => eprintln!("omawake: failed to encode response: {error}"),
    }
}

fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos}", std::process::id())
}

fn stopped_status(config: &Config, paths: &AppPaths) -> serde_json::Value {
    json!({"status_version":1,"app":"omawake","daemon":{"running":false,"state":"stopped"},
        "backend":{"kind":config.backend.kind,"requested":{"runtime":config.backend.runtime,"device":config.backend.device},"effective":null,"compiled_capabilities":compiled_capabilities(),"fallback_policy":config.backend.fallback,"fallback_used":false,"placement_verified":false,"evidence":[]},
        "model":{"family":"zipformer-kws","path":config.model_directory(paths),"loaded":false},"last_error":null,"details":{}})
}

fn schema(config: &Config, config_path: &Path, paths: &AppPaths) -> serde_json::Value {
    json!({"schema_version":1,"app":"omawake","app_version":env!("CARGO_PKG_VERSION"),"daemon_version":env!("CARGO_PKG_VERSION"),"config_path":config_path,
        "keys":[
            {"key":"backend.kind","type":"enum","section":"Backend","label":"Backend","description":"Inference engine","value":config.backend.kind,"file_value":null,"compiled":true,"restart_required":true,"choices":["sherpa-onnx"]},
            {"key":"backend.runtime","type":"enum","section":"Backend","label":"Runtime","description":"ONNX Runtime provider","value":config.backend.runtime,"file_value":null,"compiled":true,"restart_required":true,"choices":[{"value":"default","available":true,"capability":"cpu"},{"value":"openvino","available":compiled_capabilities().contains(&"openvino"),"capability":"openvino"},{"value":"cuda","available":compiled_capabilities().contains(&"cuda"),"capability":"cuda"}]},
            {"key":"backend.device","type":"string","section":"Backend","label":"Device","description":"Runtime-specific device","value":config.backend.device,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.directory","type":"path","section":"Model","label":"Directory","description":"Model asset directory","value":config.model_directory(paths),"file_value":config.model.directory,"compiled":true,"restart_required":true}],
        "collections":[{"key":"wake_words","id_key":"id","label":"Wake words","items":config.wake_words}],
        "constraints":[{"kind":"matrix","keys":["backend.runtime","backend.device"],"rows":[{"backend.runtime":"default","backend.device":["auto","cpu"]},{"backend.runtime":"cuda","backend.device":["auto","gpu"]},{"backend.runtime":"openvino","backend.device":["auto","npu","gpu","cpu","auto:<devices>","hetero:<2+ devices>","multi:<2+ devices>"]}]}]})
}

#[cfg(test)]
#[path = "../tests/unit/app_main.rs"]
mod tests;
