use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::audio::{AudioEvent, Capture, input_devices};
use crate::backend::{Fallback, Runtime, supported_capabilities};
use crate::config::{Config, WakeWord};
use crate::engine::{ActionResult, Detection, Detector, wav_duration};
use crate::keyword::{KeywordCompiler, validate_wake_words};
use crate::paths::AppPaths;
use crate::protocol::{Command, Request, Response, ResultPayload};
use crate::runtime_paths;
use crate::setup as app_setup;
use crate::setup::model::ProgressFormat;
use crate::setup::wizard::{self, RuntimeSelection, SetupMode};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::Serialize;
use serde_json::{Value, json};
use signal_hook::consts::signal::{SIGINT, SIGTERM};

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
    #[command(name = "__inventory-probe", hide = true)]
    InventoryProbe {
        candidate: String,
    },
    #[command(name = "__model-cache-prepare", hide = true)]
    ModelCachePrepare {
        candidate: String,
    },
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
    /// Install and configure a model plus the desktop launcher. Does not install a service;
    /// run `omawake setup systemd` to install one explicitly.
    All {
        #[arg(
            long,
            default_value = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
        )]
        model: String,
        #[arg(long)]
        archive: Option<PathBuf>,
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
        /// Select the runtime without opening the TUI.
        #[arg(long, value_parser = ["default", "openvino", "cuda"], conflicts_with = "json")]
        runtime: Option<String>,
        /// Select a compatible device without opening the TUI.
        #[arg(long, conflicts_with = "json")]
        device: Option<String>,
        #[arg(long, value_name = "DIRECTORY", conflicts_with = "json")]
        dir: Option<PathBuf>,
        /// Persist the candidate after its isolated probe succeeds.
        #[arg(long, conflicts_with = "json")]
        apply: bool,
    },
    /// Explicitly install or manage the optional systemd user service.
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

pub fn entry() -> ExitCode {
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
        TopCommand::InventoryProbe { candidate } => {
            let candidate = serde_json::from_str(&candidate)?;
            println!(
                "{}",
                serde_json::to_string(&crate::runtime_inventory::child(&candidate))?
            );
            return Ok(());
        }
        TopCommand::ModelCachePrepare { candidate } => {
            let mut candidate: Config = serde_json::from_str(&candidate)?;
            candidate.backend.fallback = Fallback::Error;
            println!(
                "{}",
                serde_json::to_string(&app_setup::cache::child(&candidate, &paths)?)?
            );
            return Ok(());
        }
        TopCommand::Setup { command } => return setup(command, &config_path, &paths),
        command => command,
    };
    let config = Config::load(&config_path)?;
    if command_uses_engine(&command) {
        runtime_paths::ensure_engine_library_path(&config.backend, &config_path)?;
    }
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
        TopCommand::InventoryProbe { .. }
        | TopCommand::ModelCachePrepare { .. }
        | TopCommand::Setup { .. } => unreachable!(),
        TopCommand::Config { command } => config_mutation(command, config, &config_path),
    }
}

fn command_uses_engine(command: &TopCommand) -> bool {
    matches!(
        command,
        TopCommand::Benchmark { .. }
            | TopCommand::Test { audio: Some(_), .. }
            | TopCommand::Test {
                seconds: Some(_),
                ..
            }
            | TopCommand::Daemon
    )
}

fn config_mutation(command: ConfigCommand, mut config: Config, path: &Path) -> Result<()> {
    let previous_runtime = config.backend.runtime;
    let reconcile_model = matches!(
        &command,
        ConfigCommand::Set { key, .. } | ConfigCommand::Unset { key }
            if matches!(key.as_str(), "backend.runtime" | "backend.device")
    );
    match command {
        ConfigCommand::Set { key, value } => set_config(&mut config, &key, &value)?,
        ConfigCommand::Unset { key } => unset_config(&mut config, &key)?,
        _ => unreachable!(),
    }
    let runtime_changed = config.backend.runtime != previous_runtime;
    if runtime_changed {
        config.backend.device = "auto".into();
        config.backend.provider_config.clear();
        config.backend.provider_library.clear();
        config.backend.options.clear();
        if config.backend.runtime != Runtime::Cuda {
            config.backend.device_id = 0;
        }
    }
    config.backend.validate_shape()?;
    if reconcile_model && let Some(spec) = crate::catalog::model(&config.model.name) {
        spec.apply_runtime_compatibility(&mut config);
    }
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
        "backend.library_dirs" => {
            config.backend.library_dirs = std::env::split_paths(value).collect()
        }
        "backend.onnxruntime_library" => config.backend.onnxruntime_library = value.into(),
        "backend.sherpa_library" => config.backend.sherpa_library = value.into(),
        "backend.provider_library" => config.backend.provider_library = value.into(),
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
        "backend.library_dirs" => config.backend.library_dirs = defaults.backend.library_dirs,
        "backend.onnxruntime_library" => {
            config.backend.onnxruntime_library = defaults.backend.onnxruntime_library
        }
        "backend.sherpa_library" => config.backend.sherpa_library = defaults.backend.sherpa_library,
        "backend.provider_library" => {
            config.backend.provider_library = defaults.backend.provider_library
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
            "Interactive setup needs a terminal. Run `omawake setup all` to configure the default model and launcher, or `omawake setup runtime` / `omawake setup model --list` to inspect choices. Service installation is separate: `omawake setup systemd`."
        );
    }
    match command.unwrap_or(SetupCommand::Check { json: false }) {
        SetupCommand::Check { json } => app_setup::print_checks(config_path, paths, json),
        SetupCommand::Runtime {
            json,
            runtime: None,
            device: None,
            dir: None,
            apply: false,
        } if !json && setup_is_interactive() => guided_runtime(config_path, paths),
        SetupCommand::Runtime {
            json,
            runtime,
            device,
            dir,
            apply,
        } => {
            if runtime.is_some() || device.is_some() || dir.is_some() {
                let current = Config::load(config_path)?;
                let selected_runtime = runtime
                    .as_deref()
                    .map(parse_runtime)
                    .transpose()?
                    .unwrap_or(current.backend.runtime);
                let selected_device = device.unwrap_or_else(|| {
                    current
                        .backend
                        .canonical_device()
                        .ok()
                        .filter(|_| selected_runtime == current.backend.runtime)
                        .map_or_else(|| "auto".to_owned(), |_| current.backend.device.clone())
                });
                let candidate = runtime_selection_candidate(
                    &current,
                    config_path,
                    &RuntimeSelection {
                        runtime: selected_runtime,
                        device: selected_device,
                    },
                    dir.as_deref(),
                )?;
                let evidence = crate::runtime_inventory::apply_with(
                    &candidate,
                    config_path,
                    false,
                    crate::runtime_inventory::probe,
                )?;
                if apply {
                    prepare_and_save_runtime_candidate_with(
                        &candidate,
                        config_path,
                        paths,
                        ProgressFormat::Human,
                        app_setup::cache::prepare_for_runtime,
                    )?;
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &json!({"candidate":candidate.backend,"probe":evidence,"applied":apply})
                    )?
                );
                return Ok(());
            }
            app_setup::print_runtime(&Config::load(config_path)?, config_path, json)
        }
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
                        serde_json::to_string_pretty(crate::catalog::models())?
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
                set_selected_model(
                    &id,
                    config_path,
                    paths,
                    progress_format,
                    app_setup::model::verify,
                )?;
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
                        "Supply a licensed model archive with `omawake setup model --download {default_id} --archive /path/to/model.tar.bz2`."
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
            progress_format,
        } => {
            let spec = model_spec(&model)?;
            let service_was_active = app_setup::systemd::is_active();
            install_everything(
                spec,
                config_path,
                paths,
                archive.as_deref(),
                progress_format,
                service_was_active,
                |config, path| {
                    crate::runtime_inventory::apply_with(
                        config,
                        path,
                        false,
                        crate::runtime_inventory::probe,
                    )?;
                    Ok(())
                },
                app_setup::model::install,
                app_setup::menu::install,
                app_setup::systemd::reload_if_was_active,
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
    fn probe_runtime(
        &mut self,
        candidate: &Config,
        path: &Path,
    ) -> Result<crate::runtime_inventory::Probe> {
        crate::runtime_inventory::apply_with(
            candidate,
            path,
            false,
            crate::runtime_inventory::probe,
        )
    }
    fn setup_mode(&mut self) -> Result<Option<SetupMode>>;
    fn runtime(&mut self, current: &Config) -> Result<Option<RuntimeSelection>>;
    fn runtime_library_dir(&mut self, _current: &Config) -> Result<Option<PathBuf>> {
        Ok(None)
    }
    fn confirm_runtime(
        &mut self,
        candidate: &Config,
        evidence: &crate::runtime_inventory::Probe,
    ) -> Result<bool>;
    fn model(
        &mut self,
        paths: &AppPaths,
        current: &Config,
    ) -> Result<Option<&'static crate::catalog::ModelSpec>>;
    fn model_archive(
        &mut self,
        paths: &AppPaths,
        spec: &crate::catalog::ModelSpec,
    ) -> Result<Option<PathBuf>>;
    fn confirm(
        &mut self,
        selection: &RuntimeSelection,
        model: &str,
        service_was_active: bool,
    ) -> Result<bool>;
}

struct TerminalGuidedPrompts {
    config_path: PathBuf,
}

impl GuidedPrompts for TerminalGuidedPrompts {
    fn setup_mode(&mut self) -> Result<Option<SetupMode>> {
        wizard::choose_setup_mode()
    }

    fn runtime(&mut self, current: &Config) -> Result<Option<RuntimeSelection>> {
        let libraries = runtime_paths::report(&current.backend, &self.config_path);
        wizard::choose_runtime(
            &libraries.runtime_loadable,
            &libraries.tui_context(),
            current.backend.runtime,
            &current.backend.device,
        )
    }

    fn runtime_library_dir(&mut self, current: &Config) -> Result<Option<PathBuf>> {
        wizard::choose_runtime_directory(&current.backend.library_dirs)
    }

    fn confirm_runtime(
        &mut self,
        candidate: &Config,
        evidence: &crate::runtime_inventory::Probe,
    ) -> Result<bool> {
        Ok(wizard::select(
            "Review runtime",
            &serde_json::to_string_pretty(&json!({
                "paths": candidate.backend,
                "probe": evidence,
            }))?,
            &[
                wizard::MenuItem::available(
                    "Apply",
                    "Save the verified runtime selection atomically",
                ),
                wizard::MenuItem::available("Cancel", "Leave config unchanged"),
            ],
            1,
        )? == Some(0))
    }

    fn model(
        &mut self,
        paths: &AppPaths,
        current: &Config,
    ) -> Result<Option<&'static crate::catalog::ModelSpec>> {
        choose_model(paths, &current.model.name)
    }

    fn model_archive(
        &mut self,
        paths: &AppPaths,
        spec: &crate::catalog::ModelSpec,
    ) -> Result<Option<PathBuf>> {
        if spec.downloadable || app_setup::model::verify(paths, spec).is_ok() {
            Ok(None)
        } else {
            wizard::choose_model_archive(spec.id)
        }
    }

    fn confirm(
        &mut self,
        selection: &RuntimeSelection,
        model: &str,
        service_was_active: bool,
    ) -> Result<bool> {
        wizard::confirm_apply(
            selection.runtime,
            &selection.device,
            model,
            service_was_active,
        )
    }
}

fn guided_setup(config_path: &Path, paths: &AppPaths) -> Result<()> {
    guided_setup_with(
        config_path,
        paths,
        &mut TerminalGuidedPrompts {
            config_path: config_path.to_owned(),
        },
    )
}

fn guided_setup_with(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
) -> Result<()> {
    match prompts.setup_mode()? {
        Some(SetupMode::Full) => guided_all_with(config_path, paths, prompts),
        Some(SetupMode::Runtime) => guided_runtime_with(config_path, paths, prompts),
        Some(SetupMode::Model) => guided_model_with(config_path, paths, prompts),
        Some(SetupMode::Check) => app_setup::print_checks(config_path, paths, false),
        None => {
            println!("Setup cancelled.");
            Ok(())
        }
    }
}

fn guided_runtime(config_path: &Path, paths: &AppPaths) -> Result<()> {
    guided_runtime_with(
        config_path,
        paths,
        &mut TerminalGuidedPrompts {
            config_path: config_path.to_owned(),
        },
    )
}

fn guided_runtime_with(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
) -> Result<()> {
    let current = Config::load(config_path)?;
    let Some(selection) = prompts.runtime(&current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let runtime_directory = prompts.runtime_library_dir(&current)?;
    let candidate = runtime_selection_candidate(
        &current,
        config_path,
        &selection,
        runtime_directory.as_deref(),
    )?;
    let evidence = prompts.probe_runtime(&candidate, config_path)?;
    if !prompts.confirm_runtime(&candidate, &evidence)? {
        println!("Runtime setup cancelled; no changes were made.");
        return Ok(());
    }
    prepare_and_save_runtime_candidate_with(
        &candidate,
        config_path,
        paths,
        ProgressFormat::Human,
        app_setup::cache::prepare_for_runtime,
    )?;
    println!(
        "runtime configured: {} / {}",
        runtime_name(selection.runtime),
        selection.device
    );
    Ok(())
}

fn guided_model(config_path: &Path, paths: &AppPaths) -> Result<()> {
    guided_model_with(
        config_path,
        paths,
        &mut TerminalGuidedPrompts {
            config_path: config_path.to_owned(),
        },
    )
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
    FV: FnOnce(&AppPaths, &crate::catalog::ModelSpec) -> Result<()>,
    FI: FnOnce(
        &AppPaths,
        &crate::catalog::ModelSpec,
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
        activate_model(config_path, paths, spec, ProgressFormat::Human)?;
        println!("active model: {}", spec.id);
        return Ok(());
    }
    let archive = prompts.model_archive(paths, spec)?;
    if !spec.downloadable && archive.is_none() {
        println!("Model setup cancelled; no changes were made.");
        return Ok(());
    }
    install_selected_model(
        spec.id,
        config_path,
        paths,
        archive.as_deref(),
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
        |_| app_setup::systemd::is_active(),
        app_setup::systemd::reload_if_was_active,
        |config, paths| app_setup::print_checks(config, paths, false),
        app_setup::print_checks_event,
    )
}

#[allow(clippy::too_many_arguments)]
fn guided_all_with_services<FI, FM, FA, FR, CH, CJ>(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
    install_model: FI,
    install_menu: FM,
    service_is_active: FA,
    restart_service: FR,
    check_human: CH,
    check_json: CJ,
) -> Result<()>
where
    FI: FnOnce(
        &AppPaths,
        &crate::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FA: FnOnce(&AppPaths) -> bool,
    FR: FnOnce(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    guided_all_with_services_and_validator(
        config_path,
        paths,
        prompts,
        validate_runtime_candidate,
        install_model,
        install_menu,
        service_is_active,
        restart_service,
        check_human,
        check_json,
    )
}

#[allow(clippy::too_many_arguments)]
fn guided_all_with_services_and_validator<FI, FM, FA, FR, CH, CJ, FV>(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
    validate_runtime: FV,
    install_model: FI,
    install_menu: FM,
    service_is_active: FA,
    restart_service: FR,
    check_human: CH,
    check_json: CJ,
) -> Result<()>
where
    FI: FnOnce(
        &AppPaths,
        &crate::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FA: FnOnce(&AppPaths) -> bool,
    FR: FnOnce(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
    FV: FnOnce(&Config, &Path) -> Result<()>,
{
    let current = Config::load(config_path)?;
    let Some(selection) = prompts.runtime(&current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let runtime_directory = prompts.runtime_library_dir(&current)?;
    let candidate = runtime_selection_candidate(
        &current,
        config_path,
        &selection,
        runtime_directory.as_deref(),
    )?;
    validate_runtime(&candidate, config_path)?;
    let Some(spec) = prompts.model(paths, &current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let archive = prompts.model_archive(paths, spec)?;
    if !spec.downloadable && archive.is_none() && app_setup::model::verify(paths, spec).is_err() {
        println!("Setup cancelled; no changes were made.");
        return Ok(());
    }
    let service_was_active = service_is_active(paths);
    if !prompts.confirm(&selection, spec.id, service_was_active)? {
        println!("Setup cancelled.");
        return Ok(());
    }
    install_everything_with_config(
        spec,
        candidate,
        config_path,
        paths,
        archive.as_deref(),
        ProgressFormat::Human,
        service_was_active,
        install_model,
        install_menu,
        restart_service,
        check_human,
        check_json,
    )
}

fn choose_model(
    paths: &AppPaths,
    active_model: &str,
) -> Result<Option<&'static crate::catalog::ModelSpec>> {
    choose_model_with(paths, active_model, |items, preferred| {
        wizard::select(
            "Wake-word model",
            "● active · ○ installed · unverified catalog models require a user-supplied archive",
            items,
            preferred,
        )
    })
}

fn choose_model_with<S>(
    paths: &AppPaths,
    active_model: &str,
    select: S,
) -> Result<Option<&'static crate::catalog::ModelSpec>>
where
    S: FnOnce(&[wizard::MenuItem], usize) -> Result<Option<usize>>,
{
    let items: Vec<_> = crate::catalog::models()
        .iter()
        .map(|model| {
            let installed = app_setup::model::verify(paths, model).is_ok();
            let status = if model.id == active_model && installed {
                "● active"
            } else if model.id == active_model && !model.downloadable {
                "● active · user-supplied archive required"
            } else if model.id == active_model {
                "● active · download required"
            } else if installed {
                "○ installed"
            } else if !model.downloadable {
                "· user-supplied only"
            } else {
                "· download"
            };
            let detail = format!(
                "{} · backend: {} · family: {} · license: {} ({}) · {:.1} MiB",
                model.description,
                model.backend,
                model.family,
                model.license,
                model.license_status,
                model.archive_size as f64 / 1_048_576.0
            );
            let detail = if installed || model.downloadable {
                detail
            } else {
                format!("{detail} · select to provide a licensed local archive")
            };
            wizard::MenuItem::available(format!("{status}  {}", model.id), detail)
        })
        .collect();
    let preferred = crate::catalog::models()
        .iter()
        .position(|model| model.id == active_model)
        .unwrap_or(0);
    Ok(select(&items, preferred)?.map(|index| &crate::catalog::models()[index]))
}

#[cfg(test)]
fn save_runtime_selection(config_path: &Path, selection: &RuntimeSelection) -> Result<()> {
    save_runtime_selection_impl_with(config_path, selection, None, |_, _| Ok(()))
}

#[cfg(test)]
fn save_runtime_selection_impl_with(
    config_path: &Path,
    selection: &RuntimeSelection,
    runtime_directory: Option<&Path>,
    validate: impl FnOnce(&Config, &Path) -> Result<()>,
) -> Result<()> {
    let current = Config::load(config_path)?;
    let config = runtime_selection_candidate(&current, config_path, selection, runtime_directory)?;
    validate(&config, config_path)?;
    config.save(config_path)
}

fn prepare_and_save_runtime_candidate_with<F>(
    config: &Config,
    config_path: &Path,
    paths: &AppPaths,
    progress: ProgressFormat,
    prepare_cache: F,
) -> Result<()>
where
    F: FnOnce(
        &Config,
        &Path,
        &AppPaths,
        ProgressFormat,
    ) -> Result<Option<app_setup::cache::CacheReport>>,
{
    prepare_cache(config, config_path, paths, progress)?;
    config.save(config_path)
}

fn runtime_selection_candidate(
    current: &Config,
    config_path: &Path,
    selection: &RuntimeSelection,
    runtime_directory: Option<&Path>,
) -> Result<Config> {
    let mut config = current.clone();
    let runtime_changed = config.backend.runtime != selection.runtime;
    config.backend.runtime = selection.runtime;
    config.backend.device = selection.device.clone();
    if runtime_changed {
        config.backend.provider_config.clear();
        config.backend.provider_library.clear();
        config.backend.options.clear();
    }
    if selection.runtime != Runtime::Cuda {
        config.backend.device_id = 0;
    }
    config.backend.validate_shape()?;
    if let Some(directory) = runtime_directory {
        configure_runtime_directory(&mut config, directory)?;
    }
    if let Some(spec) = crate::catalog::model(&config.model.name) {
        spec.apply_runtime_compatibility(&mut config);
    }
    // Keep packaged CPU paths relocatable. Persist exact paths for external stacks.
    let locations = runtime_paths::discover(&config.backend, config_path);
    let packaged = [&locations.onnxruntime_library, &locations.sherpa_library]
        .iter()
        .all(|library| {
            library.as_ref().is_some_and(|library| {
                locations
                    .package_library_dirs
                    .iter()
                    .any(|directory| library.starts_with(directory))
            })
        });
    if runtime_directory.is_some()
        || selection.runtime != Runtime::Default
        || !config.backend.library_dirs.is_empty()
        || !packaged
    {
        config.backend = crate::runtime_inventory::resolve(&config.backend, config_path);
    }
    Ok(config)
}

fn validate_runtime_candidate(config: &Config, config_path: &Path) -> Result<()> {
    validate_runtime_candidate_with(config, config_path, crate::runtime_inventory::probe)
}

fn validate_runtime_candidate_with(
    config: &Config,
    config_path: &Path,
    probe: impl FnOnce(&crate::backend::BackendConfig, &Path) -> crate::runtime_inventory::Probe,
) -> Result<()> {
    crate::runtime_inventory::apply_with(config, config_path, false, probe)?;
    Ok(())
}

#[cfg(test)]
fn validate_runtime_candidate_report(
    runtime: Runtime,
    report: &runtime_paths::RuntimeLibraryReport,
) -> Result<()> {
    let name = runtime_name(runtime);
    if report.runtime_loadable.get(name) == Some(&true) {
        return Ok(());
    }
    let detail = if report.remediation.is_empty() {
        "runtime ABI, provider registration, or requested device probe failed".to_owned()
    } else {
        report.remediation.join("; ")
    };
    bail!("{name} runtime validation failed; configuration was not changed: {detail}")
}

fn configure_runtime_directory(config: &mut Config, directory: &Path) -> Result<()> {
    if !directory.is_absolute() || !directory.is_dir() {
        bail!(
            "runtime library directory must be an absolute existing directory: {}",
            directory.display()
        );
    }
    let sherpa_library = runtime_paths::packaged_library("libsherpa-onnx-c-api.so").context(
        "this Omawake installation is missing its bundled extended sherpa library; reinstall the release package or copy its lib directory beside the executable",
    )?;
    configure_runtime_directory_with(config, directory, sherpa_library)
}

fn configure_runtime_directory_with(
    config: &mut Config,
    directory: &Path,
    sherpa_library: PathBuf,
) -> Result<()> {
    if !directory.is_absolute() || !directory.is_dir() {
        bail!(
            "runtime library directory must be an absolute existing directory: {}",
            directory.display()
        );
    }
    let candidates = [
        directory.to_owned(),
        directory.join("lib"),
        directory.join("lib64"),
        directory.join("runtime/lib/intel64"),
        directory.join("runtime/lib/intel64/Release"),
    ];
    let find = |prefix: &str| -> Result<PathBuf> {
        let mut matches = candidates
            .iter()
            .filter_map(|candidate| std::fs::read_dir(candidate).ok())
            .flatten()
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                (entry
                    .file_type()
                    .is_ok_and(|kind| kind.is_file() || kind.is_symlink())
                    && (name == prefix || name.starts_with(&format!("{prefix}."))))
                .then(|| entry.path())
            })
            .collect::<Vec<_>>();
        matches.sort();
        matches
            .into_iter()
            .next()
            .with_context(|| format!("{} does not contain {prefix}", directory.display()))
    };
    let onnxruntime_library = find("libonnxruntime.so")?;
    let provider_library = match config.backend.runtime {
        Runtime::Default => PathBuf::new(),
        Runtime::Openvino => find("libonnxruntime_providers_openvino.so")?,
        Runtime::Cuda => find("libonnxruntime_providers_cuda.so")?,
    };
    config.backend.library_dirs = [
        onnxruntime_library.parent(),
        sherpa_library.parent(),
        provider_library.parent(),
    ]
    .into_iter()
    .flatten()
    .map(Path::to_owned)
    .fold(Vec::new(), |mut directories, path| {
        if !directories.contains(&path) {
            directories.push(path);
        }
        directories
    });
    config.backend.onnxruntime_library = onnxruntime_library;
    config.backend.sherpa_library = sherpa_library;
    config.backend.provider_library = provider_library;
    Ok(())
}

fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
    }
}

#[allow(clippy::too_many_arguments)]
fn install_everything<FV, FI, FM, FR, CH, CJ>(
    spec: &crate::catalog::ModelSpec,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    progress_format: ProgressFormat,
    service_was_active: bool,
    validate_runtime: FV,
    install_model: FI,
    install_menu: FM,
    restart_service: FR,
    check_human: CH,
    check_json: CJ,
) -> Result<()>
where
    FV: FnOnce(&Config, &Path) -> Result<()>,
    FI: FnOnce(
        &AppPaths,
        &crate::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FR: FnOnce(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    let config = Config::load(config_path)?;
    validate_runtime(&config, config_path)?;
    install_everything_with_config(
        spec,
        config,
        config_path,
        paths,
        archive,
        progress_format,
        service_was_active,
        install_model,
        install_menu,
        restart_service,
        check_human,
        check_json,
    )
}

#[allow(clippy::too_many_arguments)]
fn install_everything_with_config<FI, FM, FR, CH, CJ>(
    spec: &crate::catalog::ModelSpec,
    config: Config,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    progress_format: ProgressFormat,
    service_was_active: bool,
    install_model: FI,
    install_menu: FM,
    restart_service: FR,
    check_human: CH,
    check_json: CJ,
) -> Result<()>
where
    FI: FnOnce(
        &AppPaths,
        &crate::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FR: FnOnce(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    install_everything_with_config_and_cache(
        spec,
        config,
        config_path,
        paths,
        archive,
        progress_format,
        service_was_active,
        install_model,
        app_setup::cache::prepare,
        install_menu,
        restart_service,
        check_human,
        check_json,
    )
}

#[allow(clippy::too_many_arguments)]
fn install_everything_with_config_and_cache<FI, FP, FM, FR, CH, CJ>(
    spec: &crate::catalog::ModelSpec,
    mut config: Config,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    progress_format: ProgressFormat,
    service_was_active: bool,
    install_model: FI,
    prepare_cache: FP,
    install_menu: FM,
    restart_service: FR,
    check_human: CH,
    check_json: CJ,
) -> Result<()>
where
    FI: FnOnce(
        &AppPaths,
        &crate::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
    FP: FnOnce(
        &Config,
        &Path,
        &AppPaths,
        ProgressFormat,
    ) -> Result<Option<app_setup::cache::CacheReport>>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FR: FnOnce(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    let original = config_snapshot(config_path)?;
    let result = (|| {
        let directory = install_model(paths, spec, archive, progress_format)?;
        spec.activate(&mut config);
        prepare_cache(&config, config_path, paths, progress_format)?;
        config.save(config_path)?;
        let launcher = install_menu(paths)?;
        match progress_format {
            ProgressFormat::Human => check_human(config_path, paths)?,
            ProgressFormat::Json => check_json(config_path, paths)?,
        }
        let service_restarted = restart_service(service_was_active)?;
        print_setup_complete(
            &directory,
            config_path,
            &launcher,
            service_was_active,
            service_restarted,
            progress_format,
        )
    })();
    if let Err(error) = result {
        if let Err(restore_error) = restore_config_snapshot(config_path, original.as_deref()) {
            return Err(error.context(format!(
                "setup also failed to restore the prior config: {restore_error:#}"
            )));
        }
        return Err(error);
    }
    Ok(())
}

fn config_snapshot(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("read config snapshot {}", path.display()))
        }
    }
}

fn restore_config_snapshot(path: &Path, bytes: Option<&[u8]>) -> Result<()> {
    let temporary = path.with_extension("toml.tmp");
    match bytes {
        Some(bytes) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&temporary, bytes)?;
            fs::rename(&temporary, path)?;
        }
        None => {
            if path.exists() {
                fs::remove_file(path)?;
            }
            if temporary.exists() {
                fs::remove_file(temporary)?;
            }
        }
    }
    Ok(())
}

fn verify_selected_model<F>(id: &str, paths: &AppPaths, verify: F) -> Result<()>
where
    F: FnOnce(&AppPaths, &crate::catalog::ModelSpec) -> Result<()>,
{
    let spec = model_spec(id)?;
    verify(paths, spec)?;
    println!(
        "verified: {}",
        app_setup::model::model_directory(paths, spec).display()
    );
    Ok(())
}

fn set_selected_model<F>(
    id: &str,
    config_path: &Path,
    paths: &AppPaths,
    progress: ProgressFormat,
    verify: F,
) -> Result<()>
where
    F: FnOnce(&AppPaths, &crate::catalog::ModelSpec) -> Result<()>,
{
    let spec = model_spec(id)?;
    verify(paths, spec)?;
    activate_model(config_path, paths, spec, progress)?;
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
        &crate::catalog::ModelSpec,
        Option<&Path>,
        ProgressFormat,
    ) -> Result<PathBuf>,
{
    let spec = model_spec(id)?;
    let directory = install(paths, spec, archive, progress_format)?;
    if !no_activate {
        activate_model(config_path, paths, spec, progress_format)?;
    }
    print_model_ready(&directory, spec, !no_activate, progress_format)
}

fn print_setup_complete(
    directory: &Path,
    config_path: &Path,
    launcher: &Path,
    service_was_active: bool,
    service_restarted: bool,
    progress_format: ProgressFormat,
) -> Result<()> {
    match progress_format {
        ProgressFormat::Human => println!(
            "model: {}\nconfig: {}\nlauncher: {}\ndaemon: run `omawake daemon`\nservice: {}",
            directory.display(),
            config_path.display(),
            launcher.display(),
            if service_restarted {
                "restarted the already-active service"
            } else {
                "unchanged (optional; run `omawake setup systemd` to install)"
            }
        ),
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "setup-complete",
                "model": directory,
                "config": config_path,
                "launcher": launcher,
                "daemon_command": "omawake daemon",
                "service": {
                    "modified": service_restarted,
                    "active_before": service_was_active,
                    "restarted": service_restarted,
                    "unit_modified": false,
                    "action": if service_restarted { "restarted" } else { "unchanged" },
                    "install_command": "omawake setup systemd",
                },
            }))?
        ),
    }
    Ok(())
}

fn activate_model(
    config_path: &Path,
    paths: &AppPaths,
    spec: &crate::catalog::ModelSpec,
    progress: ProgressFormat,
) -> Result<()> {
    activate_model_with_cache(
        config_path,
        paths,
        spec,
        progress,
        app_setup::cache::prepare,
    )
}

fn activate_model_with_cache<F>(
    config_path: &Path,
    paths: &AppPaths,
    spec: &crate::catalog::ModelSpec,
    progress: ProgressFormat,
    prepare_cache: F,
) -> Result<()>
where
    F: FnOnce(
        &Config,
        &Path,
        &AppPaths,
        ProgressFormat,
    ) -> Result<Option<app_setup::cache::CacheReport>>,
{
    let mut config = app_setup::ensure_config(config_path)?;
    spec.activate(&mut config);
    prepare_cache(&config, config_path, paths, progress)?;
    config.save(config_path)
}

fn print_model_ready(
    directory: &Path,
    spec: &crate::catalog::ModelSpec,
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

fn model_spec(id: &str) -> Result<&'static crate::catalog::ModelSpec> {
    crate::catalog::model(id)
        .ok_or_else(|| anyhow::anyhow!("unknown model {id}; run `omawake setup model --list`"))
}

fn print_models(paths: &AppPaths) {
    for model in crate::catalog::models() {
        let status = if app_setup::model::verify(paths, model).is_ok() {
            "installed"
        } else if !model.downloadable {
            "user-supplied"
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
            "placement_verified": detector.effective_runtime() == Runtime::Default,
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
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(SIGINT, Arc::clone(&shutdown_requested))?;
    signal_hook::flag::register(SIGTERM, Arc::clone(&shutdown_requested))?;
    let listener = bind_socket(paths)?;
    let socket_metadata = fs::symlink_metadata(socket_path(paths)).with_context(|| {
        format!(
            "inspect bound daemon socket {}",
            socket_path(paths).display()
        )
    })?;
    eprintln!(
        "loaded wake-word model in {} ms",
        detector.load_time.as_millis()
    );
    let serve_result = (|| -> Result<()> {
        let mut paused = false;
        let mut shutdown = false;
        while !shutdown && !shutdown_requested.load(Ordering::Relaxed) {
            if paused {
                if let Some(command) =
                    poll_control(&detector, "paused", None, || accept_control(&listener))?
                {
                    apply_daemon_command(command, &mut paused, &mut shutdown);
                }
                if shutdown_requested.load(Ordering::Relaxed) {
                    shutdown = true;
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
                || shutdown_requested.load(Ordering::Relaxed),
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
    finish_daemon(paths, Some(&socket_metadata), serve_result)
}

fn finish_daemon(
    paths: &AppPaths,
    socket_metadata: Option<&fs::Metadata>,
    serve_result: Result<()>,
) -> Result<()> {
    let path = socket_path(paths);
    let cleanup_result = socket_metadata.map_or(Ok(()), |metadata| {
        remove_socket_if_unchanged(&path, metadata)
    });
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
    mut should_shutdown: impl FnMut() -> bool,
) -> Result<(Vec<Detection>, Option<Command>)>
where
    P: FnMut(Value) -> Result<Option<Command>>,
    R: FnMut(Duration) -> std::result::Result<AudioEvent, RecvTimeoutError>,
    A: FnMut(i32, &[f32]) -> Result<Vec<Detection>>,
{
    let mut triggered = Vec::new();
    loop {
        if should_shutdown() {
            return Ok((triggered, Some(Command::Shutdown)));
        }
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
    let existing = match fs::symlink_metadata(&path) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
    };
    if let Some(metadata) = existing {
        if !metadata.file_type().is_socket() {
            bail!(
                "refusing to remove non-socket daemon path {}",
                path.display()
            );
        }
        if is_live(&path) {
            bail!("daemon is already running at {}", path.display());
        }
        remove_socket_if_unchanged(&path, &metadata)?;
    }
    let listener = bind(&path).with_context(|| format!("bind {}", path.display()))?;
    let bound_metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("inspect bound daemon socket {}", path.display()))?;
    if !bound_metadata.file_type().is_socket() {
        drop(listener);
        bail!("bound daemon path is not a socket: {}", path.display());
    }
    if let Err(error) = fs::set_permissions(&path, fs::Permissions::from_mode(0o600)) {
        drop(listener);
        let _ = remove_socket_if_unchanged(&path, &bound_metadata);
        return Err(error.into());
    }
    if let Err(error) = set_nonblocking(&listener) {
        drop(listener);
        let _ = remove_socket_if_unchanged(&path, &bound_metadata);
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
    request_with_connector(paths, command, connect_control_socket)
}

fn connect_control_socket(path: &Path) -> std::io::Result<UnixStream> {
    connect_control_socket_with(path, |path| UnixStream::connect(path))
}

fn connect_control_socket_with<S>(
    path: &Path,
    connect: impl FnOnce(&Path) -> std::io::Result<S>,
) -> std::io::Result<S> {
    let before = fs::symlink_metadata(path).ok();
    match connect(path) {
        Ok(stream) => Ok(stream),
        Err(error) => {
            if indicates_stale_socket(error.kind()) {
                remove_stale_socket_if_unchanged(path, before.as_ref());
            }
            Err(error)
        }
    }
}

fn indicates_stale_socket(kind: std::io::ErrorKind) -> bool {
    matches!(
        kind,
        std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
    )
}

fn remove_stale_socket_if_unchanged(path: &Path, before: Option<&fs::Metadata>) {
    let Some(before) = before else {
        return;
    };
    let _ = remove_socket_if_unchanged(path, before);
}

fn remove_socket_if_unchanged(path: &Path, before: &fs::Metadata) -> Result<()> {
    if !before.file_type().is_socket() {
        bail!(
            "refusing to remove non-socket daemon path {}",
            path.display()
        );
    }
    let Ok(after) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if !after.file_type().is_socket() || before.dev() != after.dev() || before.ino() != after.ino()
    {
        bail!(
            "daemon socket {} changed; refusing to remove it",
            path.display()
        );
    }
    fs::remove_file(path).with_context(|| format!("remove daemon socket {}", path.display()))
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
        "backend":{"kind":config.backend.kind,"requested":{"runtime":config.backend.runtime,"device":config.backend.device},"effective":null,"supported_capabilities":supported_capabilities(),"fallback_policy":config.backend.fallback,"fallback_used":false,"placement_verified":false,"evidence":[]},
        "model":{"family":"zipformer-kws","path":config.model_directory(paths),"loaded":false},"last_error":null,"details":{}})
}

fn schema(config: &Config, config_path: &Path, paths: &AppPaths) -> serde_json::Value {
    json!({"schema_version":1,"app":"omawake","app_version":env!("CARGO_PKG_VERSION"),"daemon_version":env!("CARGO_PKG_VERSION"),"config_path":config_path,
        "keys":[
            {"key":"backend.kind","type":"enum","section":"Backend","label":"Backend","description":"Inference engine","value":config.backend.kind,"file_value":null,"supported":true,"restart_required":true,"choices":["sherpa-onnx"]},
            {"key":"backend.runtime","type":"enum","section":"Backend","label":"Runtime","description":"ONNX Runtime provider","value":config.backend.runtime,"file_value":null,"supported":true,"restart_required":true,"choices":[{"value":"default","available":true,"capability":"cpu"},{"value":"openvino","available":supported_capabilities().contains(&"openvino"),"capability":"openvino"},{"value":"cuda","available":supported_capabilities().contains(&"cuda"),"capability":"cuda"}]},
            {"key":"backend.device","type":"string","section":"Backend","label":"Device","description":"Runtime-specific device","value":config.backend.device,"file_value":null,"supported":true,"restart_required":true},
            {"key":"backend.library_dirs","type":"path-list","section":"Backend","label":"Native library directories","description":"App-owned vendor runtime search paths","value":config.backend.library_dirs,"file_value":null,"supported":true,"restart_required":true},
            {"key":"backend.onnxruntime_library","type":"path","section":"Backend","label":"ONNX Runtime library","description":"Exact external ONNX Runtime shared library","value":config.backend.onnxruntime_library,"file_value":null,"supported":true,"restart_required":true},
            {"key":"backend.sherpa_library","type":"path","section":"Backend","label":"Sherpa library","description":"Exact patched sherpa-onnx C API shared library","value":config.backend.sherpa_library,"file_value":null,"supported":true,"restart_required":true},
            {"key":"backend.provider_library","type":"path","section":"Backend","label":"Provider library","description":"Exact OpenVINO or CUDA execution-provider plugin","value":config.backend.provider_library,"file_value":null,"supported":true,"restart_required":true},
            {"key":"model.directory","type":"path","section":"Model","label":"Directory","description":"Model asset directory","value":config.model_directory(paths),"file_value":config.model.directory,"supported":true,"restart_required":true}],
        "collections":[{"key":"wake_words","id_key":"id","label":"Wake words","items":config.wake_words}],
        "constraints":[{"kind":"matrix","keys":["backend.runtime","backend.device"],"rows":[{"backend.runtime":"default","backend.device":["auto","cpu"]},{"backend.runtime":"cuda","backend.device":["auto","gpu"]},{"backend.runtime":"openvino","backend.device":["auto","npu","gpu","cpu"]}]}]})
}

#[cfg(test)]
#[path = "../tests/unit/app_main.rs"]
mod tests;
