mod assisted;
mod feedback;
mod onboarding;
mod pause_ownership;
mod setup_home;
mod training;

use crate::setup::wizard::MenuItem;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::audio::{AudioEvent, Capture, input_devices};
use crate::backend::{Fallback, Runtime, supported_capabilities};
use crate::config::{Config, WakeWord};
use crate::engine::{ActionResult, Detection, Detector, wav_duration};
use crate::evaluation::{self, EvaluationContext, RuntimeIdentity};
use crate::keyword::validate_wake_words;
use crate::paths::AppPaths;
use crate::protocol::{Command, Request, Response, ResultPayload};
use crate::setup as app_setup;
use crate::setup::model::ProgressFormat;
#[cfg(test)]
use crate::setup::wizard::SetupMode;
use crate::setup::wizard::{self, RuntimeSelection};
use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
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
    #[command(name = "__embedding-worker", hide = true)]
    EmbeddingWorker { socket: PathBuf },
    #[command(name = "__model-cache-prepare", hide = true)]
    ModelCachePrepare {
        candidate: String,
        response: PathBuf,
    },
    #[command(name = "__native-json", hide = true)]
    NativeJson { request: String, response: PathBuf },
    #[command(name = "__whisper-probe", hide = true)]
    WhisperProbe { library: PathBuf, response: PathBuf },
    #[command(name = "__whisper-worker", hide = true)]
    WhisperWorker {
        library: PathBuf,
        verifier: PathBuf,
        vad: PathBuf,
        threads: i32,
    },
    #[command(name = "__audiocpp-worker", hide = true)]
    AudioCppWorker {
        library: PathBuf,
        verifier: PathBuf,
        vad: PathBuf,
        threads: i32,
        asr_family: String,
        backend: String,
        device: i32,
    },
    #[command(name = "__openvino-genai-worker", hide = true)]
    OpenVinoGenAiWorker {
        genai_library: PathBuf,
        core_library: PathBuf,
        audiocpp_library: PathBuf,
        model_directory: PathBuf,
        vad_model: PathBuf,
        cache_directory: PathBuf,
        device: String,
        vad_threads: u16,
        /// Language code for multilingual verifiers; empty follows the model default.
        #[arg(default_value_t = String::new())]
        language: String,
    },
    #[command(name = "__openvino-runtime-worker", hide = true)]
    OpenVinoRuntimeWorker {
        genai_library: PathBuf,
        core_library: PathBuf,
        audiocpp_library: PathBuf,
        device: String,
    },
    /// Detect configured wake phrases from a WAV file or bounded live capture.
    Test {
        /// Read audio from a WAV file without opening the microphone.
        #[arg(long, conflicts_with = "seconds")]
        audio: Option<PathBuf>,
        /// Capture from the configured microphone for this many seconds.
        #[arg(long, conflicts_with = "audio")]
        seconds: Option<u64>,
        /// Run mapped actions for detected phrases.
        #[arg(long)]
        execute: bool,
        /// Print raw verifier transcripts to stderr for alias diagnosis.
        #[arg(long, conflicts_with = "json")]
        show_transcripts: bool,
        /// Emit a machine-readable result.
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
    /// Evaluate labeled WAV files and print a versioned accuracy report as JSON.
    Evaluate {
        #[arg(value_name = "MANIFEST")]
        manifest: PathBuf,
    },
    /// Show daemon state; a stopped daemon is reported successfully.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Inspect or edit configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    #[command(visible_alias = "word")]
    /// Manage phrase-to-action mappings.
    WakeWord {
        #[command(subcommand)]
        command: WakeWordCommand,
    },
    /// List available input devices.
    AudioDevices {
        #[arg(long)]
        json: bool,
        /// Return structured device records for settings and scripts.
        #[arg(long)]
        detailed: bool,
    },
    /// Run the foreground wake-phrase daemon.
    Daemon,
    /// Pause an already-running daemon.
    Pause,
    /// Resume an already-running daemon.
    Resume,
    /// Stop an already-running daemon.
    Stop,
    /// Configure providers, models, and optional integrations.
    Setup {
        #[command(subcommand)]
        command: Option<SetupCommand>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
enum NativeJsonRequest {
    FileTest {
        audio: PathBuf,
        execute: bool,
    },
    LiveTest {
        seconds: u64,
        execute: bool,
    },
    Benchmark {
        audio: Vec<PathBuf>,
        warmup: u32,
        iterations: u32,
    },
    Evaluate {
        manifest: PathBuf,
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
    /// Get/set the trained detector threshold; use auto to restore calibration.
    Threshold {
        id: String,
        value: Option<String>,
    },
    /// Opt-in live detection clips, review labels, and bounded local retention.
    History(feedback::HistoryArgs),
    /// Train an experimental phrase head using independent labeled recordings.
    Train(training::TrainArgs),
    /// Guided setup for Whisper spellings or experimental trainable KWS.
    Onboard(onboarding::OnboardArgs),
    /// List retained local enrollment recordings, or delete one session.
    Recordings {
        id: String,
        #[arg(long)]
        remove: Option<String>,
        #[arg(long)]
        json: bool,
    },
    List {
        #[arg(long)]
        json: bool,
    },
    Add {
        #[arg(long)]
        id: String,
        #[arg(long)]
        phrase: String,
        /// Exact alternate ASR transcript to accept (repeatable).
        #[arg(long = "alias")]
        aliases: Vec<String>,
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    /// Add an exact alternate ASR transcript to an existing wake word.
    AddAlias {
        id: String,
        alias: String,
    },
    /// Remove an alternate ASR transcript from an existing wake word.
    RemoveAlias {
        id: String,
        alias: String,
    },
    Remove {
        id: String,
    },
}

#[derive(Subcommand)]
enum SetupCommand {
    /// Select or test the audio device without loading an inference model.
    Audio {
        #[arg(long)]
        device: Option<String>,
        /// Save the choice and restart an already-active service.
        #[arg(long)]
        apply: bool,
        /// Run a short microphone level check or speaker test sound.
        #[arg(long)]
        test: bool,
    },
    Check {
        #[arg(long)]
        json: bool,
    },
    /// Install and configure a model plus the desktop launcher. Does not install a service;
    /// run `omawake setup systemd` to install one explicitly.
    All {
        /// Catalog model; defaults to the selected backend's compatible profile.
        #[arg(long)]
        model: Option<String>,
        #[arg(long, value_name = "DIRECTORY")]
        source_dir: Option<PathBuf>,
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
        /// HEAD every pinned catalog URL and compare the reported size, without
        /// downloading or changing user pins; exits nonzero on any failure.
        #[arg(long, conflicts_with_all = ["download", "set", "verify", "source_dir", "no_activate"])]
        check_urls: bool,
        /// Replace the origin of pinned URLs for mirror or stub testing.
        #[arg(long, value_name = "PREFIX", requires = "check_urls")]
        url_prefix: Option<String>,
        #[arg(long, value_name = "DIRECTORY", requires = "download")]
        source_dir: Option<PathBuf>,
        #[arg(long, requires = "download")]
        no_activate: bool,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    Runtime {
        #[arg(long)]
        json: bool,
        /// Select the runtime without opening the TUI.
        #[arg(long, value_parser = ["default", "openvino", "cuda", "vulkan", "hip"], conflicts_with = "json")]
        runtime: Option<String>,
        /// Select a compatible device without opening the TUI.
        #[arg(long, conflicts_with = "json")]
        device: Option<String>,
        /// Select a zero-based GPU index for CUDA, Vulkan, or HIP.
        #[arg(long, conflicts_with = "json")]
        device_id: Option<u32>,
        /// Complete provider or vendor runtime installation directory.
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
    match run_entry(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("omawake: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_entry(cli: Cli) -> Result<()> {
    if let TopCommand::EmbeddingWorker { socket } = &cli.command {
        return crate::engine::embedding_worker::main(socket);
    }
    let paths = AppPaths::discover();
    run_entry_with_workers(
        cli,
        paths,
        crate::engine::openvino_genai::runtime_worker_main,
        crate::engine::openvino_genai::worker_main,
        crate::engine::audiocpp::worker_main,
        crate::engine::whisper::worker_main,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_entry_with_workers<OR, OG, AC, WH>(
    cli: Cli,
    paths: AppPaths,
    openvino_runtime_worker: OR,
    openvino_genai_worker: OG,
    audiocpp_worker: AC,
    whisper_worker: WH,
) -> Result<()>
where
    OR: FnOnce(&Path, &Path, &Path, &str) -> Result<()>,
    OG: FnOnce(crate::engine::openvino_genai::ProviderSpec) -> Result<()>,
    AC: FnOnce(&Path, &Path, &Path, i32, &str, &str, i32) -> Result<()>,
    WH: FnOnce(&Path, &Path, &Path, i32) -> Result<()>,
{
    if let TopCommand::OpenVinoRuntimeWorker {
        genai_library,
        core_library,
        audiocpp_library,
        device,
    } = &cli.command
    {
        return openvino_runtime_worker(genai_library, core_library, audiocpp_library, device);
    }
    if let TopCommand::OpenVinoGenAiWorker {
        genai_library,
        core_library,
        audiocpp_library,
        model_directory,
        vad_model,
        cache_directory,
        device,
        vad_threads,
        language,
    } = &cli.command
    {
        let model_name = model_directory
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned();
        return openvino_genai_worker(crate::engine::openvino_genai::ProviderSpec {
            profile: crate::engine::openvino_genai::verifier_profile_for_name(&model_name),
            language: language.clone(),
            genai_library: genai_library.clone(),
            core_library: core_library.clone(),
            audiocpp_library: audiocpp_library.clone(),
            library_dirs: Vec::new(),
            model_directory: model_directory.clone(),
            vad_model: vad_model.clone(),
            cache_directory: cache_directory.clone(),
            placement_log: cache_directory.join("placement.log"),
            device: device.clone(),
            vad_threads: *vad_threads,
        });
    }
    if let TopCommand::AudioCppWorker {
        library,
        verifier,
        vad,
        threads,
        asr_family,
        backend,
        device,
    } = &cli.command
    {
        return audiocpp_worker(
            library, verifier, vad, *threads, asr_family, backend, *device,
        );
    }
    if let TopCommand::WhisperWorker {
        library,
        verifier,
        vad,
        threads,
    } = &cli.command
    {
        return whisper_worker(library, verifier, vad, *threads);
    }
    if let Some(request) = native_json_request(&cli.command) {
        let config_path = cli.config.as_deref().unwrap_or(&paths.config_file);
        return run_native_json_worker(config_path, &paths, &request, &std::env::current_exe()?);
    }
    run_with_paths(cli, paths)
}

fn native_json_request(command: &TopCommand) -> Option<NativeJsonRequest> {
    match command {
        TopCommand::Test {
            audio: Some(audio),
            execute,
            json: true,
            ..
        } => Some(NativeJsonRequest::FileTest {
            audio: audio.clone(),
            execute: *execute,
        }),
        TopCommand::Test {
            seconds: Some(seconds),
            execute,
            json: true,
            ..
        } => Some(NativeJsonRequest::LiveTest {
            seconds: *seconds,
            execute: *execute,
        }),
        TopCommand::Benchmark {
            audio,
            warmup,
            iterations,
        } => Some(NativeJsonRequest::Benchmark {
            audio: audio.clone(),
            warmup: *warmup,
            iterations: *iterations,
        }),
        TopCommand::Evaluate { manifest } => Some(NativeJsonRequest::Evaluate {
            manifest: manifest.clone(),
        }),
        _ => None,
    }
}

fn run_native_json_worker(
    config_path: &Path,
    paths: &AppPaths,
    request: &NativeJsonRequest,
    executable: &Path,
) -> Result<()> {
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    run_native_json_worker_with(
        config_path,
        paths,
        request,
        ProcessCommand::new(executable),
        stdout.lock(),
        stderr.lock(),
    )
}

fn run_native_json_worker_with(
    config_path: &Path,
    paths: &AppPaths,
    request: &NativeJsonRequest,
    mut command: ProcessCommand,
    mut output: impl Write,
    mut diagnostics: impl Write,
) -> Result<()> {
    let response = crate::native_worker::ResponseFile::create(&paths.runtime_dir, "native-json")?;
    let child = command
        .arg("--config")
        .arg(config_path)
        .arg("__native-json")
        .arg(serde_json::to_string(request)?)
        .arg(response.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("run isolated native inference worker")?;
    if !child.status.success() {
        diagnostics.write_all(&child.stdout)?;
        diagnostics.write_all(&child.stderr)?;
        bail!("native inference worker failed: {}", child.status);
    }
    let report: Value = match response.read_json() {
        Ok(report) => report,
        Err(error) => {
            diagnostics.write_all(&child.stdout)?;
            diagnostics.write_all(&child.stderr)?;
            return Err(error);
        }
    };
    serde_json::to_writer_pretty(&mut output, &report)?;
    writeln!(output)?;
    Ok(())
}

fn execute_native_json(
    config: &Config,
    paths: &AppPaths,
    request: NativeJsonRequest,
) -> Result<Value> {
    match request {
        NativeJsonRequest::FileTest { audio, execute } => {
            let detector = Detector::load(config, paths)?;
            let detections = detector.detect_file(&audio)?;
            detection_report(&detector, detections, execute)
        }
        NativeJsonRequest::LiveTest { seconds, execute } => {
            let detector = Detector::load(config, paths)?;
            let detections = detect_live(&detector, config, Duration::from_secs(seconds))?;
            detection_report(&detector, detections, execute)
        }
        NativeJsonRequest::Benchmark {
            audio,
            warmup,
            iterations,
        } => file_benchmark_report(config, paths, &audio, warmup, iterations),
        NativeJsonRequest::Evaluate { manifest } => evaluation_report(config, paths, &manifest),
    }
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
    paths: AppPaths,
    mut send_request: F,
    list_devices: D,
) -> Result<()>
where
    F: FnMut(&AppPaths, Command) -> Result<Response>,
    D: FnOnce() -> Result<Vec<String>>,
{
    run_with_paths_services(cli, paths, &mut send_request, list_devices)
}

fn run_with_paths_services<F, D>(
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
        TopCommand::ModelCachePrepare {
            candidate,
            response,
        } => {
            let mut candidate: Config = serde_json::from_str(&candidate)?;
            candidate.backend.fallback = Fallback::Error;
            crate::native_worker::write_json(
                &response,
                &paths.runtime_dir,
                &app_setup::cache::child(&candidate, &paths)?,
            )?;
            return Ok(());
        }
        TopCommand::WhisperProbe { library, response } => {
            let result = crate::engine::whisper::probe_library(&library)
                .map_err(|error| format!("{error:#}"));
            crate::native_worker::write_json(&response, &paths.runtime_dir, &result)?;
            return Ok(());
        }
        TopCommand::NativeJson { request, response } => {
            let request = serde_json::from_str(&request)?;
            let config = Config::load(&config_path)?;
            let report = execute_native_json(&config, &paths, request)?;
            crate::native_worker::write_json(&response, &paths.runtime_dir, &report)?;
            return Ok(());
        }
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
        TopCommand::Evaluate { manifest } => {
            let report = evaluation_report(&config, &paths, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        TopCommand::Test {
            audio: Some(audio),
            execute,
            show_transcripts,
            json: as_json,
            ..
        } => {
            let _transcript_diagnostics =
                show_transcripts.then(crate::phrase::enable_transcript_diagnostics);
            let detector = Detector::load(&config, &paths)?;
            present_detections(&detector, detector.detect_file(&audio)?, execute, as_json)
        }
        TopCommand::Test {
            seconds: Some(seconds),
            execute,
            show_transcripts,
            json: as_json,
            ..
        } => {
            let _transcript_diagnostics =
                show_transcripts.then(crate::phrase::enable_transcript_diagnostics);
            let detector = Detector::load(&config, &paths)?;
            let detections = detect_live(&detector, &config, Duration::from_secs(seconds))?;
            present_detections(&detector, detections, execute, as_json)
        }
        TopCommand::Test { .. } => bail!("test requires --audio FILE or --seconds N"),
        TopCommand::Daemon => run_daemon(&config, &paths),
        TopCommand::Status { json: as_json } => match send_request(&paths, Command::Status) {
            Ok(mut response) => {
                if let ResultPayload::State { details, .. } = &mut response.result {
                    details["audio"]["saved"] = json!(config.audio.device);
                    details["audio"]["restart_required"] = json!(
                        details["audio"]["requested"]
                            .as_str()
                            .is_some_and(|active| active != config.audio.device)
                    );
                }
                print_response(response, as_json)
            }
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
        TopCommand::AudioDevices {
            json: as_json,
            detailed,
        } => {
            if detailed {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&crate::audio::device_inventory(
                        &config.audio.device
                    ))?
                );
                return Ok(());
            }
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
                for line in human_schema_lines(&value)? {
                    println!("{line}");
                }
            }
            Ok(())
        }
        TopCommand::ModelCachePrepare { .. }
        | TopCommand::NativeJson { .. }
        | TopCommand::EmbeddingWorker { .. }
        | TopCommand::AudioCppWorker { .. }
        | TopCommand::OpenVinoGenAiWorker { .. }
        | TopCommand::OpenVinoRuntimeWorker { .. }
        | TopCommand::WhisperProbe { .. }
        | TopCommand::WhisperWorker { .. }
        | TopCommand::Setup { .. } => unreachable!(),
        TopCommand::Config { command } => config_mutation(command, config, &config_path, &paths),
    }
}

fn config_mutation(
    command: ConfigCommand,
    mut config: Config,
    path: &Path,
    paths: &AppPaths,
) -> Result<()> {
    let previous_runtime = config.backend.runtime;
    match command {
        ConfigCommand::Set { key, value } => set_config(&mut config, &key, &value)?,
        ConfigCommand::Unset { key } => unset_config(&mut config, &key)?,
        _ => unreachable!(),
    }
    let runtime_changed = config.backend.runtime != previous_runtime;
    if runtime_changed {
        config.backend.device = if config.backend.runtime == Runtime::Openvino {
            "cpu".into()
        } else {
            "auto".into()
        };
        config.backend.library.clear();
        config.backend.library_dirs.clear();
        config.backend.options.clear();
        config.backend.device_id = 0;
    }
    config.backend.validate_shape()?;
    save_and_reload_active(config, path, paths).map(|_| ())
}

fn set_config(config: &mut Config, key: &str, value: &str) -> Result<()> {
    match key {
        "backend.kind" => config.backend.kind = value.into(),
        "backend.runtime" => config.backend.runtime = parse_runtime(value)?,
        "backend.device" => config.backend.device = value.into(),
        "backend.threads" => config.backend.threads = value.parse()?,
        "backend.fallback" => config.backend.fallback = parse_fallback(value)?,
        "backend.device_id" => config.backend.device_id = value.parse()?,
        "backend.library" => config.backend.library = value.into(),
        "backend.library_dirs" => {
            config.backend.library_dirs = std::env::split_paths(value).collect()
        }
        "model.name" => config.model.name = value.into(),
        "model.directory" => config.model.directory = value.into(),
        "model.verifier" => config.model.verifier = value.into(),
        "model.vad" => config.model.vad = value.into(),
        "model.sample_rate" => config.model.sample_rate = value.parse()?,
        "model.language" => config.model.language = value.trim().to_owned(),
        "audio.device" => {
            crate::audio_devices::validate(value, "input")?;
            config.audio.device = value.into();
        }
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
        "backend.library" => config.backend.library = defaults.backend.library,
        "backend.library_dirs" => config.backend.library_dirs = defaults.backend.library_dirs,
        "model.name" => config.model.name = defaults.model.name,
        "model.directory" => config.model.directory = defaults.model.directory,
        "model.verifier" => config.model.verifier = defaults.model.verifier,
        "model.vad" => config.model.vad = defaults.model.vad,
        "model.sample_rate" => config.model.sample_rate = defaults.model.sample_rate,
        "model.language" => config.model.language = defaults.model.language,
        "audio.device" => config.audio.device = defaults.audio.device,
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
        WakeWordCommand::Threshold { id, value } => {
            return feedback::threshold(id, value, config, path, paths);
        }
        WakeWordCommand::History(args) => return feedback::run(args, config, path, paths),
        WakeWordCommand::Train(args) => return training::run(args, config, path, paths),
        WakeWordCommand::Onboard(args) => return onboarding::run(args, config, path, paths),
        WakeWordCommand::Recordings {
            id,
            remove,
            json: as_json,
        } => {
            if let Some(session) = remove {
                crate::enrollment::remove_recordings(paths, &id, &session)?;
            }
            let sessions = crate::enrollment::recordings(paths, &id)?;
            if as_json {
                println!("{}", serde_json::to_string_pretty(&sessions)?);
            } else {
                for session in sessions {
                    println!(
                        "{}\t{} clip(s)\t{}",
                        session.session,
                        session.clips,
                        session.directory.display()
                    );
                }
            }
            return Ok(());
        }
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
            aliases,
            command,
        } => {
            let message = format!("added wake word: {id}");
            config.wake_words.push(WakeWord {
                engine: None,
                enrollment: None,
                id,
                phrase,
                aliases,
                enabled: true,
                command,
            });
            message
        }
        WakeWordCommand::AddAlias { id, alias } => {
            let wake_word = config
                .wake_words
                .iter_mut()
                .find(|item| item.id == id)
                .with_context(|| format!("unknown wake-word id {id}"))?;
            wake_word.aliases.push(alias);
            format!("added transcript alias to wake word: {id}")
        }
        WakeWordCommand::RemoveAlias { id, alias } => {
            let wake_word = config
                .wake_words
                .iter_mut()
                .find(|item| item.id == id)
                .with_context(|| format!("unknown wake-word id {id}"))?;
            let before = wake_word.aliases.len();
            wake_word.aliases.retain(|candidate| candidate != &alias);
            if wake_word.aliases.len() == before {
                bail!("wake word {id} has no transcript alias {alias:?}");
            }
            format!("removed transcript alias from wake word: {id}")
        }
        WakeWordCommand::Remove { id } => {
            remove_wake_word(&mut config, &id)?;
            format!("removed wake word: {id}")
        }
    };
    validate_wake_words(&config.wake_words)?;
    save_and_reload_active(config, path, paths)?;
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

fn setup(command: Option<SetupCommand>, config_path: &Path, paths: &AppPaths) -> Result<()> {
    if let Some(error) = app_setup::config_recovery(config_path)? {
        eprintln!(
            "omawake setup: the existing configuration is invalid; a successful setup apply will replace it with the current schema\n  {error}"
        );
    }
    if command.is_none() && setup_is_interactive() {
        return setup_home::run(config_path, paths);
    }
    if command.is_none() {
        println!(
            "Interactive setup needs a terminal. Run `omawake setup all` to configure the default model and launcher, or `omawake setup runtime` / `omawake setup model --list` to inspect choices. Service installation is separate: `omawake setup systemd`."
        );
    }
    match command.unwrap_or(SetupCommand::Check { json: false }) {
        SetupCommand::Audio {
            device,
            apply,
            test,
        } => setup_audio(config_path, paths, device, apply, test),
        SetupCommand::Check { json } => app_setup::print_checks(config_path, paths, json),
        SetupCommand::Runtime {
            json,
            runtime: None,
            device: None,
            device_id: None,
            dir: None,
            apply: false,
        } if !json && setup_is_interactive() => guided_runtime(config_path, paths),
        SetupCommand::Runtime {
            json,
            runtime,
            device,
            device_id,
            dir,
            apply,
        } => {
            if runtime.is_some() || device.is_some() || device_id.is_some() || dir.is_some() {
                configure_runtime_from_flags(
                    config_path,
                    paths,
                    runtime,
                    device,
                    device_id,
                    dir,
                    apply,
                    native_runtime_probe,
                )?;
                return Ok(());
            }
            app_setup::print_runtime(&app_setup::load_config(config_path)?, config_path, json)
        }
        SetupCommand::Model {
            list,
            json,
            download,
            set,
            verify,
            check_urls,
            url_prefix,
            source_dir,
            no_activate,
            progress_format,
        } => {
            if check_urls {
                let checks = app_setup::model::check_urls(url_prefix.as_deref());
                app_setup::model::print_url_checks(&checks, json);
                let failed = checks.iter().filter(|check| check.status != "ok").count();
                if failed > 0 {
                    bail!("{failed} of {} pinned catalog URLs failed", checks.len());
                }
                return Ok(());
            }
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
            let selected = match download {
                Some(id) => Some(id),
                None if !list && !json && setup_is_interactive() => {
                    return guided_model(config_path, paths);
                }
                None => {
                    let current = app_setup::load_config(config_path)?;
                    let default_id = crate::catalog::setup_model(&current)?.id;
                    print_models(paths);
                    println!(
                        "Download the default with `omawake setup model --download {default_id}` or use exact offline assets with `--source-dir /path/to/assets`."
                    );
                    None
                }
            };
            if let Some(id) = selected {
                install_selected_model(
                    &id,
                    config_path,
                    paths,
                    source_dir.as_deref(),
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
                let original = config_snapshot(config_path)?;
                let result = (|| {
                    let config = app_setup::ensure_config(config_path)?;
                    config.save(config_path)?;
                    let path = app_setup::systemd::install(paths, config_path, !no_start)?;
                    println!("installed: {}", path.display());
                    Ok(())
                })();
                if let Err(error) = result {
                    restore_config_snapshot(config_path, original.as_deref())?;
                    return Err(error);
                }
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
            source_dir,
            progress_format,
        } => {
            let current = app_setup::load_config(config_path)?;
            let spec = match model.as_deref() {
                Some(id) => model_spec(id)?,
                None => crate::catalog::setup_model(&current)?,
            };
            let edit = managed_service_active_for_config(config_path, paths)?;
            install_everything(
                spec,
                config_path,
                paths,
                source_dir.as_deref(),
                progress_format,
                edit.managed_running,
                |config, path| {
                    crate::runtime_inventory::apply_with(
                        config,
                        path,
                        false,
                        native_runtime_probe,
                    )?;
                    Ok(())
                },
                app_setup::model::install,
                prove_setup_candidate,
                app_setup::menu::install,
                app_setup::systemd::reload_if_was_active,
                |config, paths| app_setup::print_checks(config, paths, false),
                app_setup::print_checks_event,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn configure_runtime_from_flags(
    config_path: &Path,
    paths: &AppPaths,
    runtime: Option<String>,
    device: Option<String>,
    device_id: Option<u32>,
    directory: Option<PathBuf>,
    apply: bool,
    probe: impl FnOnce(&Config, &Path) -> crate::runtime_inventory::Probe,
) -> Result<()> {
    configure_runtime_from_flags_with(
        config_path,
        paths,
        runtime,
        device,
        device_id,
        directory,
        apply,
        probe,
        app_setup::cache::prepare_for_runtime,
        prove_setup_candidate,
    )
}

#[allow(clippy::too_many_arguments)]
fn configure_runtime_from_flags_with<FC, FP>(
    config_path: &Path,
    paths: &AppPaths,
    runtime: Option<String>,
    device: Option<String>,
    device_id: Option<u32>,
    directory: Option<PathBuf>,
    apply: bool,
    probe: impl FnOnce(&Config, &Path) -> crate::runtime_inventory::Probe,
    prepare_cache: FC,
    prove: FP,
) -> Result<()>
where
    FC: FnOnce(
        &Config,
        &Path,
        &AppPaths,
        ProgressFormat,
    ) -> Result<Option<app_setup::cache::CacheReport>>,
    FP: FnOnce(&Config, &AppPaths) -> Result<()>,
{
    let current = app_setup::load_config(config_path)?;
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
            .map_or_else(|| "cpu".to_owned(), |_| current.backend.device.clone())
    });
    let candidate = runtime_selection_candidate(
        &current,
        config_path,
        &RuntimeSelection {
            runtime: selected_runtime,
            device: selected_device,
        },
        device_id,
        directory.as_deref(),
    )?;
    let mut evidence = crate::runtime_inventory::apply_with(&candidate, config_path, false, probe)?;
    if apply {
        prepare_and_save_runtime_candidate_with(
            &candidate,
            config_path,
            paths,
            ProgressFormat::Human,
            prepare_cache,
            prove,
        )?;
        evidence.device_accessible = Some(true);
        evidence.ready = true;
        if evidence.evidence.available_devices.is_empty() {
            evidence.evidence.available_devices = vec![
                candidate
                    .backend
                    .canonical_device()
                    .unwrap_or_else(|_| candidate.backend.device.clone()),
            ];
        }
        evidence.evidence.model_inference_verified = true;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"candidate":candidate.backend,"probe":evidence,"applied":apply})
        )?
    );
    Ok(())
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
        crate::runtime_inventory::apply_with(candidate, path, false, native_runtime_probe)
    }
    fn audio_device(&mut self, current: &str) -> Result<Option<String>> {
        Ok(Some(current.into()))
    }
    #[cfg(test)]
    fn setup_mode(&mut self) -> Result<Option<SetupMode>>;
    fn runtime(&mut self, current: &Config) -> Result<Option<RuntimeSelection>>;
    fn runtime_library_dir(&mut self, _candidate: &Config) -> Result<Option<PathBuf>> {
        Ok(None)
    }
    fn confirm_runtime(
        &mut self,
        candidate: &Config,
        evidence: &crate::runtime_inventory::Probe,
    ) -> Result<bool>;
    fn prove_runtime(&mut self, candidate: &Config, paths: &AppPaths) -> Result<()> {
        prove_setup_candidate(candidate, paths)
    }
    fn model(
        &mut self,
        paths: &AppPaths,
        current: &Config,
    ) -> Result<Option<&'static crate::catalog::ModelSpec>>;
    fn model_source_directory(
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
    fn audio_device(&mut self, current: &str) -> Result<Option<String>> {
        choose_audio_device(current)
    }
    #[cfg(test)]
    fn setup_mode(&mut self) -> Result<Option<SetupMode>> {
        wizard::choose_setup_mode()
    }

    fn runtime(&mut self, current: &Config) -> Result<Option<RuntimeSelection>> {
        let probe = native_runtime_probe(current, &self.config_path);
        let hardware = crate::hardware::detect();
        let providers = setup_provider_availability(current, &self.config_path);
        let recommendation = crate::hardware::recommend(&hardware, providers);
        let loadable = BTreeMap::from([
            ("default", providers.packaged_cpu),
            ("openvino", providers.openvino_npu || providers.openvino_gpu),
            ("cuda", providers.cuda),
            ("vulkan", providers.vulkan),
            (
                "hip",
                current.backend.runtime == Runtime::Hip && probe.loadable,
            ),
        ]);
        let provider = probe
            .evidence
            .versions
            .first()
            .cloned()
            .unwrap_or_else(|| probe.errors.join("; "));
        wizard::choose_runtime_with_recommendation(
            &loadable,
            &format!(
                "{}\r\nHardware, provider discovery, and model proof are reported separately. Omawake loads providers through public library APIs and never invokes their CLIs.\r\nCurrent provider discovery: {provider}",
                recommendation.detail,
            ),
            current.backend.runtime,
            &current.backend.device,
            &recommendation,
            !self.config_path.exists(),
        )
    }

    fn runtime_library_dir(&mut self, candidate: &Config) -> Result<Option<PathBuf>> {
        wizard::choose_runtime_directory(&candidate.backend.library_dirs)
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
        choose_model(paths, current)
    }

    fn model_source_directory(
        &mut self,
        paths: &AppPaths,
        spec: &crate::catalog::ModelSpec,
    ) -> Result<Option<PathBuf>> {
        if spec.downloadable || app_setup::model::verify(paths, spec).is_ok() {
            Ok(None)
        } else {
            wizard::choose_model_source_directory(spec.id)
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

pub(crate) fn setup_provider_availability(
    current: &Config,
    config_path: &Path,
) -> crate::hardware::ProviderAvailability {
    let mut paths = AppPaths::discover();
    paths.config_file = config_path.to_owned();
    let audio_available = |runtime, device: &str| {
        let selection = RuntimeSelection {
            runtime,
            device: device.into(),
        };
        let mut audio_current = current.clone();
        audio_current.backend.kind = "audiocpp".into();
        let Ok(candidate) =
            runtime_selection_candidate(&audio_current, config_path, &selection, None, None)
        else {
            return false;
        };
        crate::engine::audiocpp::probe_provider(&candidate, &paths).is_ok()
    };
    let accelerated_audio_is_explicit = |runtime| {
        std::env::var_os("OMAWAKE_AUDIOCPP_LIBRARY").is_some()
            || (current.backend.runtime == runtime
                && (!current.backend.library.as_os_str().is_empty()
                    || !current.backend.library_dirs.is_empty()))
    };
    let openvino_available = |device: &str| {
        let selection = RuntimeSelection {
            runtime: Runtime::Openvino,
            device: device.into(),
        };
        runtime_selection_candidate(current, config_path, &selection, None, None)
            .and_then(|candidate| {
                crate::engine::openvino_genai::ProviderSpec::from_config(&candidate, &paths)
                    .map(|_| ())
            })
            .is_ok()
    };
    crate::hardware::ProviderAvailability {
        packaged_cpu: audio_available(Runtime::Default, "cpu"),
        cuda: accelerated_audio_is_explicit(Runtime::Cuda) && audio_available(Runtime::Cuda, "gpu"),
        openvino_npu: openvino_available("npu"),
        openvino_gpu: openvino_available("gpu"),
        vulkan: accelerated_audio_is_explicit(Runtime::Vulkan)
            && audio_available(Runtime::Vulkan, "gpu"),
    }
}

#[cfg(test)]
fn guided_setup(config_path: &Path, paths: &AppPaths) -> Result<()> {
    guided_setup_with(
        config_path,
        paths,
        &mut TerminalGuidedPrompts {
            config_path: config_path.to_owned(),
        },
    )
}

#[cfg(test)]
fn guided_setup_with(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
) -> Result<()> {
    match prompts.setup_mode()? {
        Some(SetupMode::Audio) => setup_audio(config_path, paths, None, false, false),
        Some(SetupMode::Onboard) => {
            onboarding::guided(Config::load(config_path)?, config_path, paths)
        }
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
    let current = app_setup::load_config(config_path)?;
    let Some(selection) = prompts.runtime(&current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let mut candidate = runtime_selection_candidate(&current, config_path, &selection, None, None)?;
    let evidence = match prompts.probe_runtime(&candidate, config_path) {
        Ok(evidence) => evidence,
        Err(initial_error) => {
            let Some(runtime_directory) = prompts.runtime_library_dir(&candidate)? else {
                return Err(initial_error);
            };
            candidate = runtime_selection_candidate(
                &current,
                config_path,
                &selection,
                None,
                Some(&runtime_directory),
            )?;
            prompts.probe_runtime(&candidate, config_path)?
        }
    };
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
        |candidate, paths| prompts.prove_runtime(candidate, paths),
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
    let current = app_setup::load_config(config_path)?;
    let Some(spec) = prompts.model(paths, &current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    if verify(paths, spec).is_ok() {
        activate_model(config_path, paths, spec, ProgressFormat::Human)?;
        println!("active model: {}", spec.id);
        return Ok(());
    }
    let source_directory = prompts.model_source_directory(paths, spec)?;
    if !spec.downloadable && source_directory.is_none() {
        println!("Model setup cancelled; no changes were made.");
        return Ok(());
    }
    install_selected_model(
        spec.id,
        config_path,
        paths,
        source_directory.as_deref(),
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
    let edit = managed_service_active_for_config(config_path, paths)?;
    guided_all_with_services(
        config_path,
        paths,
        prompts,
        app_setup::model::install,
        app_setup::menu::install,
        |_| edit.managed_running,
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
    FR: FnMut(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    guided_all_with_services_and_validator(
        config_path,
        paths,
        prompts,
        validate_runtime_candidate,
        install_model,
        prove_setup_candidate,
        install_menu,
        service_is_active,
        restart_service,
        check_human,
        check_json,
    )
}

#[allow(clippy::too_many_arguments)]
fn guided_all_with_services_and_validator<FI, FP, FM, FA, FR, CH, CJ, FV>(
    config_path: &Path,
    paths: &AppPaths,
    prompts: &mut impl GuidedPrompts,
    mut validate_runtime: FV,
    install_model: FI,
    prove_model: FP,
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
    FP: FnOnce(&Config, &AppPaths) -> Result<()>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FA: FnOnce(&AppPaths) -> bool,
    FR: FnMut(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
    FV: FnMut(&Config, &Path) -> Result<()>,
{
    let current = app_setup::load_config(config_path)?;
    let Some(selection) = prompts.runtime(&current)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let mut candidate = runtime_selection_candidate(&current, config_path, &selection, None, None)?;
    if let Err(initial_error) = validate_runtime(&candidate, config_path) {
        let Some(runtime_directory) = prompts.runtime_library_dir(&candidate)? else {
            return Err(initial_error);
        };
        candidate = runtime_selection_candidate(
            &current,
            config_path,
            &selection,
            None,
            Some(&runtime_directory),
        )?;
        validate_runtime(&candidate, config_path)?;
    }
    let Some(spec) = prompts.model(paths, &candidate)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    if spec.backend != candidate.backend.kind {
        bail!(
            "{} requires the {} backend; choose a model compatible with the selected {} runtime",
            spec.id,
            spec.backend,
            runtime_name(selection.runtime)
        );
    }
    let source_directory = prompts.model_source_directory(paths, spec)?;
    if !spec.downloadable
        && source_directory.is_none()
        && app_setup::model::verify(paths, spec).is_err()
    {
        println!("Setup cancelled; no changes were made.");
        return Ok(());
    }
    let Some(audio_device) = prompts.audio_device(&candidate.audio.device)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    candidate.audio.device = audio_device;
    println!("Input device: {}", candidate.audio.device);
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
        source_directory.as_deref(),
        ProgressFormat::Human,
        service_was_active,
        install_model,
        prove_model,
        install_menu,
        restart_service,
        check_human,
        check_json,
    )
}

fn choose_model(
    paths: &AppPaths,
    current: &Config,
) -> Result<Option<&'static crate::catalog::ModelSpec>> {
    choose_model_with(paths, current, |items, preferred| {
        wizard::select(
            "Wake-word model",
            "● active · ○ installed · downloadable catalog models can be installed",
            items,
            preferred,
        )
    })
}

fn choose_model_with<S>(
    paths: &AppPaths,
    current: &Config,
    select: S,
) -> Result<Option<&'static crate::catalog::ModelSpec>>
where
    S: FnOnce(&[wizard::MenuItem], usize) -> Result<Option<usize>>,
{
    let items: Vec<_> = crate::catalog::models()
        .iter()
        .map(|model| {
            let installed = app_setup::model::verify(paths, model).is_ok();
            let status = if model.id == current.model.name && installed {
                "● active"
            } else if model.id == current.model.name && !model.downloadable {
                "● active · local assets required"
            } else if model.id == current.model.name {
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
                model.total_size() as f64 / 1_048_576.0
            );
            let detail = if installed || model.downloadable {
                detail
            } else {
                format!("{detail} · select to provide an exact local asset directory")
            };
            let label = format!("{status}  {}", model.name);
            if model.compatible_with(
                &current.backend.kind,
                current.backend.runtime,
                &current.backend.device,
            ) {
                wizard::MenuItem::available(label, detail)
            } else {
                wizard::MenuItem::unavailable(
                    label,
                    format!(
                        "Requires the {} backend; incompatible with this backend/runtime selection · {detail}",
                        model.backend
                    ),
                )
            }
        })
        .collect();
    let preferred = crate::catalog::models()
        .iter()
        .position(|model| model.id == current.model.name)
        .filter(|&index| items[index].enabled)
        .or_else(|| items.iter().position(|item| item.enabled))
        .unwrap_or(0);
    let selected = select(&items, preferred)?;
    if let Some(index) = selected {
        ensure!(
            items.get(index).is_some_and(|item| item.enabled),
            "selected model is incompatible with the configured backend/runtime"
        );
    }
    Ok(selected.map(|index| &crate::catalog::models()[index]))
}

#[cfg(test)]
fn save_runtime_selection_impl_with(
    config_path: &Path,
    selection: &RuntimeSelection,
    runtime_directory: Option<&Path>,
    validate: impl FnOnce(&Config, &Path) -> Result<()>,
) -> Result<()> {
    let current = app_setup::load_config(config_path)?;
    let config =
        runtime_selection_candidate(&current, config_path, selection, None, runtime_directory)?;
    validate(&config, config_path)?;
    config.save(config_path)
}

fn prepare_and_save_runtime_candidate_with<F, P>(
    config: &Config,
    config_path: &Path,
    paths: &AppPaths,
    progress: ProgressFormat,
    prepare_cache: F,
    prove: P,
) -> Result<()>
where
    F: FnOnce(
        &Config,
        &Path,
        &AppPaths,
        ProgressFormat,
    ) -> Result<Option<app_setup::cache::CacheReport>>,
    P: FnOnce(&Config, &AppPaths) -> Result<()>,
{
    prepare_cache(config, config_path, paths, progress)?;
    prove(config, paths)
        .context("runtime candidate failed its file-only provider/model proof; config unchanged")?;
    save_and_reload_active(config.clone(), config_path, paths).map(|_| ())
}

fn runtime_selection_candidate(
    current: &Config,
    config_path: &Path,
    selection: &RuntimeSelection,
    device_id: Option<u32>,
    runtime_directory: Option<&Path>,
) -> Result<Config> {
    let mut config = current.clone();
    let backend_kind = match selection.runtime {
        Runtime::Default if current.backend.kind == "whispercpp" => "whispercpp",
        Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip => "audiocpp",
        Runtime::Openvino => "openvino-genai",
    };
    let default =
        crate::catalog::default_model(backend_kind, selection.runtime, &selection.device)?;
    if crate::catalog::model(&config.model.name).is_none_or(|model| {
        !model.compatible_with(backend_kind, selection.runtime, &selection.device)
    }) {
        default.activate(&mut config);
    }
    if config.backend.kind != backend_kind || config.backend.runtime != selection.runtime {
        config.backend.library.clear();
        config.backend.library_dirs.clear();
        config.backend.options.clear();
    }
    config.backend.kind = backend_kind.into();
    config.backend.runtime = selection.runtime;
    config.backend.device = selection.device.clone();
    config.backend.device_id = if matches!(
        selection.runtime,
        Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
    ) {
        device_id.unwrap_or_else(|| {
            if current.backend.kind == backend_kind && current.backend.runtime == selection.runtime
            {
                current.backend.device_id
            } else {
                0
            }
        })
    } else {
        0
    };
    config.backend.options.remove("audiocpp.asr_family");
    if backend_kind == "audiocpp" {
        let family = crate::catalog::setup_model(&config)?.asr_family;
        config
            .backend
            .options
            .insert("audiocpp.asr_family".into(), family.into());
    }
    config.backend.validate_shape()?;
    if let Some(directory) = runtime_directory {
        configure_runtime_directory(&mut config, directory)?;
    }
    let _ = config_path;
    Ok(config)
}

fn validate_runtime_candidate(config: &Config, config_path: &Path) -> Result<()> {
    validate_runtime_candidate_with(config, config_path, native_runtime_probe)
}

fn native_runtime_probe(config: &Config, config_path: &Path) -> crate::runtime_inventory::Probe {
    crate::runtime_inventory::probe(config, config_path)
}

#[cfg(test)]
fn native_runtime_probe_with<OV, AC>(
    config: &Config,
    config_path: &Path,
    openvino_probe: OV,
    audiocpp_probe: AC,
) -> crate::runtime_inventory::Probe
where
    OV: FnOnce(&Config, &AppPaths) -> Result<crate::engine::openvino_genai::RuntimeEvidence>,
    AC: FnOnce(&Config, &AppPaths) -> Result<(PathBuf, String)>,
{
    let backend = &config.backend;
    let mut paths = AppPaths::discover();
    paths.config_file = config_path.to_owned();
    if backend.kind == "openvino-genai" && backend.runtime == Runtime::Openvino {
        return match openvino_probe(config, &paths) {
            Ok(evidence) => crate::runtime_inventory::Probe {
                loadable: true,
                device_accessible: Some(true),
                ready: false,
                evidence: crate::runtime_inventory::Evidence {
                    versions: vec![format!(
                        "OpenVINO {} · GenAI C {} · {}",
                        evidence.runtime_build, evidence.genai_library, evidence.full_device_name
                    )],
                    provider_registration: true,
                    available_devices: vec![evidence.available_device.to_ascii_lowercase()],
                    selected_device: Some(evidence.requested_device.to_ascii_lowercase()),
                    provider_path: Some(evidence.genai_library.clone().into()),
                    model_inference_verified: false,
                },
                errors: Vec::new(),
            },
            Err(error) => crate::runtime_inventory::Probe {
                errors: vec![format!("{error:#}")],
                ..Default::default()
            },
        };
    }
    match audiocpp_probe(config, &paths) {
        Ok((library, version)) => crate::runtime_inventory::Probe {
            loadable: true,
            device_accessible: None,
            ready: false,
            evidence: crate::runtime_inventory::Evidence {
                versions: vec![format!("{version} · {}", library.display())],
                provider_registration: true,
                available_devices: Vec::new(),
                selected_device: Some(
                    backend
                        .canonical_device()
                        .unwrap_or_else(|_| backend.device.clone()),
                ),
                provider_path: Some(library),
                model_inference_verified: false,
            },
            errors: Vec::new(),
        },
        Err(error) => crate::runtime_inventory::Probe {
            errors: vec![format!("{error:#}")],
            ..Default::default()
        },
    }
}

fn validate_runtime_candidate_with(
    config: &Config,
    config_path: &Path,
    probe: impl FnOnce(&Config, &Path) -> crate::runtime_inventory::Probe,
) -> Result<()> {
    crate::runtime_inventory::apply_with(config, config_path, false, probe)?;
    Ok(())
}

fn configure_runtime_directory(config: &mut Config, directory: &Path) -> Result<()> {
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
    let find = |names: &[&str]| -> Result<PathBuf> {
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
                    && names
                        .iter()
                        .any(|prefix| name == *prefix || name.starts_with(&format!("{prefix}."))))
                .then(|| entry.path())
            })
            .collect::<Vec<_>>();
        matches.sort();
        matches.into_iter().next().with_context(|| {
            format!(
                "{} does not contain {}",
                directory.display(),
                names.join(" or ")
            )
        })
    };
    let provider = match config.backend.runtime {
        Runtime::Default if config.backend.kind == "whispercpp" => find(&["libwhisper.so"])?,
        Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip => {
            find(&["libaudiocpp.so.0.1.0", "libaudiocpp.so.0", "libaudiocpp.so"])?
        }
        Runtime::Openvino => {
            let provider = find(&["libopenvino_genai_c.so"])?;
            find(&["libopenvino_c.so"])?;
            let plugin = match config.backend.device.to_ascii_lowercase().as_str() {
                "cpu" => "libopenvino_intel_cpu_plugin.so",
                "gpu" => "libopenvino_intel_gpu_plugin.so",
                "npu" => "libopenvino_intel_npu_plugin.so",
                _ => bail!("OpenVINO setup requires an explicit CPU, GPU, or NPU device"),
            };
            find(&[plugin])?;
            if config.backend.device.eq_ignore_ascii_case("npu") {
                find(&["libopenvino_intel_npu_compiler_loader.so"])?;
                find(&["libopenvino_intel_npu_compiler.so"])?;
            }
            provider
        }
    };
    let selected_parents = std::iter::once(
        provider
            .parent()
            .context("provider library has no parent")?
            .to_owned(),
    )
    .chain(candidates.into_iter().filter(|candidate| {
        std::fs::read_dir(candidate).is_ok_and(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|entry| entry.file_name().to_string_lossy().contains(".so"))
        })
    }))
    .fold(Vec::new(), |mut directories, path| {
        if !directories.contains(&path) {
            directories.push(path);
        }
        directories
    });
    config.backend.library_dirs = selected_parents;
    config.backend.library = provider;
    Ok(())
}

fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
        Runtime::Vulkan => "vulkan",
        Runtime::Hip => "hip",
    }
}

#[allow(clippy::too_many_arguments)]
fn install_everything<FV, FI, FP, FM, FR, CH, CJ>(
    spec: &crate::catalog::ModelSpec,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    progress_format: ProgressFormat,
    service_was_active: bool,
    validate_runtime: FV,
    install_model: FI,
    prove_model: FP,
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
    FP: FnOnce(&Config, &AppPaths) -> Result<()>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FR: FnMut(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    let config = app_setup::load_config(config_path)?;
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
        prove_model,
        install_menu,
        restart_service,
        check_human,
        check_json,
    )
}

#[allow(clippy::too_many_arguments)]
fn install_everything_with_config<FI, FP, FM, FR, CH, CJ>(
    spec: &crate::catalog::ModelSpec,
    config: Config,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    progress_format: ProgressFormat,
    service_was_active: bool,
    install_model: FI,
    prove_model: FP,
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
    FP: FnOnce(&Config, &AppPaths) -> Result<()>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FR: FnMut(bool) -> Result<bool>,
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
        prove_model,
        app_setup::cache::prepare,
        install_menu,
        restart_service,
        check_human,
        check_json,
    )
}

#[allow(clippy::too_many_arguments)]
fn install_everything_with_config_and_cache<FI, FP, FC, FM, FR, CH, CJ>(
    spec: &crate::catalog::ModelSpec,
    mut config: Config,
    config_path: &Path,
    paths: &AppPaths,
    archive: Option<&Path>,
    progress_format: ProgressFormat,
    service_was_active: bool,
    install_model: FI,
    prove_model: FP,
    prepare_cache: FC,
    install_menu: FM,
    mut restart_service: FR,
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
    FP: FnOnce(&Config, &AppPaths) -> Result<()>,
    FC: FnOnce(
        &Config,
        &Path,
        &AppPaths,
        ProgressFormat,
    ) -> Result<Option<app_setup::cache::CacheReport>>,
    FM: FnOnce(&AppPaths) -> Result<PathBuf>,
    FR: FnMut(bool) -> Result<bool>,
    CH: FnOnce(&Path, &AppPaths) -> Result<()>,
    CJ: FnOnce(&Path, &AppPaths) -> Result<()>,
{
    let original = config_snapshot(config_path)?;
    let launcher_snapshot = FileSnapshot::capture(&app_setup::menu::launcher_path(paths))?;
    let mut service_restart_attempted = false;
    let result = (|| {
        let directory = install_model(paths, spec, archive, progress_format)?;
        spec.activate(&mut config);
        prove_model(&config, paths)?;
        prepare_cache(&config, config_path, paths, progress_format)?;
        config.save(config_path)?;
        let launcher = install_menu(paths)?;
        match progress_format {
            ProgressFormat::Human => check_human(config_path, paths)?,
            ProgressFormat::Json => check_json(config_path, paths)?,
        }
        service_restart_attempted = service_was_active;
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
        let mut rollback_errors = Vec::new();
        if let Err(restore_error) = restore_config_snapshot(config_path, original.as_deref()) {
            rollback_errors.push(format!("restore prior config: {restore_error:#}"));
        }
        if let Err(restore_error) = launcher_snapshot.restore() {
            rollback_errors.push(format!("restore prior launcher: {restore_error:#}"));
        }
        if service_restart_attempted && let Err(restore_error) = restart_service(true) {
            rollback_errors.push(format!(
                "restart the previously active service with restored config: {restore_error:#}"
            ));
        }
        if !rollback_errors.is_empty() {
            return Err(error.context(format!(
                "setup rollback was incomplete: {}",
                rollback_errors.join("; ")
            )));
        }
        return Err(error);
    }
    Ok(())
}

struct FileSnapshot {
    path: PathBuf,
    contents: Option<(Vec<u8>, fs::Permissions)>,
}

impl FileSnapshot {
    fn capture(path: &Path) -> Result<Self> {
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_file() => Ok(Self {
                path: path.to_owned(),
                contents: Some((
                    fs::read(path)
                        .with_context(|| format!("read setup snapshot {}", path.display()))?,
                    metadata.permissions(),
                )),
            }),
            Ok(_) => bail!(
                "setup-managed launcher is not a regular file: {}",
                path.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self {
                path: path.to_owned(),
                contents: None,
            }),
            Err(error) => {
                Err(error).with_context(|| format!("inspect setup snapshot {}", path.display()))
            }
        }
    }

    fn restore(&self) -> Result<()> {
        let temporary = self.path.with_extension("desktop.tmp");
        match &self.contents {
            Some((contents, permissions)) => {
                if let Some(parent) = self.path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&temporary, contents)?;
                fs::set_permissions(&temporary, permissions.clone())?;
                fs::rename(&temporary, &self.path)?;
            }
            None => {
                match fs::remove_file(&self.path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
                match fs::remove_file(&temporary) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
        Ok(())
    }
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

fn save_and_reload_active(config: Config, config_path: &Path, paths: &AppPaths) -> Result<bool> {
    let edit = managed_service_active_for_config(config_path, paths)?;
    let managed_running = edit.managed_running;
    save_and_reload_active_with(
        config,
        config_path,
        managed_running,
        || managed_running,
        app_setup::systemd::reload_if_was_active,
        app_setup::systemd::restart,
    )
}

struct ConfigEditState {
    managed_running: bool,
    _reservation: Option<Vec<UnixListener>>,
}

// Guided audio setup can save config while it owns the daemon reservation.
// Nested edit preflight reuses that reservation on this thread.
thread_local! {
    static AUDIO_SETUP_RESERVATIONS: RefCell<Vec<PathBuf>> = const { RefCell::new(Vec::new()) };
}

struct AudioSetupReservation {
    path: PathBuf,
    _listeners: Vec<UnixListener>,
}

impl AudioSetupReservation {
    fn new(paths: &AppPaths) -> Result<Self> {
        let listeners = bind_daemon_instance(paths)?;
        let path = paths.config_file.clone();
        AUDIO_SETUP_RESERVATIONS.with(|held| held.borrow_mut().push(path.clone()));
        Ok(Self {
            path,
            _listeners: listeners,
        })
    }
}

impl Drop for AudioSetupReservation {
    fn drop(&mut self) {
        AUDIO_SETUP_RESERVATIONS.with(|held| {
            let mut held = held.borrow_mut();
            if let Some(index) = held.iter().rposition(|path| path == &self.path) {
                held.remove(index);
            }
        });
    }
}

fn managed_service_active_for_config(
    config_path: &Path,
    paths: &AppPaths,
) -> Result<ConfigEditState> {
    let base_targets_config = app_setup::systemd::targets_config(config_path);
    let service_active = (base_targets_config || config_path == AppPaths::discover().config_file)
        && app_setup::systemd::is_active();
    let effective_targets_config = base_targets_config
        && service_active
        && app_setup::systemd::effective_targets_config(config_path);
    let managed_running =
        managed_service_active_for_config_with(paths, service_active, effective_targets_config)?;
    let audio_reservation_held = AUDIO_SETUP_RESERVATIONS
        .with(|held| held.borrow().iter().any(|path| path == &paths.config_file));
    let reservation = if managed_running || audio_reservation_held {
        None
    } else {
        Some(
            bind_daemon_instance(paths)
                .context("stop the running Omawake daemon before editing its configuration")?,
        )
    };
    Ok(ConfigEditState {
        managed_running,
        _reservation: reservation,
    })
}

fn managed_service_active_for_config_with(
    paths: &AppPaths,
    service_active: bool,
    owns_service_config: bool,
) -> Result<bool> {
    let managed_running = service_active && owns_service_config;
    if service_active && !managed_running {
        bail!(
            "a running Omawake daemon is not managed for this configuration; stop its user service before editing, then restart it to apply the change"
        );
    }
    // The control socket is shared by every config in this runtime directory.
    // For custom configs, the per-config daemon lock above identifies ownership.
    if !managed_running
        && paths.config_file == AppPaths::discover().config_file
        && connect_control_socket(&socket_path(paths)).is_ok()
    {
        bail!(
            "a running Omawake daemon is not managed for this configuration; run `omawake stop` before editing, then start it again to apply the change"
        );
    }
    if managed_running && setup_home::manual_pause_state(paths)? == Some(true) {
        bail!(
            "the running Omawake daemon is manually paused; run `omawake resume` before changing its configuration, then pause it again afterward"
        );
    }
    Ok(managed_running)
}

fn save_and_reload_active_with(
    config: Config,
    config_path: &Path,
    owns_service_config: bool,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
    restart_service: impl FnOnce() -> Result<()>,
) -> Result<bool> {
    let original = config_snapshot(config_path)?;
    let was_active = owns_service_config && service_is_active();
    config.save(config_path)?;
    if !was_active {
        return Ok(false);
    }
    let restart = reload_service(true).and_then(|restarted| {
        if restarted {
            Ok(())
        } else {
            bail!("active Omawake daemon was not restarted")
        }
    });
    if let Err(error) = restart {
        let mut rollback_errors = Vec::new();
        match restore_config_snapshot(config_path, original.as_deref()) {
            Ok(()) => {
                if let Err(restart_error) = restart_service() {
                    rollback_errors.push(format!(
                        "restart daemon with previous config: {restart_error:#}"
                    ));
                }
            }
            Err(restore_error) => {
                rollback_errors.push(format!("restore previous config: {restore_error:#}"));
            }
        }
        let rollback = if rollback_errors.is_empty() {
            "previous configuration and daemon were restored".into()
        } else {
            format!("rollback incomplete: {}", rollback_errors.join("; "))
        };
        return Err(error).context(format!(
            "reload active daemon after config update; {rollback}"
        ));
    }
    eprintln!("omawake: active daemon restarted with updated configuration");
    Ok(true)
}

fn prove_setup_candidate(config: &Config, paths: &AppPaths) -> Result<()> {
    prove_setup_candidate_with(config, paths, |config, paths, audio| {
        let detector = Detector::load(config, paths)
            .context("initialize the selected native provider and its model assets")?;
        detector
            .detect_file(audio)
            .context("run the file-only setup proof through VAD and phrase verification")
    })
}

fn prove_setup_candidate_with(
    config: &Config,
    paths: &AppPaths,
    detect: impl FnOnce(&Config, &AppPaths, &Path) -> Result<Vec<Detection>>,
) -> Result<()> {
    let directory = paths.cache_dir.join("setup-proof");
    fs::create_dir_all(&directory)?;
    let audio = directory.join(format!("silence-{}.wav", std::process::id()));
    let proof = (|| -> Result<()> {
        let mut writer = hound::WavWriter::create(
            &audio,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for _ in 0..16_000 {
            writer.write_sample(0_i16)?;
        }
        writer.finalize()?;
        let detections = detect(config, paths, &audio)?;
        if !detections.is_empty() {
            bail!("silent setup proof unexpectedly produced a wake-word detection");
        }
        Ok(())
    })();
    let _ = fs::remove_file(&audio);
    proof
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
    save_and_reload_active(config, config_path, paths).map(|_| ())
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
        "vulkan" => Ok(Runtime::Vulkan),
        "hip" => Ok(Runtime::Hip),
        _ => bail!("backend runtime must be default, openvino, cuda, vulkan, or hip"),
    }
}

fn parse_fallback(value: &str) -> Result<Fallback> {
    match value.to_ascii_lowercase().as_str() {
        "error" => Ok(Fallback::Error),
        "cpu" => Ok(Fallback::Cpu),
        _ => bail!("backend fallback must be error or cpu"),
    }
}

fn evaluation_report(config: &Config, paths: &AppPaths, manifest_path: &Path) -> Result<Value> {
    let groups = std::cell::RefCell::new(Vec::new());
    let mut report = evaluation_report_with(
        config,
        paths,
        manifest_path,
        || Detector::load(config, paths),
        |detector, path| detector.detect_file(path),
        |detector| {
            groups.replace(detector.engine_statuses());
            let (placement_verified, placement_evidence) =
                backend_placement(detector.backend_kind, detector.effective_runtime);
            Ok(RuntimeIdentity {
                backend_kind: detector.backend_kind.into(),
                requested_runtime: runtime_name(config.backend.runtime).into(),
                requested_device: config.backend.canonical_device()?,
                effective_runtime: runtime_name(detector.effective_runtime).into(),
                fallback_used: detector.fallback_used,
                placement_verified,
                placement_evidence: placement_evidence.into(),
            })
        },
    )?;
    attach_group_values(&mut report, groups.into_inner());
    Ok(report)
}

fn evaluation_report_with<D, L, F, I>(
    config: &Config,
    paths: &AppPaths,
    manifest_path: &Path,
    load: L,
    detect: F,
    identity: I,
) -> Result<Value>
where
    L: FnOnce() -> Result<D>,
    F: FnMut(&D, &Path) -> Result<Vec<Detection>>,
    I: FnMut(&D) -> Result<RuntimeIdentity>,
{
    let enabled_keywords = config
        .wake_words
        .iter()
        .filter(|keyword| keyword.enabled)
        .map(|keyword| keyword.id.clone())
        .collect::<BTreeSet<_>>();
    let prepared = evaluation::load_manifest(manifest_path, &enabled_keywords)?;
    let context = EvaluationContext {
        config_sha256: evaluation::sha256_bytes(&serde_json::to_vec(config)?),
        enabled_keyword_ids: enabled_keywords.iter().cloned().collect(),
        model_name: config.model.name.clone(),
        model_directory: config.model_directory(paths).display().to_string(),
        model_language: config.model.language.clone(),
        application_version: env!("CARGO_PKG_VERSION").into(),
    };
    let report = evaluation::evaluate_with(&prepared, context, load, detect, identity)?;
    Ok(serde_json::to_value(report)?)
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
    let report = file_benchmark_report(config, paths, audio, warmup, iterations)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn file_benchmark_report(
    config: &Config,
    paths: &AppPaths,
    audio: &[PathBuf],
    warmup: u32,
    iterations: u32,
) -> Result<Value> {
    let detector = Detector::load(config, paths)?;
    let files = benchmark_files(
        audio,
        warmup,
        iterations,
        wav_duration,
        |path| detector.detect_file(path),
        Instant::now,
    )?;
    benchmark_report(config, &detector, files, warmup, iterations)
}

fn benchmark_report(
    config: &Config,
    detector: &impl DetectorControl,
    files: Vec<BenchmarkFile>,
    warmup: u32,
    iterations: u32,
) -> Result<Value> {
    let (placement_verified, placement_evidence) =
        backend_placement(detector.backend_kind(), detector.effective_runtime());
    let summary = benchmark_summary(
        files
            .iter()
            .flat_map(|file| file.iterations.iter())
            .map(|iteration| (iteration.elapsed_milliseconds, iteration.real_time_factor)),
    );
    let mut report = json!({
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
            "placement_verified": placement_verified,
            "placement_evidence": placement_evidence,
        },
        "files": files,
        "summary": summary,
    });
    attach_engine_groups(&mut report, detector);
    Ok(report)
}

fn runtime_placement(runtime: Runtime) -> (bool, &'static str) {
    match runtime {
        Runtime::Default => (true, "native CPU session initialized"),
        Runtime::Openvino => (
            true,
            "CPU fallback was disabled while every model graph initialized on the selected OpenVINO device",
        ),
        Runtime::Cuda => (
            true,
            "audio.cpp created CUDA sessions for Silero and the phrase verifier",
        ),
        Runtime::Vulkan => (
            true,
            "audio.cpp created Vulkan sessions for Silero and the phrase verifier",
        ),
        Runtime::Hip => (
            true,
            "audio.cpp created HIP sessions for Silero and the phrase verifier",
        ),
    }
}

fn backend_placement(kind: &str, runtime: Runtime) -> (bool, &'static str) {
    match (kind, runtime) {
        ("multi-engine", _) => (false, "placement belongs to individual engine groups"),
        ("trained-whisper-encoder", Runtime::Openvino) => (
            true,
            "OpenVINO frozen encoder execution device verified; Silero VAD runs on CPU",
        ),
        ("audiocpp", Runtime::Default) => (
            true,
            "audio.cpp created explicit CPU Silero and ASR sessions",
        ),
        ("audiocpp", Runtime::Cuda) => (true, "audio.cpp created explicit CUDA sessions"),
        ("audiocpp", Runtime::Vulkan) => (true, "audio.cpp created explicit Vulkan sessions"),
        ("audiocpp", Runtime::Hip) => (true, "audio.cpp created explicit HIP sessions"),
        ("whispercpp", Runtime::Default) => (
            true,
            "whisper.cpp created explicit CPU VAD and verifier contexts",
        ),
        _ => runtime_placement(runtime),
    }
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
            Ok(AudioEvent::Error(error)) => {
                return Err(CaptureFailure(format!("audio capture failed: {error}")).into());
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(CaptureFailure("audio capture disconnected".into()).into());
            }
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
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&detection_report(detector, detections, execute)?)?
        );
        return Ok(());
    }
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

fn detection_report(
    detector: &impl DetectorControl,
    detections: Vec<Detection>,
    execute: bool,
) -> Result<Value> {
    let actions = collect_detection_actions(&detections, execute, |id| detector.run(id))?;
    let mut report = detections_report(
        detections,
        actions,
        detector.load_time(),
        detector.keywords_buffer(),
        detector.backend_kind(),
        detector.effective_runtime(),
        detector.fallback_used(),
    );
    attach_engine_groups(&mut report, detector);
    Ok(report)
}

fn attach_engine_groups(report: &mut Value, detector: &impl DetectorControl) {
    attach_group_values(report, detector.engine_groups());
}
fn attach_group_values(report: &mut Value, groups: Vec<crate::engine::GroupStatus>) {
    if !groups.is_empty() {
        report["backend"]["groups"] = json!(groups);
        // A mixed detector has no single effective runtime.
        report["backend"]["effective_runtime"] = json!("mixed");
        if report["backend"].get("requested_device").is_some() {
            report["backend"]["requested_device"] = json!("per-engine");
            report["backend"]["requested_runtime"] = json!("mixed");
            report["backend"]["placement_verified"] = json!(false);
            report["backend"]["placement_evidence"] =
                json!("placement belongs to individual engine groups");
        }
    }
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
            serde_json::to_string_pretty(&detections_report(
                detections,
                actions,
                load_time,
                keywords_buffer,
                backend_kind,
                effective_runtime,
                fallback_used,
            ))?
        );
    } else {
        for detection in detections {
            println!("{}", detection.id);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn detections_report(
    detections: Vec<Detection>,
    actions: Vec<ActionResult>,
    load_time: Duration,
    keywords_buffer: &str,
    backend_kind: &str,
    effective_runtime: Runtime,
    fallback_used: bool,
) -> Value {
    json!({
        "model_load_milliseconds": load_time.as_millis() as u64,
        "keywords_buffer": keywords_buffer,
        "detections": detections,
        "actions": actions,
        "backend": {"kind": backend_kind, "effective_runtime": effective_runtime, "fallback_used": fallback_used}
    })
}

fn run_daemon(config: &Config, paths: &AppPaths) -> Result<()> {
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(SIGINT, Arc::clone(&shutdown_requested))?;
    signal_hook::flag::register(SIGTERM, Arc::clone(&shutdown_requested))?;
    run_daemon_with_shutdown(config, paths, shutdown_requested)
}

fn run_daemon_with_shutdown(
    config: &Config,
    paths: &AppPaths,
    shutdown_requested: Arc<AtomicBool>,
) -> Result<()> {
    let _singleton = bind_daemon_instance(paths)?;
    let detector = Detector::load(config, paths)?;
    run_loaded_daemon(&detector, config, paths, shutdown_requested)
}

fn bind_daemon_instance(paths: &AppPaths) -> Result<Vec<UnixListener>> {
    crate::daemon_instance::reserve(paths)
}

fn run_loaded_daemon(
    detector: &Detector,
    config: &Config,
    paths: &AppPaths,
    shutdown_requested: Arc<AtomicBool>,
) -> Result<()> {
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
    let control = std::cell::RefCell::new(pause_ownership::OwnedControl::default());
    let poll = |state, audio| {
        let mut details = daemon_details(
            detector.backend_kind(),
            detector.effective_runtime(),
            detector.fallback_used(),
            detector.load_time(),
            audio,
            &config.model.name,
            &config.model.language,
        );
        attach_engine_groups(&mut details, detector);
        control
            .borrow_mut()
            .poll(state, &details, || accept_control(&listener))
    };
    let requested = &config.audio.device;
    let mut poll_paused = || {
        poll(
            "paused",
            Some(crate::audio_devices::status(requested, None, None)),
        )
    };
    let mut armed_cycle = || {
        retry_audio_cycle(
            || {
                let capture = Capture::start(requested, config.daemon.queue_capacity)
                    .map_err(|e| CaptureFailure(format!("{e:#}")))?;
                eprintln!(
                    "armed on {} ({} Hz, {} channel(s))",
                    capture.device_name, capture.sample_rate, capture.channels
                );
                let session = detector.live_session();
                collect_armed_detections(
                    &capture.device_name,
                    capture.sample_rate,
                    capture.channels,
                    |mut audio| {
                        audio["requested"] = json!(requested);
                        // CPAL default may be an opaque bridge, not a physical device.
                        audio["effective"] = if crate::audio_devices::is_default(requested) {
                            Value::Null
                        } else {
                            json!(&capture.device_name)
                        };
                        audio["available"] = json!(true);
                        poll("armed", Some(audio))
                    },
                    |timeout| capture.receiver().recv_timeout(timeout),
                    |sample_rate, samples| session.accept(sample_rate, samples),
                    || shutdown_requested.load(Ordering::Relaxed),
                )
            },
            |detail| {
                let until = std::time::Instant::now() + Duration::from_secs(1);
                loop {
                    if shutdown_requested.load(Ordering::Relaxed) {
                        return Ok(Some(Command::Shutdown));
                    }
                    let mut audio = crate::audio_devices::status(requested, None, Some(detail));
                    audio["available"] = json!(false);
                    if let Some(command) = poll("audio_unavailable", Some(audio))? {
                        return Ok(Some(command));
                    }
                    if std::time::Instant::now() >= until {
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                Ok(None)
            },
        )
    };
    let mut execute = |triggered| execute_detected_actions(triggered, |id| detector.run_action(id));
    let mut sleep = thread::sleep;
    let serve_result = run_daemon_state_machine(
        &shutdown_requested,
        config.daemon.cooldown_milliseconds,
        &mut poll_paused,
        &mut armed_cycle,
        &mut execute,
        &mut sleep,
    );
    drop(listener);
    finish_daemon(paths, Some(&socket_metadata), serve_result)
}

fn run_daemon_state_machine(
    shutdown_requested: &AtomicBool,
    cooldown_milliseconds: u64,
    poll_paused: &mut dyn FnMut() -> Result<Option<Command>>,
    armed_cycle: &mut dyn FnMut() -> Result<(Vec<Detection>, Option<Command>)>,
    execute: &mut dyn FnMut(Vec<Detection>),
    sleep: &mut dyn FnMut(Duration),
) -> Result<()> {
    let mut paused = false;
    let mut shutdown = false;
    while !shutdown && !shutdown_requested.load(Ordering::Relaxed) {
        if paused {
            if let Some(command) = poll_paused()? {
                apply_daemon_command(command, &mut paused, &mut shutdown);
            }
            if shutdown_requested.load(Ordering::Relaxed) {
                shutdown = true;
            }
            sleep(Duration::from_millis(50));
            continue;
        }
        let (triggered, command) = armed_cycle()?;
        if let Some(command) = command {
            apply_daemon_command(command, &mut paused, &mut shutdown);
        }
        execute(triggered);
        if !paused && !shutdown {
            sleep(Duration::from_millis(cooldown_milliseconds));
        }
    }
    Ok(())
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
                Command::Pause | Command::HoldPause | Command::Shutdown => {
                    return Ok((triggered, Some(command)));
                }
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
        Command::Pause | Command::HoldPause => *paused = true,
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
        Ok(AudioEvent::Error(error)) => {
            Err(CaptureFailure(format!("audio capture failed: {error}")).into())
        }
        Err(RecvTimeoutError::Timeout) => Ok(Vec::new()),
        Err(RecvTimeoutError::Disconnected) => {
            Err(CaptureFailure("audio capture disconnected".into()).into())
        }
    }
}

fn execute_detected_actions<F>(detections: Vec<Detection>, mut run: F)
where
    F: FnMut(&str) -> Result<ActionResult>,
{
    for detection in detections {
        match run(&detection.id) {
            Ok(result) => eprintln!(
                "detected {}; action started as pid {}",
                detection.id, result.pid
            ),
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

trait DetectorControl {
    fn engine_groups(&self) -> Vec<crate::engine::GroupStatus> {
        Vec::new()
    }
    fn backend_kind(&self) -> &str;
    fn effective_runtime(&self) -> Runtime;
    fn fallback_used(&self) -> bool;
    fn load_time(&self) -> Duration;
    fn keywords_buffer(&self) -> &str;
    fn run(&self, id: &str) -> Result<ActionResult>;
}

impl DetectorControl for Detector {
    fn engine_groups(&self) -> Vec<crate::engine::GroupStatus> {
        self.engine_statuses()
    }
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
    model_name: &str,
    model_language: &str,
) -> Value {
    json!({
        "backend": {
            "kind": backend_kind,
            "effective_runtime": effective_runtime,
            "fallback_used": fallback_used,
        },
        "model": {
            "name": model_name,
            "language": model_language,
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
        "model":{"name":config.model.name,"language":config.model.language,"path":config.model_directory(paths),"loaded":false},"last_error":null,"details":{"audio":crate::audio_devices::status(&config.audio.device,None,None)}})
}

/// Language choices for the verifier-language schema key: every token the
/// pinned multilingual Whisper model supports, plus the model default.
fn language_schema_choices() -> Vec<serde_json::Value> {
    let mut choices = vec![serde_json::Value::String(String::new())];
    for token in crate::engine::openvino_genai::WHISPER_MODEL_LANGUAGES {
        choices.push(serde_json::Value::String((*token).into()));
    }
    choices
}

fn schema(config: &Config, config_path: &Path, paths: &AppPaths) -> serde_json::Value {
    let audio_inventory = crate::audio::device_inventory(&config.audio.device);
    let audio_choices = crate::audio_devices::schema_choices(&audio_inventory);
    let selected_ready = match config.backend.kind.as_str() {
        "audiocpp" => crate::engine::audiocpp::discover_provider(config, paths).is_ok(),
        "openvino-genai" => {
            crate::engine::openvino_genai::ProviderSpec::from_config(config, paths).is_ok()
        }
        _ => false,
    };
    let mut packaged = config.clone();
    packaged.backend.kind = "audiocpp".into();
    packaged.backend.runtime = Runtime::Default;
    packaged.backend.device = "cpu".into();
    packaged.backend.device_id = 0;
    packaged.backend.library.clear();
    packaged.backend.library_dirs.clear();
    packaged.backend.options.clear();
    let packaged_ready = crate::engine::audiocpp::discover_provider(&packaged, paths).is_ok();
    let current_audio = |runtime| {
        config.backend.kind == "audiocpp" && config.backend.runtime == runtime && selected_ready
    };
    let runtime_choices = vec![
        json!({"value":"default","available":packaged_ready || current_audio(Runtime::Default),"capability":"cpu"}),
        json!({"value":"cuda","available":current_audio(Runtime::Cuda),"capability":"cuda"}),
        json!({"value":"vulkan","available":current_audio(Runtime::Vulkan),"capability":"vulkan"}),
        json!({"value":"hip","available":current_audio(Runtime::Hip),"capability":"hip"}),
        json!({"value":"openvino","available":config.backend.kind == "openvino-genai" && config.backend.runtime == Runtime::Openvino && selected_ready,"capability":"openvino"}),
    ];
    json!({"schema_version":1,"app":"omawake","app_version":env!("CARGO_PKG_VERSION"),"daemon_version":env!("CARGO_PKG_VERSION"),"config_path":config_path,
        "keys":[
            {"key":"backend.kind","type":"enum","section":"Backend","label":"Backend","description":"Complete native inference provider","value":config.backend.kind,"file_value":null,"supported":true,"restart_required":true,"choices":["audiocpp","openvino-genai","whispercpp"]},
            {"key":"backend.runtime","type":"enum","section":"Backend","label":"Runtime","description":"Qualified provider runtime; availability means a matching complete provider was detected","value":config.backend.runtime,"file_value":null,"supported":true,"restart_required":true,"choices":runtime_choices},
            {"key":"backend.device","type":"string","section":"Backend","label":"Device","description":"Runtime-specific device","value":config.backend.device,"file_value":null,"supported":true,"restart_required":true},
            {"key":"backend.device_id","type":"integer","section":"Backend","label":"Device index","description":"Zero-based GPU index for CUDA, Vulkan, or HIP","value":config.backend.device_id,"file_value":null,"supported":true,"restart_required":true,"min":0},
            {"key":"backend.threads","type":"integer","section":"Backend","label":"Threads","description":"Inference threads","value":config.backend.threads,"file_value":null,"supported":true,"restart_required":true,"min":1,"max":64},
            {"key":"backend.fallback","type":"enum","section":"Backend","label":"Fallback","description":"Fallback policy within the selected provider and model","value":config.backend.fallback,"file_value":null,"supported":true,"restart_required":true,"choices":["error","cpu"]},
            {"key":"backend.library","type":"path","section":"Backend","label":"Provider library","description":"Exact shared library for the selected native backend","value":config.backend.library,"file_value":null,"supported":true,"restart_required":true},
            {"key":"backend.library_dirs","type":"path-list","section":"Backend","label":"Native library directories","description":"Complete provider and dependency library search paths","value":config.backend.library_dirs,"file_value":null,"supported":true,"restart_required":true},
            {"key":"model.name","type":"string","section":"Model","label":"Model","description":"Active catalog or custom model profile","value":config.model.name,"file_value":null,"supported":true,"restart_required":true},
            {"key":"model.directory","type":"path","section":"Model","label":"Directory","description":"Model asset directory","value":config.model_directory(paths),"file_value":config.model.directory,"supported":true,"restart_required":true},
            {"key":"model.verifier","type":"string","section":"Model","label":"Verifier model","description":"Phrase verifier filename inside the model directory","value":config.model.verifier,"file_value":null,"supported":true,"restart_required":true},
            {"key":"model.vad","type":"string","section":"Model","label":"VAD model","description":"Silero VAD filename inside the model directory","value":config.model.vad,"file_value":null,"supported":true,"restart_required":true},
            {"key":"model.sample_rate","type":"integer","section":"Model","label":"Sample rate","description":"Native model sample rate in hertz","value":config.model.sample_rate,"file_value":null,"supported":true,"restart_required":true,"min":1},
            {"key":"model.language","type":"enum","section":"Model","label":"Language","description":"Verifier language for the multilingual Whisper profile; empty follows the model default. Accepts every language token the pinned model supports","value":config.model.language,"file_value":null,"supported":true,"restart_required":true,"choices":language_schema_choices(),},
            {"key":"audio.device","type":"enum","choices":audio_choices,"discovery_error":audio_inventory["error"],"section":"Audio","label":"Input device","description":"System default, PipeWire node, or legacy CPAL input name","choices_command":["audio-devices","--detailed","--json"],"value":config.audio.device,"file_value":null,"supported":true,"restart_required":true},
            {"key":"daemon.cooldown_milliseconds","type":"integer","section":"Daemon","label":"Cooldown","description":"Delay after launching an action before reopening capture","value":config.daemon.cooldown_milliseconds,"file_value":null,"supported":true,"restart_required":true,"min":0},
            {"key":"daemon.queue_capacity","type":"integer","section":"Daemon","label":"Capture queue","description":"Bounded live-audio queue capacity","value":config.daemon.queue_capacity,"file_value":null,"supported":true,"restart_required":true,"min":1}],
        "collections":[
            {"prefix":"backend.options.","type":"string-map","section":"Backend","label":"Provider options","description":"Provider-specific load and session options","restart_required":true},
            {"key":"wake_words","id_key":"id","label":"Wake words","description":"Phrase-to-command mappings with exact alternate ASR transcripts","items":config.wake_words}],
        "constraints":[
            {"kind":"matrix","keys":["backend.runtime","backend.device"],"rows":[{"backend.runtime":"default","backend.device":["auto","cpu"]},{"backend.runtime":"cuda","backend.device":["auto","gpu"]},{"backend.runtime":"vulkan","backend.device":["auto","gpu"]},{"backend.runtime":"hip","backend.device":["auto","gpu"]},{"backend.runtime":"openvino","backend.device":["npu","gpu","cpu"]}]},
            {"kind":"runtime-only","key":"backend.device_id","runtimes":["cuda","vulkan","hip"]}]})
}

fn human_schema_lines(schema: &Value) -> Result<Vec<String>> {
    let keys = schema["keys"]
        .as_array()
        .context("configuration schema keys are not an array")?;
    let collections = schema["collections"]
        .as_array()
        .context("configuration schema collections are not an array")?;
    keys.iter()
        .chain(collections)
        .map(|entry| {
            Ok(format!(
                "{}\t{}",
                entry
                    .get("key")
                    .or_else(|| entry.get("prefix"))
                    .and_then(Value::as_str)
                    .context("configuration schema entry has no key or prefix")?,
                entry["description"]
                    .as_str()
                    .context("configuration schema entry has no description")?
            ))
        })
        .collect()
}

#[cfg(test)]
#[path = "../tests/unit/app_main.rs"]
mod tests;

fn choose_audio_device(current: &str) -> Result<Option<String>> {
    app_setup::audio::choose(current, crate::audio::device_inventory)
}

fn setup_audio(
    config_path: &Path,
    paths: &AppPaths,
    device: Option<String>,
    apply: bool,
    test: bool,
) -> Result<()> {
    let mut config = Config::load(config_path)?;
    let interactive = device.is_none() && !test && !apply;
    let selected = if interactive {
        let Some(selected) = choose_audio_device(&config.audio.device)? else {
            println!("Setup cancelled.");
            return Ok(());
        };
        selected
    } else {
        device.unwrap_or_else(|| config.audio.device.clone())
    };
    crate::audio_devices::validate(&selected, "input")?;
    if test {
        test_audio_device(&selected)?;
    }
    let mut save = apply;
    if interactive {
        loop {
            let items = [
                MenuItem::available(
                    "Apply",
                    "Save this route and restart an already-active service.",
                ),
                MenuItem::available(
                    "Test device",
                    "Run a short audio check without loading a model.",
                ),
                MenuItem::available("Cancel", "Leave the current configuration unchanged."),
            ];
            match app_setup::wizard::select("Apply audio device", &selected, &items, 0)? {
                Some(0) => {
                    save = true;
                    break;
                }
                Some(1) => {
                    if let Err(error) = test_audio_device(&selected) {
                        eprintln!("Audio test failed: {error:#}");
                    }
                }
                _ => {
                    println!("Setup cancelled.");
                    return Ok(());
                }
            }
        }
    }
    if save {
        config.audio.device = selected;
        let restarted = save_and_reload_active(config, config_path, paths)?;
        println!(
            "Audio device saved; {}.",
            if restarted {
                "active service restarted"
            } else {
                "restart any manually launched daemon to apply"
            }
        );
    } else if !test {
        println!("Audio device: {selected}; use --apply to save or --test to check it.");
    }
    Ok(())
}

fn test_audio_device(selected: &str) -> Result<()> {
    let capture = Capture::start(selected, 32)?;
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut count = 0;
    let mut energy = 0.0_f64;
    let mut peak = 0.0_f32;
    println!("Listening for three seconds on {selected}…");
    while std::time::Instant::now() < deadline {
        match capture.receiver().recv_timeout(Duration::from_millis(50)) {
            Ok(AudioEvent::Samples { samples, .. }) => {
                for sample in samples {
                    energy += f64::from(sample).powi(2);
                    peak = peak.max(sample.abs());
                    count += 1;
                }
            }
            Ok(AudioEvent::Error(error)) => bail!("{error}"),
            Err(RecvTimeoutError::Disconnected) => bail!("microphone disconnected"),
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
    if count == 0 {
        bail!("microphone delivered no samples");
    }
    println!(
        "Captured {count} samples; RMS {:.4}, peak {peak:.4}. No audio was saved.",
        (energy / count as f64).sqrt()
    );
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct CaptureFailure(String);

/// Retry only capture failures. Each attempt owns and drops its capture and
/// recognition session before waiting, so reconnect never reuses partial audio.
fn retry_audio_cycle(
    mut attempt: impl FnMut() -> Result<(Vec<Detection>, Option<Command>)>,
    mut wait: impl FnMut(&str) -> Result<Option<Command>>,
) -> Result<(Vec<Detection>, Option<Command>)> {
    loop {
        match attempt() {
            Ok(value) => return Ok(value),
            Err(error) if error.downcast_ref::<CaptureFailure>().is_some() => {
                eprintln!("audio unavailable: {error:#}; retrying in one second");
                if let Some(command) = wait(&format!("{error:#}"))? {
                    return Ok((Vec::new(), Some(command)));
                }
            }
            Err(error) => return Err(error),
        }
    }
}
