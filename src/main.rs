use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
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
use omawake::engine::{Detection, Detector};
use omawake::keyword::{KeywordCompiler, validate_wake_words};
use omawake::paths::AppPaths;
use omawake::protocol::{Command, Request, Response, ResultPayload};
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
    Status {
        #[arg(long)]
        json: bool,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
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
        command: SetupCommand,
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
    All,
    Model,
    Runtime,
    Systemd,
    Menu,
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
    let paths = AppPaths::discover();
    let config_path = cli.config.unwrap_or_else(|| paths.config_file.clone());
    let config = Config::load(&config_path)?;
    match cli.command {
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
        TopCommand::Status { json: as_json } => match request(&paths, Command::Status) {
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
        TopCommand::Pause => print_response(request(&paths, Command::Pause)?, false),
        TopCommand::Resume => print_response(request(&paths, Command::Resume)?, false),
        TopCommand::Stop => print_response(request(&paths, Command::Shutdown)?, false),
        TopCommand::AudioDevices { json: as_json } => {
            let devices = input_devices()?;
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
        TopCommand::Setup {
            command: SetupCommand::Runtime,
        } => {
            println!("sherpa-onnx runtime is embedded in this CPU build");
            Ok(())
        }
        TopCommand::Setup {
            command: SetupCommand::Model,
        } => {
            println!(
                "model setup target: {}",
                paths.data_dir.join("models").display()
            );
            Ok(())
        }
        TopCommand::Setup { .. } => bail!("this setup target is not implemented yet"),
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
    match command {
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
        } => config.wake_words.push(WakeWord {
            id,
            phrase,
            enabled: true,
            command,
        }),
        WakeWordCommand::Remove { id } => {
            let before = config.wake_words.len();
            config.wake_words.retain(|item| item.id != id);
            if config.wake_words.len() == before {
                bail!("unknown wake-word id {id}");
            }
        }
    }
    validate_wake_words(&config.wake_words)?;
    let bpe = config.model_directory(paths).join(&config.model.bpe_model);
    if bpe.exists() {
        KeywordCompiler::open(&bpe)?.compile(&config.wake_words)?;
    }
    save_config(path, &config)
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

fn detect_live(detector: &Detector, config: &Config, duration: Duration) -> Result<Vec<Detection>> {
    let capture = Capture::start(&config.audio.device, config.daemon.queue_capacity)?;
    eprintln!(
        "capturing {} Hz, {} channel(s) from {}",
        capture.sample_rate, capture.channels, capture.device_name
    );
    let session = detector.session();
    let deadline = Instant::now() + duration;
    let mut detections = Vec::new();
    while Instant::now() < deadline {
        match capture.receiver().recv_timeout(Duration::from_millis(100)) {
            Ok(AudioEvent::Samples {
                sample_rate,
                samples,
            }) => detections.extend(session.accept(sample_rate, &samples)?),
            Ok(AudioEvent::Error(error)) => bail!("audio capture failed: {error}"),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => bail!("audio capture disconnected"),
        }
    }
    detections.extend(session.finish()?);
    Ok(detections)
}

fn present_detections(
    detector: &Detector,
    detections: Vec<Detection>,
    execute: bool,
    as_json: bool,
) -> Result<()> {
    let actions = if execute {
        detections
            .iter()
            .map(|item| detector.run_action(&item.id))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "model_load_milliseconds": detector.load_time.as_millis() as u64,
                "keywords_buffer": detector.keywords_buffer,
                "detections": detections,
                "actions": actions,
                "backend": {"effective_runtime": detector.effective_runtime, "fallback_used": detector.fallback_used}
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
    let mut paused = false;
    let mut shutdown = false;
    while !shutdown {
        if paused {
            if let Some(command) = poll_control(&listener, "paused", &detector, None)? {
                match command {
                    Command::Resume => paused = false,
                    Command::Shutdown => shutdown = true,
                    _ => {}
                }
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
        let mut triggered = Vec::new();
        loop {
            let audio_details = json!({"device":capture.device_name,"sample_rate":capture.sample_rate,"channels":capture.channels});
            if let Some(command) = poll_control(&listener, "armed", &detector, Some(audio_details))?
            {
                match command {
                    Command::Pause => paused = true,
                    Command::Shutdown => shutdown = true,
                    _ => {}
                }
                if paused || shutdown {
                    break;
                }
            }
            match capture.receiver().recv_timeout(Duration::from_millis(50)) {
                Ok(AudioEvent::Samples {
                    sample_rate,
                    samples,
                }) => {
                    triggered.extend(session.accept(sample_rate, &samples)?);
                    if !triggered.is_empty() {
                        break;
                    }
                }
                Ok(AudioEvent::Error(error)) => bail!("audio capture failed: {error}"),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => bail!("audio capture disconnected"),
            }
        }
        drop(session);
        drop(capture);
        for detection in triggered {
            match detector.run_action(&detection.id) {
                Ok(result) => {
                    eprintln!("detected {}; action exited {}", detection.id, result.status)
                }
                Err(error) => eprintln!("action for {} failed: {error:#}", detection.id),
            }
        }
        if !paused && !shutdown {
            thread::sleep(Duration::from_millis(config.daemon.cooldown_milliseconds));
        }
    }
    let _ = fs::remove_file(socket_path(paths));
    eprintln!("stopped");
    Ok(())
}

fn bind_socket(paths: &AppPaths) -> Result<UnixListener> {
    fs::create_dir_all(&paths.runtime_dir)
        .with_context(|| format!("create {}", paths.runtime_dir.display()))?;
    fs::set_permissions(&paths.runtime_dir, fs::Permissions::from_mode(0o700))?;
    let path = socket_path(paths);
    if path.exists() {
        if UnixStream::connect(&path).is_ok() {
            bail!("daemon is already running at {}", path.display());
        }
        fs::remove_file(&path)
            .with_context(|| format!("remove stale socket {}", path.display()))?;
    }
    let listener = UnixListener::bind(&path).with_context(|| format!("bind {}", path.display()))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    listener.set_nonblocking(true)?;
    Ok(listener)
}

fn poll_control(
    listener: &UnixListener,
    state: &str,
    detector: &Detector,
    audio: Option<serde_json::Value>,
) -> Result<Option<Command>> {
    loop {
        let (mut stream, _) = match listener.accept() {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        stream.set_read_timeout(Some(Duration::from_millis(500)))?;
        let mut bytes = Vec::new();
        let read_result = BufReader::new(stream.try_clone()?)
            .take(65_537)
            .read_until(b'\n', &mut bytes);
        let request: Request = match read_result.context("read daemon request").and_then(|_| {
            if bytes.len() > 65_536 {
                bail!("message exceeds 65536 bytes");
            }
            serde_json::from_slice(&bytes).context("parse daemon request")
        }) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    &mut stream,
                    &Response::error("unknown", "invalid_request", error),
                );
                continue;
            }
        };
        if request.protocol != 1 {
            write_response(
                &mut stream,
                &Response::error(
                    request.id,
                    "protocol_mismatch",
                    format!("unsupported protocol version {}", request.protocol),
                ),
            );
            continue;
        }
        let next_state = match &request.command {
            Command::Pause => "paused",
            Command::Resume => "armed",
            Command::Shutdown => "stopping",
            Command::Status => state,
        };
        let response = Response {
            protocol: 1,
            id: request.id,
            result: ResultPayload::State {
                state: next_state.into(),
                details: json!({
                    "backend":{"effective_runtime":detector.effective_runtime,"fallback_used":detector.fallback_used},
                    "model_load_milliseconds":detector.load_time.as_millis() as u64,
                    "audio":audio
                }),
            },
        };
        write_response(&mut stream, &response);
        if !matches!(request.command, Command::Status) {
            return Ok(Some(request.command));
        }
    }
}

fn request(paths: &AppPaths, command: Command) -> Result<Response> {
    let path = socket_path(paths);
    let mut stream = UnixStream::connect(&path)
        .with_context(|| format!("daemon is not running at {}", path.display()))?;
    serde_json::to_writer(
        &mut stream,
        &Request {
            protocol: 1,
            id: request_id(),
            command,
        },
    )?;
    stream.write_all(b"\n")?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
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

fn write_response(stream: &mut UnixStream, response: &Response) {
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
