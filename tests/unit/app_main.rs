use super::*;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::io::Cursor;

use crate::engine::{WakeWordBackend, WakeWordStream};

struct FakeControl;

struct InMemoryBackend;
struct InMemoryWakeWordStream;

struct DefaultProbePrompts;

impl GuidedPrompts for DefaultProbePrompts {
    fn setup_mode(&mut self) -> Result<Option<SetupMode>> {
        Ok(None)
    }
    fn runtime(&mut self, _: &Config) -> Result<Option<RuntimeSelection>> {
        Ok(None)
    }
    fn confirm_runtime(&mut self, _: &Config, _: &crate::runtime_inventory::Probe) -> Result<bool> {
        Ok(false)
    }
    fn model(
        &mut self,
        _: &AppPaths,
        _: &Config,
    ) -> Result<Option<&'static crate::catalog::ModelSpec>> {
        Ok(None)
    }
    fn model_source_directory(
        &mut self,
        _: &AppPaths,
        _: &crate::catalog::ModelSpec,
    ) -> Result<Option<PathBuf>> {
        Ok(None)
    }
    fn confirm(&mut self, _: &RuntimeSelection, _: &str, _: bool) -> Result<bool> {
        Ok(false)
    }
}

impl WakeWordBackend for InMemoryBackend {
    fn kind(&self) -> &'static str {
        "in-memory"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(InMemoryWakeWordStream)
    }

    fn detect_file(&self, _: &Path) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
}

#[derive(Default)]
struct ScriptedGuidedPrompts {
    mode: Option<SetupMode>,
    runtime: Option<RuntimeSelection>,
    runtime_library_candidates: Vec<Config>,
    runtime_probe_error: Option<&'static str>,
    model: Option<&'static crate::catalog::ModelSpec>,
    archive: Option<PathBuf>,
    confirm: bool,
}

impl GuidedPrompts for ScriptedGuidedPrompts {
    fn probe_runtime(&mut self, _: &Config, _: &Path) -> Result<crate::runtime_inventory::Probe> {
        if let Some(error) = self.runtime_probe_error {
            bail!(error);
        }
        Ok(crate::runtime_inventory::Probe {
            loadable: true,
            device_accessible: None,
            ready: false,
            ..Default::default()
        })
    }
    fn setup_mode(&mut self) -> Result<Option<SetupMode>> {
        Ok(self.mode)
    }

    fn runtime(&mut self, _: &Config) -> Result<Option<RuntimeSelection>> {
        Ok(self.runtime.clone())
    }

    fn runtime_library_dir(&mut self, candidate: &Config) -> Result<Option<PathBuf>> {
        self.runtime_library_candidates.push(candidate.clone());
        Ok(None)
    }

    fn confirm_runtime(&mut self, _: &Config, _: &crate::runtime_inventory::Probe) -> Result<bool> {
        Ok(self.confirm)
    }

    fn prove_runtime(&mut self, _: &Config, _: &AppPaths) -> Result<()> {
        Ok(())
    }

    fn model(
        &mut self,
        _: &AppPaths,
        _: &Config,
    ) -> Result<Option<&'static crate::catalog::ModelSpec>> {
        Ok(self.model)
    }

    fn model_source_directory(
        &mut self,
        _: &AppPaths,
        _: &crate::catalog::ModelSpec,
    ) -> Result<Option<PathBuf>> {
        Ok(self.archive.clone())
    }

    fn confirm(&mut self, _: &RuntimeSelection, _: &str, _: bool) -> Result<bool> {
        Ok(self.confirm)
    }
}

#[test]
fn guided_setup_dispatch_and_runtime_actions_are_testable_without_a_terminal() {
    let paths = test_paths("guided-dispatch");
    let mut cancelled = ScriptedGuidedPrompts::default();
    guided_setup_with(&paths.config_file, &paths, &mut cancelled).unwrap();

    // Each partial flow must treat Back as a clean cancellation without writing
    // a config or installing anything.
    guided_runtime_with(
        &paths.config_file,
        &paths,
        &mut ScriptedGuidedPrompts::default(),
    )
    .unwrap();
    guided_model_with(
        &paths.config_file,
        &paths,
        &mut ScriptedGuidedPrompts::default(),
    )
    .unwrap();
    guided_all_with(
        &paths.config_file,
        &paths,
        &mut ScriptedGuidedPrompts::default(),
    )
    .unwrap();

    let mut runtime = ScriptedGuidedPrompts {
        mode: Some(SetupMode::Runtime),
        runtime: Some(RuntimeSelection {
            runtime: Runtime::Default,
            device: "cpu".into(),
        }),
        confirm: true,
        ..Default::default()
    };
    guided_setup_with(&paths.config_file, &paths, &mut runtime).unwrap();
    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.runtime, Runtime::Default);
    assert_eq!(saved.backend.device, "cpu");

    let mut model_back = ScriptedGuidedPrompts {
        mode: Some(SetupMode::Model),
        ..Default::default()
    };
    guided_setup_with(&paths.config_file, &paths, &mut model_back).unwrap();

    let mut full_back = ScriptedGuidedPrompts {
        mode: Some(SetupMode::Full),
        ..Default::default()
    };
    guided_setup_with(&paths.config_file, &paths, &mut full_back).unwrap();

    let mut check = ScriptedGuidedPrompts {
        mode: Some(SetupMode::Check),
        ..Default::default()
    };
    assert!(guided_setup_with(&paths.config_file, &paths, &mut check).is_err());

    assert_eq!(runtime_name(Runtime::Default), "default");
    assert_eq!(runtime_name(Runtime::Openvino), "openvino");
    assert_eq!(runtime_name(Runtime::Cuda), "cuda");
}

#[test]
fn guided_model_activates_installed_and_downloaded_choices() {
    let spec = &crate::catalog::models()[0];
    let paths = test_paths("guided-model-installed");
    let mut installed = ScriptedGuidedPrompts {
        model: Some(spec),
        ..Default::default()
    };
    guided_model_with_services(
        &paths.config_file,
        &paths,
        &mut installed,
        |_, _| Ok(()),
        |_, _, _, _| unreachable!(),
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.name,
        spec.id
    );

    let paths = test_paths("guided-model-archive");
    let archive = paths.data_dir.join("licensed-model.tar.bz2");
    fs::create_dir_all(&paths.data_dir).unwrap();
    fs::write(&archive, b"fixture").unwrap();
    let mut missing = ScriptedGuidedPrompts {
        model: Some(spec),
        archive: Some(archive.clone()),
        ..Default::default()
    };
    guided_model_with_services(
        &paths.config_file,
        &paths,
        &mut missing,
        |_, _| bail!("missing"),
        |paths, model, selected_archive, format| {
            assert_eq!(selected_archive, Some(archive.as_path()));
            assert_eq!(format, ProgressFormat::Human);
            Ok(app_setup::model::model_directory(paths, model))
        },
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.name,
        spec.id
    );

    let paths = test_paths("guided-model-download");
    let mut downloadable = ScriptedGuidedPrompts {
        model: Some(spec),
        ..Default::default()
    };
    guided_model_with_services(
        &paths.config_file,
        &paths,
        &mut downloadable,
        |_, _| bail!("missing"),
        |paths, model, archive, _| {
            assert!(archive.is_none());
            Ok(app_setup::model::model_directory(paths, model))
        },
    )
    .unwrap();
    assert!(paths.config_file.exists());

    let mut no_model = ScriptedGuidedPrompts::default();
    guided_model_with_services(
        &paths.config_file,
        &paths,
        &mut no_model,
        |_, _| unreachable!(),
        |_, _, _, _| unreachable!(),
    )
    .unwrap();
}

#[test]
fn guided_full_setup_waits_for_review_and_then_runs_selected_plan() {
    let spec = &crate::catalog::models()[0];
    let selection = RuntimeSelection {
        runtime: Runtime::Default,
        device: "cpu".into(),
    };

    for (index, prompts) in [
        ScriptedGuidedPrompts::default(),
        ScriptedGuidedPrompts {
            runtime: Some(selection.clone()),
            ..Default::default()
        },
        ScriptedGuidedPrompts {
            runtime: Some(selection.clone()),
            model: Some(spec),
            archive: Some(PathBuf::from("/tmp/licensed-model.tar.bz2")),
            confirm: false,
            ..Default::default()
        },
    ]
    .into_iter()
    .enumerate()
    {
        let paths = test_paths(&format!("guided-full-cancel-{index}"));
        let mut prompts = prompts;
        guided_all_with_services_and_validator(
            &paths.config_file,
            &paths,
            &mut prompts,
            |_, _| Ok(()),
            |_, _, _, _| unreachable!(),
            |_, _| unreachable!(),
            |_| unreachable!(),
            |_| false,
            |_| unreachable!(),
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .unwrap();
        assert!(!paths.config_file.exists());
    }

    let paths = test_paths("guided-full-apply");
    let mut prompts = ScriptedGuidedPrompts {
        runtime: Some(RuntimeSelection {
            runtime: Runtime::Cuda,
            device: "gpu".into(),
        }),
        model: Some(spec),
        archive: Some(paths.data_dir.join("licensed-model.tar.bz2")),
        confirm: true,
        ..Default::default()
    };
    let reloads = std::cell::Cell::new(0);
    guided_all_with_services_and_validator(
        &paths.config_file,
        &paths,
        &mut prompts,
        |candidate, _| {
            assert_eq!(candidate.backend.runtime, Runtime::Cuda);
            assert_eq!(candidate.backend.device, "gpu");
            Ok(())
        },
        |paths, model, archive, format| {
            assert_eq!(
                archive,
                Some(paths.data_dir.join("licensed-model.tar.bz2").as_path())
            );
            assert_eq!(format, ProgressFormat::Human);
            Ok(app_setup::model::model_directory(paths, model))
        },
        |candidate, _| {
            assert_eq!(candidate.backend.runtime, Runtime::Cuda);
            assert_eq!(candidate.backend.device, "gpu");
            Ok(())
        },
        |paths| Ok(paths.data_dir.join("applications/omawake.desktop")),
        |found| {
            assert_eq!(found.config_file, paths.config_file);
            true
        },
        |was_active| {
            assert!(was_active);
            reloads.set(reloads.get() + 1);
            Ok(true)
        },
        |_, _| Ok(()),
        |_, _| unreachable!(),
    )
    .unwrap();
    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.runtime, Runtime::Cuda);
    assert_eq!(saved.backend.device, "gpu");
    assert_eq!(saved.model.name, spec.id);
    assert_eq!(reloads.get(), 1);
    assert!(!app_setup::systemd::service_path(&paths).exists());
}

#[test]
fn guided_full_rejects_the_runtime_before_setup_callbacks_or_config_changes() {
    let paths = test_paths("guided-full-invalid-runtime");
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    let original = b"# preserve this exact file\n[backend]\nruntime = \"default\"\n";
    fs::write(&paths.config_file, original).unwrap();
    let callbacks = std::cell::Cell::new(0);
    let mut prompts = ScriptedGuidedPrompts {
        runtime: Some(RuntimeSelection {
            runtime: Runtime::Cuda,
            device: "gpu".into(),
        }),
        ..Default::default()
    };

    let error = guided_all_with_services_and_validator(
        &paths.config_file,
        &paths,
        &mut prompts,
        |candidate, _| {
            assert_eq!(candidate.backend.runtime, Runtime::Cuda);
            bail!("candidate runtime probe failed")
        },
        |_, _, _, _| {
            callbacks.set(callbacks.get() + 1);
            unreachable!()
        },
        |_, _| {
            callbacks.set(callbacks.get() + 1);
            unreachable!()
        },
        |_| {
            callbacks.set(callbacks.get() + 1);
            unreachable!()
        },
        |_| {
            callbacks.set(callbacks.get() + 1);
            false
        },
        |_| {
            callbacks.set(callbacks.get() + 1);
            unreachable!()
        },
        |_, _| {
            callbacks.set(callbacks.get() + 1);
            unreachable!()
        },
        |_, _| {
            callbacks.set(callbacks.get() + 1);
            unreachable!()
        },
    )
    .unwrap_err();

    assert!(error.to_string().contains("candidate runtime probe failed"));
    assert_eq!(callbacks.get(), 0);
    assert_eq!(fs::read(&paths.config_file).unwrap(), original);
}

#[test]
fn guided_full_rolls_back_existing_and_new_configs_after_later_failures() {
    let spec = &crate::catalog::models()[0];
    let selection = RuntimeSelection {
        runtime: Runtime::Default,
        device: "cpu".into(),
    };

    let existing = test_paths("guided-full-rollback-existing");
    fs::create_dir_all(existing.config_file.parent().unwrap()).unwrap();
    let original =
        b"# byte-for-byte rollback\n[backend]\nruntime = \"default\"\ndevice = \"auto\"\n";
    fs::write(&existing.config_file, original).unwrap();
    let existing_launcher = app_setup::menu::launcher_path(&existing);
    fs::create_dir_all(existing_launcher.parent().unwrap()).unwrap();
    fs::write(&existing_launcher, b"prior launcher").unwrap();
    let mut prompts = ScriptedGuidedPrompts {
        runtime: Some(selection.clone()),
        model: Some(spec),
        archive: Some(existing.data_dir.join("licensed-model.tar.bz2")),
        confirm: true,
        ..Default::default()
    };
    let error = guided_all_with_services_and_validator(
        &existing.config_file,
        &existing,
        &mut prompts,
        |_, _| Ok(()),
        |paths, model, _, _| Ok(app_setup::model::model_directory(paths, model)),
        |_, _| Ok(()),
        |paths| {
            let launcher = app_setup::menu::launcher_path(paths);
            fs::write(&launcher, b"replacement launcher")?;
            Ok(launcher)
        },
        |_| false,
        |_| unreachable!(),
        |_, _| bail!("check failed after launcher install"),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("check failed"));
    assert_eq!(fs::read(&existing.config_file).unwrap(), original);
    assert_eq!(fs::read(existing_launcher).unwrap(), b"prior launcher");

    let new = test_paths("guided-full-rollback-new");
    let mut prompts = ScriptedGuidedPrompts {
        runtime: Some(selection),
        model: Some(spec),
        archive: Some(new.data_dir.join("licensed-model.tar.bz2")),
        confirm: true,
        ..Default::default()
    };
    let error = guided_all_with_services_and_validator(
        &new.config_file,
        &new,
        &mut prompts,
        |_, _| Ok(()),
        |paths, model, _, _| Ok(app_setup::model::model_directory(paths, model)),
        |_, _| Ok(()),
        |paths| {
            let launcher = app_setup::menu::launcher_path(paths);
            fs::create_dir_all(launcher.parent().unwrap())?;
            fs::write(&launcher, b"new launcher")?;
            Ok(launcher)
        },
        |_| false,
        |_| unreachable!(),
        |_, _| bail!("final setup check failed"),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("final setup check failed"));
    assert!(!new.config_file.exists());
    assert!(!app_setup::menu::launcher_path(&new).exists());

    let service = test_paths("guided-full-rollback-service");
    Config::default().save(&service.config_file).unwrap();
    let original = fs::read(&service.config_file).unwrap();
    let restarts = Cell::new(0);
    let error = install_everything(
        spec,
        &service.config_file,
        &service,
        None,
        ProgressFormat::Human,
        true,
        |_, _| Ok(()),
        |paths, model, _, _| Ok(app_setup::model::model_directory(paths, model)),
        |_, _| Ok(()),
        |paths| Ok(app_setup::menu::launcher_path(paths)),
        |_| {
            restarts.set(restarts.get() + 1);
            if restarts.get() == 1 {
                bail!("new config failed to restart")
            }
            Ok(true)
        },
        |_, _| Ok(()),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("new config failed to restart"));
    assert_eq!(restarts.get(), 2);
    assert_eq!(fs::read(&service.config_file).unwrap(), original);
}

#[test]
fn guided_model_catalog_exposes_status_metadata_and_selection() {
    let paths = test_paths("guided-model-catalog");
    let active = crate::catalog::models()[0].id;
    let selected = choose_model_with(&paths, active, |items, preferred| {
        assert_eq!(preferred, 0);
        assert!(items[0].enabled);
        assert!(items[0].label.contains("download required"));
        assert!(items[0].detail.contains("MIT (verified)"));
        Ok(Some(0))
    })
    .unwrap();
    assert_eq!(selected.map(|model| model.id), Some(active));
}

#[test]
fn terminal_prompt_adapter_reports_non_tty_errors() {
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        return;
    }
    let paths = test_paths("terminal-prompts");
    let config = Config::default();
    let mut prompts = TerminalGuidedPrompts {
        config_path: paths.config_file.clone(),
    };
    assert!(prompts.setup_mode().is_err());
    assert!(prompts.runtime(&config).is_err());
    assert!(prompts.runtime_library_dir(&config).is_err());
    assert!(
        prompts
            .confirm_runtime(&config, &crate::runtime_inventory::Probe::default())
            .is_err()
    );
    assert!(prompts.model(&paths, &config).is_err());
    assert!(
        prompts
            .model_source_directory(&paths, &crate::catalog::models()[0])
            .unwrap()
            .is_none()
    );
    assert!(
        prompts
            .confirm(
                &RuntimeSelection {
                    runtime: Runtime::Default,
                    device: "cpu".into(),
                },
                "model",
                false,
            )
            .is_err()
    );
    assert!(guided_runtime(&paths.config_file, &paths).is_err());
    assert!(guided_model(&paths.config_file, &paths).is_err());
    assert!(guided_setup(&paths.config_file, &paths).is_err());
    assert!(guided_all_with(&paths.config_file, &paths, &mut prompts).is_err());
}

impl WakeWordStream for InMemoryWakeWordStream {
    fn accept(&self, _: i32, _: &[f32]) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
}

struct ScriptedStream {
    response: Cursor<Vec<u8>>,
    request: Vec<u8>,
    fail_writes: bool,
}

impl ScriptedStream {
    fn responding_with(response: impl Into<Vec<u8>>) -> Self {
        Self {
            response: Cursor::new(response.into()),
            request: Vec::new(),
            fail_writes: false,
        }
    }
}

impl Read for ScriptedStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.response.read(buffer)
    }
}

impl Write for ScriptedStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        if self.fail_writes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "client disconnected",
            ));
        }
        self.request.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl DetectorControl for FakeControl {
    fn backend_kind(&self) -> &str {
        "fake"
    }

    fn effective_runtime(&self) -> Runtime {
        Runtime::Default
    }

    fn fallback_used(&self) -> bool {
        false
    }

    fn load_time(&self) -> Duration {
        Duration::from_millis(2)
    }

    fn keywords_buffer(&self) -> &str {
        "HELLO @hello"
    }

    fn run(&self, id: &str) -> Result<ActionResult> {
        Ok(ActionResult {
            id: id.into(),
            program: "true".into(),
            arguments: Vec::new(),
            state: "started",
            pid: 1,
        })
    }
}

fn test_paths(name: &str) -> AppPaths {
    let root = crate::test_support::unique_directory("main", name);
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

fn shell_script(path: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("/bin/sh");
    command.arg(path);
    command
}

#[test]
fn evaluation_command_loads_verified_manifest_and_reports_injected_native_identity() {
    let paths = test_paths("evaluation-command");
    fs::create_dir_all(&paths.data_dir).unwrap();
    let audio = paths.data_dir.join("negative.wav");
    let mut writer = hound::WavWriter::create(
        &audio,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..4_000 {
        writer.write_sample(0_i16).unwrap();
    }
    writer.finalize().unwrap();
    let checksum = evaluation::sha256_bytes(&fs::read(&audio).unwrap());
    let manifest = paths.data_dir.join("manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "corpus": {
                "id": "injected-negative",
                "version": "1",
                "license_spdx": "CC0-1.0",
                "source": "generated silence"
            },
            "clips": [{
                "id": "negative",
                "path": "negative.wav",
                "sha256": checksum,
                "split": "test",
                "expected": []
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let config = Config::default();
    let report = evaluation_report_with(
        &config,
        &paths,
        &manifest,
        || Ok(()),
        |_, path| {
            assert_eq!(path, audio);
            Ok(Vec::new())
        },
        |_| {
            Ok(RuntimeIdentity {
                backend_kind: "audiocpp".into(),
                requested_runtime: "default".into(),
                requested_device: "cpu".into(),
                effective_runtime: "default".into(),
                fallback_used: false,
                placement_verified: true,
                placement_evidence: "injected C ABI provider".into(),
            })
        },
    )
    .unwrap();
    assert_eq!(report["evaluation"], "omawake-phrase-verifier-accuracy");
    assert_eq!(report["backend"]["backend_kind"], "audiocpp");
    assert_eq!(report["summary"]["false_activations"], 0);
}

fn send_raw(path: &Path, bytes: Vec<u8>) -> UnixStream {
    let mut stream = UnixStream::connect(path).unwrap();
    stream.write_all(&bytes).unwrap();
    stream
}

fn encoded_request(protocol: u32, id: &str, command: Command) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(&Request {
        protocol,
        id: id.into(),
        command,
    })
    .unwrap();
    bytes.push(b'\n');
    bytes
}

fn sandbox_denied(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|error| error.kind() == std::io::ErrorKind::PermissionDenied)
}

fn anyhow_or_skip<T>(result: Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) if sandbox_denied(&error) => None,
        Err(error) => panic!("{error:#}"),
    }
}

fn io_or_skip<T>(result: std::io::Result<T>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => None,
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn word_alias_parses_remove() {
    let cli = Cli::try_parse_from(["omawake", "word", "remove", "computer"]).unwrap();
    assert!(matches!(
        cli.command,
        TopCommand::WakeWord {
            command: WakeWordCommand::Remove { ref id }
        } if id == "computer"
    ));
}

#[test]
fn last_wake_word_can_be_removed() {
    let mut config = Config::default();
    remove_wake_word(&mut config, "computer").unwrap();
    assert!(config.wake_words.is_empty());
    assert!(validate_wake_words(&config.wake_words).is_ok());
}

#[test]
fn command_line_surface_parses_representative_forms() {
    let cases = [
        vec!["omawake", "test", "--seconds", "1", "--execute", "--json"],
        vec!["omawake", "test", "--audio", "fixture.wav"],
        vec![
            "omawake",
            "benchmark",
            "--warmup",
            "2",
            "--iterations",
            "4",
            "one.wav",
            "two.wav",
        ],
        vec!["omawake", "status", "--json"],
        vec!["omawake", "config", "get", "backend.kind", "--json"],
        vec!["omawake", "config", "schema"],
        vec!["omawake", "config", "set", "backend.device", "cpu"],
        vec!["omawake", "config", "unset", "backend.device"],
        vec![
            "omawake",
            "wake-word",
            "add",
            "--id",
            "hello",
            "--phrase",
            "Hello",
            "--",
            "true",
        ],
        vec!["omawake", "wake-word", "list", "--json"],
        vec!["omawake", "audio-devices", "--json"],
        vec!["omawake", "evaluate", "manifest.json"],
        vec!["omawake", "daemon"],
        vec!["omawake", "pause"],
        vec!["omawake", "resume"],
        vec!["omawake", "stop"],
        vec![
            "omawake",
            "__audiocpp-worker",
            "libaudiocpp.so.0.1.0",
            "moonshine.gguf",
            "silero.safetensors",
            "2",
            "moonshine_asr",
            "cpu",
            "0",
        ],
        vec!["omawake", "setup", "check", "--json"],
        vec!["omawake", "setup", "runtime", "--json"],
        vec![
            "omawake",
            "setup",
            "runtime",
            "--runtime",
            "cuda",
            "--device",
            "gpu",
            "--device-id",
            "1",
            "--dir",
            "/opt/audiocpp-cuda",
            "--apply",
        ],
        vec!["omawake", "setup", "systemd", "--no-start"],
        vec!["omawake", "setup", "menu", "--status"],
        vec![
            "omawake",
            "setup",
            "model",
            "--download",
            "model",
            "--no-activate",
            "--progress-format",
            "json",
        ],
        vec![
            "omawake",
            "setup",
            "all",
            "--model",
            "model",
            "--progress-format",
            "human",
        ],
    ];
    for args in cases {
        Cli::try_parse_from(args).unwrap();
    }
    assert!(Cli::try_parse_from(["omawake", "test", "--audio", "a", "--seconds", "1"]).is_err());
    assert!(Cli::try_parse_from(["omawake", "benchmark"]).is_err());
    assert!(Cli::try_parse_from(["omawake", "benchmark", "--iterations", "0", "a.wav"]).is_err());
    assert!(
        Cli::try_parse_from(["omawake", "setup", "systemd", "--status", "--uninstall"]).is_err()
    );
    assert!(Cli::try_parse_from(["omawake", "setup", "all", "--no-start"]).is_err());
    for args in [
        ["omawake", "benchmark", "input.wav"].as_slice(),
        ["omawake", "evaluate", "manifest.json"].as_slice(),
    ] {
        assert!(native_json_request(&Cli::try_parse_from(args).unwrap().command).is_some());
    }
    for args in [
        ["omawake", "test"].as_slice(),
        ["omawake", "test", "--audio", "input.wav"].as_slice(),
        ["omawake", "test", "--seconds", "1"].as_slice(),
        ["omawake", "daemon"].as_slice(),
        ["omawake", "status"].as_slice(),
        ["omawake", "config", "get"].as_slice(),
        ["omawake", "setup", "runtime"].as_slice(),
    ] {
        assert!(native_json_request(&Cli::try_parse_from(args).unwrap().command).is_none());
    }
    let denied = || std::io::Error::from(std::io::ErrorKind::PermissionDenied);
    assert!(io_or_skip::<()>(Err(denied())).is_none());
    assert!(anyhow_or_skip::<()>(Err(denied().into())).is_none());
    assert!(
        std::panic::catch_unwind(|| { io_or_skip::<()>(Err(std::io::Error::other("unexpected"))) })
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| { anyhow_or_skip::<()>(Err(anyhow::anyhow!("unexpected"))) })
            .is_err()
    );
}

#[test]
fn hidden_native_worker_dispatch_forwards_exact_provider_arguments() {
    let paths = test_paths("hidden-worker-dispatch");
    let path = |name: &str| PathBuf::from(format!("/{name}"));

    run_entry_with_workers(
        Cli {
            config: None,
            command: TopCommand::OpenVinoRuntimeWorker {
                genai_library: path("genai"),
                core_library: path("core"),
                audiocpp_library: path("audio"),
                device: "NPU".into(),
            },
        },
        paths.clone(),
        |genai, core, audio, device| {
            assert_eq!(genai, Path::new("/genai"));
            assert_eq!(core, Path::new("/core"));
            assert_eq!(audio, Path::new("/audio"));
            assert_eq!(device, "NPU");
            Ok(())
        },
        |_| unreachable!(),
        |_, _, _, _, _, _, _| unreachable!(),
        |_, _, _, _| unreachable!(),
    )
    .unwrap();

    run_entry_with_workers(
        Cli {
            config: None,
            command: TopCommand::OpenVinoGenAiWorker {
                genai_library: path("genai"),
                core_library: path("core"),
                audiocpp_library: path("audio"),
                model_directory: path("model"),
                vad_model: path("vad"),
                cache_directory: path("cache"),
                device: "GPU".into(),
                vad_threads: 3,
            },
        },
        paths.clone(),
        |_, _, _, _| unreachable!(),
        |spec| {
            assert_eq!(spec.genai_library, path("genai"));
            assert_eq!(spec.core_library, path("core"));
            assert_eq!(spec.audiocpp_library, path("audio"));
            assert_eq!(spec.model_directory, path("model"));
            assert_eq!(spec.vad_model, path("vad"));
            assert_eq!(spec.cache_directory, path("cache"));
            assert_eq!(spec.device, "GPU");
            assert_eq!(spec.vad_threads, 3);
            Ok(())
        },
        |_, _, _, _, _, _, _| unreachable!(),
        |_, _, _, _| unreachable!(),
    )
    .unwrap();

    run_entry_with_workers(
        Cli {
            config: None,
            command: TopCommand::AudioCppWorker {
                library: path("audio"),
                verifier: path("verifier"),
                vad: path("vad"),
                threads: 4,
                asr_family: "moonshine_asr".into(),
                backend: "cuda".into(),
                device: 2,
            },
        },
        paths.clone(),
        |_, _, _, _| unreachable!(),
        |_| unreachable!(),
        |library, verifier, vad, threads, family, backend, device| {
            assert_eq!(library, Path::new("/audio"));
            assert_eq!(verifier, Path::new("/verifier"));
            assert_eq!(vad, Path::new("/vad"));
            assert_eq!(
                (threads, family, backend, device),
                (4, "moonshine_asr", "cuda", 2)
            );
            Ok(())
        },
        |_, _, _, _| unreachable!(),
    )
    .unwrap();

    run_entry_with_workers(
        Cli {
            config: None,
            command: TopCommand::WhisperWorker {
                library: path("whisper"),
                verifier: path("verifier"),
                vad: path("vad"),
                threads: 5,
            },
        },
        paths,
        |_, _, _, _| unreachable!(),
        |_| unreachable!(),
        |_, _, _, _, _, _, _| unreachable!(),
        |library, verifier, vad, threads| {
            assert_eq!(library, Path::new("/whisper"));
            assert_eq!(verifier, Path::new("/verifier"));
            assert_eq!(vad, Path::new("/vad"));
            assert_eq!(threads, 5);
            Ok(())
        },
    )
    .unwrap();
}

#[test]
fn config_helpers_cover_every_supported_key_and_validation() {
    let mut config = Config::default();
    for (key, value) in [
        ("backend.kind", "custom"),
        ("backend.runtime", "OPENVINO"),
        ("backend.device", "npu"),
        ("backend.threads", "3"),
        ("backend.fallback", "CPU"),
        ("backend.device_id", "2"),
        ("backend.library", "/opt/whisper/lib/libwhisper.so.1"),
        ("backend.library_dirs", "/opt/oma/lib:/opt/vendor/lib"),
        ("model.name", "custom-model"),
        ("model.directory", "/models/custom"),
        ("model.verifier", "ggml-tiny.en.bin"),
        ("model.vad", "ggml-silero-v6.2.0.bin"),
        ("model.sample_rate", "8000"),
        ("audio.device", "microphone"),
        ("daemon.cooldown_milliseconds", "90"),
        ("daemon.queue_capacity", "5"),
    ] {
        set_config(&mut config, key, value).unwrap();
        unset_config(&mut config, key).unwrap();
    }
    assert!(set_config(&mut config, "backend.threads", "many").is_err());
    assert!(set_config(&mut config, "missing", "value").is_err());
    assert!(unset_config(&mut config, "missing").is_err());
    assert_eq!(parse_runtime("DEFAULT").unwrap(), Runtime::Default);
    assert_eq!(parse_runtime("cuda").unwrap(), Runtime::Cuda);
    assert_eq!(parse_runtime("vulkan").unwrap(), Runtime::Vulkan);
    assert_eq!(parse_runtime("hip").unwrap(), Runtime::Hip);
    assert!(parse_runtime("rocm").is_err());
    assert_eq!(parse_fallback("error").unwrap(), Fallback::Error);
    assert!(parse_fallback("maybe").is_err());

    let value = json!({"one":{"two":3}});
    assert_eq!(dotted_get(&value, "one.two"), Some(&json!(3)));
    assert!(dotted_get(&value, "one.missing").is_none());
}

#[test]
fn config_mutation_saves_and_wake_words_validate_before_persisting() {
    let paths = test_paths("config");
    let config = Config::default();
    config_mutation(
        ConfigCommand::Set {
            key: "backend.threads".into(),
            value: "4".into(),
        },
        config.clone(),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    assert_eq!(Config::load(&paths.config_file).unwrap().backend.threads, 4);
    config_mutation(
        ConfigCommand::Set {
            key: "backend.runtime".into(),
            value: "openvino".into(),
        },
        Config::load(&paths.config_file).unwrap(),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    let accelerated = Config::load(&paths.config_file).unwrap();
    assert!(accelerated.backend.library.as_os_str().is_empty());
    config_mutation(
        ConfigCommand::Unset {
            key: "backend.threads".into(),
        },
        Config::load(&paths.config_file).unwrap(),
        &paths.config_file,
        &paths,
    )
    .unwrap();

    wake_word_command(
        WakeWordCommand::Add {
            id: "lovely-child".into(),
            phrase: "Lovely Child".into(),
            aliases: Vec::new(),
            command: vec!["true".into()],
        },
        config.clone(),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().wake_words.len(),
        2
    );
    assert!(
        wake_word_command(
            WakeWordCommand::Add {
                id: "computer".into(),
                phrase: "Duplicate".into(),
                aliases: Vec::new(),
                command: vec!["true".into()],
            },
            config,
            &paths.config_file,
            &paths,
        )
        .is_err()
    );
}

#[test]
fn config_refresh_skips_inactive_and_unowned_services() {
    let paths = test_paths("config-refresh-inactive");
    let mut config = Config::default();
    config.audio.device = "updated".into();
    let reloads = std::cell::Cell::new(0);
    let restarts = std::cell::Cell::new(0);
    assert!(
        !save_and_reload_active_with(
            config.clone(),
            &paths.config_file,
            true,
            || false,
            |_| {
                reloads.set(reloads.get() + 1);
                Ok(true)
            },
            || {
                restarts.set(restarts.get() + 1);
                Ok(())
            },
        )
        .unwrap()
    );
    assert_eq!(reloads.get(), 0);
    assert_eq!(restarts.get(), 0);
    assert_eq!(
        Config::load(&paths.config_file).unwrap().audio.device,
        "updated"
    );

    let activity_checks = std::cell::Cell::new(0);
    config.audio.device = "custom-config".into();
    assert!(
        !save_and_reload_active_with(
            config,
            &paths.config_file,
            false,
            || {
                activity_checks.set(activity_checks.get() + 1);
                true
            },
            |_| unreachable!(),
            || unreachable!(),
        )
        .unwrap()
    );
    assert_eq!(activity_checks.get(), 0);
}

#[test]
fn config_refresh_restarts_once_and_explicitly_resurrects_after_rollback() {
    let paths = test_paths("config-refresh-rollback");
    let mut original = Config::default();
    original.audio.device = "old".into();
    original.save(&paths.config_file).unwrap();

    let mut updated = original.clone();
    updated.audio.device = "new".into();
    let reloads = std::cell::Cell::new(0);
    let restarted = save_and_reload_active_with(
        updated,
        &paths.config_file,
        true,
        || true,
        |_| {
            reloads.set(reloads.get() + 1);
            assert_eq!(
                Config::load(&paths.config_file).unwrap().audio.device,
                "new"
            );
            Ok(true)
        },
        || unreachable!(),
    )
    .unwrap();
    assert!(restarted);
    assert_eq!(reloads.get(), 1);
    let previous_bytes = fs::read(&paths.config_file).unwrap();

    let mut rejected = Config::load(&paths.config_file).unwrap();
    rejected.audio.device = "broken".into();
    let explicit_restarts = std::cell::Cell::new(0);
    let error = save_and_reload_active_with(
        rejected,
        &paths.config_file,
        true,
        || true,
        |_| {
            assert_eq!(
                Config::load(&paths.config_file).unwrap().audio.device,
                "broken"
            );
            bail!("injected failed startup left service inactive")
        },
        || {
            explicit_restarts.set(explicit_restarts.get() + 1);
            assert_eq!(fs::read(&paths.config_file).unwrap(), previous_bytes);
            Ok(())
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("reload active daemon"));
    assert!(
        error
            .to_string()
            .contains("previous configuration and daemon were restored")
    );
    assert_eq!(explicit_restarts.get(), 1);
    assert_eq!(fs::read(&paths.config_file).unwrap(), previous_bytes);
}

#[test]
fn runtime_candidate_validation_failure_preserves_the_original_config() {
    let paths = test_paths("runtime-candidate-validation");
    let original = Config::default();
    original.save(&paths.config_file).unwrap();
    let original_bytes = fs::read(&paths.config_file).unwrap();

    let error = save_runtime_selection_impl_with(
        &paths.config_file,
        &RuntimeSelection {
            runtime: Runtime::Default,
            device: "cpu".into(),
        },
        None,
        |staged, path| {
            assert_eq!(path, paths.config_file);
            assert_eq!(staged.backend.runtime, Runtime::Default);
            assert_eq!(staged.backend.device, "cpu");
            bail!("injected ABI probe failure")
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("injected ABI probe failure"));
    assert_eq!(fs::read(&paths.config_file).unwrap(), original_bytes);

    let mut default = Config::default();
    default.backend.device = "cpu".into();
    let error = validate_runtime_candidate_with(&default, &paths.config_file, |backend, path| {
        assert_eq!(backend.backend.runtime, Runtime::Default);
        assert_eq!(backend.backend.device, "cpu");
        assert_eq!(path, paths.config_file);
        crate::runtime_inventory::Probe {
            errors: vec!["default CPU payload is not staged".into()],
            ..Default::default()
        }
    })
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("default CPU payload is not staged")
    );
}

#[test]
fn focused_runtime_preview_failure_and_apply_recover_invalid_config_transactionally() {
    let paths = test_paths("runtime-invalid-config-recovery");
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    let invalid = b"# obsolete pre-release config\n[backend]\nremoved_option = \"\"\n";
    fs::write(&paths.config_file, invalid).unwrap();

    let discovered = |_: &Config, _: &Path| crate::runtime_inventory::Probe {
        loadable: true,
        device_accessible: None,
        ready: false,
        ..Default::default()
    };
    configure_runtime_from_flags_with(
        &paths.config_file,
        &paths,
        Some("default".into()),
        Some("cpu".into()),
        None,
        None,
        false,
        discovered,
        |_, _, _, _| Ok(None),
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(fs::read(&paths.config_file).unwrap(), invalid);

    let rejected = |_: &Config, _: &Path| crate::runtime_inventory::Probe {
        errors: vec!["injected runtime validation failure".into()],
        ..Default::default()
    };
    let error = configure_runtime_from_flags_with(
        &paths.config_file,
        &paths,
        Some("default".into()),
        Some("cpu".into()),
        None,
        None,
        true,
        rejected,
        |_, _, _, _| Ok(None),
        |_, _| Ok(()),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected runtime validation failure")
    );
    assert_eq!(fs::read(&paths.config_file).unwrap(), invalid);

    configure_runtime_from_flags_with(
        &paths.config_file,
        &paths,
        Some("default".into()),
        Some("cpu".into()),
        None,
        None,
        true,
        discovered,
        |_, _, _, _| Ok(None),
        |_, _| Ok(()),
    )
    .unwrap();
    let repaired = Config::load(&paths.config_file).unwrap();
    assert_eq!(repaired.backend.kind, "audiocpp");
    assert_eq!(repaired.backend.device, "cpu");
    assert!(
        !fs::read_to_string(&paths.config_file)
            .unwrap()
            .contains("removed_option")
    );
}

#[test]
fn focused_openvino_runtime_probe_and_proof_receive_the_compatible_model() {
    let paths = test_paths("runtime-openvino-model-candidate");
    Config::default().save(&paths.config_file).unwrap();

    configure_runtime_from_flags_with(
        &paths.config_file,
        &paths,
        Some("openvino".into()),
        Some("cpu".into()),
        None,
        None,
        true,
        |candidate, path| {
            assert_eq!(path, paths.config_file);
            assert_eq!(candidate.backend.kind, "openvino-genai");
            assert_eq!(candidate.backend.runtime, Runtime::Openvino);
            assert_eq!(candidate.model.name, crate::catalog::OPENVINO_MODEL_ID);
            crate::runtime_inventory::Probe {
                loadable: true,
                device_accessible: Some(true),
                ready: false,
                ..Default::default()
            }
        },
        |candidate, _, _, _| {
            assert_eq!(candidate.model.name, crate::catalog::OPENVINO_MODEL_ID);
            Ok(None)
        },
        |candidate, _| {
            assert_eq!(candidate.model.name, crate::catalog::OPENVINO_MODEL_ID);
            Ok(())
        },
    )
    .unwrap();

    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.kind, "openvino-genai");
    assert_eq!(saved.backend.runtime, Runtime::Openvino);
    assert_eq!(saved.model.name, crate::catalog::OPENVINO_MODEL_ID);
}

#[test]
fn npu_runtime_cache_preparation_finishes_before_config_commit() {
    let paths = test_paths("runtime-cache-transaction");
    let original = Config::default();
    original.save(&paths.config_file).unwrap();
    let original_bytes = fs::read(&paths.config_file).unwrap();
    let mut candidate = original.clone();
    candidate.backend.runtime = Runtime::Openvino;
    candidate.backend.device = "npu".into();

    let error = prepare_and_save_runtime_candidate_with(
        &candidate,
        &paths.config_file,
        &paths,
        ProgressFormat::Json,
        |received, path, received_paths, progress| {
            assert_eq!(received.backend.runtime, Runtime::Openvino);
            assert_eq!(path, paths.config_file);
            assert_eq!(received_paths, &paths);
            assert_eq!(progress, ProgressFormat::Json);
            assert_eq!(fs::read(path).unwrap(), original_bytes);
            bail!("cold NPU compile failed")
        },
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("cold NPU compile failed"));
    assert_eq!(fs::read(&paths.config_file).unwrap(), original_bytes);

    let proof_error = prepare_and_save_runtime_candidate_with(
        &candidate,
        &paths.config_file,
        &paths,
        ProgressFormat::Human,
        |_, path, _, _| {
            assert_eq!(fs::read(path).unwrap(), original_bytes);
            Ok(None)
        },
        |received, received_paths| {
            assert_eq!(received.backend.runtime, Runtime::Openvino);
            assert_eq!(received_paths, &paths);
            assert_eq!(fs::read(&paths.config_file).unwrap(), original_bytes);
            bail!("injected file-only provider/model proof failure")
        },
    )
    .unwrap_err();
    assert!(
        proof_error
            .to_string()
            .contains("runtime candidate failed its file-only provider/model proof")
    );
    assert_eq!(fs::read(&paths.config_file).unwrap(), original_bytes);

    prepare_and_save_runtime_candidate_with(
        &candidate,
        &paths.config_file,
        &paths,
        ProgressFormat::Human,
        |_, path, _, _| {
            assert_eq!(fs::read(path).unwrap(), original_bytes);
            Ok(None)
        },
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().backend.runtime,
        Runtime::Openvino
    );
}

#[test]
fn guided_runtime_apply_and_cancel_share_the_review_step() {
    let paths = test_paths("guided-runtime-review");
    let selection = RuntimeSelection {
        runtime: Runtime::Default,
        device: "cpu".into(),
    };
    let mut cancel = ScriptedGuidedPrompts {
        runtime: Some(selection.clone()),
        confirm: false,
        ..Default::default()
    };
    guided_runtime_with(&paths.config_file, &paths, &mut cancel).unwrap();
    assert!(!paths.config_file.exists());

    let mut apply = ScriptedGuidedPrompts {
        runtime: Some(selection),
        confirm: true,
        ..Default::default()
    };
    guided_runtime_with(&paths.config_file, &paths, &mut apply).unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().backend.device,
        "cpu"
    );
}

#[test]
fn guided_runtime_does_not_offer_openvino_paths_after_selecting_default_audiocpp() {
    let paths = test_paths("guided-runtime-provider-switch");
    let openvino = paths.data_dir.join("openvino_genai/runtime/lib/intel64");
    fs::create_dir_all(&openvino).unwrap();
    let mut current = Config::default();
    current.backend.kind = "openvino-genai".into();
    current.backend.runtime = Runtime::Openvino;
    current.backend.device = "npu".into();
    current.backend.library = openvino.join("libopenvino_genai_c.so");
    current.backend.library_dirs = vec![openvino.clone()];
    current.model.name = crate::catalog::OPENVINO_MODEL_ID.into();
    current.save(&paths.config_file).unwrap();

    let mut prompts = ScriptedGuidedPrompts {
        runtime: Some(RuntimeSelection {
            runtime: Runtime::Default,
            device: "cpu".into(),
        }),
        confirm: true,
        ..Default::default()
    };
    guided_runtime_with(&paths.config_file, &paths, &mut prompts).unwrap();

    assert!(
        prompts.runtime_library_candidates.is_empty(),
        "a ready packaged provider must skip the runtime-directory prompt"
    );

    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.kind, "audiocpp");
    assert_eq!(saved.backend.runtime, Runtime::Default);
    assert!(saved.backend.library.as_os_str().is_empty());
    assert!(saved.backend.library_dirs.is_empty());
    assert!(!format!("{saved:?}").contains("openvino_genai"));
}

#[test]
fn guided_runtime_prompts_for_each_missing_external_provider() {
    for (runtime, device) in [
        (Runtime::Cuda, "gpu"),
        (Runtime::Vulkan, "gpu"),
        (Runtime::Hip, "gpu"),
        (Runtime::Openvino, "npu"),
    ] {
        let paths = test_paths(&format!("guided-missing-{runtime:?}"));
        let mut prompts = ScriptedGuidedPrompts {
            runtime: Some(RuntimeSelection {
                runtime,
                device: device.into(),
            }),
            runtime_probe_error: Some("selected provider is missing"),
            ..Default::default()
        };
        let error = guided_runtime_with(&paths.config_file, &paths, &mut prompts).unwrap_err();
        assert!(error.to_string().contains("selected provider is missing"));
        assert_eq!(prompts.runtime_library_candidates.len(), 1);
        let prompted = &prompts.runtime_library_candidates[0];
        assert_eq!(prompted.backend.runtime, runtime);
        assert!(prompted.backend.library_dirs.is_empty());
    }
}

#[test]
fn guided_cancellation_preserves_invalid_config_bytes() {
    let paths = test_paths("guided-invalid-config-cancel");
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    let invalid = b"# keep exactly\n[backend]\nremoved_option = \"\"\n";
    fs::write(&paths.config_file, invalid).unwrap();

    let mut runtime = ScriptedGuidedPrompts {
        runtime: Some(RuntimeSelection {
            runtime: Runtime::Default,
            device: "cpu".into(),
        }),
        confirm: false,
        ..Default::default()
    };
    guided_runtime_with(&paths.config_file, &paths, &mut runtime).unwrap();
    assert_eq!(fs::read(&paths.config_file).unwrap(), invalid);

    let mut model = ScriptedGuidedPrompts::default();
    guided_model_with(&paths.config_file, &paths, &mut model).unwrap();
    assert_eq!(fs::read(&paths.config_file).unwrap(), invalid);

    let mut full = ScriptedGuidedPrompts {
        runtime: Some(RuntimeSelection {
            runtime: Runtime::Default,
            device: "cpu".into(),
        }),
        ..Default::default()
    };
    guided_all_with_services_and_validator(
        &paths.config_file,
        &paths,
        &mut full,
        |_, _| Ok(()),
        |_, _, _, _| unreachable!(),
        |_, _| unreachable!(),
        |_| unreachable!(),
        |_| false,
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap();
    assert_eq!(fs::read(&paths.config_file).unwrap(), invalid);
}

#[test]
fn full_setup_replaces_invalid_config_only_after_transaction_success() {
    let spec = &crate::catalog::models()[0];
    let invalid = b"[backend]\nremoved_option = \"\"\n";

    let failed = test_paths("full-invalid-config-failure");
    fs::create_dir_all(failed.config_file.parent().unwrap()).unwrap();
    fs::write(&failed.config_file, invalid).unwrap();
    let validation_error = install_everything(
        spec,
        &failed.config_file,
        &failed,
        None,
        ProgressFormat::Human,
        false,
        |candidate, _| {
            assert_eq!(candidate.backend.kind, "audiocpp");
            bail!("injected full runtime validation failure")
        },
        |_, _, _, _| unreachable!(),
        |_, _| unreachable!(),
        |_| unreachable!(),
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(
        validation_error
            .to_string()
            .contains("injected full runtime validation failure")
    );
    assert_eq!(fs::read(&failed.config_file).unwrap(), invalid);

    let error = install_everything(
        spec,
        &failed.config_file,
        &failed,
        None,
        ProgressFormat::Human,
        false,
        |_, _| Ok(()),
        |_, _, _, _| Ok(failed.data_dir.join("model")),
        |_, _| Ok(()),
        |_| bail!("injected launcher failure"),
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("injected launcher failure"));
    assert_eq!(fs::read(&failed.config_file).unwrap(), invalid);

    let applied = test_paths("full-invalid-config-success");
    fs::create_dir_all(applied.config_file.parent().unwrap()).unwrap();
    fs::write(&applied.config_file, invalid).unwrap();
    install_everything(
        spec,
        &applied.config_file,
        &applied,
        None,
        ProgressFormat::Json,
        false,
        |candidate, _| {
            assert_eq!(candidate.backend.kind, "audiocpp");
            Ok(())
        },
        |_, _, _, _| Ok(applied.data_dir.join("model")),
        |_, _| Ok(()),
        |_| Ok(applied.data_dir.join("omawake-settings.desktop")),
        |_| Ok(false),
        |_, _| unreachable!(),
        |_, _| Ok(()),
    )
    .unwrap();
    let repaired = Config::load(&applied.config_file).unwrap();
    assert_eq!(repaired.model.name, spec.id);
    assert!(
        !fs::read_to_string(&applied.config_file)
            .unwrap()
            .contains("removed_option")
    );
}

#[test]
fn fresh_full_setup_proves_both_native_assets_before_its_first_config_save() {
    let spec = &crate::catalog::models()[0];
    let paths = test_paths("fresh-full-native-proof");
    let proof_ran = Cell::new(false);

    install_everything(
        spec,
        &paths.config_file,
        &paths,
        None,
        ProgressFormat::Human,
        false,
        |candidate, path| {
            assert!(!path.exists());
            assert_eq!(candidate.backend.kind, "audiocpp");
            assert_eq!(candidate.backend.runtime, Runtime::Default);
            assert_eq!(candidate.backend.device, "cpu");
            assert_eq!(candidate.model.name, crate::catalog::DEFAULT_MODEL_ID);
            Ok(())
        },
        |received_paths, received_spec, source, _| {
            assert!(source.is_none());
            let directory = app_setup::model::model_directory(received_paths, received_spec);
            fs::create_dir_all(&directory)?;
            for asset in received_spec.assets {
                fs::write(directory.join(asset.path), asset.role)?;
            }
            Ok(directory)
        },
        |candidate, received_paths| {
            assert!(!received_paths.config_file.exists());
            assert_eq!(candidate.backend.kind, "audiocpp");
            let directory = candidate.model_directory(received_paths);
            assert!(directory.join(candidate.model.verifier.as_str()).is_file());
            assert!(directory.join(candidate.model.vad.as_str()).is_file());
            proof_ran.set(true);
            Ok(())
        },
        |received_paths| {
            assert!(proof_ran.get());
            assert!(received_paths.config_file.is_file());
            Ok(received_paths.data_dir.join("omawake-settings.desktop"))
        },
        |active| {
            assert!(!active);
            Ok(false)
        },
        |_, _| Ok(()),
        |_, _| unreachable!(),
    )
    .unwrap();

    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.kind, "audiocpp");
    assert_eq!(saved.model.name, crate::catalog::DEFAULT_MODEL_ID);
    assert!(proof_ran.get());
    assert!(!app_setup::systemd::service_path(&paths).exists());

    let rollback = test_paths("fresh-full-native-proof-rollback");
    let mut original = Config::default();
    original.audio.device = "keep-me".into();
    original.save(&rollback.config_file).unwrap();
    let original_bytes = fs::read(&rollback.config_file).unwrap();
    let error = install_everything(
        spec,
        &rollback.config_file,
        &rollback,
        None,
        ProgressFormat::Human,
        false,
        |_, _| Ok(()),
        |received_paths, received_spec, _, _| {
            let directory = app_setup::model::model_directory(received_paths, received_spec);
            fs::create_dir_all(&directory)?;
            for asset in received_spec.assets {
                fs::write(directory.join(asset.path), asset.role)?;
            }
            Ok(directory)
        },
        |_, _| bail!("injected native file proof failure"),
        |_| unreachable!(),
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("native file proof failure"));
    assert_eq!(fs::read(&rollback.config_file).unwrap(), original_bytes);
}

#[test]
fn setup_file_proof_uses_silence_and_removes_the_temporary_audio_on_every_outcome() {
    let paths = test_paths("file-proof-helper");
    let config = Config::default();
    let observed = std::cell::RefCell::new(None);
    prove_setup_candidate_with(&config, &paths, |received, received_paths, audio| {
        assert_eq!(received.backend.kind, "audiocpp");
        assert_eq!(received_paths, &paths);
        let reader = hound::WavReader::open(audio)?;
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.duration(), 16_000);
        observed.replace(Some(audio.to_owned()));
        Ok(Vec::new())
    })
    .unwrap();
    assert!(!observed.borrow().as_ref().unwrap().exists());

    let detected_audio = std::cell::RefCell::new(None);
    let error = prove_setup_candidate_with(&config, &paths, |_, _, audio| {
        detected_audio.replace(Some(audio.to_owned()));
        Ok(vec![Detection {
            id: "computer".into(),
            tokens: vec!["computer".into()],
            timestamps: Vec::new(),
            start_time: 0.0,
        }])
    })
    .unwrap_err();
    assert!(error.to_string().contains("unexpectedly produced"));
    assert!(!detected_audio.borrow().as_ref().unwrap().exists());

    let failed_audio = std::cell::RefCell::new(None);
    let error = prove_setup_candidate_with(&config, &paths, |_, _, audio| {
        failed_audio.replace(Some(audio.to_owned()));
        bail!("controlled inference failure")
    })
    .unwrap_err();
    assert!(error.to_string().contains("controlled inference failure"));
    assert!(!failed_audio.borrow().as_ref().unwrap().exists());
}

#[test]
fn metadata_helpers_return_stable_shapes() {
    let paths = test_paths("metadata");
    let config = Config::default();
    let status = stopped_status(&config, &paths);
    assert_eq!(status["app"], "omawake");
    assert_eq!(status["daemon"]["state"], "stopped");
    let value = schema(&config, &paths.config_file, &paths);
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["collections"][0]["prefix"], "backend.options.");
    assert_eq!(value["collections"][1]["key"], "wake_words");
    assert!(
        value["keys"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["key"] == "backend.device_id")
    );
    assert_eq!(value["constraints"][0]["rows"].as_array().unwrap().len(), 5);
    assert_eq!(
        human_schema_lines(&value).unwrap().len(),
        value["keys"].as_array().unwrap().len() + value["collections"].as_array().unwrap().len()
    );
    assert!(model_spec(&config.model.name).is_ok());
    assert!(model_spec("missing").is_err());
    let spec = model_spec(&config.model.name).unwrap();
    verify_selected_model(spec.id, &paths, |_, found| {
        assert_eq!(found.id, spec.id);
        Ok(())
    })
    .unwrap();
    assert!(verify_selected_model("missing", &paths, |_, _| Ok(())).is_err());
    assert!(verify_selected_model(spec.id, &paths, |_, _| bail!("invalid model")).is_err());
    set_selected_model(
        spec.id,
        &paths.config_file,
        &paths,
        ProgressFormat::Human,
        |_, _| Ok(()),
    )
    .unwrap();
    assert!(
        set_selected_model(
            spec.id,
            &paths.config_file,
            &paths,
            ProgressFormat::Human,
            |_, _| bail!("invalid model"),
        )
        .is_err()
    );
    activate_model(&paths.config_file, &paths, spec, ProgressFormat::Human).unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.name,
        spec.id
    );
    print_model_ready(&paths.data_dir, spec, true, ProgressFormat::Human).unwrap();
    print_model_ready(&paths.data_dir, spec, false, ProgressFormat::Json).unwrap();
    install_selected_model(
        spec.id,
        &paths.config_file,
        &paths,
        Some(Path::new("archive.tar.bz2")),
        false,
        ProgressFormat::Human,
        |_, found, archive, progress| {
            assert_eq!(found.id, spec.id);
            assert_eq!(archive, Some(Path::new("archive.tar.bz2")));
            assert_eq!(progress, ProgressFormat::Human);
            Ok(paths.data_dir.clone())
        },
    )
    .unwrap();
    install_selected_model(
        spec.id,
        &paths.config_file,
        &paths,
        None,
        true,
        ProgressFormat::Json,
        |_, _, _, _| Ok(paths.data_dir.clone()),
    )
    .unwrap();
    assert!(
        install_selected_model(
            spec.id,
            &paths.config_file,
            &paths,
            None,
            true,
            ProgressFormat::Human,
            |_, _, _, _| bail!("install failed"),
        )
        .is_err()
    );
    for progress in [ProgressFormat::Human, ProgressFormat::Json] {
        print_setup_complete(
            &paths.data_dir,
            &paths.config_file,
            Path::new("launcher.desktop"),
            false,
            false,
            progress,
        )
        .unwrap();
        install_everything(
            spec,
            &paths.config_file,
            &paths,
            None,
            progress,
            false,
            |_, _| Ok(()),
            |_, _, _, _| Ok(paths.data_dir.join("model")),
            |_, _| Ok(()),
            |_| Ok(paths.data_dir.join("launcher.desktop")),
            |was_active| {
                assert!(!was_active);
                Ok(false)
            },
            |_, _| Ok(()),
            |_, _| Ok(()),
        )
        .unwrap();
        assert!(!app_setup::systemd::service_path(&paths).exists());
    }
    let config_before_runtime_failure = fs::read(&paths.config_file).unwrap();
    assert!(
        install_everything(
            spec,
            &paths.config_file,
            &paths,
            None,
            ProgressFormat::Human,
            false,
            |_, _| bail!("runtime failed"),
            |_, _, _, _| unreachable!(),
            |_, _| unreachable!(),
            |_| unreachable!(),
            |_| unreachable!(),
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .is_err()
    );
    assert_eq!(
        fs::read(&paths.config_file).unwrap(),
        config_before_runtime_failure
    );
    assert!(
        install_everything(
            spec,
            &paths.config_file,
            &paths,
            None,
            ProgressFormat::Human,
            false,
            |_, _| Ok(()),
            |_, _, _, _| bail!("model failed"),
            |_, _| unreachable!(),
            |_| unreachable!(),
            |_| unreachable!(),
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .is_err()
    );
    assert_eq!(socket_path(&paths), paths.runtime_dir.join("control.sock"));
    let first = request_id();
    let second = request_id();
    assert!(first.starts_with(&format!("{}-", std::process::id())));
    assert_ne!(first, second);
    let details = daemon_details(
        "fake",
        Runtime::Cuda,
        true,
        Duration::from_millis(12),
        Some(json!({"device":"test"})),
    );
    assert_eq!(details["backend"]["kind"], "fake");
    assert_eq!(details["model_load_milliseconds"], 12);
}

#[test]
fn focused_model_apply_replaces_invalid_config_only_after_cache_success() {
    let spec = &crate::catalog::models()[0];

    let failed = test_paths("model-invalid-config-failure");
    fs::create_dir_all(failed.config_file.parent().unwrap()).unwrap();
    let invalid = b"[backend]\nremoved_option = \"\"\n";
    fs::write(&failed.config_file, invalid).unwrap();
    let error = activate_model_with_cache(
        &failed.config_file,
        &failed,
        spec,
        ProgressFormat::Human,
        |_, _, _, _| bail!("injected cache failure"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("injected cache failure"));
    assert_eq!(fs::read(&failed.config_file).unwrap(), invalid);

    let applied = test_paths("model-invalid-config-success");
    fs::create_dir_all(applied.config_file.parent().unwrap()).unwrap();
    fs::write(&applied.config_file, invalid).unwrap();
    activate_model_with_cache(
        &applied.config_file,
        &applied,
        spec,
        ProgressFormat::Human,
        |_, _, _, _| Ok(None),
    )
    .unwrap();
    let repaired = Config::load(&applied.config_file).unwrap();
    assert_eq!(repaired.model.name, spec.id);
    assert!(
        !fs::read_to_string(&applied.config_file)
            .unwrap()
            .contains("removed_option")
    );
}

#[test]
fn detection_output_supports_human_and_diagnostic_json_forms() {
    let detection = || Detection {
        id: "hello".into(),
        tokens: vec!["HELLO".into()],
        timestamps: vec![0.1],
        start_time: 0.0,
    };
    print_detections(
        vec![detection()],
        Vec::new(),
        false,
        Duration::from_millis(3),
        "HELLO @hello",
        "fake",
        Runtime::Default,
        false,
    )
    .unwrap();
    print_detections(
        vec![detection()],
        vec![ActionResult {
            id: "hello".into(),
            program: "true".into(),
            arguments: Vec::new(),
            state: "started",
            pid: 1,
        }],
        true,
        Duration::from_millis(3),
        "HELLO @hello",
        "fake",
        Runtime::Cuda,
        true,
    )
    .unwrap();

    let detections = vec![detection()];
    assert!(
        collect_detection_actions(&detections, false, |_| unreachable!())
            .unwrap()
            .is_empty()
    );
    let actions = collect_detection_actions(&detections, true, |id| {
        Ok(ActionResult {
            id: id.into(),
            program: "true".into(),
            arguments: Vec::new(),
            state: "started",
            pid: 1,
        })
    })
    .unwrap();
    assert_eq!(actions[0].id, "hello");
    assert!(collect_detection_actions(&detections, true, |_| bail!("action failed")).is_err());
    present_detections(&FakeControl, vec![detection()], true, true).unwrap();
}

#[test]
fn native_json_worker_suppresses_nonactionable_success_diagnostics() {
    let paths = test_paths("native-json-success");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    let executable = paths.runtime_dir.join("native-json-success.sh");
    fs::write(
        &executable,
        "#!/bin/sh\nfor response do :; done\nprintf '[ERROR] native provider diagnostic\\n'\nprintf 'native warning\\n' >&2\nprintf '%s' '{\"ok\":true}' > \"$response\"\n",
    )
    .unwrap();

    let mut output = Vec::new();
    let mut diagnostics = Vec::new();
    run_native_json_worker_with(
        &paths.config_file,
        &paths,
        &NativeJsonRequest::Benchmark {
            audio: vec![PathBuf::from("probe.wav")],
            warmup: 1,
            iterations: 2,
        },
        shell_script(&executable),
        &mut output,
        &mut diagnostics,
    )
    .unwrap();

    let parsed: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(parsed, json!({"ok": true}));
    let output = String::from_utf8(output).unwrap();
    assert!(!output.contains("native provider diagnostic"));
    assert!(diagnostics.is_empty());
    assert_eq!(
        fs::read_dir(&paths.runtime_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".native-json-")
            })
            .count(),
        0
    );
}

#[test]
fn native_json_worker_reports_failure_without_polluting_stdout() {
    let paths = test_paths("native-json-failure");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    let executable = paths.runtime_dir.join("native-json-failure.sh");
    fs::write(
        &executable,
        "#!/bin/sh\nprintf 'provider stdout failure\\n'\nprintf 'provider stderr failure\\n' >&2\nexit 7\n",
    )
    .unwrap();

    let mut output = Vec::new();
    let mut diagnostics = Vec::new();
    let error = run_native_json_worker_with(
        &paths.config_file,
        &paths,
        &NativeJsonRequest::FileTest {
            audio: PathBuf::from("probe.wav"),
            execute: false,
        },
        shell_script(&executable),
        &mut output,
        &mut diagnostics,
    )
    .unwrap_err();
    assert!(output.is_empty());
    assert!(error.to_string().contains("worker failed"));
    assert!(error.to_string().contains('7'));
    let diagnostics = String::from_utf8(diagnostics).unwrap();
    assert!(diagnostics.contains("provider stdout failure"));
    assert!(diagnostics.contains("provider stderr failure"));
}

#[test]
fn native_json_worker_reports_a_missing_response_with_child_diagnostics() {
    let paths = test_paths("native-json-missing-response");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    let executable = paths.runtime_dir.join("native-json-missing-response.sh");
    fs::write(
        &executable,
        "#!/bin/sh\nprintf 'provider returned no response\\n'\nprintf 'response diagnostic\\n' >&2\n",
    )
    .unwrap();

    let mut output = Vec::new();
    let mut diagnostics = Vec::new();
    assert!(
        run_native_json_worker_with(
            &paths.config_file,
            &paths,
            &NativeJsonRequest::FileTest {
                audio: PathBuf::from("probe.wav"),
                execute: false,
            },
            shell_script(&executable),
            &mut output,
            &mut diagnostics,
        )
        .is_err()
    );
    assert!(output.is_empty());
    let diagnostics = String::from_utf8(diagnostics).unwrap();
    assert!(diagnostics.contains("provider returned no response"));
    assert!(diagnostics.contains("response diagnostic"));
}

#[test]
fn native_json_requests_only_wrap_inference_commands_with_json_output() {
    assert!(matches!(
        native_json_request(&TopCommand::Evaluate {
            manifest: PathBuf::from("manifest.json"),
        }),
        Some(NativeJsonRequest::Evaluate { .. })
    ));
    assert!(matches!(
        native_json_request(&TopCommand::Benchmark {
            audio: vec![PathBuf::from("probe.wav")],
            warmup: 1,
            iterations: 1,
        }),
        Some(NativeJsonRequest::Benchmark { .. })
    ));
    assert!(matches!(
        native_json_request(&TopCommand::Test {
            audio: Some(PathBuf::from("probe.wav")),
            seconds: None,
            execute: false,
            show_transcripts: false,
            json: true,
        }),
        Some(NativeJsonRequest::FileTest { .. })
    ));
    assert!(matches!(
        native_json_request(&TopCommand::Test {
            audio: None,
            seconds: Some(3),
            execute: true,
            show_transcripts: false,
            json: true,
        }),
        Some(NativeJsonRequest::LiveTest {
            seconds: 3,
            execute: true
        })
    ));
    assert!(
        native_json_request(&TopCommand::Test {
            audio: Some(PathBuf::from("probe.wav")),
            seconds: None,
            execute: false,
            show_transcripts: false,
            json: false,
        })
        .is_none()
    );
}

#[test]
fn file_benchmark_reports_warmups_iterations_percentiles_and_rtf() {
    let paths = [PathBuf::from("short.wav"), PathBuf::from("long.wav")];
    let base = Instant::now();
    let mut clock = VecDeque::from([
        base,
        base + Duration::from_millis(10),
        base + Duration::from_millis(10),
        base + Duration::from_millis(30),
        base + Duration::from_millis(30),
        base + Duration::from_millis(35),
        base + Duration::from_millis(35),
        base + Duration::from_millis(55),
    ]);
    let mut calls = Vec::new();
    let files = benchmark_files(
        &paths,
        1,
        2,
        |path| {
            Ok(if path == Path::new("short.wav") {
                Duration::from_millis(100)
            } else {
                Duration::from_millis(200)
            })
        },
        |path| {
            calls.push(path.to_owned());
            Ok(vec![Detection {
                id: path.file_stem().unwrap().to_string_lossy().into_owned(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                start_time: 0.0,
            }])
        },
        || clock.pop_front().unwrap(),
    )
    .unwrap();

    assert_eq!(calls.len(), 6);
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].audio_duration_milliseconds, 100.0);
    assert_eq!(files[0].iterations[0].iteration, 1);
    assert_eq!(files[0].iterations[0].elapsed_milliseconds, 10.0);
    assert!((files[0].iterations[0].real_time_factor.unwrap() - 0.1).abs() < f64::EPSILON);
    assert_eq!(files[0].iterations[0].detections[0].id, "short");
    assert_eq!(files[0].summary.samples, 2);
    assert_eq!(files[0].summary.p50_milliseconds, Some(10.0));
    assert_eq!(files[0].summary.p95_milliseconds, Some(20.0));
    assert_eq!(files[1].summary.p50_milliseconds, Some(5.0));
    assert_eq!(files[1].summary.p95_milliseconds, Some(20.0));
    assert!((files[1].summary.p50_real_time_factor.unwrap() - 0.025).abs() < f64::EPSILON);
    assert!((files[1].summary.p95_real_time_factor.unwrap() - 0.1).abs() < f64::EPSILON);

    let encoded = serde_json::to_value(&files).unwrap();
    assert_eq!(encoded[0]["path"], "short.wav");
    assert_eq!(encoded[1]["iterations"][1]["detections"][0]["id"], "long");

    let report = benchmark_report(&Config::default(), &FakeControl, files, 1, 2).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["benchmark"], "omawake-file-detection");
    assert_eq!(report["model_load_milliseconds"], 2.0);
    assert_eq!(report["warmup_iterations"], 1);
    assert_eq!(report["measured_iterations"], 2);
    assert_eq!(report["backend"]["kind"], "fake");
    assert_eq!(report["backend"]["requested_device"], "cpu");
    assert_eq!(report["backend"]["placement_verified"], true);
    assert!(
        report["backend"]["placement_evidence"]
            .as_str()
            .unwrap()
            .contains("CPU session")
    );
    assert!(runtime_placement(Runtime::Openvino).0);
    assert!(
        runtime_placement(Runtime::Openvino)
            .1
            .contains("fallback was disabled")
    );
    assert!(runtime_placement(Runtime::Cuda).0);
    assert!(runtime_placement(Runtime::Cuda).1.contains("CUDA sessions"));
    assert!(
        backend_placement("audiocpp", Runtime::Default)
            .1
            .contains("Silero and ASR")
    );
    assert_eq!(report["summary"]["samples"], 4);
    assert_eq!(report["summary"]["p50_milliseconds"], 10.0);
    assert_eq!(report["summary"]["p95_milliseconds"], 20.0);
}

#[test]
fn file_benchmark_validates_inputs_and_handles_zero_duration() {
    fn duration(_: &Path) -> Result<Duration> {
        Ok(Duration::from_millis(1))
    }
    fn detect(_: &Path) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
    assert!(benchmark_files(&[], 0, 1, duration, detect, Instant::now).is_err());
    assert!(
        benchmark_files(
            &[PathBuf::from("a.wav")],
            0,
            0,
            duration,
            detect,
            Instant::now
        )
        .is_err()
    );
    assert!(
        benchmark_files(
            &[PathBuf::from("a.wav")],
            0,
            1,
            |_: &Path| bail!("bad WAV"),
            detect,
            Instant::now
        )
        .is_err()
    );
    assert!(
        benchmark_files(
            &[PathBuf::from("a.wav")],
            1,
            1,
            duration,
            |_: &Path| bail!("warmup failed"),
            Instant::now
        )
        .is_err()
    );

    let base = Instant::now();
    let mut clock = [base, base + Duration::from_millis(1)].into_iter();
    let zero = benchmark_files(
        &[PathBuf::from("empty.wav")],
        0,
        1,
        |_: &Path| Ok(Duration::ZERO),
        detect,
        || clock.next().unwrap(),
    )
    .unwrap();
    assert_eq!(zero[0].iterations[0].real_time_factor, None);
    assert_eq!(zero[0].summary.p50_real_time_factor, None);
    assert_eq!(percentile(&[], 0.5), None);
    assert_eq!(percentile(&[30.0, 10.0, 20.0], 0.5), Some(20.0));
    assert_eq!(milliseconds(Duration::from_micros(1_500)), 1.5);
}

#[test]
fn live_collection_handles_samples_timeouts_and_capture_failures() {
    fn no_detections(_: i32, _: &[f32]) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
    let detection = || Detection {
        id: "hello".into(),
        tokens: Vec::new(),
        timestamps: Vec::new(),
        start_time: 0.0,
    };
    let base = Instant::now();
    let mut times = [base, base, base + Duration::from_millis(3)].into_iter();
    let found = collect_live_detections_with_clock(
        Duration::from_millis(2),
        |_| {
            Ok(AudioEvent::Samples {
                sample_rate: 16_000,
                samples: vec![0.0],
            })
        },
        |sample_rate, samples| {
            assert_eq!(sample_rate, 16_000);
            assert_eq!(samples, &[0.0]);
            Ok(vec![detection()])
        },
        || Ok(vec![detection()]),
        || times.next().unwrap(),
    )
    .unwrap();
    assert_eq!(found.len(), 2);

    assert!(
        collect_live_detections(
            Duration::from_secs(1),
            |_| Ok(AudioEvent::Error("lost microphone".into())),
            no_detections,
            || Ok(Vec::new()),
        )
        .is_err()
    );
    assert!(
        collect_live_detections(
            Duration::from_secs(1),
            |_| Err(RecvTimeoutError::Disconnected),
            no_detections,
            || Ok(Vec::new()),
        )
        .is_err()
    );
    assert!(
        collect_live_detections(
            Duration::ZERO,
            |_| Err(RecvTimeoutError::Timeout),
            no_detections,
            || bail!("finish failed"),
        )
        .is_err()
    );
    assert!(
        collect_live_detections(
            Duration::from_millis(1),
            |_| {
                Ok(AudioEvent::Samples {
                    sample_rate: 16_000,
                    samples: Vec::new(),
                })
            },
            |_, _| bail!("decode failed"),
            || Ok(Vec::new()),
        )
        .is_err()
    );

    let base = Instant::now();
    let mut times = [base, base, base + Duration::from_millis(2)].into_iter();
    assert!(
        collect_live_detections_with_clock(
            Duration::from_millis(1),
            |_| Err(RecvTimeoutError::Timeout),
            no_detections,
            || Ok(Vec::new()),
            || times.next().unwrap(),
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn daemon_event_helpers_cover_controls_audio_and_action_results() {
    let mut paused = false;
    let mut shutdown = false;
    apply_daemon_command(Command::Pause, &mut paused, &mut shutdown);
    assert!(paused);
    apply_daemon_command(Command::Resume, &mut paused, &mut shutdown);
    assert!(!paused);
    apply_daemon_command(Command::Status, &mut paused, &mut shutdown);
    assert!(!paused && !shutdown);
    apply_daemon_command(Command::Shutdown, &mut paused, &mut shutdown);
    assert!(shutdown);

    let samples = handle_audio_event(
        Ok(AudioEvent::Samples {
            sample_rate: 8_000,
            samples: vec![0.25],
        }),
        |rate, samples| {
            assert_eq!(rate, 8_000);
            assert_eq!(samples, &[0.25]);
            Ok(vec![Detection {
                id: "hello".into(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                start_time: 0.0,
            }])
        },
    )
    .unwrap();
    assert_eq!(samples.len(), 1);
    assert!(
        handle_audio_event(Err(RecvTimeoutError::Timeout), |_, _| unreachable!())
            .unwrap()
            .is_empty()
    );
    assert!(
        handle_audio_event(Ok(AudioEvent::Error("capture".into())), |_, _| Ok(
            Vec::new()
        ))
        .is_err()
    );
    assert!(
        handle_audio_event(Err(RecvTimeoutError::Disconnected), |_, _| Ok(Vec::new())).is_err()
    );

    execute_detected_actions(samples.clone(), |id| {
        Ok(ActionResult {
            id: id.into(),
            program: "true".into(),
            arguments: Vec::new(),
            state: "started",
            pid: 1,
        })
    });
    execute_detected_actions(samples, |_| bail!("action failure"));
    execute_detected_actions(Vec::new(), |_| unreachable!());
}

#[test]
fn armed_cycle_reports_control_commands_and_detections_without_hardware() {
    let mut receives = 0;
    let (detections, command) = collect_armed_detections(
        "test microphone",
        16_000,
        1,
        |audio| {
            assert_eq!(audio["device"], "test microphone");
            assert_eq!(audio["sample_rate"], 16_000);
            Ok(None)
        },
        |_| {
            receives += 1;
            if receives == 1 {
                Err(RecvTimeoutError::Timeout)
            } else {
                Ok(AudioEvent::Samples {
                    sample_rate: 16_000,
                    samples: vec![0.5],
                })
            }
        },
        |rate, samples| {
            assert_eq!(rate, 16_000);
            assert_eq!(samples, &[0.5]);
            Ok(vec![Detection {
                id: "hello".into(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                start_time: 0.0,
            }])
        },
        || false,
    )
    .unwrap();
    assert_eq!(detections.len(), 1);
    assert!(command.is_none());

    let (detections, command) = collect_armed_detections(
        "test microphone",
        8_000,
        2,
        |_| Ok(Some(Command::Pause)),
        |_| unreachable!(),
        |_, _| unreachable!(),
        || false,
    )
    .unwrap();
    assert!(detections.is_empty());
    assert!(matches!(command, Some(Command::Pause)));

    let mut polls = 0;
    let (detections, command) = collect_armed_detections(
        "test microphone",
        16_000,
        1,
        |_| {
            polls += 1;
            Ok((polls == 1).then_some(Command::Resume))
        },
        |_| {
            Ok(AudioEvent::Samples {
                sample_rate: 16_000,
                samples: vec![0.25],
            })
        },
        |_, _| {
            Ok(vec![Detection {
                id: "hello".into(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                start_time: 0.0,
            }])
        },
        || false,
    )
    .unwrap();
    assert_eq!(detections.len(), 1);
    assert!(command.is_none());

    assert!(
        collect_armed_detections(
            "test microphone",
            16_000,
            1,
            |_| bail!("control failed"),
            |_| unreachable!(),
            |_, _| unreachable!(),
            || false,
        )
        .is_err()
    );

    let (detections, command) = collect_armed_detections(
        "test microphone",
        16_000,
        1,
        |_| unreachable!(),
        |_| unreachable!(),
        |_, _| unreachable!(),
        || true,
    )
    .unwrap();
    assert!(detections.is_empty());
    assert!(matches!(command, Some(Command::Shutdown)));
}

#[test]
fn daemon_state_machine_and_loaded_shutdown_are_hardware_independent() {
    let detection = || Detection {
        id: "computer".into(),
        tokens: vec!["computer".into()],
        timestamps: Vec::new(),
        start_time: 0.0,
    };
    let armed = RefCell::new(VecDeque::from([
        (vec![detection()], Some(Command::Status)),
        (vec![detection()], Some(Command::Pause)),
        (Vec::new(), Some(Command::Shutdown)),
    ]));
    let paused = RefCell::new(VecDeque::from([
        Some(Command::Status),
        Some(Command::Resume),
    ]));
    let executed = RefCell::new(Vec::new());
    let sleeps = RefCell::new(Vec::new());
    let mut poll_paused = || Ok(paused.borrow_mut().pop_front().unwrap());
    let mut armed_cycle = || Ok(armed.borrow_mut().pop_front().unwrap());
    let mut execute = |detections: Vec<Detection>| executed.borrow_mut().push(detections.len());
    let mut sleep = |duration| sleeps.borrow_mut().push(duration);
    run_daemon_state_machine(
        &AtomicBool::new(false),
        123,
        &mut poll_paused,
        &mut armed_cycle,
        &mut execute,
        &mut sleep,
    )
    .unwrap();
    assert_eq!(&*executed.borrow(), &[1, 1, 0]);
    assert_eq!(
        sleeps
            .borrow()
            .iter()
            .filter(|duration| **duration == Duration::from_millis(50))
            .count(),
        2
    );
    assert!(sleeps.borrow().contains(&Duration::from_millis(123)));

    let called = Cell::new(false);
    let mut poll_paused = || {
        called.set(true);
        Ok(None)
    };
    let mut armed_cycle = || unreachable!();
    let mut execute = |_| unreachable!();
    let mut sleep = |_| unreachable!();
    run_daemon_state_machine(
        &AtomicBool::new(true),
        0,
        &mut poll_paused,
        &mut armed_cycle,
        &mut execute,
        &mut sleep,
    )
    .unwrap();
    assert!(!called.get());

    let paths = test_paths("loaded-daemon-shutdown");
    let mut config = Config::default();
    config.wake_words[0].command = vec!["true".into()];
    let detector = Detector::from_backend(
        &config,
        Box::new(InMemoryBackend),
        "COMPUTER @computer".into(),
        Runtime::Default,
        false,
        Duration::ZERO,
    )
    .unwrap();
    let result = run_loaded_daemon(&detector, &config, &paths, Arc::new(AtomicBool::new(true)));
    if result.as_ref().is_err_and(sandbox_denied) {
        return;
    }
    result.unwrap();
    assert!(!socket_path(&paths).exists());
}

#[test]
fn socket_binding_replaces_stale_socket_and_rejects_live_daemon() {
    let paths = test_paths("bind");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    fs::write(socket_path(&paths), "stale").unwrap();
    assert!(bind_socket(&paths).is_err());
    assert_eq!(fs::read_to_string(socket_path(&paths)).unwrap(), "stale");
    fs::remove_file(socket_path(&paths)).unwrap();
    let Some(stale) = io_or_skip(UnixListener::bind(socket_path(&paths))) else {
        return;
    };
    drop(stale);
    let Some(listener) = anyhow_or_skip(bind_socket(&paths)) else {
        return;
    };
    assert_eq!(
        fs::metadata(&paths.runtime_dir)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(socket_path(&paths))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(bind_socket(&paths).is_err());
    drop(listener);
}

#[test]
fn socket_preparation_is_testable_without_creating_a_unix_socket() {
    let paths = test_paths("bind-memory");
    let Some(listener) = anyhow_or_skip(bind_socket_with(
        &paths,
        |_| false,
        |path| UnixListener::bind(path),
        |listener| listener.set_nonblocking(true),
    )) else {
        return;
    };
    assert_eq!(
        fs::metadata(&paths.runtime_dir)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(socket_path(&paths))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );

    assert!(
        bind_socket_with(
            &paths,
            |_| true,
            |_| unreachable!(),
            |_: &()| unreachable!(),
        )
        .is_err()
    );
    drop(listener);

    let failing_paths = test_paths("bind-configure-failure");
    assert!(
        bind_socket_with(
            &failing_paths,
            |_| false,
            |path| UnixListener::bind(path),
            |_| Err(std::io::Error::other("nonblocking failed")),
        )
        .is_err()
    );
    assert!(!socket_path(&failing_paths).exists());
}

#[test]
fn daemon_socket_cleanup_runs_after_success_and_failure() {
    for (name, serve_result) in [
        ("success", Ok(())),
        ("failure", Err(anyhow::anyhow!("serve failed"))),
    ] {
        let paths = test_paths(name);
        fs::create_dir_all(&paths.runtime_dir).unwrap();
        let Some(listener) = io_or_skip(UnixListener::bind(socket_path(&paths))) else {
            return;
        };
        let metadata = fs::symlink_metadata(socket_path(&paths)).unwrap();
        drop(listener);
        let result = finish_daemon(&paths, Some(&metadata), serve_result);
        assert!(!socket_path(&paths).exists());
        assert_eq!(result.is_err(), name == "failure");
    }

    let absent = test_paths("already-absent");
    finish_daemon(&absent, None, Ok(())).unwrap();

    let cleanup_failure = test_paths("cleanup-failure");
    fs::create_dir_all(socket_path(&cleanup_failure)).unwrap();
    let metadata = fs::symlink_metadata(socket_path(&cleanup_failure)).unwrap();
    assert!(finish_daemon(&cleanup_failure, Some(&metadata), Ok(())).is_err());

    let original_error = test_paths("cleanup-and-serve-failure");
    fs::create_dir_all(socket_path(&original_error)).unwrap();
    let metadata = fs::symlink_metadata(socket_path(&original_error)).unwrap();
    assert_eq!(
        finish_daemon(
            &original_error,
            Some(&metadata),
            Err(anyhow::anyhow!("original failure")),
        )
        .unwrap_err()
        .to_string(),
        "original failure"
    );
}

#[test]
fn request_framing_accepts_valid_and_rejects_invalid_messages() {
    let Some((mut sender, receiver)) = io_or_skip(UnixStream::pair()) else {
        return;
    };
    if io_or_skip(sender.write_all(&encoded_request(1, "valid", Command::Pause))).is_none() {
        return;
    }
    let request = read_request(&receiver).unwrap();
    assert_eq!(request.id, "valid");
    assert!(matches!(request.command, Command::Pause));

    let (mut sender, receiver) = UnixStream::pair().unwrap();
    sender.write_all(b"not-json\n").unwrap();
    assert!(read_request(&receiver).is_err());

    let (mut sender, receiver) = UnixStream::pair().unwrap();
    let writer = thread::spawn(move || sender.write_all(&vec![b'x'; 65_537]).unwrap());
    assert!(read_request(&receiver).is_err());
    writer.join().unwrap();
}

#[test]
fn control_socket_handles_status_commands_and_bad_clients() {
    let paths = test_paths("control");
    let Some(listener) = anyhow_or_skip(bind_socket(&paths)) else {
        return;
    };
    let details = json!({"backend":{"kind":"fake"}});

    let mut invalid = send_raw(&socket_path(&paths), b"bad\n".to_vec());
    let mut mismatch = send_raw(
        &socket_path(&paths),
        encoded_request(2, "old", Command::Status),
    );
    let mut status = send_raw(
        &socket_path(&paths),
        encoded_request(1, "status", Command::Status),
    );
    assert!(
        poll_control(&FakeControl, "armed", None, || accept_control(&listener))
            .unwrap()
            .is_none()
    );
    for (stream, code) in [
        (&mut invalid, "invalid_request"),
        (&mut mismatch, "protocol_mismatch"),
    ] {
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        assert!(line.contains(code));
    }
    let mut line = String::new();
    BufReader::new(&mut status).read_line(&mut line).unwrap();
    let response: Response = serde_json::from_str(&line).unwrap();
    assert!(matches!(response.result, ResultPayload::State { ref state, .. } if state == "armed"));

    for (command, expected_state) in [
        (Command::Pause, "paused"),
        (Command::Resume, "armed"),
        (Command::Shutdown, "stopping"),
    ] {
        let mut client = send_raw(
            &socket_path(&paths),
            encoded_request(1, expected_state, command),
        );
        let returned = poll_control_connections(|| accept_control(&listener), "current", &details)
            .unwrap()
            .unwrap();
        assert!(matches!(
            (returned, expected_state),
            (Command::Pause, "paused")
                | (Command::Resume, "armed")
                | (Command::Shutdown, "stopping")
        ));
        let mut line = String::new();
        BufReader::new(&mut client).read_line(&mut line).unwrap();
        assert!(line.contains(expected_state));
    }
}

#[test]
fn request_round_trips_responses_and_reports_malformed_server_data() {
    for valid in [true, false] {
        let paths = test_paths(if valid {
            "request-valid"
        } else {
            "request-invalid"
        });
        fs::create_dir_all(&paths.runtime_dir).unwrap();
        let Some(listener) = io_or_skip(UnixListener::bind(socket_path(&paths))) else {
            return;
        };
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&stream).unwrap();
            assert!(matches!(request.command, Command::Status));
            if valid {
                write_response(
                    &mut stream,
                    &Response {
                        protocol: 1,
                        id: request.id,
                        result: ResultPayload::State {
                            state: "armed".into(),
                            details: json!({}),
                        },
                    },
                );
            } else {
                stream.write_all(b"invalid\n").unwrap();
            }
        });
        let result = request(&paths, Command::Status);
        if valid {
            assert!(
                matches!(result.unwrap().result, ResultPayload::State { state, .. } if state == "armed")
            );
        } else {
            assert!(result.is_err());
        }
        server.join().unwrap();
    }
    assert!(request(&test_paths("not-running"), Command::Status).is_err());
}

#[test]
fn response_output_handles_state_json_and_daemon_errors() {
    let state = || Response {
        protocol: 1,
        id: "id".into(),
        result: ResultPayload::State {
            state: "armed".into(),
            details: json!({}),
        },
    };
    print_response(state(), false).unwrap();
    print_response(state(), true).unwrap();
    assert!(print_response(Response::error("id", "broken", "failure"), false).is_err());

    let Some((mut left, right)) = io_or_skip(UnixStream::pair()) else {
        return;
    };
    drop(right);
    write_response(&mut left, &state());
}

#[test]
fn top_level_dispatch_runs_file_only_commands_with_injected_paths() {
    let paths = test_paths("dispatch");
    Config::default().save(&paths.config_file).unwrap();
    let invoke = |command| {
        run_with_paths(
            Cli {
                config: Some(paths.config_file.clone()),
                command,
            },
            paths.clone(),
        )
    };

    invoke(TopCommand::Status { json: false }).unwrap();
    invoke(TopCommand::Status { json: true }).unwrap();
    invoke(TopCommand::Config {
        command: ConfigCommand::Get {
            key: Some("backend.kind".into()),
            json: false,
        },
    })
    .unwrap();
    invoke(TopCommand::Config {
        command: ConfigCommand::Get {
            key: None,
            json: true,
        },
    })
    .unwrap();
    assert!(
        invoke(TopCommand::Config {
            command: ConfigCommand::Get {
                key: Some("missing.key".into()),
                json: false,
            },
        })
        .is_err()
    );
    for json in [false, true] {
        invoke(TopCommand::Config {
            command: ConfigCommand::Schema { json },
        })
        .unwrap();
        invoke(TopCommand::WakeWord {
            command: WakeWordCommand::List { json },
        })
        .unwrap();
    }
    invoke(TopCommand::Config {
        command: ConfigCommand::Set {
            key: "backend.threads".into(),
            value: "2".into(),
        },
    })
    .unwrap();
    invoke(TopCommand::Config {
        command: ConfigCommand::Unset {
            key: "backend.threads".into(),
        },
    })
    .unwrap();
    assert!(
        invoke(TopCommand::Test {
            audio: None,
            seconds: None,
            execute: false,
            show_transcripts: false,
            json: false,
        })
        .is_err()
    );
    assert!(
        invoke(TopCommand::Test {
            audio: Some(paths.data_dir.join("missing.wav")),
            seconds: None,
            execute: false,
            show_transcripts: false,
            json: false,
        })
        .is_err()
    );
    assert!(
        invoke(TopCommand::Test {
            audio: None,
            seconds: Some(0),
            execute: false,
            show_transcripts: false,
            json: false,
        })
        .is_err()
    );
    assert!(
        invoke(TopCommand::Benchmark {
            audio: vec![paths.data_dir.join("missing.wav")],
            warmup: 0,
            iterations: 1,
        })
        .is_err()
    );
}

#[test]
fn top_level_control_dispatch_handles_successful_injected_daemon_responses() {
    let paths = test_paths("control-dispatch");
    let selected_config = paths.config_file.with_file_name("selected.toml");
    Config::default().save(&selected_config).unwrap();
    for (command, expected_command, state) in [
        (TopCommand::Status { json: false }, Command::Status, "armed"),
        (TopCommand::Status { json: true }, Command::Status, "armed"),
        (TopCommand::Pause, Command::Pause, "paused"),
        (TopCommand::Resume, Command::Resume, "armed"),
        (TopCommand::Stop, Command::Shutdown, "stopping"),
    ] {
        run_with_paths_and_request(
            Cli {
                config: Some(selected_config.clone()),
                command,
            },
            paths.clone(),
            |received_paths, received_command| {
                assert_eq!(received_paths.config_file, selected_config);
                assert_eq!(received_paths.socket(), paths.socket());
                assert_eq!(
                    std::mem::discriminant(&received_command),
                    std::mem::discriminant(&expected_command)
                );
                Ok(Response {
                    protocol: 1,
                    id: "in-memory".into(),
                    result: ResultPayload::State {
                        state: state.into(),
                        details: json!({}),
                    },
                })
            },
        )
        .unwrap();
    }
}

#[test]
fn top_level_audio_device_dispatch_uses_injected_enumerator() {
    let paths = test_paths("audio-device-dispatch");
    Config::default().save(&paths.config_file).unwrap();
    for json in [false, true] {
        run_with_paths_and_services(
            Cli {
                config: Some(paths.config_file.clone()),
                command: TopCommand::AudioDevices { json },
            },
            paths.clone(),
            |_, _| unreachable!(),
            || Ok(vec!["Microphone One".into(), "Microphone Two".into()]),
        )
        .unwrap();
    }
    assert!(
        run_with_paths_and_services(
            Cli {
                config: Some(paths.config_file.clone()),
                command: TopCommand::AudioDevices { json: false },
            },
            paths,
            |_, _| unreachable!(),
            || bail!("enumeration failed"),
        )
        .is_err()
    );
}

#[test]
fn setup_dispatch_covers_checks_catalog_and_safe_failure_paths() {
    let paths = test_paths("setup-dispatch");
    let config = &paths.config_file;

    for json in [false, true] {
        assert!(setup(Some(SetupCommand::Check { json }), config, &paths).is_err());
        setup(
            Some(SetupCommand::Runtime {
                json,
                runtime: None,
                device: None,
                device_id: None,
                dir: None,
                apply: false,
            }),
            config,
            &paths,
        )
        .unwrap();
        setup(
            Some(SetupCommand::Model {
                list: true,
                json,
                download: None,
                set: None,
                verify: None,
                source_dir: None,
                no_activate: false,
                progress_format: ProgressFormat::Human,
            }),
            config,
            &paths,
        )
        .unwrap();
    }

    let model = Config::default().model.name;
    let model_command = |download, set, verify, archive| SetupCommand::Model {
        list: false,
        json: false,
        download,
        set,
        verify,
        source_dir: archive,
        no_activate: false,
        progress_format: ProgressFormat::Human,
    };
    assert!(
        setup(
            Some(model_command(None, None, Some(model.clone()), None)),
            config,
            &paths,
        )
        .is_err()
    );
    assert!(
        setup(
            Some(model_command(None, Some(model.clone()), None, None)),
            config,
            &paths,
        )
        .is_err()
    );
    assert!(
        setup(
            Some(model_command(
                Some(model.clone()),
                None,
                None,
                Some(paths.data_dir.join("missing.tar.bz2")),
            )),
            config,
            &paths,
        )
        .is_err()
    );
    assert!(
        setup(
            Some(SetupCommand::All {
                model,
                source_dir: Some(paths.data_dir.join("missing-model-assets")),
                progress_format: ProgressFormat::Json,
            }),
            config,
            &paths,
        )
        .is_err()
    );

    setup(
        Some(SetupCommand::Menu {
            uninstall: false,
            status: false,
        }),
        config,
        &paths,
    )
    .unwrap();
    setup(
        Some(SetupCommand::Menu {
            uninstall: false,
            status: true,
        }),
        config,
        &paths,
    )
    .unwrap();
    setup(
        Some(SetupCommand::Menu {
            uninstall: true,
            status: false,
        }),
        config,
        &paths,
    )
    .unwrap();
}

#[test]
fn noninteractive_setup_covers_safe_runtime_model_and_service_decisions() {
    let paths = test_paths("setup-safe-decisions");
    // With no terminal and no subcommand, setup runs its read-only check path.
    assert!(setup(None, &paths.config_file, &paths).is_err());
    Config::default().save(&paths.config_file).unwrap();

    let empty_runtime = paths.data_dir.join("empty-runtime");
    fs::create_dir_all(&empty_runtime).unwrap();
    let runtime_error = setup(
        Some(SetupCommand::Runtime {
            json: false,
            runtime: Some("default".into()),
            device: Some("cpu".into()),
            device_id: None,
            dir: Some(empty_runtime),
            apply: false,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap_err();
    assert!(
        runtime_error.to_string().contains("libaudiocpp"),
        "unexpected runtime setup error: {runtime_error:#}"
    );

    for command in [
        SetupCommand::Model {
            list: false,
            json: false,
            download: None,
            set: None,
            verify: Some("unknown".into()),
            source_dir: None,
            no_activate: false,
            progress_format: ProgressFormat::Human,
        },
        SetupCommand::Model {
            list: false,
            json: false,
            download: None,
            set: Some("unknown".into()),
            verify: None,
            source_dir: None,
            no_activate: false,
            progress_format: ProgressFormat::Human,
        },
    ] {
        assert!(setup(Some(command), &paths.config_file, &paths).is_err());
    }
    setup(
        Some(SetupCommand::Model {
            list: false,
            json: false,
            download: None,
            set: None,
            verify: None,
            source_dir: None,
            no_activate: false,
            progress_format: ProgressFormat::Human,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    assert!(
        setup(
            Some(SetupCommand::Systemd {
                uninstall: false,
                status: true,
                no_start: false,
            }),
            &paths.config_file,
            &paths,
        )
        .is_err()
    );

    let discovered = |_: &Config, _: &Path| crate::runtime_inventory::Probe {
        loadable: true,
        device_accessible: None,
        ready: false,
        ..Default::default()
    };
    configure_runtime_from_flags_with(
        &paths.config_file,
        &paths,
        Some("openvino".into()),
        None,
        None,
        None,
        false,
        discovered,
        |_, _, _, _| Ok(None),
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().backend.runtime,
        Runtime::Default
    );
    configure_runtime_from_flags_with(
        &paths.config_file,
        &paths,
        Some("default".into()),
        Some("cpu".into()),
        None,
        None,
        true,
        discovered,
        |_, _, _, _| Ok(None),
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().backend.device,
        "cpu"
    );
}

#[test]
fn default_runtime_probe_and_os_socket_wrappers_report_real_failures_cleanly() {
    let paths = test_paths("default-probe-and-sockets");
    let mut config = Config::default();
    config.backend.library = paths.data_dir.join("missing-audiocpp.so");
    let mut prompts = DefaultProbePrompts;
    assert!(prompts.probe_runtime(&config, &paths.config_file).is_err());
    assert!(validate_runtime_candidate(&config, &paths.config_file).is_err());

    let listener = match bind_socket(&paths) {
        Ok(listener) => listener,
        Err(error) if sandbox_denied(&error) => return,
        Err(error) => panic!("{error:#}"),
    };
    assert!(accept_control(&listener).unwrap().is_none());
    let metadata = fs::symlink_metadata(socket_path(&paths)).unwrap();
    drop(listener);
    finish_daemon(&paths, Some(&metadata), Ok(())).unwrap();
    assert!(!socket_path(&paths).exists());
    assert!(remove_socket_if_unchanged(&socket_path(&paths), &metadata).is_ok());
    assert!(connect_control_socket(&socket_path(&paths)).is_err());

    fs::write(socket_path(&paths), b"ordinary file").unwrap();
    let ordinary = fs::symlink_metadata(socket_path(&paths)).unwrap();
    assert!(remove_socket_if_unchanged(&socket_path(&paths), &ordinary).is_err());
}

#[test]
fn native_runtime_probe_reports_real_safe_audio_cpp_and_openvino_evidence() {
    let paths = test_paths("native-runtime-evidence");
    let mut audio_cpp = Config::default();
    audio_cpp.backend.library = crate::engine::audiocpp::tests::fake_library().to_path_buf();
    let probe = native_runtime_probe(&audio_cpp, &paths.config_file);
    assert!(probe.loadable);
    assert!(!probe.ready);
    assert_eq!(probe.device_accessible, None);
    assert!(probe.evidence.available_devices.is_empty());
    assert!(!probe.evidence.model_inference_verified);
    assert_eq!(probe.evidence.selected_device.as_deref(), Some("cpu"));
    assert_eq!(
        probe.evidence.provider_path.as_deref(),
        Some(crate::engine::audiocpp::tests::fake_library())
    );
    assert!(probe.evidence.versions[0].contains("fake-provider-1"));

    let mut openvino = Config::default();
    openvino.backend.kind = "openvino-genai".into();
    openvino.backend.runtime = Runtime::Openvino;
    openvino.backend.device = "npu".into();
    let probe = native_runtime_probe_with(
        &openvino,
        &paths.config_file,
        |_, _| {
            Ok(crate::engine::openvino_genai::RuntimeEvidence {
                runtime_build: "2026.3.fake".into(),
                runtime_description: "safe fake runtime".into(),
                requested_device: "NPU".into(),
                available_device: "NPU".into(),
                full_device_name: "Safe Fake NPU".into(),
                device_architecture: "fake".into(),
                driver_version: "fake".into(),
                genai_library: "/fake/libopenvino_genai_c.so".into(),
                core_library: "/fake/libopenvino_c.so".into(),
                audiocpp_library: "/fake/libaudiocpp.so".into(),
            })
        },
        |_, _| unreachable!(),
    );
    assert!(probe.loadable, "{:?}", probe.errors);
    assert!(!probe.ready);
    assert_eq!(probe.device_accessible, Some(true));
    assert!(!probe.evidence.model_inference_verified);
    assert_eq!(probe.evidence.selected_device.as_deref(), Some("npu"));
    assert_eq!(
        probe.evidence.provider_path.as_deref(),
        Some(Path::new("/fake/libopenvino_genai_c.so"))
    );
    assert!(probe.evidence.versions[0].contains("OpenVINO"));

    let failed = native_runtime_probe_with(
        &openvino,
        &paths.config_file,
        |_, _| bail!("controlled OpenVINO probe failure"),
        |_, _| unreachable!(),
    );
    assert!(!failed.ready);
    assert!(failed.errors[0].contains("controlled OpenVINO"));
}

#[test]
fn runtime_directory_and_config_snapshot_cover_default_cuda_and_cleanup_edges() {
    let paths = test_paths("runtime-and-snapshot-edges");
    let runtime = paths.data_dir.join("runtime");
    fs::create_dir_all(&runtime).unwrap();
    fs::write(runtime.join("libaudiocpp.so.0.1.0"), b"provider").unwrap();

    let mut config = Config::default();
    configure_runtime_directory(&mut config, &runtime).unwrap();
    assert!(config.backend.library.ends_with("libaudiocpp.so.0.1.0"));
    assert!(configure_runtime_directory(&mut config, Path::new("relative")).is_err());

    let openvino = paths.data_dir.join("openvino/runtime/lib/intel64/Release");
    fs::create_dir_all(&openvino).unwrap();
    for library in [
        "libopenvino_genai_c.so",
        "libopenvino_c.so",
        "libopenvino_intel_cpu_plugin.so",
        "libopenvino_intel_gpu_plugin.so",
        "libopenvino_intel_npu_plugin.so",
        "libopenvino_intel_npu_compiler_loader.so",
        "libopenvino_intel_npu_compiler.so",
    ] {
        fs::write(openvino.join(library), b"provider").unwrap();
    }
    for device in ["cpu", "gpu", "npu"] {
        let mut candidate = Config::default();
        candidate.backend.kind = "openvino-genai".into();
        candidate.backend.runtime = Runtime::Openvino;
        candidate.backend.device = device.into();
        configure_runtime_directory(&mut candidate, &paths.data_dir.join("openvino")).unwrap();
        assert!(
            candidate
                .backend
                .library
                .ends_with("libopenvino_genai_c.so")
        );
        assert!(candidate.backend.library_dirs.contains(&openvino));
    }
    fs::remove_file(openvino.join("libopenvino_intel_npu_compiler.so")).unwrap();
    let mut incomplete_npu = Config::default();
    incomplete_npu.backend.kind = "openvino-genai".into();
    incomplete_npu.backend.runtime = Runtime::Openvino;
    incomplete_npu.backend.device = "npu".into();
    assert!(
        configure_runtime_directory(&mut incomplete_npu, &paths.data_dir.join("openvino")).is_err()
    );

    let snapshot = paths.data_dir.join("snapshot.toml");
    let temporary = snapshot.with_extension("toml.tmp");
    fs::write(&snapshot, b"current").unwrap();
    fs::write(&temporary, b"partial").unwrap();
    restore_config_snapshot(&snapshot, None).unwrap();
    assert!(!snapshot.exists() && !temporary.exists());

    let directory = paths.data_dir.join("directory-as-config");
    fs::create_dir_all(&directory).unwrap();
    assert!(config_snapshot(&directory).is_err());
}

#[test]
fn request_reader_is_testable_without_a_unix_socket() {
    let request = read_request(std::io::Cursor::new(encoded_request(
        1,
        "cursor",
        Command::Resume,
    )))
    .unwrap();
    assert_eq!(request.id, "cursor");
    assert!(matches!(request.command, Command::Resume));
    assert!(read_request(std::io::Cursor::new(b"invalid\n")).is_err());
    assert!(read_request(std::io::Cursor::new(vec![b'x'; 65_537])).is_err());
}

#[test]
fn control_protocol_maps_every_request_to_a_response_and_transition() {
    let details = json!({"backend": {"kind": "fake"}});
    let request = |protocol: u32, id: &str, command: Command| {
        Ok(Request {
            protocol,
            id: id.into(),
            command,
        })
    };

    for (command, expected_state, transition) in [
        (Command::Status, "current", false),
        (Command::Pause, "paused", true),
        (Command::Resume, "armed", true),
        (Command::Shutdown, "stopping", true),
    ] {
        let (response, next) =
            control_response(request(1, expected_state, command), "current", &details);
        assert_eq!(response.id, expected_state);
        assert_eq!(next.is_some(), transition);
        assert!(matches!(
            response.result,
            ResultPayload::State { ref state, ref details }
                if state == expected_state && details["backend"]["kind"] == "fake"
        ));
    }

    let (response, next) =
        control_response(request(9, "old", Command::Status), "current", &details);
    assert!(next.is_none());
    assert!(
        matches!(response.result, ResultPayload::Error { ref code, .. } if code == "protocol_mismatch")
    );

    let (response, next) =
        control_response(Err(anyhow::anyhow!("broken JSON")), "current", &details);
    assert!(next.is_none());
    assert!(
        matches!(response.result, ResultPayload::Error { ref code, .. } if code == "invalid_request")
    );
}

#[test]
fn control_stream_processes_messages_entirely_in_memory() {
    let details = json!({"backend": {"kind": "fake"}});
    for (bytes, expected_code, expected_command) in [
        (b"not-json\n".to_vec(), Some("invalid_request"), None),
        (
            encoded_request(2, "old", Command::Status),
            Some("protocol_mismatch"),
            None,
        ),
        (encoded_request(1, "status", Command::Status), None, None),
        (
            encoded_request(1, "pause", Command::Pause),
            None,
            Some(Command::Pause),
        ),
    ] {
        let mut stream = ScriptedStream::responding_with(bytes);
        let command = handle_control_stream(&mut stream, "armed", &details);
        assert_eq!(command.is_some(), expected_command.is_some());
        let response: Response = serde_json::from_slice(&stream.request).unwrap();
        match expected_code {
            Some(expected) => assert!(
                matches!(response.result, ResultPayload::Error { ref code, .. } if code == expected)
            ),
            None => assert!(matches!(response.result, ResultPayload::State { .. })),
        }
    }

    let mut disconnected =
        ScriptedStream::responding_with(encoded_request(1, "disconnected", Command::Status));
    disconnected.fail_writes = true;
    assert!(handle_control_stream(&mut disconnected, "armed", &details).is_none());
}

#[test]
fn control_polling_skips_non_commands_and_stops_at_transition_in_memory() {
    let details = json!({"backend": {"kind": "fake"}});
    let mut clients = VecDeque::from([
        ScriptedStream::responding_with(b"invalid\n".to_vec()),
        ScriptedStream::responding_with(encoded_request(1, "status", Command::Status)),
        ScriptedStream::responding_with(encoded_request(1, "stop", Command::Shutdown)),
    ]);
    let command = poll_control_connections(|| Ok(clients.pop_front()), "armed", &details).unwrap();
    assert!(matches!(command, Some(Command::Shutdown)));
    assert!(clients.is_empty());

    assert!(
        poll_control_connections(
            || Ok::<Option<ScriptedStream>, anyhow::Error>(None),
            "armed",
            &details,
        )
        .unwrap()
        .is_none()
    );
    assert!(
        poll_control_connections(
            || Err::<Option<ScriptedStream>, _>(anyhow::anyhow!("accept failed")),
            "armed",
            &details,
        )
        .is_err()
    );

    let command = poll_control(
        &FakeControl,
        "armed",
        Some(json!({"device": "test microphone"})),
        || {
            Ok(Some(ScriptedStream::responding_with(encoded_request(
                1,
                "pause",
                Command::Pause,
            ))))
        },
    )
    .unwrap();
    assert!(matches!(command, Some(Command::Pause)));

    assert!(
        prepare_accepted_control(Ok(()), |_| Ok(()))
            .unwrap()
            .is_some()
    );
    assert!(
        prepare_accepted_control::<(), _>(
            Err(std::io::Error::from(std::io::ErrorKind::WouldBlock)),
            |_| unreachable!(),
        )
        .unwrap()
        .is_none()
    );
    assert!(
        prepare_accepted_control::<(), _>(Err(std::io::Error::other("accept failed")), |_| {
            unreachable!()
        })
        .is_err()
    );
    assert!(
        prepare_accepted_control(Ok(()), |_| Err(std::io::Error::other("configure failed")))
            .is_err()
    );
}

#[test]
fn real_detector_forwards_control_metadata_without_native_backend() {
    let mut config = Config::default();
    config.wake_words[0].command = vec!["true".into()];
    let detector = Detector::from_backend(
        &config,
        Box::new(InMemoryBackend),
        "COMPUTER @computer".into(),
        Runtime::Cuda,
        true,
        Duration::from_millis(17),
    )
    .unwrap();

    assert_eq!(DetectorControl::backend_kind(&detector), "in-memory");
    assert_eq!(DetectorControl::effective_runtime(&detector), Runtime::Cuda);
    assert!(DetectorControl::fallback_used(&detector));
    assert_eq!(
        DetectorControl::load_time(&detector),
        Duration::from_millis(17)
    );
    assert_eq!(
        DetectorControl::keywords_buffer(&detector),
        "COMPUTER @computer"
    );
    assert_eq!(
        DetectorControl::run(&detector, "computer").unwrap().state,
        "started"
    );

    let mut streams = VecDeque::from([ScriptedStream::responding_with(encoded_request(
        1,
        "pause",
        Command::Pause,
    ))]);
    assert!(matches!(
        poll_control(&detector, "armed", None, || Ok(streams.pop_front())).unwrap(),
        Some(Command::Pause)
    ));
}

#[test]
fn client_request_exchange_is_testable_without_a_socket() {
    let response = Response {
        protocol: 1,
        id: "server-id".into(),
        result: ResultPayload::State {
            state: "armed".into(),
            details: json!({}),
        },
    };
    let mut encoded_response = serde_json::to_vec(&response).unwrap();
    encoded_response.push(b'\n');
    let mut stream = ScriptedStream::responding_with(encoded_response);
    let received = request_over_stream(&mut stream, Command::Status).unwrap();
    assert!(matches!(
        received.result,
        ResultPayload::State { ref state, .. } if state == "armed"
    ));
    let sent: Request = serde_json::from_slice(&stream.request).unwrap();
    assert_eq!(sent.protocol, 1);
    assert!(matches!(sent.command, Command::Status));

    let mut malformed = ScriptedStream::responding_with(b"invalid\n".to_vec());
    assert!(request_over_stream(&mut malformed, Command::Pause).is_err());

    let mut disconnected = ScriptedStream::responding_with(Vec::new());
    disconnected.fail_writes = true;
    assert!(request_over_stream(&mut disconnected, Command::Resume).is_err());

    let paths = test_paths("request-connector");
    let mut encoded_response = serde_json::to_vec(&response).unwrap();
    encoded_response.push(b'\n');
    let received = request_with_connector(&paths, Command::Status, |path| {
        assert_eq!(path, socket_path(&paths));
        Ok(ScriptedStream::responding_with(encoded_response))
    })
    .unwrap();
    assert!(matches!(received.result, ResultPayload::State { .. }));
    assert!(
        request_with_connector::<ScriptedStream, _>(&paths, Command::Status, |_| {
            Err(std::io::Error::other("connect failed"))
        })
        .is_err()
    );
}

#[test]
fn refused_client_connection_removes_only_the_same_stale_socket() {
    let paths = test_paths("stale-client");
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    let path = socket_path(&paths);
    let Some(stale) = io_or_skip(std::os::unix::net::UnixDatagram::bind(&path)) else {
        return;
    };
    let error = connect_control_socket_with::<UnixStream>(&path, |_| {
        Err(std::io::Error::from(std::io::ErrorKind::ConnectionRefused))
    })
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::ConnectionRefused);
    assert!(!path.exists());
    drop(stale);

    let untouched_path = paths.runtime_dir.join("other-error.sock");
    let untouched = std::os::unix::net::UnixDatagram::bind(&untouched_path).unwrap();
    assert!(
        connect_control_socket_with::<UnixStream>(&untouched_path, |_| {
            Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
        })
        .is_err()
    );
    assert!(untouched_path.exists());
    drop(untouched);

    let replacement_paths = test_paths("replaced-client");
    fs::create_dir_all(&replacement_paths.runtime_dir).unwrap();
    let replacement_path = socket_path(&replacement_paths);
    let Some(original) = io_or_skip(UnixListener::bind(&replacement_path)) else {
        return;
    };
    let replacement_source = replacement_paths.runtime_dir.join("replacement.sock");
    let replacement = UnixListener::bind(&replacement_source).unwrap();
    let error = connect_control_socket_with::<UnixStream>(&replacement_path, |_| {
        fs::remove_file(&replacement_path).unwrap();
        fs::rename(&replacement_source, &replacement_path).unwrap();
        Err(std::io::Error::from(std::io::ErrorKind::ConnectionRefused))
    })
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::ConnectionRefused);
    assert!(replacement_path.exists());
    drop(original);
    drop(replacement);

    let regular_file = paths.runtime_dir.join("regular-file");
    fs::write(&regular_file, "keep").unwrap();
    let metadata = fs::metadata(&regular_file).unwrap();
    remove_stale_socket_if_unchanged(&regular_file, Some(&metadata));
    assert!(regular_file.exists());

    let symlink_target = paths.runtime_dir.join("symlink-target.sock");
    let symlink_socket = std::os::unix::net::UnixDatagram::bind(&symlink_target).unwrap();
    let symlink_path = paths.runtime_dir.join("symlink.sock");
    std::os::unix::fs::symlink(&symlink_target, &symlink_path).unwrap();
    let error = connect_control_socket_with::<UnixStream>(&symlink_path, |_| {
        Err(std::io::Error::from(std::io::ErrorKind::ConnectionRefused))
    })
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::ConnectionRefused);
    assert!(symlink_path.exists());
    drop(symlink_socket);
}

#[test]
fn stale_socket_connection_errors_cover_kernel_variants() {
    for kind in [
        std::io::ErrorKind::ConnectionRefused,
        std::io::ErrorKind::ConnectionReset,
        std::io::ErrorKind::ConnectionAborted,
    ] {
        assert!(indicates_stale_socket(kind));
    }
    assert!(!indicates_stale_socket(
        std::io::ErrorKind::PermissionDenied
    ));
}
