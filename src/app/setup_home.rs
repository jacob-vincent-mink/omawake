use super::*;
use std::io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HomeAction {
    Guided,
    Words,
    Teach,
    Audio,
    Runtime,
    Model,
    Test,
    Service,
    Advanced,
    Check,
    Exit,
}

const HOME_ACTIONS: [HomeAction; 11] = [
    HomeAction::Guided,
    HomeAction::Words,
    HomeAction::Teach,
    HomeAction::Audio,
    HomeAction::Runtime,
    HomeAction::Model,
    HomeAction::Test,
    HomeAction::Service,
    HomeAction::Advanced,
    HomeAction::Check,
    HomeAction::Exit,
];

struct Snapshot {
    config: Option<Config>,
    invalid_config: bool,
    installed: bool,
    managed: bool,
    active: bool,
}

impl Snapshot {
    fn read(path: &Path, paths: &AppPaths) -> Self {
        let installed = app_setup::systemd::service_path(paths).is_file();
        let config = path.is_file().then(|| Config::load(path));
        Self {
            invalid_config: config.as_ref().is_some_and(Result::is_err),
            config: config.and_then(Result::ok),
            installed,
            managed: installed && app_setup::systemd::is_managed(paths, path),
            active: app_setup::systemd::is_active(),
        }
    }

    fn summary(&self) -> String {
        let Some(config) = &self.config else {
            if self.invalid_config {
                return "The existing configuration could not be loaded. Run guided setup to repair it; the current file is replaced only after setup succeeds.".into();
            }
            return "First run: start with Guided setup. Runtime, model, microphone, and wake words can be changed later. The service is optional and requires a separate choice.".into();
        };
        let words = if has_example_word(config) {
            "example 'Computer' — review its phrase and action".to_owned()
        } else {
            format!("{} configured", config.wake_words.len())
        };
        format!(
            "Runtime: {} / {}   Model: {}   Mic: {}\nWake words: {}   Service: {}\nChoose an area to configure it. Run Checks to verify the full setup.",
            runtime_name(config.backend.runtime),
            config.backend.device,
            config.model.name,
            config.audio.device,
            words,
            if self.installed && !self.managed {
                "unmanaged user unit"
            } else if self.active && !self.installed {
                "running outside setup"
            } else if self.active {
                "running"
            } else if self.installed {
                "installed, stopped"
            } else {
                "not installed"
            },
        )
    }

    fn items(&self) -> Vec<MenuItem> {
        HOME_ACTIONS
            .iter()
            .map(|action| match action {
                HomeAction::Guided => MenuItem::available(
                    if self.config.is_some() { "Run guided setup again" } else { "Start guided setup" },
                    "Choose a runtime, model, and microphone; verify assets and install the desktop launcher.",
                ),
                HomeAction::Words => available_if_config(
                    self,
                    if self.config.as_ref().is_some_and(has_example_word) { "Review example wake word & action" } else { "Wake words & actions" },
                    "Add a phrase, edit its action or aliases, enable/disable it, or remove it.",
                ),
                HomeAction::Teach => available_if_config(
                    self,
                    "Teach a wake word",
                    "Record examples and review alternate spellings. Actions are not run while teaching.",
                ),
                HomeAction::Audio => available_if_config(
                    self,
                    "Microphone",
                    "Select and test an input device, then apply it.",
                ),
                HomeAction::Runtime => MenuItem::available(
                    "Runtime & device",
                    "Inspect available providers, choose a CPU/GPU/NPU device, and verify it before saving.",
                ),
                HomeAction::Model => MenuItem::available(
                    "Model",
                    "Browse compatible models; activate an installed model or install verified assets.",
                ),
                HomeAction::Test => available_if_config(
                    self,
                    "Try recognition (5 seconds)",
                    "Listen for wake words without executing their actions.",
                ),
                HomeAction::Service => MenuItem::available(
                    "Background service",
                    "Install/start at login, stop, restart, or uninstall a setup-managed user service.",
                ),
                HomeAction::Advanced => available_if_config(
                    self,
                    "Advanced settings",
                    "Edit CPU threads, cooldown, and capture queue capacity.",
                ),
                HomeAction::Check => MenuItem::available(
                    "Run setup checks",
                    "Verify configuration, model, runtime, microphone, launcher, and service.",
                ),
                HomeAction::Exit => MenuItem::available("Exit setup", "Return to the shell."),
            })
            .collect()
    }
}

fn has_example_word(config: &Config) -> bool {
    let words = &config.wake_words;
    words.len() == 1
        && words[0].id == "computer"
        && words[0].phrase == "Computer"
        && words[0].enabled
        && words[0].aliases.is_empty()
        && words[0].command == ["notify-send", "Wake word heard"]
}

fn available_if_config(snapshot: &Snapshot, label: &str, detail: &str) -> MenuItem {
    if snapshot.config.is_some() {
        MenuItem::available(label, detail)
    } else {
        MenuItem::unavailable(
            label,
            format!(
                "{} {detail}",
                if snapshot.invalid_config {
                    "Repair the configuration with guided setup first."
                } else {
                    "Run guided setup first."
                }
            ),
        )
    }
}

pub(super) fn run(config_path: &Path, paths: &AppPaths) -> Result<()> {
    let mut preferred = 0;
    loop {
        let snapshot = Snapshot::read(config_path, paths);
        let items = snapshot.items();
        let Some(index) = wizard::select("Omawake setup", &snapshot.summary(), &items, preferred)?
        else {
            return Ok(());
        };
        preferred = index;
        let action = HOME_ACTIONS[index];
        if action == HomeAction::Exit {
            return Ok(());
        }
        if let Err(error) = perform(action, config_path, paths) {
            notice("Setup action failed", &format!("{error:#}"))?;
        }
    }
}

fn perform(action: HomeAction, config_path: &Path, paths: &AppPaths) -> Result<()> {
    match action {
        HomeAction::Guided => with_daemon_paused(paths, || {
            guided_all_with(
                config_path,
                paths,
                &mut TerminalGuidedPrompts {
                    config_path: config_path.to_owned(),
                },
            )
        }),
        HomeAction::Words => return words(config_path, paths),
        HomeAction::Teach => with_daemon_paused(paths, || {
            onboarding::guided(Config::load(config_path)?, config_path, paths)
        }),
        HomeAction::Audio => with_daemon_paused(paths, || {
            setup_audio(config_path, paths, None, false, false)
        }),
        HomeAction::Runtime => guided_runtime(config_path, paths),
        HomeAction::Model => guided_model(config_path, paths),
        HomeAction::Test => with_daemon_paused(paths, || {
            let config = Config::load(config_path)?;
            println!("Listening for five seconds. Wake-word actions will not run.");
            let detector = Detector::load(&config, paths)?;
            let detections = detect_live(&detector, &config, Duration::from_secs(5))?;
            present_detections(&detector, detections, false, false)
        }),
        HomeAction::Service => return service(config_path, paths),
        HomeAction::Advanced => return advanced(config_path, paths),
        HomeAction::Check => app_setup::print_checks(config_path, paths, false),
        HomeAction::Exit => return Ok(()),
    }?;
    pause()?;
    Ok(())
}

fn with_daemon_paused<T>(paths: &AppPaths, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let hold = match connect_control_socket(&socket_path(paths)) {
        Ok(mut stream) => {
            stream.set_read_timeout(Some(Duration::from_secs(10)))?;
            stream.set_write_timeout(Some(Duration::from_secs(10)))?;
            match request_over_stream(&mut stream, Command::HoldPause)
                .context("wait for the daemon to release its microphone")?
                .result
            {
                ResultPayload::State { state, .. } if state == "paused" => Some(stream),
                ResultPayload::Error { code, message } => {
                    bail!("could not pause the running daemon: {code}: {message}")
                }
                _ => bail!("the running daemon did not confirm its microphone was released"),
            }
        }
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::ConnectionRefused
                    | io::ErrorKind::ConnectionReset
            ) =>
        {
            ensure!(
                !app_setup::systemd::active_state()?,
                "an Omawake systemd service is active but its control socket is unavailable; stop it before recording or testing"
            );
            None
        }
        Err(error) => {
            return Err(error).context("connect to the Omawake daemon before audio setup");
        }
    };
    let result = work();
    drop(hold);
    result
}

fn notice(title: &str, body: &str) -> Result<()> {
    wizard::select(
        title,
        body,
        &[MenuItem::available("Back", "Return to setup")],
        0,
    )?;
    Ok(())
}

fn pause() -> Result<()> {
    print!("\nPress Enter to return to setup...");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServiceAction {
    Install,
    Start,
    Stop,
    Restart,
    Status,
    Uninstall,
    Back,
}

fn service_items(
    installed: bool,
    managed: bool,
    active: bool,
    configured: bool,
) -> Vec<(ServiceAction, MenuItem)> {
    let mut items = Vec::new();
    if installed && !managed || active && !installed {
        items.push((
            ServiceAction::Install,
            MenuItem::unavailable(
                "Service is managed outside setup",
                "An existing Omawake user unit is active or was not installed by setup. Use systemctl to manage it.",
            ),
        ));
        items.push((
            ServiceAction::Status,
            MenuItem::available(
                "Show service status",
                "Show the detected service state and local unit path.",
            ),
        ));
        items.push((
            ServiceAction::Back,
            MenuItem::available("Back", "Return to setup."),
        ));
        return items;
    }
    if !installed {
        items.push((
            ServiceAction::Install,
            if configured {
                MenuItem::available(
                    "Install and start at login",
                    "Verify the model and runtime, enable the user service, then start it now.",
                )
            } else {
                MenuItem::unavailable("Install and start at login", "Complete guided setup first.")
            },
        ));
    } else {
        if !active {
            items.push((
                ServiceAction::Start,
                MenuItem::available(
                    "Start service",
                    "Start listening now; login startup remains enabled.",
                ),
            ));
        } else {
            items.push((
                ServiceAction::Stop,
                MenuItem::available(
                    "Stop service",
                    "Stop listening now; it can start at the next login.",
                ),
            ));
            items.push((
                ServiceAction::Restart,
                MenuItem::available(
                    "Restart service",
                    "Reload the current configuration and confirm the daemon remains active.",
                ),
            ));
        }
        items.push((
            ServiceAction::Status,
            MenuItem::available(
                "Show service status",
                "Show whether the service is running and where its unit is installed.",
            ),
        ));
        items.push((
            ServiceAction::Uninstall,
            MenuItem::available(
                "Uninstall service",
                "Disable login startup, stop listening, and remove the app-owned unit.",
            ),
        ));
    }
    items.push((
        ServiceAction::Back,
        MenuItem::available("Back", "Return to setup."),
    ));
    items
}

fn service(config_path: &Path, paths: &AppPaths) -> Result<()> {
    loop {
        let installed = app_setup::systemd::service_path(paths).is_file();
        let managed = installed && app_setup::systemd::is_managed(paths, config_path);
        let active = app_setup::systemd::is_active();
        let entries = service_items(
            installed,
            managed,
            active,
            config_path.is_file() && Config::load(config_path).is_ok(),
        );
        let items: Vec<_> = entries.iter().map(|(_, item)| item.clone()).collect();
        let state = if installed && !managed || active && !installed {
            "managed outside setup"
        } else if active {
            "running"
        } else if installed {
            "installed, stopped"
        } else {
            "not installed"
        };
        let Some(index) = wizard::select(
            "Background service",
            &format!("Current state: {state}. This is an optional systemd user service."),
            &items,
            0,
        )?
        else {
            return Ok(());
        };
        let action = entries[index].0;
        if action == ServiceAction::Back {
            return Ok(());
        }
        let mut changed = true;
        let result: Result<()> = (|| match action {
            ServiceAction::Install => {
                ensure!(
                    !app_setup::systemd::is_active(),
                    "an Omawake service is already running outside setup; stop it before installing a setup-managed unit"
                );
                ensure!(
                    config_path.is_file(),
                    "finish guided setup before installing the service"
                );
                let config = Config::load(config_path)
                    .context("finish guided setup before installing the service")?;
                prove_setup_candidate(&config, paths)
                    .context("verify model and runtime before starting the service")?;
                app_setup::systemd::install(paths, config_path, true).map(|_| ())
            }
            ServiceAction::Start => {
                ensure_managed_service(paths, config_path)?;
                app_setup::systemd::start(paths)
            }
            ServiceAction::Stop => {
                ensure_managed_service(paths, config_path)?;
                app_setup::systemd::stop(paths)
            }
            ServiceAction::Restart => {
                ensure_managed_service(paths, config_path)?;
                app_setup::systemd::restart()
            }
            ServiceAction::Status => {
                changed = false;
                notice(
                    "Service status",
                    &format!(
                        "{} · {}",
                        if active { "Running" } else { "Stopped" },
                        if managed {
                            format!(
                                "Setup-managed unit: {}",
                                app_setup::systemd::service_path(paths).display()
                            )
                        } else if installed {
                            format!(
                                "Unmanaged local unit: {}",
                                app_setup::systemd::service_path(paths).display()
                            )
                        } else {
                            "No setup-managed local unit".into()
                        }
                    ),
                )
            }
            ServiceAction::Uninstall => {
                let confirmed = wizard::select(
                    "Uninstall service?",
                    "The service will stop and no longer start at login. Your model and configuration remain.",
                    &[
                        MenuItem::available("Keep service", "Return without changing anything."),
                        MenuItem::available("Uninstall", "Disable, stop, and remove the unit."),
                    ],
                    0,
                )? == Some(1);
                if confirmed {
                    ensure_managed_service(paths, config_path)?;
                    app_setup::systemd::uninstall(paths)
                } else {
                    changed = false;
                    Ok(())
                }
            }
            ServiceAction::Back => Ok(()),
        })();
        match result {
            Ok(()) => {
                if changed {
                    notice(
                        "Service",
                        &format!(
                            "{} completed. Current state: {}.",
                            service_action_name(action),
                            if app_setup::systemd::is_active() {
                                "running"
                            } else {
                                "stopped"
                            }
                        ),
                    )?;
                }
            }
            Err(error) => notice("Service action failed", &format!("{error:#}"))?,
        }
    }
}

fn ensure_managed_service(paths: &AppPaths, config_path: &Path) -> Result<()> {
    ensure!(
        app_setup::systemd::is_managed(paths, config_path),
        "refusing to change an Omawake unit not managed by this setup"
    );
    Ok(())
}

fn service_action_name(action: ServiceAction) -> &'static str {
    match action {
        ServiceAction::Install => "Install and start",
        ServiceAction::Start => "Start",
        ServiceAction::Stop => "Stop",
        ServiceAction::Restart => "Restart",
        ServiceAction::Status => "Status",
        ServiceAction::Uninstall => "Uninstall",
        ServiceAction::Back => "Back",
    }
}

fn advanced(config_path: &Path, paths: &AppPaths) -> Result<()> {
    loop {
        let config = Config::load(config_path)?;
        let items = [
            MenuItem::available(
                "CPU threads",
                format!("Current: {}", config.backend.threads),
            ),
            MenuItem::available(
                "Cooldown",
                format!(
                    "Current: {} ms after an action",
                    config.daemon.cooldown_milliseconds
                ),
            ),
            MenuItem::available(
                "Capture queue",
                format!("Current: {} audio chunks", config.daemon.queue_capacity),
            ),
            MenuItem::available("Back", "Return to setup."),
        ];
        let Some(index) = wizard::select(
            "Advanced settings",
            "Changes to an active app-owned service restart it automatically.",
            &items,
            0,
        )?
        else {
            return Ok(());
        };
        let (key, label, current) = match index {
            0 => (
                "backend.threads",
                "CPU threads (1–64)",
                config.backend.threads.to_string(),
            ),
            1 => (
                "daemon.cooldown_milliseconds",
                "Cooldown in milliseconds",
                config.daemon.cooldown_milliseconds.to_string(),
            ),
            2 => (
                "daemon.queue_capacity",
                "Capture queue (at least 1)",
                config.daemon.queue_capacity.to_string(),
            ),
            _ => return Ok(()),
        };
        let Some(value) = input(label, Some(&current), true)? else {
            continue;
        };
        let result =
            if key == "daemon.queue_capacity" && value.parse::<usize>().is_ok_and(|n| n == 0) {
                Err(anyhow::anyhow!("capture queue must be at least 1"))
            } else {
                config_mutation(
                    ConfigCommand::Set {
                        key: key.into(),
                        value,
                    },
                    config,
                    config_path,
                    paths,
                )
            };
        match result {
            Ok(()) => notice(
                "Setting saved",
                "The setting is saved. An active app-owned service, if present, was restarted.",
            )?,
            Err(error) => notice("Setting not saved", &format!("{error:#}"))?,
        }
    }
}

fn words(config_path: &Path, paths: &AppPaths) -> Result<()> {
    loop {
        let config = Config::load(config_path)?;
        let mut items: Vec<_> = config
            .wake_words
            .iter()
            .map(|word| {
                MenuItem::available(
                    format!("{}  {}", if word.enabled { "●" } else { "○" }, word.phrase),
                    format!(
                        "ID: {} · action: {} · {} alias(es)",
                        word.id,
                        word.command.join(" "),
                        word.aliases.len()
                    ),
                )
            })
            .collect();
        items.push(MenuItem::available(
            "Add wake word",
            "Enter a phrase and build an action from a program and arguments.",
        ));
        items.push(MenuItem::available("Back", "Return to setup."));
        let Some(index) = wizard::select(
            "Wake words & actions",
            "● enabled · ○ disabled. Selecting a word lets you edit it. Saving restarts an active app-owned service.",
            &items,
            0,
        )?
        else {
            return Ok(());
        };
        if index == config.wake_words.len() + 1 {
            return Ok(());
        }
        if index == config.wake_words.len() {
            add_word(config_path, paths)?;
        } else {
            edit_word(config_path, paths, &config.wake_words[index].id)?;
        }
    }
}

fn add_word(config_path: &Path, paths: &AppPaths) -> Result<()> {
    let Some(id) = input("Word ID (lowercase letters, numbers, hyphens)", None, false)? else {
        return Ok(());
    };
    let Some(phrase) = input("Wake phrase", None, false)? else {
        return Ok(());
    };
    let Some(program) = input("Action program (for example, notify-send)", None, false)? else {
        return Ok(());
    };
    let mut command = vec![program];
    loop {
        let Some(argument) = input("Next action argument (blank finishes)", None, true)? else {
            break;
        };
        command.push(argument);
    }
    let mut config = Config::load(config_path)?;
    config.wake_words.push(WakeWord {
        engine: None,
        enrollment: None,
        id,
        phrase,
        aliases: Vec::new(),
        enabled: true,
        command,
    });
    save_words(config, config_path, paths)?;
    notice(
        "Wake word saved",
        "The new phrase and action are configured. Use Teach a wake word to record examples if needed.",
    )
}

fn edit_word(config_path: &Path, paths: &AppPaths, id: &str) -> Result<()> {
    loop {
        let config = Config::load(config_path)?;
        let word = config
            .wake_words
            .iter()
            .find(|word| word.id == id)
            .with_context(|| format!("wake word {id} was removed"))?;
        let items = edit_word_items(word);
        let Some(index) = wizard::select(
            &format!("Wake word: {}", word.phrase),
            "Edit one part at a time. Changes are validated before saving.",
            &items,
            0,
        )?
        else {
            return Ok(());
        };
        if index == 5 {
            return Ok(());
        }
        if index == 1 {
            edit_action(config_path, paths, id)?;
            continue;
        }
        if index == 2 {
            aliases(config_path, paths, id)?;
            continue;
        }
        if index == 4 {
            let confirmed = wizard::select(
                "Remove wake word?",
                &format!("Remove {id} and its action? This cannot be undone."),
                &[
                    MenuItem::available("Keep word", "Return without changing it."),
                    MenuItem::available("Remove", "Delete this wake word."),
                ],
                0,
            )? == Some(1);
            if confirmed {
                let mut config = config;
                remove_wake_word(&mut config, id)?;
                save_words(config, config_path, paths)?;
                notice("Wake word removed", "The phrase and action were removed.")?;
                return Ok(());
            }
            continue;
        }
        let mut config = config;
        let word = config
            .wake_words
            .iter_mut()
            .find(|word| word.id == id)
            .context("wake word missing")?;
        if index == 0 {
            let Some(phrase) = input("Wake phrase", Some(&word.phrase), false)? else {
                continue;
            };
            word.phrase = phrase;
        } else {
            word.enabled = !word.enabled;
        }
        save_words(config, config_path, paths)?;
        notice("Wake word saved", "The change is active.")?;
    }
}

fn edit_word_items(word: &WakeWord) -> [MenuItem; 6] {
    let trained = word.uses_trained_head();
    [
        if trained {
            MenuItem::unavailable(
                "Phrase locked (trained)",
                "Add a new wake word with the desired phrase, teach it, then remove this trained word.",
            )
        } else {
            MenuItem::available("Edit phrase", format!("Current: {}", word.phrase))
        },
        MenuItem::available(
            "Edit action",
            format!("Current: {}", word.command.join(" ")),
        ),
        if trained {
            MenuItem::unavailable(
                "Aliases",
                "A trained detector does not use transcript aliases. Use Teach a wake word to change recognition.",
            )
        } else {
            MenuItem::available(
                "Aliases",
                format!("{} alternate spelling(s)", word.aliases.len()),
            )
        },
        MenuItem::available(
            if word.enabled { "Disable" } else { "Enable" },
            "Toggle recognition without deleting this word.",
        ),
        MenuItem::available(
            "Remove word",
            "Delete this phrase and its action after confirmation.",
        ),
        MenuItem::available("Back", "Return to all wake words."),
    ]
}

fn edit_action(config_path: &Path, paths: &AppPaths, id: &str) -> Result<()> {
    let mut config = Config::load(config_path)?;
    let word = config
        .wake_words
        .iter_mut()
        .find(|word| word.id == id)
        .context("wake word missing")?;
    loop {
        let mut items = vec![MenuItem::available(
            "Program",
            word.command.first().cloned().unwrap_or_default(),
        )];
        for (index, argument) in word.command.iter().enumerate().skip(1) {
            items.push(MenuItem::available(format!("Argument {index}"), argument));
        }
        items.push(MenuItem::available(
            "Add argument",
            "Append one argument to the action.",
        ));
        items.push(MenuItem::available(
            "Save action",
            "Validate and apply all changes.",
        ));
        items.push(MenuItem::available("Cancel", "Discard all action edits."));
        let Some(index) = wizard::select(
            "Edit action",
            "Each argument is passed directly to the program; no shell is used. Select an argument to edit or remove it.",
            &items,
            0,
        )?
        else {
            return Ok(());
        };
        let add = word.command.len();
        if index == add {
            if let Some(argument) = input("New argument", None, false)? {
                word.command.push(argument);
            }
        } else if index == add + 1 {
            save_words(config, config_path, paths)?;
            return notice(
                "Action saved",
                "The action is configured. It was not executed while editing.",
            );
        } else if index == add + 2 {
            return Ok(());
        } else if index == 0 {
            if let Some(program) = input(
                "Action program",
                word.command.first().map(String::as_str),
                false,
            )? {
                word.command[0] = program;
            }
        } else {
            let selected = wizard::select(
                "Edit argument",
                &format!("Current: {}", word.command[index]),
                &[
                    MenuItem::available("Change", "Enter a replacement value."),
                    MenuItem::available("Remove", "Delete this argument."),
                    MenuItem::available("Back", "Leave it unchanged."),
                ],
                0,
            )?;
            match selected {
                Some(0) => {
                    if let Some(argument) = input("Argument", Some(&word.command[index]), false)? {
                        word.command[index] = argument;
                    }
                }
                Some(1) => {
                    word.command.remove(index);
                }
                _ => {}
            }
        }
    }
}

fn aliases(config_path: &Path, paths: &AppPaths, id: &str) -> Result<()> {
    loop {
        let mut config = Config::load(config_path)?;
        let word = config
            .wake_words
            .iter()
            .find(|word| word.id == id)
            .context("wake word missing")?;
        let mut items: Vec<_> = word
            .aliases
            .iter()
            .map(|alias| {
                MenuItem::available(alias, "Select to remove this exact alternate transcript.")
            })
            .collect();
        let count = items.len();
        items.push(MenuItem::available(
            "Add alias",
            "Accept an exact alternate transcript spelling.",
        ));
        items.push(MenuItem::available("Back", "Return to the wake word."));
        let Some(index) = wizard::select(
            "Alternate spellings",
            "Aliases are exact recognized transcripts, not extra wake phrases.",
            &items,
            0,
        )?
        else {
            return Ok(());
        };
        if index == count + 1 {
            return Ok(());
        }
        if index == count {
            let Some(alias) = input("Exact alternate transcript", None, false)? else {
                continue;
            };
            config
                .wake_words
                .iter_mut()
                .find(|word| word.id == id)
                .context("wake word missing")?
                .aliases
                .push(alias);
        } else {
            let alias = config
                .wake_words
                .iter()
                .find(|word| word.id == id)
                .context("wake word missing")?
                .aliases[index]
                .clone();
            let confirmed = wizard::select(
                "Remove alias?",
                &format!("Stop accepting {alias:?} for {id}?"),
                &[
                    MenuItem::available("Keep", "Leave this alias configured."),
                    MenuItem::available("Remove", "Delete only this alias."),
                ],
                0,
            )? == Some(1);
            if !confirmed {
                continue;
            }
            config
                .wake_words
                .iter_mut()
                .find(|word| word.id == id)
                .context("wake word missing")?
                .aliases
                .remove(index);
        }
        save_words(config, config_path, paths)?;
        notice("Aliases saved", "The accepted spellings were updated.")?;
    }
}

fn save_words(config: Config, path: &Path, paths: &AppPaths) -> Result<()> {
    validate_wake_words(&config.wake_words)?;
    save_and_reload_active(config, path, paths)?;
    Ok(())
}

fn input(label: &str, current: Option<&str>, blank_allowed: bool) -> Result<Option<String>> {
    match current {
        Some(value) => print!("{label} [current: {value}; blank keeps it]: "),
        None if blank_allowed => print!("{label}: "),
        None => print!("{label} [blank cancels]: "),
    }
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().lock().take(4097).read_line(&mut line)?;
    ensure!(line.len() <= 4096, "input is too long");
    let value = line.trim().to_owned();
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/setup_home.rs"]
mod tests;
