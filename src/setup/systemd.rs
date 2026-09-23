use std::fs;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};

use crate::paths::AppPaths;
const UNIT: &str = "omawake.service";
const MANAGED_MARKER: &str = "# Managed by Omawake setup\n";

pub fn service_path(paths: &AppPaths) -> PathBuf {
    paths
        .config_file
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join("systemd/user/omawake.service")
}

/// Decide whether the active user unit owns a configuration file. A local
/// unit is authoritative; without one, packaged units use the XDG default.
pub fn targets_config(config: &Path) -> bool {
    let defaults = AppPaths::discover();
    match fs::symlink_metadata(service_path(&defaults)) {
        Ok(_) => read_managed_unit(&defaults, Some(config)).is_ok(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            config == defaults.config_file
        }
        Err(_) => false,
    }
}

/// Check the unit systemd actually loaded before restarting it after an edit.
/// A base file alone is insufficient because a drop-in can replace ExecStart.
pub fn effective_targets_config(config: &Path) -> bool {
    let defaults = AppPaths::discover();
    let Some(fragment) = loaded_fragment_without_overrides() else {
        return false;
    };
    let local = service_path(&defaults);
    match fs::symlink_metadata(&local) {
        Ok(_) => fragment == local && read_managed_unit(&defaults, Some(config)).is_ok(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            config == defaults.config_file
                && fs::read_to_string(fragment).is_ok_and(|body| {
                    body == include_str!("../../packaging/systemd/omawake.service")
                })
        }
        Err(_) => false,
    }
}

/// A loaded, unmodified setup unit can prove that an active service belongs
/// to a different config, allowing this config to be edited independently.
pub fn loaded_managed_unit_targets_another_config(config: &Path) -> bool {
    let defaults = AppPaths::discover();
    let local = service_path(&defaults);
    local.is_file()
        && loaded_fragment_without_overrides().as_deref() == Some(local.as_path())
        && read_managed_unit(&defaults, None).is_ok()
        && read_managed_unit(&defaults, Some(config)).is_err()
}

fn loaded_fragment_without_overrides() -> Option<PathBuf> {
    let Ok(fragment) = unit_property("FragmentPath") else {
        return None;
    };
    let Ok(drop_ins) = unit_property("DropInPaths") else {
        return None;
    };
    let Ok(needs_reload) = unit_property("NeedDaemonReload") else {
        return None;
    };
    if !drop_ins.is_empty() || needs_reload != "no" {
        return None;
    }
    let fragment = PathBuf::from(fragment);
    if !fragment.is_absolute() {
        return None;
    }
    Some(fragment)
}

fn unit_property(property: &str) -> Result<String> {
    let output = Command::new("systemctl")
        .args(["--user", "show", UNIT, "--property", property, "--value"])
        .output()
        .context("inspect active Omawake user service")?;
    if !output.status.success() {
        bail!("systemctl could not inspect the active Omawake user service");
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(test)]
fn targets_config_with_unit(config: &Path, default_config: &Path, unit: Option<&str>) -> bool {
    unit.map_or(config == default_config, |unit| {
        has_managed_template(unit, Some(config))
    })
}

pub fn generate(binary: &Path, config: &Path) -> String {
    format!(
        "{MANAGED_MARKER}[Unit]\nDescription=Omawake local wake-word daemon\nPartOf=graphical-session.target\nAfter=graphical-session.target pipewire.service\n\n[Service]\nType=simple\nExecStart={} --config {} daemon\nRestart=on-failure\nRestartSec=1\nEnvironment=XDG_RUNTIME_DIR=%t\n\n[Install]\nWantedBy=graphical-session.target\n",
        quote(binary),
        quote(config)
    )
}

/// Whether the local unit has Omawake's generated shape and uses this config.
/// Legacy units without the managed marker are also recognized.
pub fn is_managed(paths: &AppPaths, config: &Path) -> bool {
    read_managed_unit(paths, Some(config)).is_ok()
}

fn read_managed_unit(paths: &AppPaths, config: Option<&Path>) -> Result<String> {
    let path = service_path(paths);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            bail!(
                "Omawake user service is not installed at {}",
                path.display()
            )
        }
        Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
    };
    if !metadata.is_file() {
        bail!(
            "Omawake user service unit is not a file: {}",
            path.display()
        );
    }
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("read Omawake user service unit at {}", path.display()))?;
    if !has_managed_template(&contents, config) {
        bail!(
            "refusing to manage an unrecognized Omawake user service unit at {}",
            path.display()
        );
    }
    Ok(contents)
}

fn has_managed_template(contents: &str, config: Option<&Path>) -> bool {
    let Some(command) = contents
        .lines()
        .find_map(|line| line.strip_prefix("ExecStart="))
    else {
        return false;
    };
    let Some((binary, rest)) = command.split_once(" --config ") else {
        return false;
    };
    let Some(config_arg) = rest.strip_suffix(" daemon") else {
        return false;
    };
    if !valid_quoted_argument(binary)
        || !valid_quoted_argument(config_arg)
        || config.is_some_and(|config| config_arg != quote(config))
    {
        return false;
    }

    let actual_command = format!("ExecStart={command}");
    let template_command =
        "ExecStart=\"/__omawake_binary__\" --config \"/__omawake_config__\" daemon";
    let canonical = contents.replacen(&actual_command, template_command, 1);
    let template = generate(
        Path::new("/__omawake_binary__"),
        Path::new("/__omawake_config__"),
    );
    canonical == template || canonical == template[MANAGED_MARKER.len()..]
}

fn valid_quoted_argument(argument: &str) -> bool {
    let Some(inner) = argument
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    else {
        return false;
    };
    let mut chars = inner.chars();
    while let Some(character) = chars.next() {
        match character {
            '\\' => {
                if !matches!(chars.next(), Some('\\' | '"' | 'n' | 'r' | 't')) {
                    return false;
                }
            }
            '"' | '\n' | '\r' | '\t' => return false,
            '%' if chars.next() != Some('%') => return false,
            _ => {}
        }
    }
    true
}

pub fn install(paths: &AppPaths, config: &Path, start: bool) -> Result<PathBuf> {
    let path = service_path(paths);
    let binary = std::env::current_exe()?.canonicalize()?;
    let unit = generate(&binary, config);
    let previous = match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_file() => {
            Some(fs::read(&path).context("snapshot existing Omawake service unit")?)
        }
        Ok(_) => bail!(
            "Omawake user service unit is not a regular file: {}",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("snapshot existing Omawake service unit"),
    };
    if previous.is_some() {
        read_managed_unit(paths, None)?;
    }
    let was_active = is_active();
    if !was_active {
        ensure_no_direct_daemon(paths)?;
    }
    let was_enabled = is_enabled();
    let mut enable_attempted = false;
    let mut restart_attempted = false;
    let result = (|| {
        write_atomic(&path, unit.as_bytes())?;
        systemctl(["daemon-reload"])?;
        if !effective_targets_config(config) {
            bail!(
                "systemd did not load the setup-managed unit, or a unit override changes its effective command"
            );
        }
        enable_attempted = true;
        systemctl(["enable", UNIT])?;
        if start {
            restart_attempted = true;
            restart()?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        let restored = (|| {
            if restart_attempted {
                systemctl(["stop", UNIT]).context("stop service before restoring its unit")?;
                ensure!(
                    !active_state()?,
                    "service is still active; leave its unit installed for recovery"
                );
            }
            match previous {
                Some(bytes) => write_atomic(&path, &bytes),
                None => match fs::remove_file(&path) {
                    Ok(()) => Ok(()),
                    Err(remove) if remove.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    Err(remove) => Err(remove.into()),
                },
            }?;
            systemctl(["daemon-reload"])?;
            if enable_attempted {
                systemctl([if was_enabled { "enable" } else { "disable" }, UNIT])?;
            }
            if was_active && restart_attempted {
                restart()?;
            }
            Ok(())
        })();
        return match restored {
            Ok(()) => Err(error),
            Err(restore) => Err(error.context(format!(
                "service installation also failed to restore the previous unit: {restore:#}"
            ))),
        };
    }
    Ok(path)
}

pub fn uninstall(paths: &AppPaths) -> Result<()> {
    read_managed_unit(paths, None)?;
    let local = service_path(paths);
    if loaded_fragment_without_overrides().as_deref() != Some(local.as_path()) {
        bail!("refusing to uninstall an Omawake unit that systemd has overridden or not loaded");
    }
    uninstall_with(
        paths,
        || systemctl(["disable", "--now", UNIT]),
        active_state,
        || systemctl(["daemon-reload"]),
    )
}

fn uninstall_with(
    paths: &AppPaths,
    disable: impl FnOnce() -> Result<()>,
    active_after: impl FnOnce() -> Result<bool>,
    reload: impl FnOnce() -> Result<()>,
) -> Result<()> {
    read_managed_unit(paths, None)?;
    disable()?;
    if active_after()? {
        bail!("{UNIT} remains active after disable --now");
    }
    let path = service_path(paths);
    fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
    reload()
}

pub fn status(paths: &AppPaths) -> Result<()> {
    let path = service_path(paths);
    println!("unit: {}", path.display());
    if !path.exists() {
        bail!("omawake systemd user service is not installed");
    }
    let status = Command::new("systemctl")
        .args(["--user", "status", UNIT, "--no-pager"])
        .status()
        .context("run systemctl --user status")?;
    if !status.success() {
        bail!("omawake.service is not running");
    }
    Ok(())
}

pub fn is_active() -> bool {
    Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", UNIT])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn is_enabled() -> bool {
    Command::new("systemctl")
        .args(["--user", "is-enabled", "--quiet", UNIT])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn active_state() -> Result<bool> {
    let output = Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", UNIT])
        .stdout(Stdio::null())
        .env("LC_ALL", "C")
        .output();
    let output = match output {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).context("run systemctl --user is-active"),
    };
    if String::from_utf8_lossy(&output.stderr).contains("Failed to connect to bus:") {
        return Ok(false);
    }
    match output.status.code() {
        Some(0) => Ok(true),
        Some(3 | 4) => Ok(false),
        _ => bail!("systemctl --user could not determine whether {UNIT} is active"),
    }
}

pub fn restart() -> Result<()> {
    systemctl(["restart", UNIT])?;
    if !is_active() {
        bail!("{UNIT} did not remain active after restart");
    }
    Ok(())
}

/// Start the Omawake user unit installed by setup.
pub fn start(paths: &AppPaths) -> Result<()> {
    if !is_active() {
        ensure_no_direct_daemon(paths)?;
    }
    set_active_with(
        paths,
        "start",
        true,
        |action| systemctl([action, UNIT]),
        active_state,
    )
}

pub(crate) fn ensure_no_direct_daemon(paths: &AppPaths) -> Result<()> {
    match UnixStream::connect(paths.socket()) {
        Ok(_) => bail!(
            "a directly launched Omawake daemon is running; run `omawake stop` before starting the user service"
        ),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound
                    | std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::ConnectionReset
            ) => {}
        Err(error) => return Err(error).context("check for a directly launched Omawake daemon"),
    }
    let _reservation = crate::daemon_instance::reserve(paths)
        .context("stop the running Omawake daemon before starting the user service")?;
    Ok(())
}

/// Stop the Omawake user unit installed by setup.
pub fn stop(paths: &AppPaths) -> Result<()> {
    set_active_with(
        paths,
        "stop",
        false,
        |action| systemctl([action, UNIT]),
        active_state,
    )
}

fn set_active_with(
    paths: &AppPaths,
    action: &str,
    expected_active: bool,
    control: impl FnOnce(&str) -> Result<()>,
    active_after: impl FnOnce() -> Result<bool>,
) -> Result<()> {
    read_managed_unit(paths, None)?;
    control(action)?;
    if active_after()? != expected_active {
        bail!(
            "{UNIT} did not {} after {action}",
            if expected_active { "start" } else { "stop" }
        );
    }
    Ok(())
}

pub fn reload_if_was_active(was_active: bool) -> Result<bool> {
    reload_if_was_active_with(was_active, || systemctl(["try-restart", UNIT]), is_active)
}

fn reload_if_was_active_with(
    was_active: bool,
    try_restart: impl FnOnce() -> Result<()>,
    active_after: impl FnOnce() -> bool,
) -> Result<bool> {
    if !was_active {
        return Ok(false);
    }
    try_restart()?;
    if !active_after() {
        bail!("{UNIT} stopped while applying setup changes");
    }
    Ok(true)
}

fn systemctl<const N: usize>(args: [&str; N]) -> Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
        .context("run systemctl --user")?;
    if !status.success() {
        bail!("systemctl --user command failed");
    }
    Ok(())
}

fn quote(path: &Path) -> String {
    let mut escaped = String::with_capacity(path.as_os_str().len() + 2);
    escaped.push('"');
    for character in path.display().to_string().chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '%' => escaped.push_str("%%"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_systemd.rs"]
mod tests;
