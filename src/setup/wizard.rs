use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{self, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};

use crate::backend::Runtime;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupMode {
    Full,
    Runtime,
    Model,
    Check,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeSelection {
    pub runtime: Runtime,
    pub device: String,
}

trait Prompter {
    fn choose(
        &mut self,
        title: &str,
        help: &str,
        items: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>>;
}

struct TerminalPrompter;

impl Prompter for TerminalPrompter {
    fn choose(
        &mut self,
        title: &str,
        help: &str,
        items: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>> {
        select(title, help, items, preferred)
    }
}

/// One row in an interactive setup selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuItem {
    pub label: String,
    pub detail: String,
    pub enabled: bool,
}

impl MenuItem {
    pub fn available(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            enabled: true,
        }
    }

    pub fn unavailable(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            enabled: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Up,
    Down,
    Accept,
    Cancel,
    Ignore,
}

#[derive(Debug)]
struct MenuState {
    selected: usize,
}

impl MenuState {
    fn new(items: &[MenuItem], preferred: usize) -> Result<Self> {
        if items.is_empty() {
            bail!("interactive menu has no choices");
        }
        if !items.iter().any(|item| item.enabled) {
            bail!("interactive menu has no available choices");
        }
        let selected = if items.get(preferred).is_some_and(|item| item.enabled) {
            preferred
        } else {
            items
                .iter()
                .position(|item| item.enabled)
                .unwrap_or_default()
        };
        Ok(Self { selected })
    }

    fn apply(&mut self, action: Action, items: &[MenuItem]) -> Option<Option<usize>> {
        match action {
            Action::Up => self.move_by(items, -1),
            Action::Down => self.move_by(items, 1),
            Action::Accept => return Some(Some(self.selected)),
            Action::Cancel => return Some(None),
            Action::Ignore => {}
        }
        None
    }

    fn move_by(&mut self, items: &[MenuItem], direction: isize) {
        let mut next = self.selected;
        loop {
            next = (next as isize + direction).rem_euclid(items.len() as isize) as usize;
            if items[next].enabled {
                self.selected = next;
                return;
            }
        }
    }
}

/// Show an arrow-key selector. Enter accepts; Escape or q cancels.
pub fn select(
    title: &str,
    help: &str,
    items: &[MenuItem],
    preferred: usize,
) -> Result<Option<usize>> {
    let mut stdout = io::stdout();
    let _terminal = TerminalSession::enter(&mut stdout)?;
    run_menu(&mut stdout, title, help, items, preferred, || {
        event::read().context("read terminal input")
    })
}

struct TerminalSession;

impl TerminalSession {
    fn enter(output: &mut impl Write) -> Result<Self> {
        terminal::enable_raw_mode().context("enable terminal raw mode")?;
        if let Err(error) = execute!(output, EnterAlternateScreen, cursor::Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(error).context("open interactive setup screen");
        }
        Ok(Self)
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

fn run_menu(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    preferred: usize,
    mut read: impl FnMut() -> Result<Event>,
) -> Result<Option<usize>> {
    let mut state = MenuState::new(items, preferred)?;
    loop {
        render(output, title, help, items, state.selected)?;
        let Event::Key(key) = read()? else {
            continue;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }
        if let Some(result) = state.apply(action(key), items) {
            return Ok(result);
        }
    }
}

fn action(key: KeyEvent) -> Action {
    match (key.code, key.modifiers) {
        (KeyCode::Up | KeyCode::Char('k'), KeyModifiers::NONE) => Action::Up,
        (KeyCode::Down | KeyCode::Char('j'), KeyModifiers::NONE) => Action::Down,
        (KeyCode::Enter, KeyModifiers::NONE) => Action::Accept,
        (KeyCode::Esc | KeyCode::Char('q'), KeyModifiers::NONE)
        | (KeyCode::Char('c'), KeyModifiers::CONTROL) => Action::Cancel,
        _ => Action::Ignore,
    }
}

fn render(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    selected: usize,
) -> Result<()> {
    queue!(
        output,
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0),
        SetAttribute(Attribute::Bold),
        Print(title),
        SetAttribute(Attribute::Reset),
        Print("\r\n\r\n"),
        Print(help),
        Print("\r\n"),
        SetForegroundColor(Color::DarkGrey),
        Print("↑↓ navigate · Enter select · Esc cancel"),
        ResetColor,
        Print("\r\n\r\n")
    )?;

    for (index, item) in items.iter().enumerate() {
        if !item.enabled {
            queue!(output, SetForegroundColor(Color::DarkGrey))?;
        } else if index == selected {
            queue!(
                output,
                SetForegroundColor(Color::Cyan),
                SetAttribute(Attribute::Bold)
            )?;
        }
        queue!(
            output,
            Print(if index == selected { "  › " } else { "    " }),
            Print(&item.label),
            SetAttribute(Attribute::Reset),
            ResetColor,
            Print("\r\n      "),
            SetForegroundColor(Color::DarkGrey),
            Print(&item.detail),
            ResetColor,
            Print("\r\n\r\n")
        )?;
    }
    output.flush()?;
    Ok(())
}

pub fn choose_setup_mode() -> Result<Option<SetupMode>> {
    choose_setup_mode_with(&mut TerminalPrompter)
}

fn choose_setup_mode_with(prompter: &mut impl Prompter) -> Result<Option<SetupMode>> {
    let items = setup_mode_items();
    Ok(prompter
        .choose("Omawake setup", "Choose a guided setup flow.", &items, 0)?
        .map(setup_mode))
}

fn setup_mode_items() -> [MenuItem; 4] {
    [
        MenuItem::available(
            "Full setup",
            "Choose runtime and model, then install the model and launcher.",
        ),
        MenuItem::available(
            "Runtime",
            "Choose an inference runtime and compatible device.",
        ),
        MenuItem::available("Model", "Browse, download, and activate a wake-word model."),
        MenuItem::available(
            "Check",
            "Verify configuration, model, launcher, and optional service status.",
        ),
    ]
}

fn setup_mode(index: usize) -> SetupMode {
    [
        SetupMode::Full,
        SetupMode::Runtime,
        SetupMode::Model,
        SetupMode::Check,
    ][index]
}

pub fn choose_runtime(
    loadable: &BTreeMap<&str, bool>,
    library_context: &str,
    current_runtime: Runtime,
    current_device: &str,
) -> Result<Option<RuntimeSelection>> {
    choose_runtime_with(
        &mut TerminalPrompter,
        loadable,
        library_context,
        current_runtime,
        current_device,
    )
}

pub fn choose_runtime_directory(current: &[PathBuf]) -> Result<Option<PathBuf>> {
    let mut input = io::stdin().lock();
    choose_runtime_directory_with(&mut TerminalPrompter, current, || {
        let mut line = String::new();
        print!("Runtime library directory: ");
        io::stdout().flush()?;
        input.read_line(&mut line)?;
        Ok(line)
    })
}

pub fn choose_model_archive(model: &str) -> Result<Option<PathBuf>> {
    let mut input = io::stdin().lock();
    choose_model_archive_with(&mut TerminalPrompter, model, || {
        let mut line = String::new();
        print!("Licensed model archive: ");
        io::stdout().flush()?;
        input.read_line(&mut line)?;
        Ok(line)
    })
}

fn choose_model_archive_with(
    prompter: &mut impl Prompter,
    model: &str,
    mut read_line: impl FnMut() -> Result<String>,
) -> Result<Option<PathBuf>> {
    let items = [
        MenuItem::available(
            "Choose licensed archive",
            format!("Provide a local archive for {model}; Omawake verifies its pinned hash"),
        ),
        MenuItem::available("Back", "Return without installing a model."),
    ];
    if prompter.choose(
        "Model archive",
        "This model cannot be downloaded until its license terms are verified.",
        &items,
        0,
    )? != Some(0)
    {
        return Ok(None);
    }
    let raw = read_line()?;
    let archive = Path::new(raw.trim());
    if !archive.is_absolute() || !archive.is_file() {
        bail!(
            "model archive must be an absolute existing file: {}",
            archive.display()
        );
    }
    Ok(Some(archive.to_owned()))
}

fn choose_runtime_directory_with(
    prompter: &mut impl Prompter,
    current: &[PathBuf],
    mut read_line: impl FnMut() -> Result<String>,
) -> Result<Option<PathBuf>> {
    let configured = if current.is_empty() {
        "No configured runtime directory".to_owned()
    } else {
        format!(
            "Keep {}",
            current
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(":")
        )
    };
    let items = [
        MenuItem::available("Use current discovery", configured),
        MenuItem::available(
            "Choose runtime directory",
            "Point Omawake at external ONNX Runtime, patched sherpa, and provider libraries",
        ),
    ];
    if prompter.choose(
        "Runtime libraries",
        "You can keep detected/configured paths or enter an absolute runtime directory.",
        &items,
        0,
    )? != Some(1)
    {
        return Ok(None);
    }
    let raw = read_line()?;
    let directory = Path::new(raw.trim());
    if !directory.is_absolute() || !directory.is_dir() {
        bail!(
            "runtime library directory must be an absolute existing directory: {}",
            directory.display()
        );
    }
    Ok(Some(directory.to_owned()))
}

fn choose_runtime_with(
    prompter: &mut impl Prompter,
    loadable: &BTreeMap<&str, bool>,
    library_context: &str,
    current_runtime: Runtime,
    current_device: &str,
) -> Result<Option<RuntimeSelection>> {
    let items = runtime_items(loadable);
    let preferred = runtime_index(current_runtime);
    let help = format!(
        "All runtimes remain selectable so you can configure an external stack after choosing.\r\n{library_context}"
    );
    let Some(index) = prompter.choose("Inference runtime", &help, &items, preferred)? else {
        return Ok(None);
    };
    let runtime = runtime_at(index);
    let devices = device_items(runtime);
    let preferred = device_values(runtime)
        .iter()
        .position(|(value, _)| value.eq_ignore_ascii_case(current_device))
        .unwrap_or(0);
    let Some(index) = prompter.choose(
        "Inference device",
        "Only devices compatible with the chosen runtime are shown.",
        &devices,
        preferred,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(RuntimeSelection {
        runtime,
        device: device_values(runtime)[index].0.into(),
    }))
}

fn runtime_items(loadable: &BTreeMap<&str, bool>) -> [MenuItem; 3] {
    [
        if loadable.get("default").copied().unwrap_or(false) {
            MenuItem::available("Default", "ONNX Runtime and sherpa detected · CPU")
        } else {
            MenuItem::available(
                "Default",
                "Runtime not detected · release archives include it, or choose a library directory",
            )
        },
        if loadable.get("openvino").copied().unwrap_or(false) {
            MenuItem::available(
                "OpenVINO",
                "External OpenVINO provider detected · Intel CPU, GPU, or NPU",
            )
        } else {
            MenuItem::available(
                "OpenVINO",
                "External OpenVINO provider not detected · select to configure its library directory",
            )
        },
        if loadable.get("cuda").copied().unwrap_or(false) {
            MenuItem::available("CUDA", "External CUDA provider detected · NVIDIA GPU")
        } else {
            MenuItem::available(
                "CUDA",
                "External CUDA provider not detected · select to configure its library directory",
            )
        },
    ]
}

fn runtime_index(runtime: Runtime) -> usize {
    match runtime {
        Runtime::Default => 0,
        Runtime::Openvino => 1,
        Runtime::Cuda => 2,
    }
}

fn runtime_at(index: usize) -> Runtime {
    [Runtime::Default, Runtime::Openvino, Runtime::Cuda][index]
}

pub fn confirm_apply(
    runtime: Runtime,
    device: &str,
    model: &str,
    service_was_active: bool,
) -> Result<bool> {
    confirm_apply_with(
        &mut TerminalPrompter,
        runtime,
        device,
        model,
        service_was_active,
    )
}

pub fn confirm_runtime_apply(
    runtime: Runtime,
    device: &str,
    runtime_directory: Option<&Path>,
) -> Result<bool> {
    confirm_runtime_apply_with(&mut TerminalPrompter, runtime, device, runtime_directory)
}

fn confirm_runtime_apply_with(
    prompter: &mut impl Prompter,
    runtime: Runtime,
    device: &str,
    runtime_directory: Option<&Path>,
) -> Result<bool> {
    let source = runtime_directory.map_or_else(
        || "Use configured, packaged, or system runtime discovery".to_owned(),
        |directory| format!("Use runtime libraries from {}", directory.display()),
    );
    let items = [
        MenuItem::available(
            "Apply runtime",
            format!("Runtime: {} / {device} · {source}", runtime_name(runtime)),
        ),
        MenuItem::available("Cancel", "Return without changing the config."),
    ];
    Ok(matches!(
        prompter.choose(
            "Review runtime setup",
            "The selected stack is validated before the config is saved.",
            &items,
            0,
        )?,
        Some(0)
    ))
}

fn confirm_apply_with(
    prompter: &mut impl Prompter,
    runtime: Runtime,
    device: &str,
    model: &str,
    service_was_active: bool,
) -> Result<bool> {
    let items = apply_items(runtime, device, model, service_was_active);
    Ok(matches!(
        prompter.choose(
            "Review full setup",
            "Nothing has been changed yet.",
            &items,
            0,
        )?,
        Some(0)
    ))
}

fn apply_items(
    runtime: Runtime,
    device: &str,
    model: &str,
    service_was_active: bool,
) -> [MenuItem; 2] {
    [
        MenuItem::available(
            "Apply setup",
            format!(
                "Runtime: {} / {device} · Model: {model} · install model and launcher · {}",
                runtime_name(runtime),
                if service_was_active {
                    "restart the already-active service"
                } else {
                    "leave the optional service unchanged"
                }
            ),
        ),
        MenuItem::available("Cancel", "Return without changing files."),
    ]
}

fn device_items(runtime: Runtime) -> Vec<MenuItem> {
    device_values(runtime)
        .iter()
        .map(|(value, description)| MenuItem::available(*value, *description))
        .collect()
}

fn device_values(runtime: Runtime) -> &'static [(&'static str, &'static str)] {
    match runtime {
        Runtime::Default => &[("auto", "Let sherpa-onnx choose"), ("cpu", "CPU execution")],
        Runtime::Openvino => &[
            ("auto", "Let OpenVINO choose"),
            ("npu", "Intel NPU"),
            ("gpu", "Intel integrated or discrete GPU"),
            ("cpu", "CPU through OpenVINO"),
        ],
        Runtime::Cuda => &[("auto", "Default CUDA device"), ("gpu", "NVIDIA GPU")],
    }
}

fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
    }
}

#[cfg(test)]
#[path = "../../tests/unit/setup_wizard.rs"]
mod tests;
