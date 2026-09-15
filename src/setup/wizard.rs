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
    let width = terminal::size()
        .ok()
        .filter(|(width, _)| *width > 0)
        .map_or(80, |(width, _)| width as usize);
    render_at_width(output, title, help, items, selected, width)
}

fn render_at_width(
    output: &mut impl Write,
    title: &str,
    help: &str,
    items: &[MenuItem],
    selected: usize,
    width: usize,
) -> Result<()> {
    queue!(
        output,
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0),
        SetAttribute(Attribute::Bold),
        Print(wrap(title, width, 0)),
        SetAttribute(Attribute::Reset),
        Print("\r\n\r\n"),
        Print(wrap(help, width, 0)),
        Print("\r\n"),
        SetForegroundColor(Color::DarkGrey),
        Print(wrap("↑↓ navigate · Enter select · Esc cancel", width, 0)),
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
            Print(wrap(&item.label, width, 4)),
            SetAttribute(Attribute::Reset),
            ResetColor,
            Print("\r\n      "),
            SetForegroundColor(Color::DarkGrey),
            Print(wrap(&item.detail, width, 6)),
            ResetColor,
            Print("\r\n\r\n")
        )?;
    }
    output.flush()?;
    Ok(())
}

fn wrap(text: &str, width: usize, indent: usize) -> String {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

    let limit = width.saturating_sub(indent + 1).max(1);
    let text: String = text
        .chars()
        .filter(|character| *character == '\n' || !character.is_control())
        .collect();
    let mut output = String::new();
    for (paragraph_index, paragraph) in text.split('\n').enumerate() {
        if paragraph_index > 0 {
            wrapped_newline(&mut output, indent);
        }
        let mut column = 0;
        for word in paragraph.split_whitespace() {
            let word_width = word.width();
            if column > 0 && column + 1 + word_width <= limit {
                output.push(' ');
                output.push_str(word);
                column += 1 + word_width;
                continue;
            }
            if column > 0 {
                wrapped_newline(&mut output, indent);
                column = 0;
            }
            for character in word.chars() {
                let character_width = character.width().unwrap_or(0);
                if column > 0 && column + character_width > limit {
                    wrapped_newline(&mut output, indent);
                    column = 0;
                }
                output.push(character);
                column += character_width;
            }
        }
    }
    output
}

fn wrapped_newline(output: &mut String, indent: usize) {
    output.push_str("\r\n");
    output.push_str(&" ".repeat(indent));
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
        None,
        false,
    )
}

pub fn choose_runtime_with_recommendation(
    loadable: &BTreeMap<&str, bool>,
    library_context: &str,
    current_runtime: Runtime,
    current_device: &str,
    recommendation: &crate::hardware::Recommendation,
    prefer_recommendation: bool,
) -> Result<Option<RuntimeSelection>> {
    choose_runtime_with(
        &mut TerminalPrompter,
        loadable,
        library_context,
        current_runtime,
        current_device,
        Some(recommendation),
        prefer_recommendation,
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

pub fn choose_model_source_directory(model: &str) -> Result<Option<PathBuf>> {
    let mut input = io::stdin().lock();
    choose_model_source_directory_with(&mut TerminalPrompter, model, || {
        let mut line = String::new();
        print!("Model source directory: ");
        io::stdout().flush()?;
        input.read_line(&mut line)?;
        Ok(line)
    })
}

fn choose_model_source_directory_with(
    prompter: &mut impl Prompter,
    model: &str,
    mut read_line: impl FnMut() -> Result<String>,
) -> Result<Option<PathBuf>> {
    let items = [
        MenuItem::available(
            "Choose local asset directory",
            format!(
                "Provide all files for {model}; Omawake verifies every pinned size and SHA-256"
            ),
        ),
        MenuItem::available("Back", "Return without installing a model."),
    ];
    if prompter.choose(
        "Model source directory",
        "Use exact local assets when catalog download is unavailable.",
        &items,
        0,
    )? != Some(0)
    {
        return Ok(None);
    }
    let raw = read_line()?;
    let directory = Path::new(raw.trim());
    if !directory.is_absolute() || !directory.is_dir() {
        bail!(
            "model source must be an absolute existing directory: {}",
            directory.display()
        );
    }
    Ok(Some(directory.to_owned()))
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
            "Select a complete audio.cpp build or OpenVINO root; setup discovers its libraries and plugins",
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
    recommendation: Option<&crate::hardware::Recommendation>,
    prefer_recommendation: bool,
) -> Result<Option<RuntimeSelection>> {
    let mut items = runtime_items(loadable);
    if let Some(recommendation) = recommendation {
        let item = &mut items[runtime_index(recommendation.runtime)];
        item.label.push_str(" · Recommended");
        item.detail = format!("{} · {}", recommendation.detail, item.detail);
    }
    let preferred_runtime = recommendation
        .filter(|_| prefer_recommendation)
        .map_or(current_runtime, |item| item.runtime);
    let preferred = runtime_index(preferred_runtime);
    let help = format!(
        "Select a complete, supported provider. Setup discovers libraries but never installs system runtimes.\r\n{library_context}"
    );
    let Some(index) = prompter.choose("Inference runtime", &help, &items, preferred)? else {
        return Ok(None);
    };
    let runtime = runtime_at(index);
    let mut devices = device_items(runtime);
    if let Some(recommendation) = recommendation.filter(|item| item.runtime == runtime)
        && let Some(index) = device_values(runtime)
            .iter()
            .position(|(value, _)| value.eq_ignore_ascii_case(&recommendation.device))
    {
        devices[index].label.push_str(" · Recommended");
        devices[index].detail = format!("{} · {}", recommendation.detail, devices[index].detail);
    }
    let preferred_device = recommendation
        .filter(|item| prefer_recommendation && item.runtime == runtime)
        .map_or(current_device, |item| item.device.as_str());
    let preferred = device_values(runtime)
        .iter()
        .position(|(value, _)| value.eq_ignore_ascii_case(preferred_device))
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

fn runtime_items(loadable: &BTreeMap<&str, bool>) -> [MenuItem; 5] {
    [
        if loadable.get("default").copied().unwrap_or(false) {
            MenuItem::available(
                "audio.cpp · CPU",
                "Integrated public C ABI provider detected",
            )
        } else {
            MenuItem::available(
                "audio.cpp · CPU",
                "Provider not detected · release packages include it, or choose a complete build directory",
            )
        },
        if loadable.get("openvino").copied().unwrap_or(false) {
            MenuItem::available(
                "OpenVINO GenAI · Intel CPU/GPU/NPU",
                "Official Whisper C API provider detected",
            )
        } else {
            MenuItem::available(
                "OpenVINO GenAI · Intel CPU/GPU/NPU",
                "Choose a complete OpenVINO installation root; setup never installs it",
            )
        },
        accelerated_runtime_item(loadable, "cuda", "audio.cpp · CUDA", "NVIDIA GPU"),
        accelerated_runtime_item(loadable, "vulkan", "audio.cpp · Vulkan", "Vulkan GPU"),
        accelerated_runtime_item(loadable, "hip", "audio.cpp · HIP", "AMD GPU"),
    ]
}

fn accelerated_runtime_item(
    loadable: &BTreeMap<&str, bool>,
    runtime: &str,
    label: &str,
    device: &str,
) -> MenuItem {
    if loadable.get(runtime).copied().unwrap_or(false) {
        MenuItem::available(label, format!("Complete provider detected for {device}"))
    } else {
        MenuItem::available(
            label,
            format!("Choose a complete provider build for {device}; setup never installs it"),
        )
    }
}

fn runtime_index(runtime: Runtime) -> usize {
    match runtime {
        Runtime::Default => 0,
        Runtime::Openvino => 1,
        Runtime::Cuda => 2,
        Runtime::Vulkan => 3,
        Runtime::Hip => 4,
    }
}

fn runtime_at(index: usize) -> Runtime {
    [
        Runtime::Default,
        Runtime::Openvino,
        Runtime::Cuda,
        Runtime::Vulkan,
        Runtime::Hip,
    ][index]
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
        Runtime::Default => &[("cpu", "CPU execution")],
        Runtime::Openvino => &[
            ("npu", "Intel NPU"),
            ("gpu", "Intel integrated or discrete GPU"),
            ("cpu", "CPU through OpenVINO"),
        ],
        Runtime::Cuda => &[("auto", "Default CUDA device"), ("gpu", "NVIDIA GPU")],
        Runtime::Vulkan => &[("auto", "Default Vulkan device"), ("gpu", "Vulkan GPU")],
        Runtime::Hip => &[("auto", "Default HIP device"), ("gpu", "AMD GPU")],
    }
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

#[cfg(test)]
#[path = "../../tests/unit/setup_wizard.rs"]
mod tests;
