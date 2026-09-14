use super::*;
use std::collections::VecDeque;
use std::io::IsTerminal;

struct ScriptedPrompter {
    choices: VecDeque<Option<usize>>,
    preferred: Vec<usize>,
}

impl Prompter for ScriptedPrompter {
    fn choose(
        &mut self,
        _: &str,
        _: &str,
        _: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>> {
        self.preferred.push(preferred);
        Ok(self.choices.pop_front().unwrap())
    }
}

fn key(code: KeyCode) -> Event {
    Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn assert_no_bare_line_feeds(output: &[u8]) {
    assert!(
        output
            .iter()
            .enumerate()
            .all(|(index, byte)| *byte != b'\n' || index > 0 && output[index - 1] == b'\r'),
        "raw-mode rendering must return to column zero before every line feed"
    );
}

#[test]
fn item_constructors_preserve_metadata() {
    assert_eq!(
        MenuItem::available("CPU", "portable"),
        MenuItem {
            label: "CPU".into(),
            detail: "portable".into(),
            enabled: true,
        }
    );
    assert!(!MenuItem::unavailable("NPU", "not built").enabled);
}

#[test]
fn state_uses_available_preference_and_skips_disabled_rows() {
    let items = [
        MenuItem::unavailable("zero", "disabled"),
        MenuItem::available("one", "enabled"),
        MenuItem::unavailable("two", "disabled"),
        MenuItem::available("three", "enabled"),
    ];
    let mut state = MenuState::new(&items, 0).unwrap();
    assert_eq!(state.selected, 1);
    assert_eq!(state.apply(Action::Down, &items), None);
    assert_eq!(state.selected, 3);
    assert_eq!(state.apply(Action::Down, &items), None);
    assert_eq!(state.selected, 1);
    assert_eq!(state.apply(Action::Up, &items), None);
    assert_eq!(state.selected, 3);
    assert_eq!(state.apply(Action::Ignore, &items), None);
    assert_eq!(state.apply(Action::Accept, &items), Some(Some(3)));
    assert_eq!(state.apply(Action::Cancel, &items), Some(None));

    assert!(MenuState::new(&[], 0).is_err());
    assert!(MenuState::new(&[MenuItem::unavailable("x", "x")], 0).is_err());
}

#[test]
fn keyboard_mapping_covers_navigation_selection_and_cancel() {
    for code in [KeyCode::Up, KeyCode::Char('k')] {
        assert_eq!(action(KeyEvent::new(code, KeyModifiers::NONE)), Action::Up);
    }
    for code in [KeyCode::Down, KeyCode::Char('j')] {
        assert_eq!(
            action(KeyEvent::new(code, KeyModifiers::NONE)),
            Action::Down
        );
    }
    assert_eq!(
        action(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        Action::Accept
    );
    assert_eq!(
        action(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        Action::Cancel
    );
    assert_eq!(
        action(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        Action::Cancel
    );
    assert_eq!(
        action(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
        Action::Ignore
    );
}

#[test]
fn menu_renders_metadata_and_processes_arrow_enter_and_cancel() {
    let items = [
        MenuItem::available("CPU", "Built in"),
        MenuItem::unavailable("CUDA", "Unavailable in this build"),
        MenuItem::available("OpenVINO", "Intel CPU, GPU, and NPU"),
    ];
    let mut events = VecDeque::from([
        Event::Resize(80, 24),
        Event::Key(KeyEvent::new_with_kind(
            KeyCode::Down,
            KeyModifiers::NONE,
            KeyEventKind::Release,
        )),
        key(KeyCode::Down),
        key(KeyCode::Enter),
    ]);
    let mut output = Vec::new();
    let selected = run_menu(
        &mut output,
        "Runtime",
        "Choose a runtime.",
        &items,
        0,
        || Ok(events.pop_front().unwrap()),
    )
    .unwrap();
    assert_eq!(selected, Some(2));
    let rendered = String::from_utf8(output).unwrap();
    assert!(rendered.contains("Runtime"));
    assert!(rendered.contains("CUDA"));
    assert!(rendered.contains("Unavailable in this build"));
    assert!(rendered.contains("\r\n      "));
    assert_no_bare_line_feeds(rendered.as_bytes());

    let mut events = VecDeque::from([key(KeyCode::Char('q'))]);
    assert_eq!(
        run_menu(&mut Vec::new(), "x", "y", &items, 0, || {
            Ok(events.pop_front().unwrap())
        })
        .unwrap(),
        None
    );
}

#[test]
fn guided_setup_metadata_covers_modes_runtimes_devices_and_review() {
    let modes = setup_mode_items();
    assert_eq!(modes[0].label, "Full setup");
    assert!(modes[0].detail.contains("model and launcher"));
    assert!(!modes[0].detail.contains("service"));
    assert_eq!(setup_mode(0), SetupMode::Full);
    assert_eq!(setup_mode(1), SetupMode::Runtime);
    assert_eq!(setup_mode(2), SetupMode::Model);
    assert_eq!(setup_mode(3), SetupMode::Check);

    let cpu_only = runtime_items(&["cpu"]);
    assert!(cpu_only[0].enabled);
    assert!(!cpu_only[1].enabled);
    assert!(!cpu_only[2].enabled);
    let all = runtime_items(&["cpu", "openvino", "cuda"]);
    assert!(all.iter().all(|item| item.enabled));

    for (index, runtime) in [Runtime::Default, Runtime::Openvino, Runtime::Cuda]
        .into_iter()
        .enumerate()
    {
        assert_eq!(runtime_index(runtime), index);
        assert_eq!(runtime_at(index), runtime);
        assert!(!device_items(runtime).is_empty());
    }
    assert_eq!(runtime_name(Runtime::Default), "default");
    assert_eq!(runtime_name(Runtime::Openvino), "openvino");
    assert_eq!(runtime_name(Runtime::Cuda), "cuda");
    assert_eq!(device_values(Runtime::Openvino)[1].0, "npu");

    let review = apply_items(Runtime::Openvino, "npu", "wake-model", false);
    assert!(review[0].detail.contains("openvino / npu"));
    assert!(review[0].detail.contains("wake-model"));
    assert!(review[0].detail.contains("model and launcher"));
    assert!(
        review[0]
            .detail
            .contains("leave the optional service unchanged")
    );
    let active_review = apply_items(Runtime::Openvino, "npu", "wake-model", true);
    assert!(
        active_review[0]
            .detail
            .contains("restart the already-active service")
    );
    assert_eq!(review[1].label, "Cancel");
}

#[test]
fn guided_flows_map_scripted_choices_and_preserve_preferences() {
    for (index, expected) in [
        SetupMode::Full,
        SetupMode::Runtime,
        SetupMode::Model,
        SetupMode::Check,
    ]
    .into_iter()
    .enumerate()
    {
        let mut prompt = ScriptedPrompter {
            choices: VecDeque::from([Some(index)]),
            preferred: vec![],
        };
        assert_eq!(choose_setup_mode_with(&mut prompt).unwrap(), Some(expected));
    }
    let mut cancelled = ScriptedPrompter {
        choices: VecDeque::from([None]),
        preferred: vec![],
    };
    assert_eq!(choose_setup_mode_with(&mut cancelled).unwrap(), None);

    let mut npu = ScriptedPrompter {
        choices: VecDeque::from([Some(1), Some(1)]),
        preferred: vec![],
    };
    assert_eq!(
        choose_runtime_with(&mut npu, &["cpu", "openvino"], Runtime::Openvino, "npu").unwrap(),
        Some(RuntimeSelection {
            runtime: Runtime::Openvino,
            device: "npu".into(),
        })
    );
    assert_eq!(npu.preferred, [1, 1]);

    for choices in [VecDeque::from([None]), VecDeque::from([Some(0), None])] {
        let mut prompt = ScriptedPrompter {
            choices,
            preferred: vec![],
        };
        assert_eq!(
            choose_runtime_with(&mut prompt, &["cpu"], Runtime::Default, "unknown").unwrap(),
            None
        );
    }

    for (choice, expected) in [(Some(0), true), (Some(1), false), (None, false)] {
        let mut prompt = ScriptedPrompter {
            choices: VecDeque::from([choice]),
            preferred: vec![],
        };
        assert_eq!(
            confirm_apply_with(&mut prompt, Runtime::Default, "cpu", "model", false).unwrap(),
            expected
        );
    }
}

#[test]
fn terminal_entry_points_fail_cleanly_without_a_tty() {
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        return;
    }
    assert!(choose_setup_mode().is_err());
    assert!(choose_runtime(&["cpu"], Runtime::Default, "auto").is_err());
    assert!(confirm_apply(Runtime::Default, "cpu", "model", false).is_err());
}
