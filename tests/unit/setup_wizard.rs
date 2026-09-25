use super::*;

#[test]
fn rows_wrap_at_narrow_and_normal_widths_without_losing_unicode_alignment() {
    use unicode_width::UnicodeWidthStr;
    for width in [24, 80] {
        for indent in [0, 4, 6] {
            let text = wrap(
                "OpenVINO / GPU — /opt/运行时/lib/intel64/Release\nprovider registration failed; configure external libraries",
                width,
                indent,
            );
            for (index, line) in text.split("\r\n").enumerate() {
                assert!(line.width() + if index == 0 { indent } else { 0 } < width);
            }
            assert_no_bare_line_feeds(text.as_bytes());
        }
        let mut output = Vec::new();
        render_at_width(
            &mut output,
            "Runtime",
            "Choose an external installation",
            &[MenuItem::available(
                "OpenVINO",
                "/opt/runtime/lib/intel64/Release",
            )],
            0,
            width,
        )
        .unwrap();
        assert!(!output.is_empty());
    }
    assert!(wrap("\x1bhello\tworld", 80, 0).contains("helloworld"));
    let prose = wrap(
        "This never installs or starts a service; an active daemon restarts after Apply.",
        32,
        6,
    );
    assert!(!prose.contains("\r\n      his"));
    assert!(!prose.contains("\r\n      pply"));
}
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
    assert!(rendered.contains("OMAWAKE  /  RUNTIME"));
    assert!(rendered.contains("SELECTED  /  OpenVINO"));
    assert!(rendered.contains("Intel CPU, GPU, and NPU"));
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
    let cpu_loadable = BTreeMap::from([("default", true)]);
    let modes = setup_mode_items();
    assert_eq!(modes[0].label, "Full setup");
    assert!(modes[0].detail.contains("model and launcher"));
    assert!(!modes[0].detail.contains("service"));
    assert_eq!(setup_mode(0), SetupMode::Full);
    assert_eq!(setup_mode(1), SetupMode::Runtime);
    assert_eq!(setup_mode(2), SetupMode::Model);
    assert_eq!(setup_mode(3), SetupMode::Check);
    assert_eq!(setup_mode(4), SetupMode::Audio);
    assert_eq!(modes[4].label, "Audio");
    assert_eq!(setup_mode(5), SetupMode::Onboard);
    assert_eq!(modes[5].label, "Teach a wake word");

    let cpu_only = runtime_items(&cpu_loadable);
    assert!(cpu_only[0].enabled);
    assert!(cpu_only[1].enabled);
    assert!(cpu_only.iter().all(|item| item.enabled));
    let all_loadable = BTreeMap::from([
        ("default", true),
        ("openvino", true),
        ("cuda", true),
        ("vulkan", true),
        ("hip", true),
    ]);
    let all = runtime_items(&all_loadable);
    assert!(all[0].enabled);
    assert!(all.iter().all(|item| item.enabled));
    assert!(all[1].detail.contains("Official"));
    let compiled_only = runtime_items(&BTreeMap::from([("default", true)]));
    assert!(compiled_only[1].enabled);
    assert!(compiled_only[1].detail.contains("complete OpenVINO"));

    for (index, runtime) in [
        Runtime::Default,
        Runtime::Openvino,
        Runtime::Cuda,
        Runtime::Vulkan,
        Runtime::Hip,
    ]
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
    assert_eq!(runtime_name(Runtime::Vulkan), "vulkan");
    assert_eq!(runtime_name(Runtime::Hip), "hip");
    assert_eq!(device_values(Runtime::Openvino)[0].0, "npu");

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
fn runtime_directory_prompt_keeps_or_validates_an_absolute_directory() {
    let root =
        std::env::temp_dir().join(format!("omawake-wizard-runtime-dir-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut keep = ScriptedPrompter {
        choices: VecDeque::from([Some(0)]),
        preferred: vec![],
    };
    assert_eq!(
        choose_runtime_directory_with(&mut keep, std::slice::from_ref(&root), || {
            unreachable!()
        })
        .unwrap(),
        None
    );

    let mut choose = ScriptedPrompter {
        choices: VecDeque::from([Some(1)]),
        preferred: vec![],
    };
    assert_eq!(
        choose_runtime_directory_with(&mut choose, &[], || { Ok(format!("{}\n", root.display())) })
            .unwrap(),
        Some(root.clone())
    );

    let mut invalid = ScriptedPrompter {
        choices: VecDeque::from([Some(1)]),
        preferred: vec![],
    };
    assert!(choose_runtime_directory_with(&mut invalid, &[], || Ok("relative\n".into())).is_err());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn model_source_prompt_supports_back_and_validates_an_absolute_directory() {
    let root = std::env::temp_dir().join(format!("omawake-wizard-model-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("model-assets");
    std::fs::create_dir_all(&source).unwrap();

    for choice in [Some(1), None] {
        let mut back = ScriptedPrompter {
            choices: VecDeque::from([choice]),
            preferred: vec![],
        };
        assert_eq!(
            choose_model_source_directory_with(&mut back, "wake-model", || unreachable!()).unwrap(),
            None
        );
    }

    let mut choose = ScriptedPrompter {
        choices: VecDeque::from([Some(0)]),
        preferred: vec![],
    };
    assert_eq!(
        choose_model_source_directory_with(&mut choose, "wake-model", || {
            Ok(format!("{}\n", source.display()))
        })
        .unwrap(),
        Some(source.clone())
    );

    let mut relative = ScriptedPrompter {
        choices: VecDeque::from([Some(0)]),
        preferred: vec![],
    };
    assert!(
        choose_model_source_directory_with(&mut relative, "wake-model", || Ok("relative\n".into()))
            .is_err()
    );

    let mut missing = ScriptedPrompter {
        choices: VecDeque::from([Some(0)]),
        preferred: vec![],
    };
    assert!(
        choose_model_source_directory_with(&mut missing, "wake-model", || {
            Ok(format!("{}\n", root.join("missing.tar").display()))
        })
        .is_err()
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn guided_flows_map_scripted_choices_and_preserve_preferences() {
    let loadable = BTreeMap::from([("openvino", true)]);
    for (index, expected) in [
        SetupMode::Full,
        SetupMode::Runtime,
        SetupMode::Model,
        SetupMode::Check,
        SetupMode::Audio,
        SetupMode::Onboard,
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
        choices: VecDeque::from([Some(1), Some(0)]),
        preferred: vec![],
    };
    assert_eq!(
        choose_runtime_with(
            &mut npu,
            &loadable,
            "Configured: /opt/oma\r\nEffective: /opt/oma\r\nRemediation: none",
            Runtime::Openvino,
            "npu",
            None,
            false,
        )
        .unwrap(),
        Some(RuntimeSelection {
            runtime: Runtime::Openvino,
            device: "npu".into(),
        })
    );
    assert_eq!(npu.preferred, [1, 0]);

    for choices in [VecDeque::from([None]), VecDeque::from([Some(0), None])] {
        let mut prompt = ScriptedPrompter {
            choices,
            preferred: vec![],
        };
        assert_eq!(
            choose_runtime_with(
                &mut prompt,
                &BTreeMap::new(),
                "Configured: none\r\nEffective: none\r\nRemediation: none",
                Runtime::Default,
                "unknown",
                None,
                false,
            )
            .unwrap(),
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

        let mut prompt = ScriptedPrompter {
            choices: VecDeque::from([choice]),
            preferred: vec![],
        };
        assert_eq!(
            confirm_runtime_apply_with(
                &mut prompt,
                Runtime::Openvino,
                "npu",
                Some(Path::new("/opt/openvino")),
            )
            .unwrap(),
            expected
        );
    }
}

#[test]
fn runtime_recommendation_prefers_fresh_accelerator_but_preserves_existing_selection() {
    let recommendation = crate::hardware::recommend(
        &crate::hardware::HardwareReport {
            intel_npu: true,
            ..Default::default()
        },
        crate::hardware::ProviderAvailability {
            packaged_cpu: true,
            openvino_npu: true,
            ..Default::default()
        },
    );
    let loadable = BTreeMap::from([("default", true), ("openvino", true)]);

    let mut fresh = ScriptedPrompter {
        choices: VecDeque::from([Some(1), Some(0)]),
        preferred: Vec::new(),
    };
    let selected = choose_runtime_with(
        &mut fresh,
        &loadable,
        "discovery",
        Runtime::Default,
        "cpu",
        Some(&recommendation),
        true,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        (selected.runtime, selected.device.as_str()),
        (Runtime::Openvino, "npu")
    );
    assert_eq!(fresh.preferred, [1, 0]);

    let mut existing = ScriptedPrompter {
        choices: VecDeque::from([Some(0), Some(0)]),
        preferred: Vec::new(),
    };
    let selected = choose_runtime_with(
        &mut existing,
        &loadable,
        "discovery",
        Runtime::Default,
        "cpu",
        Some(&recommendation),
        false,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        (selected.runtime, selected.device.as_str()),
        (Runtime::Default, "cpu")
    );
    assert_eq!(existing.preferred, [0, 0]);
}

#[test]
fn terminal_entry_points_fail_cleanly_without_a_tty() {
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        return;
    }
    assert!(choose_setup_mode().is_err());
    assert!(
        choose_runtime(
            &BTreeMap::new(),
            "Configured: none\r\nEffective: none\r\nRemediation: none",
            Runtime::Default,
            "auto"
        )
        .is_err()
    );
    assert!(confirm_apply(Runtime::Default, "cpu", "model", false).is_err());
    assert!(confirm_runtime_apply(Runtime::Default, "cpu", None).is_err());
}

fn plain_terminal_output(output: &[u8]) -> String {
    let mut text = String::new();
    let mut escape = false;
    for character in String::from_utf8_lossy(output).chars() {
        if character == '\x1b' {
            escape = true;
        } else if escape {
            if character.is_ascii_alphabetic() {
                escape = false;
            }
        } else {
            text.push(character);
        }
    }
    text
}

#[test]
fn viewport_keeps_selection_visible_without_overflow_on_resize() {
    use unicode_width::UnicodeWidthStr;
    let items = (0..100)
        .map(|n| {
            MenuItem::available(
                format!("Choice {n}"),
                "Long details with Unicode 运行时 and many words to wrap across a narrow terminal.",
            )
        })
        .collect::<Vec<_>>();
    for (width, height) in [(24, 8), (80, 24), (12, 4), (4, 2), (1, 1)] {
        for selected in [0, 50, 99] {
            let mut output = Vec::new();
            render_at_size(
                &mut output,
                "Choose a model",
                "Navigate the installed and downloadable catalog.",
                &items,
                selected,
                width,
                height,
            )
            .unwrap();
            let text = plain_terminal_output(&output);
            assert!(
                text.split("\r\n").count() <= height,
                "{width}x{height}: {text:?}"
            );
            assert!(text.split("\r\n").all(|line| line.width() < width));
            if width >= 24 {
                assert!(text.contains(&format!("› Choice {selected}")), "{text:?}");
            }
            assert_no_bare_line_feeds(&output);
        }
    }
}
