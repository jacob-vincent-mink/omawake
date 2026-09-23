use super::*;

fn paths() -> AppPaths {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "omawake-home-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    AppPaths {
        config_file: root.join("config/omawake/config.toml"),
        data_dir: root.join("data/omawake"),
        cache_dir: root.join("cache/omawake"),
        state_dir: root.join("state/omawake"),
        runtime_dir: root.join("run/omawake"),
    }
}

#[test]
fn first_run_has_a_clear_start_and_disables_actions_that_need_config() {
    let paths = paths();
    let snapshot = Snapshot::read(&paths.config_file, &paths);
    assert!(snapshot.config.is_none());
    let items = snapshot.items();
    assert_eq!(items.len(), HOME_ACTIONS.len());
    assert_eq!(items[0].label, "Start guided setup");
    assert!(items[0].enabled);
    for action in [
        HomeAction::Words,
        HomeAction::Teach,
        HomeAction::Audio,
        HomeAction::Test,
        HomeAction::Advanced,
    ] {
        let index = HOME_ACTIONS
            .iter()
            .position(|candidate| *candidate == action)
            .unwrap();
        assert!(!items[index].enabled);
    }
    assert!(snapshot.summary().contains("service is optional"));
}

#[test]
fn invalid_config_is_presented_as_repair_instead_of_first_run() {
    let paths = paths();
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    fs::write(&paths.config_file, "not = [valid").unwrap();
    let snapshot = Snapshot::read(&paths.config_file, &paths);
    assert!(snapshot.invalid_config);
    assert!(
        snapshot
            .summary()
            .contains("existing configuration could not be loaded")
    );
    assert!(
        snapshot.items()[1]
            .detail
            .contains("Repair the configuration")
    );
    fs::remove_dir_all(
        paths
            .config_file
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    )
    .unwrap();
}

#[test]
fn default_example_is_presented_for_review_not_as_completed_onboarding() {
    let paths = paths();
    Config::default().save(&paths.config_file).unwrap();
    let snapshot = Snapshot::read(&paths.config_file, &paths);
    assert!(snapshot.summary().contains("example 'Computer'"));
    assert!(snapshot.items()[1].label.starts_with("Review example"));
    fs::remove_dir_all(
        paths
            .config_file
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    )
    .unwrap();
}

#[test]
fn service_menu_only_offers_actions_for_the_current_state() {
    let actions = |installed, active| {
        service_items(installed, installed, active, true)
            .into_iter()
            .map(|entry| entry.0)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        actions(false, false),
        vec![ServiceAction::Install, ServiceAction::Back]
    );
    assert_eq!(
        actions(true, false),
        vec![
            ServiceAction::Start,
            ServiceAction::Status,
            ServiceAction::Uninstall,
            ServiceAction::Back
        ]
    );
    assert_eq!(
        actions(true, true),
        vec![
            ServiceAction::Stop,
            ServiceAction::Restart,
            ServiceAction::Status,
            ServiceAction::Uninstall,
            ServiceAction::Back
        ]
    );
    assert!(!service_items(false, false, false, false)[0].1.enabled);
    let unmanaged = service_items(true, false, true, true);
    assert!(!unmanaged[0].1.enabled);
    assert_eq!(unmanaged[1].0, ServiceAction::Status);
    assert_eq!(unmanaged[2].0, ServiceAction::Back);
}

#[test]
fn wake_word_edits_validate_and_persist_without_touching_invalid_config() {
    let paths = paths();
    let path = &paths.config_file;
    let mut config = Config::default();
    config.save(path).unwrap();
    let initial = fs::read(path).unwrap();

    config.wake_words[0].phrase = "Computer, please".into();
    config.wake_words[0].command = vec!["notify-send".into(), "Ready".into()];
    save_words(config.clone(), path, &paths).unwrap();
    let saved = Config::load(path).unwrap();
    assert_eq!(saved.wake_words[0].phrase, "Computer, please");
    assert_eq!(saved.wake_words[0].command[1], "Ready");
    assert_ne!(fs::read(path).unwrap(), initial);

    let stable = fs::read(path).unwrap();
    config.wake_words.push(config.wake_words[0].clone());
    assert!(save_words(config, path, &paths).is_err());
    assert_eq!(fs::read(path).unwrap(), stable);
    fs::remove_dir_all(path.parent().unwrap().parent().unwrap().parent().unwrap()).unwrap();
}

#[test]
fn trained_words_do_not_offer_transcript_only_edits() {
    let mut word = Config::default().wake_words.remove(0);
    assert!(edit_word_items(&word)[0].enabled);
    assert!(edit_word_items(&word)[2].enabled);

    word.enrollment = Some(crate::enrollment::artifact::EnrollmentBinding::default());
    let items = edit_word_items(&word);
    assert!(!items[0].enabled);
    assert!(items[0].detail.contains("Add a new wake word"));
    assert!(!items[2].enabled);
    assert!(items[1].enabled);
    assert!(items[3].enabled);

    word.enrollment.as_mut().unwrap().active = false;
    assert!(edit_word_items(&word)[0].enabled);
    assert!(edit_word_items(&word)[2].enabled);
}
