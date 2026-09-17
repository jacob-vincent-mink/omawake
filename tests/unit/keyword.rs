use super::*;

fn word(id: &str, phrase: &str, enabled: bool, command: &[&str]) -> WakeWord {
    WakeWord {
        engine: None,
        enrollment: None,
        id: id.into(),
        phrase: phrase.into(),
        aliases: Vec::new(),
        enabled,
        command: command.iter().map(|item| (*item).into()).collect(),
    }
}

#[test]
fn validates_empty_and_multiple_mapping_sets() {
    assert!(validate_wake_words(&[]).is_ok());
    assert!(
        validate_wake_words(&[
            word("one", "Hello", true, &["true"]),
            word("two", " hello ", false, &["true"]),
        ])
        .is_ok()
    );
    assert!(
        validate_wake_words(&[
            word("one", "Hello", true, &["true"]),
            word("one", "Other", false, &["true"]),
        ])
        .is_err()
    );
    assert!(
        validate_wake_words(&[
            word("one", "Hello", true, &["true"]),
            word("two", " hello ", true, &["true"]),
        ])
        .is_err()
    );
}

#[test]
fn rejects_invalid_entries() {
    for invalid in [
        word("", "Hello", true, &["true"]),
        word("UPPER", "Hello", true, &["true"]),
        word("-start", "Hello", true, &["true"]),
        word("end-", "Hello", true, &["true"]),
        word("valid", " ", true, &["true"]),
        word("valid", "Hello", true, &[]),
        word("valid", "Hello", true, &[""]),
    ] {
        assert!(validate_wake_words(&[invalid]).is_err());
    }
}

#[test]
fn validates_exact_aliases_and_rejects_ambiguous_or_empty_ones() {
    let mut jarvis = word("jarvis", "hey jarvis", true, &["true"]);
    jarvis.aliases = vec!["hey jar viss".into()];
    assert!(validate_wake_words(&[jarvis.clone()]).is_ok());

    jarvis.aliases.push(" ".into());
    assert!(validate_wake_words(&[jarvis]).is_err());

    let mut duplicate = word("jarvis", "hey jarvis", true, &["true"]);
    duplicate.aliases = vec!["hey jar vis".into()];
    assert!(validate_wake_words(&[duplicate]).is_err());
}

#[test]
fn rejects_variants_that_normalize_to_empty_before_saving() {
    for text in ["---", "!!!", "\u{200b}"] {
        assert!(validate_wake_words(&[word("empty", text, true, &["true"])]).is_err());
        let mut entry = word("valid", "hey jarvis", true, &["true"]);
        entry.aliases.push(text.into());
        assert!(validate_wake_words(&[entry]).is_err());
    }
}
