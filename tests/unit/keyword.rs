use super::*;

fn word(id: &str, phrase: &str, enabled: bool, command: &[&str]) -> WakeWord {
    WakeWord {
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
    let mut atreyu = word("atreyu", "hey atreyu", true, &["true"]);
    atreyu.aliases = vec!["hey a tray you".into()];
    assert!(validate_wake_words(&[atreyu.clone()]).is_ok());

    atreyu.aliases.push(" ".into());
    assert!(validate_wake_words(&[atreyu]).is_err());

    let mut duplicate = word("atreyu", "hey atreyu", true, &["true"]);
    duplicate.aliases = vec!["hey at rey u".into()];
    assert!(validate_wake_words(&[duplicate]).is_err());
}
