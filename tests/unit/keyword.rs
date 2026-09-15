use super::*;

fn word(id: &str, phrase: &str, enabled: bool, command: &[&str]) -> WakeWord {
    WakeWord {
        id: id.into(),
        phrase: phrase.into(),
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
