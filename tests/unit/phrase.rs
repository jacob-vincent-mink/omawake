use super::*;

fn wake_word(id: &str, phrase: &str) -> WakeWord {
    WakeWord {
        id: id.into(),
        phrase: phrase.into(),
        aliases: Vec::new(),
        enabled: true,
        command: vec!["true".into()],
    }
}

#[test]
fn normalizes_unicode_case_width_and_punctuation() {
    assert_eq!(normalize_tokens("  ＨＥＹ, Café!  "), ["hey", "café"]);
}

#[test]
fn requires_whole_tokens() {
    let matcher = PhraseMatcher::compile(&[wake_word("computer", "computer")]).unwrap();
    assert!(
        matcher
            .matches("the computer is ready")
            .iter()
            .any(|item| item.id == "computer")
    );
    assert!(matcher.matches("computerized controls").is_empty());
}

#[test]
fn ignores_asr_word_boundary_variation() {
    let matcher = PhraseMatcher::compile(&[
        wake_word("forever", "forever"),
        wake_word("lights", "light up"),
    ])
    .unwrap();

    assert_eq!(
        matcher.matches("registered for ever in our temples"),
        [PhraseMatch {
            id: "forever".into(),
            start_token: 1,
            end_token: 3,
        }]
    );
    assert_eq!(
        matcher.matches("the lamps would lightup here"),
        [PhraseMatch {
            id: "lights".into(),
            start_token: 3,
            end_token: 4,
        }]
    );
}

#[test]
fn matches_separate_phrases_and_ignores_disabled_entries() {
    let mut disabled = wake_word("sleep", "go to sleep");
    disabled.enabled = false;
    let matcher = PhraseMatcher::compile(&[
        wake_word("computer", "hey computer"),
        wake_word("lights", "lights on"),
        disabled,
    ])
    .unwrap();
    assert_eq!(
        matcher.matches("Hey, computer. Please turn the LIGHTS ON!"),
        [
            PhraseMatch {
                id: "computer".into(),
                start_token: 0,
                end_token: 2,
            },
            PhraseMatch {
                id: "lights".into(),
                start_token: 5,
                end_token: 7,
            },
        ]
    );
}

#[test]
fn longest_phrase_wins_an_overlap() {
    let matcher = PhraseMatcher::compile(&[
        wake_word("short", "computer"),
        wake_word("long", "hey computer"),
    ])
    .unwrap();
    assert_eq!(
        matcher.matches("hey computer"),
        [PhraseMatch {
            id: "long".into(),
            start_token: 0,
            end_token: 2,
        }]
    );
}

#[test]
fn a_later_non_overlapping_occurrence_can_still_match() {
    let matcher = PhraseMatcher::compile(&[
        wake_word("short", "computer"),
        wake_word("long", "hey computer"),
    ])
    .unwrap();
    assert_eq!(
        matcher.matches("hey computer then computer"),
        [
            PhraseMatch {
                id: "long".into(),
                start_token: 0,
                end_token: 2,
            },
            PhraseMatch {
                id: "short".into(),
                start_token: 3,
                end_token: 4,
            },
        ]
    );
}

#[test]
fn exact_transcript_aliases_map_to_the_same_action_without_fuzzy_matching() {
    let mut atreyu = wake_word("atreyu", "hey atreyu");
    atreyu.aliases = vec!["hey a tray you".into(), "hey atre you".into()];
    let matcher = PhraseMatcher::compile(&[atreyu]).unwrap();

    assert_eq!(matcher.matches("Hey, a tray you! ")[0].id, "atreyu");
    assert_eq!(matcher.matches("hey atre you")[0].id, "atreyu");
    assert!(matcher.matches("hey atrium").is_empty());
}

#[test]
fn rejects_ambiguous_or_empty_enabled_phrases() {
    assert!(PhraseMatcher::compile(&[wake_word("empty", "---")]).is_err());
    assert!(
        PhraseMatcher::compile(&[
            wake_word("first", "Hey, Computer"),
            wake_word("second", "hey computer"),
        ])
        .is_err()
    );
    let mut duplicate_alias = wake_word("atreyu", "hey atreyu");
    duplicate_alias.aliases = vec!["hey a tre yu".into(), "hey atre yu".into()];
    assert!(PhraseMatcher::compile(&[duplicate_alias]).is_err());
    assert!(
        PhraseMatcher::compile(&[
            wake_word("first", "forever"),
            wake_word("second", "for ever"),
        ])
        .is_err()
    );
    assert!(
        PhraseMatcher::compile(&[wake_word("same", "computer"), wake_word("same", "lights"),])
            .is_err()
    );
}
