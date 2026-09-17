use anyhow::{Result, bail};
use std::collections::HashSet;

use crate::config::WakeWord;
use crate::phrase::normalize_tokens;

pub fn validate_wake_words(wake_words: &[WakeWord]) -> Result<()> {
    let mut ids = HashSet::new();
    let mut phrases = HashSet::new();
    for wake_word in wake_words {
        validate_entry(wake_word)?;
        if !ids.insert(wake_word.id.as_str()) {
            bail!("duplicate wake-word id {}", wake_word.id);
        }
        if wake_word.enabled {
            for variant in std::iter::once(&wake_word.phrase).chain(&wake_word.aliases) {
                let normalized = normalize_tokens(variant).concat();
                if !phrases.insert(normalized) {
                    bail!("duplicate wake-word phrase or alias {variant}");
                }
            }
        }
    }
    Ok(())
}

fn validate_entry(wake_word: &WakeWord) -> Result<()> {
    if let Some(binding) = &wake_word.enrollment {
        binding.validate()?;
    }
    let valid_id = !wake_word.id.is_empty()
        && wake_word
            .id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && !wake_word.id.starts_with('-')
        && !wake_word.id.ends_with('-');
    if !valid_id {
        bail!("wake-word id must be a lowercase slug: {}", wake_word.id);
    }
    if normalize_tokens(&wake_word.phrase).is_empty() {
        bail!("wake-word phrase must contain letters or numbers");
    }
    if wake_word
        .aliases
        .iter()
        .any(|alias| normalize_tokens(alias).is_empty())
    {
        bail!("wake-word aliases must contain letters or numbers");
    }
    if wake_word.command.is_empty() || wake_word.command[0].is_empty() {
        bail!("wake-word command must not be empty");
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/keyword.rs"]
mod tests;
