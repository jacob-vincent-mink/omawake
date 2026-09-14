use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use sentencepiece_rs::SentencePieceProcessor;

use crate::config::WakeWord;

pub struct KeywordCompiler {
    processor: SentencePieceProcessor,
}

impl KeywordCompiler {
    pub fn open(path: &Path) -> Result<Self> {
        let processor = SentencePieceProcessor::open(path)
            .with_context(|| format!("open SentencePiece model {}", path.display()))?;
        Ok(Self { processor })
    }

    pub fn compile(&self, wake_words: &[WakeWord]) -> Result<String> {
        validate_wake_words(wake_words)?;
        let mut ids = HashSet::new();
        let mut phrases = HashSet::new();
        let mut lines = Vec::new();
        for wake_word in wake_words.iter().filter(|entry| entry.enabled) {
            if !ids.insert(wake_word.id.as_str()) {
                bail!("duplicate wake-word id {}", wake_word.id);
            }
            let normalized = wake_word.phrase.trim().to_uppercase();
            if !phrases.insert(normalized.clone()) {
                bail!("duplicate wake-word phrase {}", wake_word.phrase);
            }
            let pieces = self
                .processor
                .encode(&normalized)
                .with_context(|| format!("tokenize wake phrase {}", wake_word.phrase))?;
            if pieces.is_empty() {
                bail!("wake phrase {} produced no tokens", wake_word.phrase);
            }
            lines.push(format!("{} @{}", pieces.join(" "), wake_word.id));
        }
        if lines.is_empty() {
            bail!("at least one enabled wake word is required");
        }
        Ok(lines.join("\n"))
    }
}

pub fn validate_wake_words(wake_words: &[WakeWord]) -> Result<()> {
    let mut ids = HashSet::new();
    let mut phrases = HashSet::new();
    for wake_word in wake_words {
        validate_entry(wake_word)?;
        if !ids.insert(wake_word.id.as_str()) {
            bail!("duplicate wake-word id {}", wake_word.id);
        }
        if wake_word.enabled {
            let normalized = wake_word.phrase.trim().to_uppercase();
            if !phrases.insert(normalized) {
                bail!("duplicate wake-word phrase {}", wake_word.phrase);
            }
        }
    }
    Ok(())
}

fn validate_entry(wake_word: &WakeWord) -> Result<()> {
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
    if wake_word.phrase.trim().is_empty() {
        bail!("wake-word phrase must not be empty");
    }
    if wake_word.command.is_empty() || wake_word.command[0].is_empty() {
        bail!("wake-word command must not be empty");
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/keyword.rs"]
mod tests;
