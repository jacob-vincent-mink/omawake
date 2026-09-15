//! Provider-independent matching of verifier transcripts to configured phrases.

use anyhow::{Result, bail};
use unicode_normalization::UnicodeNormalization;

use crate::config::WakeWord;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhraseMatch {
    pub id: String,
    pub start_token: usize,
    pub end_token: usize,
}

#[derive(Clone, Debug)]
struct CompiledPhrase {
    id: String,
    tokens: Vec<String>,
    order: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PhraseMatcher {
    phrases: Vec<CompiledPhrase>,
}

impl PhraseMatcher {
    pub fn compile(wake_words: &[WakeWord]) -> Result<Self> {
        let mut phrases = Vec::new();
        for (order, wake_word) in wake_words.iter().enumerate() {
            if !wake_word.enabled {
                continue;
            }
            let tokens = normalize_tokens(&wake_word.phrase);
            if tokens.is_empty() {
                bail!(
                    "enabled wake-word phrase for id {:?} contains no letters or numbers",
                    wake_word.id
                );
            }
            if phrases
                .iter()
                .any(|existing: &CompiledPhrase| existing.id == wake_word.id)
            {
                bail!("duplicate enabled wake-word id {:?}", wake_word.id);
            }
            if phrases.iter().any(|existing| existing.tokens == tokens) {
                bail!(
                    "enabled wake-word phrase {:?} normalizes to the same phrase as another entry",
                    wake_word.phrase
                );
            }
            phrases.push(CompiledPhrase {
                id: wake_word.id.clone(),
                tokens,
                order,
            });
        }
        if phrases.is_empty() {
            bail!("at least one wake word must be enabled");
        }
        Ok(Self { phrases })
    }

    /// Match complete token sequences in a verifier transcript.
    ///
    /// Each configured phrase is emitted at most once. When configured phrases
    /// overlap, the phrase with more tokens wins; configuration order breaks a
    /// tie. Separate, non-overlapping phrases can both fire from one utterance.
    pub fn matches(&self, transcript: &str) -> Vec<PhraseMatch> {
        let transcript = normalize_tokens(transcript);
        let mut candidates = Vec::new();
        for phrase in &self.phrases {
            if phrase.tokens.len() > transcript.len() {
                continue;
            }
            for (start_token, window) in transcript.windows(phrase.tokens.len()).enumerate() {
                if window == phrase.tokens {
                    candidates.push((phrase, start_token, start_token + phrase.tokens.len()));
                }
            }
        }

        candidates.sort_by(|(left, left_start, _), (right, right_start, _)| {
            left_start
                .cmp(right_start)
                .then_with(|| right.tokens.len().cmp(&left.tokens.len()))
                .then_with(|| left.order.cmp(&right.order))
        });

        let mut accepted: Vec<PhraseMatch> = Vec::new();
        for (phrase, start_token, end_token) in candidates {
            if accepted.iter().any(|item| item.id == phrase.id) {
                continue;
            }
            let overlaps = accepted
                .iter()
                .any(|item| item.start_token < end_token && start_token < item.end_token);
            if !overlaps {
                accepted.push(PhraseMatch {
                    id: phrase.id.clone(),
                    start_token,
                    end_token,
                });
            }
        }
        accepted
    }
}

pub fn normalize_tokens(input: &str) -> Vec<String> {
    let normalized: String = input.nfkc().collect();
    let mut tokens = Vec::new();
    let mut token = String::new();
    for character in normalized.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            token.push(character);
        } else if !token.is_empty() {
            tokens.push(std::mem::take(&mut token));
        }
    }
    if !token.is_empty() {
        tokens.push(token);
    }
    tokens
}

#[cfg(test)]
#[path = "../tests/unit/phrase.rs"]
mod tests;
