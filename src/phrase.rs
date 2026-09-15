//! Provider-independent matching of verifier transcripts to configured phrases.

use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, Ordering};
use unicode_normalization::UnicodeNormalization;

use crate::config::WakeWord;

static SHOW_TRANSCRIPTS: AtomicBool = AtomicBool::new(false);

pub(crate) struct TranscriptDiagnosticsGuard(bool);

pub(crate) fn enable_transcript_diagnostics() -> TranscriptDiagnosticsGuard {
    TranscriptDiagnosticsGuard(SHOW_TRANSCRIPTS.swap(true, Ordering::Relaxed))
}

pub(crate) fn record_transcript(transcript: &str) {
    if SHOW_TRANSCRIPTS.load(Ordering::Relaxed) {
        eprintln!("verifier transcript: {transcript:?}");
    }
}

impl Drop for TranscriptDiagnosticsGuard {
    fn drop(&mut self) {
        SHOW_TRANSCRIPTS.store(self.0, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhraseMatch {
    pub id: String,
    pub start_token: usize,
    pub end_token: usize,
}

#[derive(Clone, Debug)]
struct CompiledPhrase {
    id: String,
    canonical: String,
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
            if phrases
                .iter()
                .any(|existing: &CompiledPhrase| existing.id == wake_word.id)
            {
                bail!("duplicate enabled wake-word id {:?}", wake_word.id);
            }
            for variant in std::iter::once(&wake_word.phrase).chain(&wake_word.aliases) {
                let tokens = normalize_tokens(variant);
                if tokens.is_empty() {
                    bail!(
                        "enabled wake-word phrase or alias for id {:?} contains no letters or numbers",
                        wake_word.id
                    );
                }
                let canonical = tokens.concat();
                if phrases
                    .iter()
                    .any(|existing| existing.canonical == canonical)
                {
                    bail!(
                        "enabled wake-word phrase or alias {:?} normalizes to the same text as another variant, including word boundaries",
                        variant
                    );
                }
                phrases.push(CompiledPhrase {
                    id: wake_word.id.clone(),
                    canonical,
                    order,
                });
            }
        }
        if phrases.is_empty() {
            bail!("at least one wake word must be enabled");
        }
        Ok(Self { phrases })
    }

    /// Match complete phrase text in a verifier transcript. Word boundaries
    /// are ignored because ASR output can render the same speech as, for
    /// example, either `forever` or `for ever`. Character boundaries are still
    /// exact, so `computer` does not match `computerized`.
    ///
    /// Each configured phrase is emitted at most once. When configured phrases
    /// overlap, the phrase with more tokens wins; configuration order breaks a
    /// tie. Separate, non-overlapping phrases can both fire from one utterance.
    pub fn matches(&self, transcript: &str) -> Vec<PhraseMatch> {
        let transcript = normalize_tokens(transcript);
        let mut candidates = Vec::new();
        for phrase in &self.phrases {
            for start_token in 0..transcript.len() {
                let mut canonical = String::new();
                for (offset, token) in transcript[start_token..].iter().enumerate() {
                    canonical.push_str(token);
                    if canonical == phrase.canonical {
                        candidates.push((phrase, start_token, start_token + offset + 1));
                        break;
                    }
                    if canonical.len() >= phrase.canonical.len() {
                        break;
                    }
                }
            }
        }

        candidates.sort_by(|(left, left_start, _), (right, right_start, _)| {
            left_start
                .cmp(right_start)
                .then_with(|| right.canonical.len().cmp(&left.canonical.len()))
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
