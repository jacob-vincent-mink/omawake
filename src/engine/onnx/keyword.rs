//! Open-vocabulary keyword beam search following icefall's KWS decoder semantics.
//!
//! This implementation was authored from the Apache-2.0 icefall sources at
//! commit `aac7df064a6d1529f3bf4acccc6c550bd260b7b3` (`context_graph.py` and
//! `keywords_search` in the GigaSpeech Zipformer recipe).

use std::collections::{HashMap, VecDeque};

use anyhow::{Context, Result, bail};

const DECODER_CONTEXT: usize = 2;
const ICEFALL_BLANK: i64 = 0;

#[derive(Clone, Debug)]
struct KeywordEnd {
    id: String,
    acoustic_threshold: f32,
}

#[derive(Clone, Debug)]
struct State {
    next: HashMap<i64, usize>,
    failure: usize,
    output: Option<usize>,
    token_score: f32,
    path_score: f32,
    output_score: f32,
    depth: usize,
    keyword: Option<KeywordEnd>,
}

impl State {
    fn root() -> Self {
        Self {
            next: HashMap::new(),
            failure: 0,
            output: None,
            token_score: 0.0,
            path_score: 0.0,
            output_score: 0.0,
            depth: 0,
            keyword: None,
        }
    }

    fn child(token_score: f32, path_score: f32, depth: usize) -> Self {
        Self {
            next: HashMap::new(),
            failure: 0,
            output: None,
            token_score,
            path_score,
            output_score: 0.0,
            depth,
            keyword: None,
        }
    }

    fn is_terminal(&self) -> bool {
        self.keyword.is_some()
    }
}

/// A trie with Aho-Corasick failure and output links plus icefall context scores.
#[derive(Clone, Debug)]
pub struct KeywordGraph {
    states: Vec<State>,
    vocabulary_size: usize,
    unknown_id: i64,
}

impl KeywordGraph {
    /// Builds a keyword graph from compiled lines such as `▁HEL LO @hello` and
    /// a `tokens.txt` symbol table.
    pub fn from_buffer(buffer: &str, tokens: &str, score: f32, threshold: f32) -> Result<Self> {
        if !score.is_finite() {
            bail!("keyword score must be finite");
        }
        if !threshold.is_finite() {
            bail!("keyword acoustic threshold must be finite");
        }

        let (symbols, vocabulary_size) = parse_symbol_table(tokens)?;
        let blank_id = symbols
            .get("<blk>")
            .copied()
            .context("tokens.txt does not define <blk>")?;
        if blank_id != ICEFALL_BLANK {
            bail!(
                "the icefall decoder interface requires <blk> id {ICEFALL_BLANK}, got {blank_id}"
            );
        }
        // The reference decoder falls back to treating only blank as unknown
        // when the model does not expose a separate unknown ID.
        let unknown_id = symbols.get("<unk>").copied().unwrap_or(blank_id);

        let mut graph = Self {
            states: vec![State::root()],
            vocabulary_size,
            unknown_id,
        };
        let mut keyword_count = 0usize;

        for (line_index, raw_line) in buffer.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let mut fields: Vec<&str> = line.split_whitespace().collect();
            let tagged_id = fields
                .pop()
                .with_context(|| format!("keyword definition {} has no fields", line_index + 1))?;
            let id = tagged_id
                .strip_prefix('@')
                .filter(|id| !id.is_empty())
                .with_context(|| {
                    format!(
                        "keyword definition {} needs a trailing @name",
                        line_index + 1
                    )
                })?;
            if fields.is_empty() {
                bail!("keyword @{id} has no acoustic symbols");
            }

            let token_ids = fields
                .iter()
                .map(|symbol| {
                    symbols.get(*symbol).copied().with_context(|| {
                        format!("keyword definition references undefined symbol {symbol:?}")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            graph.insert_keyword(&token_ids, id, score, threshold);
            keyword_count += 1;
        }
        if keyword_count == 0 {
            bail!("no keyword definitions were supplied");
        }

        graph.build_failure_links();
        Ok(graph)
    }

    fn insert_keyword(&mut self, tokens: &[i64], id: &str, score: f32, threshold: f32) {
        let mut state = 0usize;
        for (offset, &token) in tokens.iter().enumerate() {
            let existing = self.states[state].next.get(&token).copied();
            let next = match existing {
                Some(next) => {
                    // icefall assigns the maximum per-token boost to a shared trie arc.
                    let token_score = self.states[next].token_score.max(score);
                    self.states[next].token_score = token_score;
                    self.states[next].path_score = self.states[state].path_score + token_score;
                    next
                }
                None => {
                    let next = self.states.len();
                    let path_score = self.states[state].path_score + score;
                    self.states
                        .push(State::child(score, path_score, offset + 1));
                    self.states[state].next.insert(token, next);
                    next
                }
            };

            if offset + 1 == tokens.len() {
                self.states[next].keyword = Some(KeywordEnd {
                    id: id.to_owned(),
                    acoustic_threshold: threshold,
                });
                self.states[next].output_score = self.states[next].path_score;
            } else if !self.states[next].is_terminal() {
                self.states[next].output_score = 0.0;
            }
            state = next;
        }
    }

    fn build_failure_links(&mut self) {
        let mut queue = VecDeque::new();
        let root_children: Vec<usize> = self.states[0].next.values().copied().collect();
        for child in root_children {
            self.states[child].failure = 0;
            queue.push_back(child);
        }

        while let Some(parent) = queue.pop_front() {
            let edges: Vec<(i64, usize)> = self.states[parent]
                .next
                .iter()
                .map(|(&token, &child)| (token, child))
                .collect();
            for (token, child) in edges {
                let mut fallback = self.states[parent].failure;
                while fallback != 0 && !self.states[fallback].next.contains_key(&token) {
                    fallback = self.states[fallback].failure;
                }
                let failure = self.states[fallback].next.get(&token).copied().unwrap_or(0);
                self.states[child].failure = failure;

                let output = if self.states[failure].is_terminal() {
                    Some(failure)
                } else {
                    self.states[failure].output
                };
                self.states[child].output = output;
                if let Some(output) = output {
                    self.states[child].output_score += self.states[output].output_score;
                }
                queue.push_back(child);
            }
        }
    }

    fn step(&self, state: usize, token: i64) -> (f32, usize) {
        if let Some(&next) = self.states[state].next.get(&token) {
            let score = self.states[next].token_score + self.states[next].output_score;
            return (score, next);
        }

        let mut fallback = self.states[state].failure;
        while fallback != 0 && !self.states[fallback].next.contains_key(&token) {
            fallback = self.states[fallback].failure;
        }
        let next = self.states[fallback].next.get(&token).copied().unwrap_or(0);
        let score = self.states[next].path_score - self.states[state].path_score
            + self.states[next].output_score;
        (score, next)
    }

    fn matched_state(&self, state: usize) -> Option<usize> {
        if self.states[state].is_terminal() {
            Some(state)
        } else {
            self.states[state].output
        }
    }
}

fn parse_symbol_table(tokens: &str) -> Result<(HashMap<String, i64>, usize)> {
    let mut symbols = HashMap::new();
    let mut ids = HashMap::<i64, String>::new();
    let mut maximum_id = None::<usize>;

    for (line_index, raw_line) in tokens.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let split = line
            .char_indices()
            .rev()
            .find(|(_, character)| character.is_whitespace())
            .map(|(index, _)| index)
            .with_context(|| format!("malformed tokens.txt line {}", line_index + 1))?;
        let symbol = line[..split].trim_end();
        let encoded_id = line[split..].trim();
        if symbol.is_empty() || encoded_id.is_empty() {
            bail!("malformed tokens.txt line {}", line_index + 1);
        }
        let id: i64 = encoded_id
            .parse()
            .with_context(|| format!("invalid token id on tokens.txt line {}", line_index + 1))?;
        let index = usize::try_from(id)
            .with_context(|| format!("negative token id on tokens.txt line {}", line_index + 1))?;
        if symbols.insert(symbol.to_owned(), id).is_some() {
            bail!("duplicate token symbol {symbol:?}");
        }
        if let Some(previous) = ids.insert(id, symbol.to_owned()) {
            bail!("token id {id} is assigned to both {previous:?} and {symbol:?}");
        }
        maximum_id = Some(maximum_id.map_or(index, |old| old.max(index)));
    }

    let vocabulary_size = maximum_id
        .and_then(|id| id.checked_add(1))
        .context("tokens.txt contains no usable token ids")?;
    Ok((symbols, vocabulary_size))
}

#[derive(Clone, Debug)]
struct Hypothesis {
    sequence: Vec<i64>,
    log_probability: f32,
    acoustic_probabilities: Vec<f32>,
    timestamps: Vec<usize>,
    graph_state: usize,
    trailing_blanks: usize,
}

impl Hypothesis {
    fn initial() -> Self {
        Self {
            sequence: vec![-1, ICEFALL_BLANK],
            log_probability: 0.0,
            acoustic_probabilities: Vec::new(),
            timestamps: Vec::new(),
            graph_state: 0,
            trailing_blanks: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    pub id: String,
    pub tokens: Vec<i64>,
    pub timestamps: Vec<usize>,
}

/// Stateful one-utterance keyword beam.
pub struct KeywordBeam {
    width: usize,
    required_trailing_blanks: usize,
    hypotheses: Vec<Hypothesis>,
}

impl KeywordBeam {
    pub fn new(width: usize, required_trailing_blanks: usize) -> Self {
        Self {
            width,
            required_trailing_blanks,
            hypotheses: vec![Hypothesis::initial()],
        }
    }

    pub fn reset(&mut self) {
        self.hypotheses.clear();
        self.hypotheses.push(Hypothesis::initial());
    }

    pub fn contexts(&self) -> Vec<i64> {
        self.hypotheses
            .iter()
            .flat_map(|hypothesis| {
                hypothesis.sequence[hypothesis.sequence.len() - DECODER_CONTEXT..]
                    .iter()
                    .copied()
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    pub fn trailing_blanks(&self) -> usize {
        self.best_index()
            .map(|index| self.hypotheses[index].trailing_blanks)
            .unwrap_or(0)
    }

    pub fn advance(
        &mut self,
        logits: &[f32],
        frame: usize,
        graph: &KeywordGraph,
    ) -> Result<Option<Match>> {
        if self.width == 0 {
            bail!("keyword beam width must be greater than zero");
        }
        let expected = self
            .hypotheses
            .len()
            .checked_mul(graph.vocabulary_size)
            .context("keyword beam logit count overflow")?;
        if logits.len() != expected {
            bail!(
                "keyword beam expected {expected} logits ({} paths x {} tokens), got {}",
                self.hypotheses.len(),
                graph.vocabulary_size,
                logits.len()
            );
        }

        let mut expansions = Vec::with_capacity(expected);
        for (hypothesis_index, (hypothesis, row)) in self
            .hypotheses
            .iter()
            .zip(logits.chunks_exact(graph.vocabulary_size))
            .enumerate()
        {
            let normalizer = log_sum_exp(row)?;
            for (token, &logit) in row.iter().enumerate() {
                let acoustic_log_probability = logit - normalizer;
                expansions.push(Expansion {
                    rank_score: hypothesis.log_probability + acoustic_log_probability,
                    acoustic_probability: acoustic_log_probability.exp(),
                    hypothesis_index,
                    token: token as i64,
                    flat_index: hypothesis_index * graph.vocabulary_size + token,
                });
            }
        }
        expansions.sort_by(|left, right| {
            right
                .rank_score
                .total_cmp(&left.rank_score)
                .then_with(|| left.flat_index.cmp(&right.flat_index))
        });
        expansions.truncate(self.width.min(expansions.len()));

        // HypothesisList in the reference keys paths by the complete token sequence.
        // Preserve the first (highest-ranked) path's metadata while log-adding scores.
        let mut merged = Vec::<Hypothesis>::with_capacity(expansions.len());
        let mut positions = HashMap::<Vec<i64>, usize>::new();
        for expansion in expansions {
            let source = &self.hypotheses[expansion.hypothesis_index];
            let mut candidate = source.clone();
            candidate.log_probability = expansion.rank_score;

            if expansion.token == ICEFALL_BLANK || expansion.token == graph.unknown_id {
                candidate.trailing_blanks = candidate.trailing_blanks.saturating_add(1);
            } else {
                candidate.sequence.push(expansion.token);
                candidate.timestamps.push(frame);
                candidate
                    .acoustic_probabilities
                    .push(expansion.acoustic_probability);
                let (context_score, next_state) =
                    graph.step(candidate.graph_state, expansion.token);
                candidate.log_probability += context_score;
                candidate.graph_state = next_state;
                candidate.trailing_blanks = 0;

                // This reset is part of the KWS recipe: a token that leaves the
                // context graph at root also resets the transducer's two-token input.
                if next_state == 0 {
                    let tail = candidate.sequence.len() - DECODER_CONTEXT;
                    candidate.sequence[tail..].copy_from_slice(&[-1, ICEFALL_BLANK]);
                }
            }

            if let Some(&position) = positions.get(&candidate.sequence) {
                merged[position].log_probability =
                    log_add_exp(merged[position].log_probability, candidate.log_probability);
            } else {
                let position = merged.len();
                positions.insert(candidate.sequence.clone(), position);
                merged.push(candidate);
            }
        }
        self.hypotheses = merged;

        let result = self.accepted_match(graph, true)?;
        if result.is_some() {
            self.reset();
        }
        Ok(result)
    }

    /// Performs icefall's end-of-input keyword check.
    ///
    /// Unlike [`Self::advance`], the final check has no trailing-blank gate. It
    /// still selects the length-normalized best path and applies the keyword's
    /// mean acoustic-probability threshold.
    pub fn finish(&mut self, graph: &KeywordGraph) -> Result<Option<Match>> {
        let result = self.accepted_match(graph, false)?;
        // A finished beam must not return the same terminal match a second time.
        self.reset();
        Ok(result)
    }

    fn accepted_match(
        &self,
        graph: &KeywordGraph,
        require_trailing_blanks: bool,
    ) -> Result<Option<Match>> {
        let best_index = self
            .best_index()
            .context("keyword beam produced no hypotheses")?;
        let best = &self.hypotheses[best_index];
        let Some(matched_state) = graph.matched_state(best.graph_state) else {
            return Ok(None);
        };
        let matched = &graph.states[matched_state];
        let keyword = matched
            .keyword
            .as_ref()
            .context("context graph output link does not point to a keyword")?;
        if require_trailing_blanks && best.trailing_blanks <= self.required_trailing_blanks {
            return Ok(None);
        }
        if matched.depth > best.acoustic_probabilities.len()
            || matched.depth > best.timestamps.len()
            || matched.depth > best.sequence.len()
        {
            bail!("context depth exceeds the retained emission history");
        }
        let probability_start = best.acoustic_probabilities.len() - matched.depth;
        let average_acoustic_probability = best.acoustic_probabilities[probability_start..]
            .iter()
            .sum::<f32>()
            / matched.depth as f32;
        if average_acoustic_probability < keyword.acoustic_threshold {
            return Ok(None);
        }

        let token_start = best.sequence.len() - matched.depth;
        let time_start = best.timestamps.len() - matched.depth;
        let result = Match {
            id: keyword.id.clone(),
            tokens: best.sequence[token_start..].to_vec(),
            timestamps: best.timestamps[time_start..].to_vec(),
        };
        Ok(Some(result))
    }

    fn best_index(&self) -> Option<usize> {
        let mut best = None::<(usize, f32)>;
        for (index, hypothesis) in self.hypotheses.iter().enumerate() {
            let normalized = hypothesis.log_probability / hypothesis.sequence.len() as f32;
            if best.is_none_or(|(_, best_score)| normalized > best_score) {
                best = Some((index, normalized));
            }
        }
        best.map(|(index, _)| index)
    }
}

struct Expansion {
    rank_score: f32,
    acoustic_probability: f32,
    hypothesis_index: usize,
    token: i64,
    flat_index: usize,
}

fn log_sum_exp(values: &[f32]) -> Result<f32> {
    let maximum = values
        .iter()
        .copied()
        .try_fold(f32::NEG_INFINITY, |maximum, value| {
            if value.is_finite() {
                Ok(maximum.max(value))
            } else {
                bail!("keyword logits must be finite")
            }
        })?;
    let scaled_sum = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f32>();
    Ok(maximum + scaled_sum.ln())
}

fn log_add_exp(left: f32, right: f32) -> f32 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENS: &str = "<blk> 0\nunused 1\n<unk> 2\nA 3\nB 4\nS 5\n";

    fn peaked(vocabulary_size: usize, token: usize) -> Vec<f32> {
        let mut logits = vec![-12.0; vocabulary_size];
        logits[token] = 12.0;
        logits
    }

    #[test]
    fn emits_after_strictly_more_than_the_requested_trailing_blanks() {
        let graph = KeywordGraph::from_buffer("S A B @signal", TOKENS, 1.25, 0.2).unwrap();
        let mut beam = KeywordBeam::new(1, 2);

        assert!(beam.advance(&peaked(6, 5), 41, &graph).unwrap().is_none());
        assert!(beam.advance(&peaked(6, 3), 43, &graph).unwrap().is_none());
        assert!(beam.advance(&peaked(6, 4), 47, &graph).unwrap().is_none());
        assert!(beam.advance(&peaked(6, 0), 50, &graph).unwrap().is_none());
        assert!(beam.advance(&peaked(6, 0), 51, &graph).unwrap().is_none());
        let found = beam.advance(&peaked(6, 0), 52, &graph).unwrap().unwrap();

        assert_eq!(found.id, "signal");
        assert_eq!(found.tokens, vec![5, 3, 4]);
        assert_eq!(found.timestamps, vec![41, 43, 47]);
        assert_eq!(beam.contexts(), vec![-1, 0]);
    }

    #[test]
    fn finish_accepts_a_terminal_keyword_without_a_blank_frame() {
        let graph = KeywordGraph::from_buffer("S B @eof", TOKENS, 1.0, 0.5).unwrap();
        let mut beam = KeywordBeam::new(1, 9);
        assert!(beam.advance(&peaked(6, 5), 70, &graph).unwrap().is_none());
        assert!(beam.advance(&peaked(6, 4), 71, &graph).unwrap().is_none());

        let found = beam.finish(&graph).unwrap().unwrap();
        assert_eq!(found.id, "eof");
        assert_eq!(found.tokens, vec![5, 4]);
        assert_eq!(found.timestamps, vec![70, 71]);
        assert!(beam.finish(&graph).unwrap().is_none());
    }

    #[test]
    fn graph_refunds_an_unfinished_prefix_and_prefers_the_longest_output() {
        let graph = KeywordGraph::from_buffer("A B @ab\nB @b", TOKENS, 1.5, 0.2).unwrap();
        let (prefix_score, prefix) = graph.step(0, 3);
        let (refund_score, root) = graph.step(prefix, 1);
        assert_eq!(prefix_score, 1.5);
        assert_eq!(refund_score, -1.5);
        assert_eq!(root, 0);

        let (_, prefix) = graph.step(0, 3);
        let (completion_score, completed) = graph.step(prefix, 4);
        assert_eq!(completion_score, 6.0); // arc 1.5 + AB output 3.0 + B output 1.5
        let terminal = graph.matched_state(completed).unwrap();
        assert_eq!(graph.states[terminal].keyword.as_ref().unwrap().id, "ab");
    }

    #[test]
    fn finish_rejects_a_terminal_path_below_its_acoustic_threshold() {
        let graph = KeywordGraph::from_buffer("A @quiet", TOKENS, 100.0, 0.9).unwrap();
        let mut beam = KeywordBeam::new(1, 0);
        let mut weak_a = vec![0.0; 6];
        weak_a[3] = 1.0;
        assert!(beam.advance(&weak_a, 1, &graph).unwrap().is_none());
        assert!(beam.finish(&graph).unwrap().is_none());
    }

    #[test]
    fn finish_returns_none_for_an_unfinished_prefix() {
        let graph = KeywordGraph::from_buffer("A B @complete", TOKENS, 1.0, 0.2).unwrap();
        let mut beam = KeywordBeam::new(1, 0);
        assert!(beam.advance(&peaked(6, 3), 8, &graph).unwrap().is_none());
        assert!(beam.finish(&graph).unwrap().is_none());
    }

    #[test]
    fn most_probable_hypothesis_is_length_normalized() {
        let beam = KeywordBeam {
            width: 2,
            required_trailing_blanks: 0,
            hypotheses: vec![
                Hypothesis {
                    sequence: vec![-1, 0],
                    log_probability: -2.0,
                    acoustic_probabilities: vec![],
                    timestamps: vec![],
                    graph_state: 0,
                    trailing_blanks: 1,
                },
                Hypothesis {
                    sequence: vec![-1, 0, 3, 4],
                    log_probability: -3.0,
                    acoustic_probabilities: vec![1.0, 1.0],
                    timestamps: vec![1, 2],
                    graph_state: 0,
                    trailing_blanks: 7,
                },
            ],
        };
        assert_eq!(beam.best_index(), Some(1));
        assert_eq!(beam.trailing_blanks(), 7);
    }

    #[test]
    fn derives_vocabulary_size_and_rejects_bad_compiled_input() {
        let graph = KeywordGraph::from_buffer("A @wake", TOKENS, 1.0, 0.25).unwrap();
        assert_eq!(graph.vocabulary_size, 6);
        let mut beam = KeywordBeam::new(1, 0);
        assert!(beam.advance(&[0.0; 5], 0, &graph).is_err());
        assert!(KeywordGraph::from_buffer("NO_SUCH_SYMBOL @x", TOKENS, 1.0, 0.25).is_err());
        assert!(KeywordGraph::from_buffer("B missing-tag", TOKENS, 1.0, 0.25).is_err());
        assert!(KeywordGraph::from_buffer("A @wake", "A 3\n", 1.0, 0.25).is_err());
    }
}
