//! Keyword beam search for the exported transducer model.
//!
//! Behavioral specification: icefall's open-vocabulary KWS recipe and its
//! modified Aho-Corasick context graph:
//! https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py
//! https://github.com/k2-fsa/icefall/blob/master/icefall/context_graph.py
//! This is an independent Rust implementation over plain vectors and indexes.

use anyhow::{Context, Result, bail};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug)]
struct Node {
    token_score: f32,
    path_score: f32,
    output_score: f32,
    level: usize,
    phrase: String,
    threshold: f32,
    terminal: bool,
    next: HashMap<i64, usize>,
    fail: usize,
    output: Option<usize>,
}

impl Node {
    fn root() -> Self {
        Self {
            token_score: 0.0,
            path_score: 0.0,
            output_score: 0.0,
            level: 0,
            phrase: String::new(),
            threshold: 0.0,
            terminal: false,
            next: HashMap::new(),
            fail: 0,
            output: None,
        }
    }
}

#[derive(Debug)]
pub struct KeywordGraph {
    nodes: Vec<Node>,
}

impl KeywordGraph {
    pub fn from_buffer(buffer: &str, tokens: &str, score: f32, threshold: f32) -> Result<Self> {
        let symbols = load_symbol_ids(tokens)?;
        let mut graph = Self {
            nodes: vec![Node::root()],
        };
        for line in buffer
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            let mut fields: Vec<&str> = line.split_whitespace().collect();
            let id = fields
                .pop()
                .filter(|field| field.starts_with('@') && field.len() > 1)
                .context("compiled keyword line must end in @id")?[1..]
                .to_owned();
            let ids: Vec<i64> = fields
                .into_iter()
                .map(|symbol| {
                    symbols.get(symbol).copied().with_context(|| {
                        format!("keyword token {symbol:?} is absent from tokens.txt")
                    })
                })
                .collect::<Result<_>>()?;
            if ids.is_empty() {
                bail!("keyword {id} contains no tokens");
            }
            graph.insert(&ids, id, score, threshold);
        }
        if graph.nodes.len() == 1 {
            bail!("compiled keyword buffer contains no tokens");
        }
        graph.fill_failure_links();
        Ok(graph)
    }

    fn insert(&mut self, tokens: &[i64], phrase: String, score: f32, threshold: f32) {
        let mut current = 0;
        for (offset, &token) in tokens.iter().enumerate() {
            let terminal = offset + 1 == tokens.len();
            let next = if let Some(&next) = self.nodes[current].next.get(&token) {
                let token_score = self.nodes[next].token_score.max(score);
                self.nodes[next].token_score = token_score;
                self.nodes[next].path_score = self.nodes[current].path_score + token_score;
                self.nodes[next].terminal |= terminal;
                if self.nodes[next].terminal {
                    self.nodes[next].output_score = self.nodes[next].path_score;
                }
                if terminal {
                    self.nodes[next].phrase = phrase.clone();
                    self.nodes[next].threshold = threshold;
                }
                next
            } else {
                let next = self.nodes.len();
                let path_score = self.nodes[current].path_score + score;
                self.nodes.push(Node {
                    token_score: score,
                    path_score,
                    output_score: if terminal { path_score } else { 0.0 },
                    level: offset + 1,
                    phrase: if terminal {
                        phrase.clone()
                    } else {
                        String::new()
                    },
                    threshold: if terminal { threshold } else { 0.0 },
                    terminal,
                    next: HashMap::new(),
                    fail: 0,
                    output: None,
                });
                self.nodes[current].next.insert(token, next);
                next
            };
            current = next;
        }
    }

    fn fill_failure_links(&mut self) {
        let mut queue = VecDeque::new();
        for child in self.nodes[0].next.values().copied().collect::<Vec<_>>() {
            self.nodes[child].fail = 0;
            queue.push_back(child);
        }
        while let Some(current) = queue.pop_front() {
            let edges: Vec<(i64, usize)> = self.nodes[current]
                .next
                .iter()
                .map(|(&a, &b)| (a, b))
                .collect();
            for (token, child) in edges {
                let mut fail = self.nodes[current].fail;
                while fail != 0 && !self.nodes[fail].next.contains_key(&token) {
                    fail = self.nodes[fail].fail;
                }
                if let Some(&next) = self.nodes[fail].next.get(&token) {
                    fail = next;
                }
                self.nodes[child].fail = fail;
                let mut output = fail;
                while output != 0 && !self.nodes[output].terminal {
                    output = self.nodes[output].fail;
                }
                if self.nodes[output].terminal {
                    self.nodes[child].output = Some(output);
                    self.nodes[child].output_score += self.nodes[output].output_score;
                }
                queue.push_back(child);
            }
        }
    }

    fn advance(&self, state: usize, token: i64) -> (f32, usize) {
        if let Some(&next) = self.nodes[state].next.get(&token) {
            return (
                self.nodes[next].token_score + self.nodes[next].output_score,
                next,
            );
        }
        let mut next = self.nodes[state].fail;
        while next != 0 && !self.nodes[next].next.contains_key(&token) {
            next = self.nodes[next].fail;
        }
        if let Some(&matched) = self.nodes[next].next.get(&token) {
            next = matched;
        }
        (
            self.nodes[next].path_score - self.nodes[state].path_score
                + self.nodes[next].output_score,
            next,
        )
    }

    fn matched(&self, state: usize) -> Option<&Node> {
        if self.nodes[state].terminal {
            Some(&self.nodes[state])
        } else {
            self.nodes[state].output.map(|node| &self.nodes[node])
        }
    }
}

fn load_symbol_ids(contents: &str) -> Result<HashMap<String, i64>> {
    let mut symbols = HashMap::new();
    for line in contents.lines() {
        let (symbol, id) = line.rsplit_once(' ').context("malformed tokens.txt line")?;
        symbols.insert(symbol.to_owned(), id.parse()?);
    }
    Ok(symbols)
}

#[derive(Clone, Debug)]
struct Hypothesis {
    tokens: Vec<i64>,
    score: f32,
    acoustic_probabilities: Vec<f32>,
    timestamps: Vec<usize>,
    context: usize,
    trailing_blanks: usize,
}

#[derive(Clone, Debug)]
pub struct Match {
    pub id: String,
    pub tokens: Vec<i64>,
    pub timestamps: Vec<usize>,
}

#[derive(Debug)]
pub struct KeywordBeam {
    hypotheses: Vec<Hypothesis>,
    width: usize,
    required_trailing_blanks: usize,
}

impl KeywordBeam {
    pub fn new(width: usize, required_trailing_blanks: usize) -> Self {
        let mut beam = Self {
            hypotheses: Vec::new(),
            width,
            required_trailing_blanks,
        };
        beam.reset();
        beam
    }

    pub fn reset(&mut self) {
        self.hypotheses = vec![Hypothesis {
            tokens: vec![-1, 0],
            score: 0.0,
            acoustic_probabilities: Vec::new(),
            timestamps: Vec::new(),
            context: 0,
            trailing_blanks: 0,
        }];
    }

    pub fn contexts(&self) -> Vec<i64> {
        self.hypotheses
            .iter()
            .flat_map(|hypothesis| {
                hypothesis.tokens[hypothesis.tokens.len() - 2..]
                    .iter()
                    .copied()
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    pub fn trailing_blanks(&self) -> usize {
        self.best(false).trailing_blanks
    }

    pub fn advance(
        &mut self,
        logits: &[f32],
        frame: usize,
        graph: &KeywordGraph,
    ) -> Result<Option<Match>> {
        let paths = self.hypotheses.len();
        if logits.len() != paths * 500 {
            bail!(
                "joiner returned {} logits for {paths} hypotheses",
                logits.len()
            );
        }
        let mut probabilities = Vec::with_capacity(logits.len());
        let mut candidates = Vec::with_capacity(logits.len());
        let (rows, remainder) = logits.as_chunks::<500>();
        debug_assert!(remainder.is_empty());
        for (path, row) in rows.iter().enumerate() {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let normalizer = row.iter().map(|value| (*value - max).exp()).sum::<f32>();
            for (token, &value) in row.iter().enumerate() {
                let probability = (value - max).exp() / normalizer;
                probabilities.push(probability);
                candidates.push((self.hypotheses[path].score + probability.ln(), path, token));
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut merged: HashMap<Vec<i64>, Hypothesis> = HashMap::new();
        for &(candidate_score, path, token) in candidates.iter().take(self.width) {
            let mut hypothesis = self.hypotheses[path].clone();
            let mut context_score = 0.0;
            if token != 0 && token != 2 {
                hypothesis.tokens.push(token as i64);
                hypothesis.timestamps.push(frame);
                hypothesis
                    .acoustic_probabilities
                    .push(probabilities[path * 500 + token]);
                let (score, context) = graph.advance(hypothesis.context, token as i64);
                context_score = score;
                hypothesis.context = context;
                hypothesis.trailing_blanks = 0;
                if context == 0 {
                    hypothesis.tokens = vec![-1, 0];
                    hypothesis.timestamps.clear();
                    hypothesis.acoustic_probabilities.clear();
                }
            } else {
                hypothesis.trailing_blanks += 1;
            }
            hypothesis.score = candidate_score + context_score;
            merged
                .entry(hypothesis.tokens.clone())
                .and_modify(|existing| existing.score = log_add(existing.score, hypothesis.score))
                .or_insert(hypothesis);
        }
        self.hypotheses = merged.into_values().collect();

        let best = self.best(false);
        let Some(matched) = graph.matched(best.context) else {
            return Ok(None);
        };
        if best.trailing_blanks <= self.required_trailing_blanks {
            return Ok(None);
        }
        let begin = best
            .acoustic_probabilities
            .len()
            .checked_sub(matched.level)
            .context("matched keyword is longer than hypothesis acoustic history")?;
        let average =
            best.acoustic_probabilities[begin..].iter().sum::<f32>() / matched.level as f32;
        if average < matched.threshold {
            return Ok(None);
        }
        let token_begin = best.tokens.len() - matched.level;
        let timestamp_begin = best.timestamps.len() - matched.level;
        let result = Match {
            id: matched.phrase.clone(),
            tokens: best.tokens[token_begin..].to_vec(),
            timestamps: best.timestamps[timestamp_begin..].to_vec(),
        };
        self.reset();
        Ok(Some(result))
    }

    fn best(&self, normalize_length: bool) -> &Hypothesis {
        self.hypotheses
            .iter()
            .max_by(|a, b| {
                let a_score = if normalize_length {
                    a.score / a.tokens.len() as f32
                } else {
                    a.score
                };
                let b_score = if normalize_length {
                    b.score / b.tokens.len() as f32
                } else {
                    b.score
                };
                a_score.total_cmp(&b_score)
            })
            .expect("beam is never empty")
    }
}

fn log_add(a: f32, b: f32) -> f32 {
    let high = a.max(b);
    high + ((a - high).exp() + (b - high).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(token: usize) -> Vec<f32> {
        let mut logits = vec![-20.0; 500];
        logits[token] = 20.0;
        logits
    }

    #[test]
    fn emits_keyword_only_after_required_trailing_blanks() {
        let graph =
            KeywordGraph::from_buffer("A B @wake", "<blk> 0\n<unk> 2\nA 3\nB 4\n", 1.5, 0.25)
                .unwrap();
        let mut beam = KeywordBeam::new(1, 1);
        assert!(beam.advance(&logits(3), 10, &graph).unwrap().is_none());
        assert!(beam.advance(&logits(4), 11, &graph).unwrap().is_none());
        assert!(beam.advance(&logits(0), 12, &graph).unwrap().is_none());
        let matched = beam.advance(&logits(0), 13, &graph).unwrap().unwrap();
        assert_eq!(matched.id, "wake");
        assert_eq!(matched.tokens, [3, 4]);
        assert_eq!(matched.timestamps, [10, 11]);
    }

    #[test]
    fn rejects_malformed_compiled_keywords() {
        assert!(KeywordGraph::from_buffer("A", "A 3\n", 1.5, 0.25).is_err());
        assert!(KeywordGraph::from_buffer("MISSING @wake", "A 3\n", 1.5, 0.25).is_err());
    }
}
