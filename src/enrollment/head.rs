//! Independently implemented bounded logistic-regression heads. The frozen
//! encoder supplies content embeddings; this module never learns an encoder.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_DIMENSIONS: usize = 4096;
const MAX_EXAMPLES: usize = 4096;
const EPOCHS: usize = 600;

#[derive(Clone, Debug)]
pub struct Example {
    /// Fingerprint of the source recording, used to enforce split separation.
    pub id: String,
    pub values: Vec<f32>,
    pub positive: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Head {
    pub format_version: u32,
    pub encoder_contract: String,
    pub dimensions: usize,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub threshold: f32,
    pub training_ids: Vec<String>,
    pub calibration_ids: Vec<String>,
    pub validation_ids: Vec<String>,
    pub training_positives: usize,
    pub calibration_positives: usize,
    pub validation: Option<Validation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Validation {
    pub positives: usize,
    pub negatives: usize,
    pub misses: usize,
    pub false_activations: usize,
    pub minimum_margin: f32,
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn normalized(values: &[f32]) -> Result<Vec<f64>> {
    ensure!(
        (1..=MAX_DIMENSIONS).contains(&values.len()),
        "embedding dimensions must be 1..={MAX_DIMENSIONS}"
    );
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "embedding contains non-finite values"
    );
    let norm = values
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && norm > 1e-12,
        "embedding has no finite nonzero content"
    );
    Ok(values.iter().map(|x| *x as f64 / norm).collect())
}

fn examples(samples: &[Example], dimensions: usize) -> Result<Vec<Vec<f64>>> {
    ensure!(
        (4..=MAX_EXAMPLES).contains(&samples.len()),
        "each split needs at least four examples and at most {MAX_EXAMPLES}"
    );
    let positives = samples.iter().filter(|s| s.positive).count();
    ensure!(
        positives >= 2 && samples.len() - positives >= 2,
        "each split needs at least two positives and two negatives"
    );
    let mut ids = BTreeSet::new();
    samples
        .iter()
        .map(|sample| {
            ensure!(
                !sample.id.is_empty() && sample.id.len() <= 128 && ids.insert(&sample.id),
                "duplicate or invalid sample fingerprint"
            );
            ensure!(
                sample.values.len() == dimensions,
                "embedding dimension mismatch"
            );
            normalized(&sample.values)
        })
        .collect()
}

fn disjoint(left: &[String], right: &[Example]) -> Result<()> {
    let ids: BTreeSet<_> = left.iter().collect();
    ensure!(
        right.iter().all(|s| !ids.contains(&s.id)),
        "training, calibration and validation recordings must be disjoint"
    );
    Ok(())
}

impl Head {
    pub fn train(contract: &str, training: &[Example], calibration: &[Example]) -> Result<Self> {
        ensure!(
            !contract.is_empty() && contract.len() <= 256,
            "invalid encoder contract"
        );
        let dimensions = training.first().map_or(0, |s| s.values.len());
        let train = examples(training, dimensions)?;
        let calibrate = examples(calibration, dimensions)?;
        let training_ids: Vec<_> = training.iter().map(|s| s.id.clone()).collect();
        disjoint(&training_ids, calibration)?;
        ensure!(
            dimensions as u64 * training.len() as u64 * EPOCHS as u64 <= 256_000_000,
            "training work budget exceeded"
        );
        let positives = training.iter().filter(|s| s.positive).count();
        let negatives = training.len() - positives;
        let mut weights = vec![0.0_f64; dimensions];
        let mut bias = 0.0;
        for _ in 0..EPOCHS {
            let mut gradient = vec![0.0; dimensions];
            let mut bias_gradient = 0.0;
            for (sample, vector) in training.iter().zip(&train) {
                let score =
                    sigmoid(weights.iter().zip(vector).map(|(w, x)| w * x).sum::<f64>() + bias);
                let target = if sample.positive { 1.0 } else { 0.0 };
                let class_weight = 0.5
                    / if sample.positive {
                        positives as f64
                    } else {
                        negatives as f64
                    };
                let error = (score - target) * class_weight;
                for (g, x) in gradient.iter_mut().zip(vector) {
                    *g += error * x;
                }
                bias_gradient += error;
            }
            for (w, g) in weights.iter_mut().zip(gradient) {
                *w -= 0.8 * (g + 0.0001 * *w);
            }
            bias -= 0.8 * bias_gradient;
        }
        ensure!(
            weights.iter().all(|w| w.is_finite()) && bias.is_finite(),
            "training produced non-finite weights"
        );
        let trained_loss: f64 = train
            .iter()
            .zip(training)
            .map(|(v, s)| {
                let p = sigmoid(weights.iter().zip(v).map(|(w, x)| w * x).sum::<f64>() + bias)
                    .clamp(1e-15, 1.0 - 1e-15);
                if s.positive { -p.ln() } else { -(1.0 - p).ln() }
            })
            .sum::<f64>()
            / training.len() as f64;
        ensure!(
            trained_loss < std::f64::consts::LN_2 - 0.001,
            "training did not learn a useful boundary"
        );
        let mut head = Self {
            format_version: 1,
            encoder_contract: contract.into(),
            dimensions,
            weights: weights.iter().map(|w| *w as f32).collect(),
            bias: bias as f32,
            threshold: 0.5,
            training_ids,
            calibration_ids: calibration.iter().map(|s| s.id.clone()).collect(),
            validation_ids: Vec::new(),
            training_positives: positives,
            calibration_positives: calibration.iter().filter(|s| s.positive).count(),
            validation: None,
        };
        let mut minimum_positive = 1.0_f32;
        let mut maximum_negative = 0.0_f32;
        for (sample, vector) in calibration.iter().zip(&calibrate) {
            let full = sigmoid(weights.iter().zip(vector).map(|(w, x)| w * x).sum::<f64>() + bias);
            let score = head.score_normalized(vector);
            ensure!(
                (full - score as f64).abs() <= 0.00001,
                "f32 head loses score parity"
            );
            if sample.positive {
                minimum_positive = minimum_positive.min(score);
            } else {
                maximum_negative = maximum_negative.max(score);
            }
        }
        ensure!(
            minimum_positive - maximum_negative >= 0.05,
            "calibration examples overlap; record more varied positives and confusable negatives"
        );
        head.threshold = (minimum_positive + maximum_negative) * 0.5;
        head.validate()?;
        Ok(head)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.format_version == 1, "unsupported head format");
        ensure!(
            !self.encoder_contract.is_empty() && self.encoder_contract.len() <= 256,
            "invalid head encoder contract"
        );
        ensure!(
            (1..=MAX_DIMENSIONS).contains(&self.dimensions)
                && self.weights.len() == self.dimensions,
            "invalid head dimensions"
        );
        ensure!(
            self.weights.iter().all(|w| w.is_finite() && w.abs() <= 1e6)
                && self.bias.is_finite()
                && self.bias.abs() <= 1e6,
            "invalid head weights"
        );
        ensure!(
            self.threshold.is_finite() && self.threshold > 0.0 && self.threshold < 1.0,
            "invalid head threshold"
        );
        let mut ids = BTreeSet::new();
        for split in [
            &self.training_ids,
            &self.calibration_ids,
            &self.validation_ids,
        ] {
            ensure!(
                split.len() <= MAX_EXAMPLES,
                "too many head sample fingerprints"
            );
            for id in split {
                ensure!(
                    !id.is_empty() && id.len() <= 128 && ids.insert(id),
                    "duplicate or invalid head sample fingerprint"
                );
            }
        }
        ensure!(
            self.training_positives >= 2
                && self.training_positives <= self.training_ids.len().saturating_sub(2),
            "invalid training counts"
        );
        ensure!(
            self.calibration_positives >= 2
                && self.calibration_positives <= self.calibration_ids.len().saturating_sub(2),
            "invalid calibration counts"
        );
        if let Some(v) = &self.validation {
            ensure!(
                v.positives >= 2
                    && v.negatives >= 2
                    && v.positives.checked_add(v.negatives) == Some(self.validation_ids.len()),
                "invalid held-out validation counts"
            );
            ensure!(
                v.misses <= v.positives
                    && v.false_activations <= v.negatives
                    && v.minimum_margin.is_finite()
                    && (0.0..=1.0).contains(&v.minimum_margin),
                "invalid held-out validation metrics"
            );
        } else {
            ensure!(
                self.validation_ids.is_empty(),
                "validation fingerprints without results"
            );
        }
        Ok(())
    }

    fn score_normalized(&self, values: &[f64]) -> f32 {
        sigmoid(
            self.weights
                .iter()
                .zip(values)
                .map(|(w, x)| *w as f64 * x)
                .sum::<f64>()
                + self.bias as f64,
        ) as f32
    }

    pub fn score(&self, contract: &str, values: &[f32]) -> Result<f32> {
        self.validate()?;
        ensure!(
            contract == self.encoder_contract && values.len() == self.dimensions,
            "head is incompatible with the selected encoder; retain the previous engine or retrain"
        );
        Ok(self.score_normalized(&normalized(values)?))
    }

    pub fn validate_held_out(&mut self, samples: &[Example]) -> Result<()> {
        self.validate()?;
        ensure!(
            self.validation.is_none(),
            "held-out validation is single-use; train a fresh candidate to evaluate another split"
        );
        let vectors = examples(samples, self.dimensions)?;
        disjoint(&self.training_ids, samples)?;
        disjoint(&self.calibration_ids, samples)?;
        let mut result = Validation {
            positives: 0,
            negatives: 0,
            misses: 0,
            false_activations: 0,
            minimum_margin: 1.0,
        };
        for (sample, vector) in samples.iter().zip(&vectors) {
            let score = self.score_normalized(vector);
            let wake = score >= self.threshold;
            if sample.positive {
                result.positives += 1;
                result.misses += usize::from(!wake);
            } else {
                result.negatives += 1;
                result.false_activations += usize::from(wake);
            }
            result.minimum_margin = result.minimum_margin.min((score - self.threshold).abs());
        }
        ensure!(
            result.misses == 0 && result.false_activations == 0 && result.minimum_margin >= 0.025,
            "held-out local recordings failed validation; keep the previous detector and record more examples"
        );
        self.validation_ids = samples.iter().map(|s| s.id.clone()).collect();
        self.validation = Some(result);
        Ok(())
    }

    pub fn locally_validated(&self) -> bool {
        self.validation
            .as_ref()
            .is_some_and(|v| v.misses == 0 && v.false_activations == 0 && v.minimum_margin >= 0.025)
    }
}

/// One normalized embedding scores the complete head matrix; no per-word
/// encoder, worker process, or accelerator submission is created.
pub fn score_many(heads: &[Head], contract: &str, values: &[f32]) -> Result<Vec<f32>> {
    ensure!(heads.len() <= 256, "at most 256 heads can share an encoder");
    let vector = normalized(values)?;
    heads
        .iter()
        .map(|head| {
            head.validate()?;
            ensure!(
                head.encoder_contract == contract && head.dimensions == vector.len(),
                "head/encoder contract mismatch"
            );
            Ok(head.score_normalized(&vector))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn split(prefix: &str) -> Vec<Example> {
        (0..8)
            .map(|i| Example {
                id: format!("{prefix}{i}"),
                values: vec![if i < 4 { 1.0 } else { -1.0 }, (i as f32 + 1.0) * 0.03],
                positive: i < 4,
            })
            .collect()
    }
    #[test]
    fn training_calibration_and_heldout_are_distinct_and_deterministic() {
        let a = Head::train("encoder-v1", &split("train"), &split("cal")).unwrap();
        let mut b = Head::train("encoder-v1", &split("train"), &split("cal")).unwrap();
        assert_eq!(a.weights, b.weights);
        assert!(!b.locally_validated());
        b.validate_held_out(&split("test")).unwrap();
        assert!(b.locally_validated());
        let c: Head = serde_json::from_slice(&serde_json::to_vec(&b).unwrap()).unwrap();
        c.validate().unwrap();
        assert!(c.locally_validated());
        assert!(Head::train("encoder-v1", &split("same"), &split("same")).is_err());
        assert!(b.validate_held_out(&split("train")).is_err());
    }
    #[test]
    fn calibration_confusions_and_invalid_artifacts_fail() {
        let training = split("train");
        let mut cal = split("cal");
        cal[0].values = cal[4].values.clone();
        assert!(Head::train("encoder", &training, &cal).is_err());
        let h = Head::train("encoder", &training, &split("cal")).unwrap();
        assert!(h.score("different", &[1.0, 0.0]).is_err());
        assert!(h.score("encoder", &[f32::NAN, 0.0]).is_err());
        assert!(h.score("encoder", &[0.0, 0.0]).is_err());
        let mut bad = h.clone();
        bad.weights.push(0.0);
        assert!(bad.validate().is_err());
        let mut bad = h.clone();
        bad.threshold = f32::INFINITY;
        assert!(bad.validate().is_err());
        let mut bad = h.clone();
        bad.weights[0] = f32::NAN;
        assert!(bad.validate().is_err());
        let mut bad = h;
        bad.format_version = 99;
        assert!(bad.validate().is_err());
    }
    #[test]
    fn all_heads_share_one_embedding_and_match_individual_scores() {
        let h = Head::train("encoder", &split("train"), &split("cal")).unwrap();
        let heads = vec![h; 12];
        let many = score_many(&heads, "encoder", &[0.8, 0.1]).unwrap();
        for (score, head) in many.iter().zip(&heads) {
            assert_eq!(*score, head.score("encoder", &[0.8, 0.1]).unwrap());
        }
        assert!(score_many(&heads, "wrong", &[0.8, 0.1]).is_err());
    }
}
