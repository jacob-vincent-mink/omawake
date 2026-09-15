//! Independent implementation of OpenAI Whisper's original audio.py feature
//! contract: Slaney80, periodic Hann400, hop160, centered reflect padding,
//! 30-second right padding, log10 clamp8, then (log+4)/4.
use anyhow::{Result, ensure};
use std::sync::OnceLock;
const FFT: usize = 400;
const BINS: usize = 201;
const MELS: usize = 80;
const FRAMES: usize = 3000;
const FULL: usize = 480_000;
struct Tables {
    window: [f64; FFT],
    cos: [f64; FFT],
    sin: [f64; FFT],
    filters: Vec<Vec<(usize, f64)>>,
}
fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let to_mel = |hz: f64| {
            if hz < 1000.0 {
                hz / (200.0 / 3.0)
            } else {
                15.0 + (hz / 1000.0).ln() / (6.4_f64.ln() / 27.0)
            }
        };
        let to_hz = |mel: f64| {
            if mel < 15.0 {
                mel * (200.0 / 3.0)
            } else {
                1000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp()
            }
        };
        let edges: Vec<_> = (0..82)
            .map(|i| to_hz(to_mel(8000.0) * i as f64 / 81.0))
            .collect();
        let filters = (0..MELS)
            .map(|m| {
                (0..BINS)
                    .filter_map(|k| {
                        let hz = k as f64 * 40.0;
                        let weight = ((hz - edges[m]) / (edges[m + 1] - edges[m]))
                            .min((edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]))
                            .max(0.0);
                        // Original librosa filter table stores each triangular weight
                        // as f32, then applies Slaney area scaling into that f32 array.
                        let scaled =
                            ((weight as f32) as f64 * 2.0 / (edges[m + 2] - edges[m])) as f32;
                        (scaled > 0.0).then_some((k, scaled as f64))
                    })
                    .collect()
            })
            .collect();
        Tables {
            window: std::array::from_fn(|i| {
                0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT as f64).cos()
            }),
            cos: std::array::from_fn(|i| (std::f64::consts::TAU * i as f64 / FFT as f64).cos()),
            sin: std::array::from_fn(|i| (std::f64::consts::TAU * i as f64 / FFT as f64).sin()),
            filters,
        }
    })
}
pub(crate) fn log_mel(samples: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !samples.is_empty() && samples.len() <= FULL,
        "Whisper expects 1..480000 samples at 16 kHz"
    );
    ensure!(
        samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
        "invalid Whisper audio"
    );
    let t = tables();
    let mut result = vec![-10.0_f64; MELS * FRAMES];
    let mut maximum = -10.0_f64;
    // Reflect at the padded 30s boundary, not at the end of the short clip.
    let sample_at = |mut index: isize| {
        if index < 0 {
            index = -index;
        }
        if index >= FULL as isize {
            index = 2 * (FULL as isize - 1) - index;
        }
        samples.get(index as usize).copied().unwrap_or(0.0) as f64
    };
    for frame in 0..FRAMES {
        let input: [f64; FFT] =
            std::array::from_fn(|i| sample_at((frame * 160 + i) as isize - 200) * t.window[i]);
        if input.iter().all(|x| *x == 0.0) {
            continue;
        }
        let mut power = [0.0; BINS];
        // A bounded DFT is deliberate: 400 is not a radix-2 FFT size. Tables
        // are shared across calls; all-zero tail frames require no transform.
        for (k, value) in power.iter_mut().enumerate() {
            let mut re = 0.0;
            let mut im = 0.0;
            for (i, sample) in input.iter().enumerate() {
                let phase = k * i % FFT;
                re += sample * t.cos[phase];
                im -= sample * t.sin[phase];
            }
            *value = re * re + im * im;
        }
        for (m, filter) in t.filters.iter().enumerate() {
            let value = filter
                .iter()
                .map(|(k, w)| power[*k] * w)
                .sum::<f64>()
                .max(1e-10)
                .log10();
            maximum = maximum.max(value);
            result[m * FRAMES + frame] = value;
        }
    }
    Ok(result
        .into_iter()
        .map(|value| ((value.max(maximum - 8.0) + 4.0) / 4.0) as f32)
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_independent_openai_whisper_features() {
        let wave = include_bytes!("../../tests/fixtures/whisper-golden.wav");
        let mut reader = hound::WavReader::new(std::io::Cursor::new(wave)).unwrap();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let expected: Vec<f32> =
            include_bytes!("../../tests/fixtures/whisper-golden-mel-80frames.f32")
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
        let actual = log_mel(&samples).unwrap();
        let mut maximum = 0_f32;
        for m in 0..80 {
            for frame in 0..3000 {
                let reference = expected[m * 80 + frame.min(79)];
                maximum = maximum.max((actual[m * 3000 + frame] - reference).abs());
            }
        }
        eprintln!("Whisper feature max absolute error: {maximum}");
        assert!(maximum <= 0.0001, "Whisper feature mismatch: {maximum}");
    }
    #[test]
    fn silence_and_invalid_input() {
        let silent = log_mel(&[0.0; 1600]).unwrap();
        assert_eq!(silent.len(), MELS * FRAMES);
        assert!(silent.iter().all(|v| *v == -1.5));
        assert!(log_mel(&[]).is_err());
        assert!(log_mel(&[f32::NAN]).is_err());
        assert!(log_mel(&[1.1]).is_err());
    }
}
