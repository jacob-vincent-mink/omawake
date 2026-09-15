//! Kaldi-compatible online log Mel filterbank extraction for the bundled model.
//!
//! The feature contract comes from the GigaSpeech/icefall training recipe:
//! 16 kHz audio, 80 Mel bins, 25 ms Povey windows, 10 ms frame shift,
//! reflected edges, no dither, DC removal and 0.97 preemphasis. This module is
//! an independent Rust implementation of that published feature definition.

use anyhow::{Result, bail};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::{collections::VecDeque, sync::Arc};

pub const SAMPLE_RATE: usize = 16_000;
pub const BINS: usize = 80;
const WINDOW: usize = 400;
const SHIFT: usize = 160;
const FFT_SIZE: usize = 512;

pub struct OnlineFbank {
    samples: Vec<f32>,
    sample_base: usize,
    sample_count: usize,
    next_frame: usize,
    frames: VecDeque<[f32; BINS]>,
    weights: Vec<Vec<(usize, f32)>>,
    povey: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    spectrum: Vec<Complex32>,
    finished: bool,
}

impl OnlineFbank {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            samples: Vec::new(),
            sample_base: 0,
            sample_count: 0,
            next_frame: 0,
            frames: VecDeque::new(),
            weights: mel_weights(),
            povey: (0..WINDOW)
                .map(|index| {
                    let phase = std::f64::consts::TAU * index as f64 / (WINDOW - 1) as f64;
                    (0.5 - 0.5 * phase.cos()).powf(0.85) as f32
                })
                .collect(),
            fft: planner.plan_fft_forward(FFT_SIZE),
            spectrum: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            finished: false,
        }
    }

    pub fn accept(&mut self, samples: &[f32]) -> Result<()> {
        if self.finished {
            bail!("audio was supplied after the stream finished");
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("audio contains a non-finite sample");
        }
        self.samples.extend_from_slice(samples);
        self.sample_count += samples.len();
        self.extract_ready(false);
        Ok(())
    }

    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if self.sample_count != 0 {
            self.extract_ready(true);
        }
    }

    pub fn frames_ready(&self) -> usize {
        self.frames.len()
    }

    pub fn chunk(&self, count: usize) -> Result<Vec<f32>> {
        if self.frames.len() < count {
            bail!(
                "requested {count} feature frames with only {} ready",
                self.frames.len()
            );
        }
        Ok(self.frames.iter().take(count).flatten().copied().collect())
    }

    pub fn advance(&mut self, count: usize) {
        for _ in 0..count.min(self.frames.len()) {
            self.frames.pop_front();
        }
    }

    fn extract_ready(&mut self, final_input: bool) {
        let final_frames = (self.sample_count + SHIFT / 2) / SHIFT;
        loop {
            if final_input {
                if self.next_frame >= final_frames {
                    break;
                }
            } else {
                let right = self.next_frame * SHIFT + SHIFT / 2 + WINDOW / 2 - 1;
                if right >= self.sample_count {
                    break;
                }
            }
            let feature = self.extract_frame(self.next_frame);
            self.frames.push_back(feature);
            self.next_frame += 1;
        }

        // Retain only samples a future window or final edge reflection can use.
        let keep_from = self.next_frame.saturating_mul(SHIFT).saturating_sub(WINDOW);
        if keep_from > self.sample_base {
            let drain = (keep_from - self.sample_base).min(self.samples.len());
            self.samples.drain(..drain);
            self.sample_base += drain;
        }
    }

    fn extract_frame(&mut self, frame: usize) -> [f32; BINS] {
        let first = frame as isize * SHIFT as isize + SHIFT as isize / 2 - WINDOW as isize / 2;
        for index in 0..FFT_SIZE {
            self.spectrum[index] = Complex32::new(
                if index < WINDOW {
                    self.sample(reflect(first + index as isize, self.sample_count))
                } else {
                    0.0
                },
                0.0,
            );
        }
        let mean = self.spectrum[..WINDOW]
            .iter()
            .map(|point| point.re)
            .sum::<f32>()
            / WINDOW as f32;
        for point in &mut self.spectrum[..WINDOW] {
            point.re -= mean;
        }
        for index in (1..WINDOW).rev() {
            self.spectrum[index].re -= 0.97 * self.spectrum[index - 1].re;
        }
        self.spectrum[0].re -= 0.97 * self.spectrum[0].re;
        for (point, weight) in self.spectrum[..WINDOW].iter_mut().zip(&self.povey) {
            point.re *= *weight;
        }
        self.fft.process(&mut self.spectrum);
        std::array::from_fn(|bin| {
            self.weights[bin]
                .iter()
                .map(|&(index, weight)| self.spectrum[index].norm_sqr() * weight)
                .sum::<f32>()
                .max(f32::EPSILON)
                .ln()
        })
    }

    fn sample(&self, global: usize) -> f32 {
        self.samples[global - self.sample_base]
    }
}

fn reflect(mut index: isize, samples: usize) -> usize {
    let samples = samples as isize;
    while index < 0 || index >= samples {
        index = if index < 0 {
            -index - 1
        } else {
            2 * samples - 1 - index
        };
    }
    index as usize
}

fn mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_weights() -> Vec<Vec<(usize, f32)>> {
    let low = mel(20.0);
    let high = mel(7600.0);
    let delta = (high - low) / (BINS + 1) as f32;
    (0..BINS)
        .map(|bin| {
            let left = low + bin as f32 * delta;
            let center = left + delta;
            let right = center + delta;
            (0..=FFT_SIZE / 2)
                .filter_map(|fft_bin| {
                    let point = mel(fft_bin as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32);
                    (left < point && point < right).then(|| {
                        let weight = if point <= center {
                            (point - left) / (center - left)
                        } else {
                            (right - point) / (right - center)
                        };
                        (fft_bin, weight)
                    })
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_does_not_change_features() {
        let samples: Vec<f32> = (0..16_321).map(|n| ((n as f32) * 0.017).sin()).collect();
        let mut whole = OnlineFbank::new();
        whole.accept(&samples).unwrap();
        whole.finish();
        let expected = whole.chunk(whole.frames_ready()).unwrap();

        let mut split = OnlineFbank::new();
        for chunk in samples.chunks(317) {
            split.accept(chunk).unwrap();
        }
        split.finish();
        assert_eq!(split.chunk(split.frames_ready()).unwrap(), expected);
    }

    #[test]
    fn bounds_memory_for_long_streams() {
        let mut fbank = OnlineFbank::new();
        for _ in 0..100 {
            fbank.accept(&vec![0.0; SAMPLE_RATE]).unwrap();
            fbank.advance(fbank.frames_ready());
        }
        assert!(fbank.samples.len() <= WINDOW * 2);
    }
}
