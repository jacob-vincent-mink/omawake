use anyhow::{Context, Result, bail};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use super::fbank::SAMPLE_RATE;

pub struct AudioResampler {
    source_rate: Option<usize>,
    sinc: Option<SincFixedIn<f32>>,
    pending: Vec<f32>,
    source_samples: usize,
    output_samples: usize,
    delay_remaining: usize,
    finished: bool,
}

impl AudioResampler {
    pub fn new() -> Self {
        Self {
            source_rate: None,
            sinc: None,
            pending: Vec::new(),
            source_samples: 0,
            output_samples: 0,
            delay_remaining: 0,
            finished: false,
        }
    }

    pub fn accept(&mut self, sample_rate: i32, samples: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            bail!("audio was supplied after the stream finished");
        }
        let rate = usize::try_from(sample_rate)
            .ok()
            .filter(|rate| *rate > 0)
            .context("audio sample rate must be positive")?;
        self.ensure_rate(rate)?;
        self.source_samples += samples.len();
        if rate == SAMPLE_RATE {
            self.output_samples += samples.len();
            return Ok(samples.to_vec());
        }
        self.pending.extend_from_slice(samples);
        let mut output = Vec::new();
        loop {
            let needed = self.sinc.as_ref().unwrap().input_frames_next();
            if self.pending.len() < needed {
                break;
            }
            let input = vec![self.pending[..needed].to_vec()];
            self.pending.drain(..needed);
            let block = self.sinc.as_mut().unwrap().process(&input, None)?;
            self.append_compensated(&mut output, &block[0], None);
        }
        Ok(output)
    }

    pub fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let Some(rate) = self.source_rate else {
            return Ok(Vec::new());
        };
        if rate == SAMPLE_RATE {
            return Ok(Vec::new());
        }
        let expected = ((self.source_samples as u128 * SAMPLE_RATE as u128 + rate as u128 / 2)
            / rate as u128) as usize;
        let mut output = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        let block = self
            .sinc
            .as_mut()
            .unwrap()
            .process_partial(Some(&[pending]), None)?;
        self.append_compensated(&mut output, &block[0], Some(expected));
        for _ in 0..4 {
            if self.output_samples >= expected {
                break;
            }
            let block = self
                .sinc
                .as_mut()
                .unwrap()
                .process_partial::<Vec<f32>>(None, None)?;
            self.append_compensated(&mut output, &block[0], Some(expected));
        }
        if self.output_samples < expected {
            bail!(
                "resampler produced {} of {expected} expected samples",
                self.output_samples
            );
        }
        Ok(output)
    }

    fn ensure_rate(&mut self, rate: usize) -> Result<()> {
        if let Some(existing) = self.source_rate {
            if existing != rate {
                bail!("audio sample rate changed from {existing} to {rate} during a stream");
            }
            return Ok(());
        }
        self.source_rate = Some(rate);
        if rate != SAMPLE_RATE {
            let parameters = SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.95,
                oversampling_factor: 128,
                interpolation: SincInterpolationType::Cubic,
                window: WindowFunction::BlackmanHarris2,
            };
            let sinc =
                SincFixedIn::new(SAMPLE_RATE as f64 / rate as f64, 1.0, parameters, 1024, 1)?;
            self.delay_remaining = sinc.output_delay();
            self.sinc = Some(sinc);
        }
        Ok(())
    }

    fn append_compensated(&mut self, target: &mut Vec<f32>, block: &[f32], limit: Option<usize>) {
        let skip = self.delay_remaining.min(block.len());
        self.delay_remaining -= skip;
        let available = &block[skip..];
        let take = limit.map_or(available.len(), |limit| {
            limit
                .saturating_sub(self.output_samples)
                .min(available.len())
        });
        target.extend_from_slice(&available[..take]);
        self.output_samples += take;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_16khz_through_across_chunks() {
        let mut resampler = AudioResampler::new();
        assert_eq!(resampler.accept(16_000, &[1.0, 2.0]).unwrap(), [1.0, 2.0]);
        assert_eq!(resampler.accept(16_000, &[3.0]).unwrap(), [3.0]);
        assert!(resampler.finish().unwrap().is_empty());
    }

    #[test]
    fn produces_expected_duration_from_48khz() {
        let mut resampler = AudioResampler::new();
        let mut output = Vec::new();
        for chunk in vec![0.0; 48_003].chunks(317) {
            output.extend(resampler.accept(48_000, chunk).unwrap());
        }
        output.extend(resampler.finish().unwrap());
        assert_eq!(output.len(), 16_001);
    }
}
