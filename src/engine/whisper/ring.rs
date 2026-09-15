use std::collections::VecDeque;

pub(super) const FRAME_SAMPLES: usize = 512;
const START_THRESHOLD: f32 = 0.50;
const END_THRESHOLD: f32 = 0.35;
const PRE_ROLL_FRAMES: usize = 10;
const END_SILENCE_FRAMES: usize = 10;
const MAX_UTTERANCE_SAMPLES: usize = 16_000 * 30;

#[derive(Debug)]
pub(super) struct Utterance {
    pub samples: Vec<f32>,
    pub start_sample: u64,
    pub end_sample: u64,
}

struct Active {
    samples: Vec<f32>,
    start_sample: u64,
    silence_frames: usize,
}

pub(super) struct EndpointBuffer {
    idle: VecDeque<Vec<f32>>,
    active: Option<Active>,
    cursor: u64,
}

impl EndpointBuffer {
    pub fn new() -> Self {
        Self {
            idle: VecDeque::with_capacity(PRE_ROLL_FRAMES),
            active: None,
            cursor: 0,
        }
    }

    pub fn push(&mut self, frame: &[f32], probability: f32) -> Option<Utterance> {
        debug_assert_eq!(frame.len(), FRAME_SAMPLES);
        let frame_end = self.cursor + frame.len() as u64;
        if let Some(active) = &mut self.active {
            let remaining = MAX_UTTERANCE_SAMPLES.saturating_sub(active.samples.len());
            active
                .samples
                .extend_from_slice(&frame[..remaining.min(frame.len())]);
            if probability < END_THRESHOLD {
                active.silence_frames += 1;
            } else {
                active.silence_frames = 0;
            }
            let complete = active.silence_frames >= END_SILENCE_FRAMES
                || active.samples.len() >= MAX_UTTERANCE_SAMPLES;
            self.cursor = frame_end;
            if complete {
                return self.finish();
            }
            return None;
        }

        self.idle.push_back(frame.to_vec());
        while self.idle.len() > PRE_ROLL_FRAMES {
            self.idle.pop_front();
        }
        if probability >= START_THRESHOLD {
            let buffered = self.idle.len() * FRAME_SAMPLES;
            let start_sample = frame_end.saturating_sub(buffered as u64);
            let mut samples = Vec::with_capacity(MAX_UTTERANCE_SAMPLES.min(buffered * 4));
            for frame in self.idle.drain(..) {
                samples.extend(frame);
            }
            self.active = Some(Active {
                samples,
                start_sample,
                silence_frames: 0,
            });
        }
        self.cursor = frame_end;
        None
    }

    pub fn finish(&mut self) -> Option<Utterance> {
        self.active.take().map(|active| Utterance {
            samples: active.samples,
            start_sample: active.start_sample,
            end_sample: self.cursor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_pre_roll_hangover_and_utterance_size() {
        let frame = [0.0; FRAME_SAMPLES];
        let mut buffer = EndpointBuffer::new();
        for _ in 0..20 {
            assert!(buffer.push(&frame, 0.0).is_none());
        }
        assert!(buffer.push(&frame, 0.9).is_none());
        for _ in 0..END_SILENCE_FRAMES - 1 {
            assert!(buffer.push(&frame, 0.0).is_none());
        }
        let utterance = buffer.push(&frame, 0.0).unwrap();
        assert_eq!(utterance.start_sample, 11 * FRAME_SAMPLES as u64);
        assert_eq!(utterance.samples.len(), 20 * FRAME_SAMPLES);
        assert_eq!(utterance.end_sample, 31 * FRAME_SAMPLES as u64);
        assert!(utterance.samples.len() <= MAX_UTTERANCE_SAMPLES);
    }
}
