use std::collections::VecDeque;

use super::protocol::FRAME_SAMPLES;

const PRE_ROLL_SAMPLES: usize = 16_000 * 320 / 1_000;
const POST_ROLL_SAMPLES: usize = 16_000 * 320 / 1_000;
const RETAINED_IDLE_FRAMES: usize = PRE_ROLL_SAMPLES / FRAME_SAMPLES + 2;
const MAX_UTTERANCE_SAMPLES: usize = 16_000 * 30;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Activity {
    pub start_before_frame_end: Option<usize>,
    pub end_before_frame_end: Option<usize>,
}

#[derive(Debug)]
pub(super) struct Utterance {
    pub samples: Vec<f32>,
    pub start_sample: u64,
    pub end_sample: u64,
}

struct Active {
    samples: Vec<f32>,
    start_sample: u64,
    pending_end_sample: Option<u64>,
}

pub(super) struct ActivityBuffer {
    idle: VecDeque<Vec<f32>>,
    active: Option<Active>,
    cursor: u64,
}

impl ActivityBuffer {
    pub fn new() -> Self {
        Self {
            idle: VecDeque::with_capacity(RETAINED_IDLE_FRAMES),
            active: None,
            cursor: 0,
        }
    }

    pub fn push(&mut self, frame: &[f32], activity: Activity) -> Option<Utterance> {
        debug_assert_eq!(frame.len(), FRAME_SAMPLES);
        let frame_end = self.cursor + frame.len() as u64;
        if let Some(active) = &mut self.active {
            let remaining = MAX_UTTERANCE_SAMPLES.saturating_sub(active.samples.len());
            active
                .samples
                .extend_from_slice(&frame[..remaining.min(frame.len())]);
            self.cursor = frame_end;
            // Silero may open another segment during the post-roll window.
            // Treat that close speech as one utterance for phrase verification.
            if activity.start_before_frame_end.is_some() {
                active.pending_end_sample = None;
            }
            if let Some(trim) = activity.end_before_frame_end {
                let end_sample = frame_end.saturating_sub(trim as u64);
                active.pending_end_sample = Some(end_sample);
            }
            if let Some(end_sample) = active.pending_end_sample {
                let close_sample = end_sample.saturating_add(POST_ROLL_SAMPLES as u64);
                if frame_end >= close_sample {
                    return self.finish_at(close_sample);
                }
            }
            if active.samples.len() >= MAX_UTTERANCE_SAMPLES {
                return self.finish_at(frame_end);
            }
            return None;
        }

        self.idle.push_back(frame.to_vec());
        while self.idle.len() > RETAINED_IDLE_FRAMES {
            self.idle.pop_front();
        }
        if let Some(start_back) = activity.start_before_frame_end {
            let buffered = self.idle.len() * FRAME_SAMPLES;
            let available_start = frame_end.saturating_sub(buffered as u64);
            let requested_start = frame_end
                .saturating_sub(start_back as u64)
                .saturating_sub(PRE_ROLL_SAMPLES as u64);
            let start_sample = available_start.max(requested_start);
            let mut samples = Vec::with_capacity(MAX_UTTERANCE_SAMPLES.min(buffered * 4));
            for frame in self.idle.drain(..) {
                samples.extend(frame);
            }
            let discard = usize::try_from(start_sample.saturating_sub(available_start))
                .unwrap_or(usize::MAX)
                .min(samples.len());
            samples.drain(..discard);
            self.active = Some(Active {
                samples,
                start_sample,
                pending_end_sample: None,
            });
        }
        self.cursor = frame_end;
        if let Some(trim) = activity.end_before_frame_end
            && self.active.is_some()
        {
            let end_sample = frame_end.saturating_sub(trim as u64);
            if let Some(active) = &mut self.active {
                active.pending_end_sample = Some(end_sample);
            }
        }
        None
    }

    pub fn finish(&mut self) -> Option<Utterance> {
        self.finish_at(self.cursor)
    }

    fn finish_at(&mut self, end_sample: u64) -> Option<Utterance> {
        self.active.take().map(|mut active| {
            let end_sample = end_sample.max(active.start_sample);
            let expected = usize::try_from(end_sample - active.start_sample)
                .unwrap_or(usize::MAX)
                .min(active.samples.len());
            active.samples.truncate(expected);
            Utterance {
                samples: active.samples,
                start_sample: active.start_sample,
                end_sample: active.start_sample + expected as u64,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_bounded_and_adds_post_roll_to_provider_endpoint() {
        let frame = [0.0; FRAME_SAMPLES];
        let mut ring = ActivityBuffer::new();
        for _ in 0..20 {
            assert!(ring.push(&frame, Activity::default()).is_none());
        }
        assert!(
            ring.push(
                &frame,
                Activity {
                    start_before_frame_end: Some(0),
                    end_before_frame_end: None
                }
            )
            .is_none()
        );
        assert!(
            ring.push(
                &frame,
                Activity {
                    start_before_frame_end: None,
                    end_before_frame_end: Some(32),
                },
            )
            .is_none()
        );
        let endpoint = 22 * FRAME_SAMPLES as u64 - 32;
        let mut utterance = None;
        for _ in 0..11 {
            utterance = ring.push(&frame, Activity::default());
            if utterance.is_some() {
                break;
            }
        }
        let utterance = utterance.unwrap();
        assert_eq!(utterance.end_sample, endpoint + POST_ROLL_SAMPLES as u64);
        assert_eq!(
            utterance.samples.len(),
            (utterance.end_sample - utterance.start_sample) as usize
        );
        assert!(utterance.samples.len() <= MAX_UTTERANCE_SAMPLES);
    }

    #[test]
    fn close_speech_start_cancels_pending_end_and_merges_segments() {
        let frame = [0.0; FRAME_SAMPLES];
        let mut ring = ActivityBuffer::new();
        assert!(
            ring.push(
                &frame,
                Activity {
                    start_before_frame_end: Some(0),
                    end_before_frame_end: None,
                },
            )
            .is_none()
        );
        assert!(
            ring.push(
                &frame,
                Activity {
                    start_before_frame_end: None,
                    end_before_frame_end: Some(0),
                },
            )
            .is_none()
        );
        for _ in 0..3 {
            assert!(ring.push(&frame, Activity::default()).is_none());
        }
        assert!(
            ring.push(
                &frame,
                Activity {
                    start_before_frame_end: Some(0),
                    end_before_frame_end: None,
                },
            )
            .is_none()
        );
        for _ in 0..10 {
            assert!(ring.push(&frame, Activity::default()).is_none());
        }
        assert!(
            ring.push(
                &frame,
                Activity {
                    start_before_frame_end: None,
                    end_before_frame_end: Some(0),
                },
            )
            .is_none()
        );
        let mut utterance = None;
        for _ in 0..10 {
            utterance = ring.push(&frame, Activity::default());
        }
        let utterance = utterance.expect("second endpoint closes after post-roll");
        assert_eq!(utterance.start_sample, 0);
        assert_eq!(utterance.end_sample, 27 * FRAME_SAMPLES as u64);
    }
}
