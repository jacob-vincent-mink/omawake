use std::collections::VecDeque;

use super::protocol::FRAME_SAMPLES;

const PRE_ROLL_SAMPLES: usize = 16_000 * 320 / 1_000;
// Moonshine can lose the final consonant when audio stops exactly at Silero's
// speech-end timestamp. Retain a small, bounded tail for verifier context.
const POST_ROLL_SAMPLES: usize = 16_000 * 320 / 1_000;
// Keep two extra frames so Silero's timestamp can precede the frame that emits
// SPEECH_START while still retaining the full configured pre-roll.
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
    speech_end_sample: Option<u64>,
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
            if activity.start_before_frame_end.is_some() && activity.end_before_frame_end.is_none()
            {
                // Merge speech that resumes during the trailing-context window.
                active.speech_end_sample = None;
            }
            if let Some(trim) = activity.end_before_frame_end {
                active.speech_end_sample = Some(frame_end.saturating_sub(trim as u64));
            }
            if let Some(end_sample) = active.speech_end_sample {
                let finish_sample = end_sample.saturating_add(POST_ROLL_SAMPLES as u64);
                if frame_end >= finish_sample {
                    return self.finish_at(finish_sample);
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
                speech_end_sample: None,
            });
        }
        self.cursor = frame_end;
        if let Some(trim) = activity.end_before_frame_end
            && self.active.is_some()
        {
            let end_sample = frame_end.saturating_sub(trim as u64);
            if let Some(active) = &mut self.active {
                active.speech_end_sample = Some(end_sample);
            }
            let finish_sample = end_sample.saturating_add(POST_ROLL_SAMPLES as u64);
            if frame_end >= finish_sample {
                return self.finish_at(finish_sample);
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
    fn keeps_bounded_pre_and_post_roll_around_provider_endpoints() {
        let frame = [0.0; FRAME_SAMPLES];
        let mut buffer = ActivityBuffer::new();
        for _ in 0..20 {
            assert!(buffer.push(&frame, Activity::default()).is_none());
        }
        assert!(
            buffer
                .push(
                    &frame,
                    Activity {
                        start_before_frame_end: Some(0),
                        end_before_frame_end: None,
                    },
                )
                .is_none()
        );
        for _ in 0..4 {
            assert!(buffer.push(&frame, Activity::default()).is_none());
        }
        assert!(
            buffer
                .push(
                    &frame,
                    Activity {
                        start_before_frame_end: None,
                        end_before_frame_end: Some(0),
                    },
                )
                .is_none()
        );
        for _ in 0..(POST_ROLL_SAMPLES / FRAME_SAMPLES - 1) {
            assert!(buffer.push(&frame, Activity::default()).is_none());
        }
        let utterance = buffer.push(&frame, Activity::default()).unwrap();
        assert_eq!(utterance.start_sample, 11 * FRAME_SAMPLES as u64);
        assert_eq!(utterance.end_sample, 36 * FRAME_SAMPLES as u64);
        assert_eq!(utterance.samples.len(), 25 * FRAME_SAMPLES);
        assert!(utterance.samples.len() <= MAX_UTTERANCE_SAMPLES);
    }

    #[test]
    fn a_single_segment_event_keeps_bounded_trailing_context() {
        let frame = [0.0; FRAME_SAMPLES];
        let mut buffer = ActivityBuffer::new();
        for _ in 0..RETAINED_IDLE_FRAMES {
            assert!(buffer.push(&frame, Activity::default()).is_none());
        }
        let frame_end = (RETAINED_IDLE_FRAMES + 1) * FRAME_SAMPLES;
        assert!(
            buffer
                .push(
                    &frame,
                    Activity {
                        start_before_frame_end: Some(128),
                        end_before_frame_end: Some(32),
                    },
                )
                .is_none()
        );
        for _ in 0..(POST_ROLL_SAMPLES / FRAME_SAMPLES - 1) {
            assert!(buffer.push(&frame, Activity::default()).is_none());
        }
        let utterance = buffer.push(&frame, Activity::default()).unwrap();
        assert_eq!(
            utterance.start_sample,
            (frame_end - 128 - PRE_ROLL_SAMPLES) as u64
        );
        assert_eq!(
            utterance.end_sample,
            (frame_end - 32 + POST_ROLL_SAMPLES) as u64
        );
        assert_eq!(
            utterance.samples.len(),
            (utterance.end_sample - utterance.start_sample) as usize
        );
    }

    #[test]
    fn speech_resuming_during_post_roll_merges_into_one_utterance() {
        let frame = [0.0; FRAME_SAMPLES];
        let mut buffer = ActivityBuffer::new();
        assert!(
            buffer
                .push(
                    &frame,
                    Activity {
                        start_before_frame_end: Some(0),
                        end_before_frame_end: None,
                    },
                )
                .is_none()
        );
        assert!(
            buffer
                .push(
                    &frame,
                    Activity {
                        start_before_frame_end: None,
                        end_before_frame_end: Some(0),
                    },
                )
                .is_none()
        );
        assert!(
            buffer
                .push(
                    &frame,
                    Activity {
                        start_before_frame_end: Some(0),
                        end_before_frame_end: None,
                    },
                )
                .is_none()
        );
        for _ in 0..POST_ROLL_SAMPLES / FRAME_SAMPLES + 1 {
            assert!(buffer.push(&frame, Activity::default()).is_none());
        }
        assert!(buffer.finish().is_some());
    }
}
