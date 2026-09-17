use super::{
    Detection, WakeWordBackend, WakeWordStream,
    audio::{AudioResampler, read_wave},
    detect_samples,
    embedding_worker::{EmbeddingSession, EncodedUtterance, load_session},
};
use crate::{
    config::{Config, WakeWord},
    enrollment::{
        artifact,
        head::{Head, score_many},
    },
    paths::AppPaths,
};
use anyhow::{Context, Result, ensure};
use std::{
    cell::{Cell, RefCell},
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

pub(crate) struct TrainedBackend {
    worker: RefCell<Box<dyn EmbeddingSession>>,
    words: Vec<WakeWord>,
    heads: Vec<Head>,
    paths: AppPaths,
    active: Cell<bool>,
}
impl TrainedBackend {
    pub fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        Self::load_with(config, paths, load_session)
    }
    fn load_with(
        config: &Config,
        paths: &AppPaths,
        load: impl FnOnce(&Config, &AppPaths, Arc<AtomicBool>) -> Result<Box<dyn EmbeddingSession>>,
    ) -> Result<Self> {
        let worker = load(config, paths, Arc::new(AtomicBool::new(false)))?;
        let words: Vec<_> = config
            .wake_words
            .iter()
            .filter(|w| w.enabled)
            .cloned()
            .collect();
        ensure!(!words.is_empty(), "trained engine has no enabled words");
        let mut heads = Vec::new();
        for word in &words {
            let binding = word
                .enrollment
                .as_ref()
                .context("trained engine received a transcript-only word")?;
            binding.validate()?;
            let path = binding.heads.get(worker.contract()).with_context(|| format!("word {} has no head for this encoder; its existing heads are preserved. Retrain from retained recordings or select its previous engine", word.id))?;
            let path = if path.is_absolute() {
                path.clone()
            } else {
                paths
                    .config_file
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(path)
            };
            let head = artifact::load(&path)?;
            ensure!(
                head.encoder_contract == worker.contract(),
                "head artifact encoder does not match its binding"
            );
            heads.push(head);
        }
        ensure!(heads.len() <= 256, "at most 256 heads can share an encoder");
        Ok(Self {
            worker: RefCell::new(worker),
            words,
            heads,
            paths: paths.clone(),
            active: Cell::new(false),
        })
    }
    fn detections(&self, utterances: Vec<EncodedUtterance>, live: bool) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        for utterance in utterances {
            let scores = score_many(
                &self.heads,
                &utterance.embedding.encoder_contract,
                &utterance.embedding.values,
            )?;
            for ((word, head), score) in self.words.iter().zip(&self.heads).zip(scores) {
                let binding = word.enrollment.as_ref().context("missing enrollment")?;
                let threshold = binding.threshold.unwrap_or(head.threshold);
                if score >= threshold {
                    eprintln!(
                        "trained detection {}: score={score:.6} threshold={threshold:.6} device={}",
                        word.id, utterance.embedding.execution_devices
                    );
                    if live && binding.history.enabled {
                        let event = crate::enrollment::history::Event {
                            id: String::new(),
                            word_id: word.id.clone(),
                            created_ms: 0,
                            score,
                            threshold,
                            encoder_contract: head.encoder_contract.clone(),
                            head: binding.heads[&head.encoder_contract].display().to_string(),
                            device: utterance.embedding.execution_devices.clone(),
                            label: crate::enrollment::history::Label::Unreviewed,
                            audio: Default::default(),
                        };
                        if let Err(error) = crate::enrollment::history::record(
                            &self.paths,
                            &binding.history,
                            event,
                            &utterance.audio,
                        ) {
                            eprintln!("history for {} could not be saved: {error:#}", word.id);
                        }
                    }
                    detections.push(Detection {
                        id: word.id.clone(),
                        tokens: crate::phrase::normalize_tokens(&word.phrase),
                        timestamps: Vec::new(),
                        start_time: utterance.start_sample as f32 / 16_000.0,
                    });
                }
            }
        }
        Ok(detections)
    }
}
impl WakeWordBackend for TrainedBackend {
    fn kind(&self) -> &'static str {
        "trained-whisper-encoder"
    }
    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(Stream {
            backend: self,
            live: false,
            resampler: RefCell::new(AudioResampler::new()),
            started: Cell::new(false),
            finished: Cell::new(false),
        })
    }
    fn live_stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(Stream {
            backend: self,
            live: true,
            resampler: RefCell::new(AudioResampler::new()),
            started: Cell::new(false),
            finished: Cell::new(false),
        })
    }
    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (rate, samples) = read_wave(path)?;
        detect_samples(self.stream().as_ref(), rate, &samples)
    }
}
struct Stream<'a> {
    backend: &'a TrainedBackend,
    live: bool,
    resampler: RefCell<AudioResampler>,
    started: Cell<bool>,
    finished: Cell<bool>,
}
impl Stream<'_> {
    fn start(&self) -> Result<()> {
        ensure!(!self.finished.get(), "trained stream is already finished");
        if !self.started.get() {
            ensure!(
                !self.backend.active.get(),
                "trained encoder already has an active stream"
            );
            self.backend.worker.borrow_mut().start()?;
            self.backend.active.set(true);
            self.started.set(true);
        }
        Ok(())
    }
    fn send(&self, samples: &[f32]) -> Result<Vec<Detection>> {
        let mut detected = Vec::new();
        for chunk in samples.chunks(16_000) {
            let utterances = self.backend.worker.borrow_mut().audio(chunk)?;
            detected.extend(self.backend.detections(utterances, self.live)?);
        }
        Ok(detected)
    }
}
impl WakeWordStream for Stream<'_> {
    fn accept(&self, rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        self.start()?;
        self.send(&self.resampler.borrow_mut().accept(rate, samples)?)
    }
    fn finish(&self) -> Result<Vec<Detection>> {
        self.start()?;
        let mut detected = self.send(&self.resampler.borrow_mut().finish()?)?;
        let utterances = self.backend.worker.borrow_mut().finish()?;
        detected.extend(self.backend.detections(utterances, self.live)?);
        self.finished.set(true);
        self.backend.active.set(false);
        Ok(detected)
    }
}
impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if self.started.get() && !self.finished.get() {
            self.backend.active.set(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        enrollment::head::Example,
        test_support::{FakeEmbeddingSession, isolated_paths, unique_directory},
    };
    use std::fs;
    fn split(label: &str) -> Vec<Example> {
        (0..4)
            .map(|i| {
                let mut values = vec![0.0; 512];
                values[0] = if i < 2 { 1.0 } else { -1.0 };
                values[1] = 0.1;
                Example {
                    id: format!("{label}-{i}"),
                    values,
                    positive: i < 2,
                }
            })
            .collect()
    }
    #[test]
    fn threshold_controls_detection_and_only_opted_in_live_streams_save_audio() {
        let root = unique_directory("trained-history", "live");
        let paths = isolated_paths(&root);
        let mut head = Head::train("test-encoder", &split("train"), &split("cal")).unwrap();
        head.validate_held_out(&split("validation")).unwrap();
        let artifact = artifact::install(&root.join("heads"), &head).unwrap();
        let mut config = Config::default();
        config.wake_words[0].enrollment = Some(crate::enrollment::artifact::EnrollmentBinding {
            heads: [("test-encoder".into(), artifact)].into(),
            ..Default::default()
        });
        for (enabled, threshold, live, expected) in [
            (false, None, true, 1),
            (true, None, false, 1),
            (true, Some(1.0), true, 0),
            (true, None, true, 1),
        ] {
            let binding = config.wake_words[0].enrollment.as_mut().unwrap();
            binding.history.enabled = enabled;
            binding.threshold = threshold;
            let backend = TrainedBackend::load_with(&config, &paths, |_, _, _| {
                Ok(Box::new(FakeEmbeddingSession::new("test-encoder")))
            })
            .unwrap();
            let stream = if live {
                backend.live_stream()
            } else {
                backend.stream()
            };
            stream.accept(16000, &vec![0.2; 1600]).unwrap();
            assert_eq!(stream.finish().unwrap().len(), expected);
            let events = crate::enrollment::history::list(&paths, "computer").unwrap();
            assert_eq!(events.len(), usize::from(enabled && live && expected > 0));
        }
        let event = &crate::enrollment::history::list(&paths, "computer").unwrap()[0];
        assert!(event.score >= event.threshold);
        assert_eq!(event.device, "TEST");
        assert!(event.audio.is_file());
        config.wake_words[0].enrollment.as_mut().unwrap().threshold = Some(f32::NAN);
        assert!(
            TrainedBackend::load_with(&config, &paths, |_, _, _| Ok(Box::new(
                FakeEmbeddingSession::new("test-encoder")
            )))
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compatible_heads_share_stream_and_reject_incompatible_encoder() {
        let root = unique_directory("trained-runtime", "heads");
        let paths = isolated_paths(&root);
        let mut head = Head::train("test-encoder", &split("train"), &split("cal")).unwrap();
        head.validate_held_out(&split("heldout")).unwrap();
        let artifact = artifact::install(&root.join("heads"), &head).unwrap();
        let mut config = Config::default();
        let mut second = config.wake_words[0].clone();
        second.id = "second".into();
        config.wake_words.push(second);
        for word in &mut config.wake_words {
            word.enrollment = Some(crate::enrollment::artifact::EnrollmentBinding {
                threshold: None,
                history: Default::default(),
                active: true,
                heads: [("test-encoder".into(), artifact.clone())].into(),
            });
        }
        let backend = TrainedBackend::load_with(&config, &paths, |_, _, _| {
            Ok(Box::new(FakeEmbeddingSession::new("test-encoder")))
        })
        .unwrap();
        let stream = backend.stream();
        assert!(stream.accept(16000, &vec![0.1; 4000]).unwrap().is_empty());
        assert!(backend.stream().accept(16000, &[0.1]).is_err());
        assert_eq!(
            stream
                .finish()
                .unwrap()
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            ["computer", "second"]
        );
        assert!(stream.finish().is_err());
        let negative = backend.stream();
        negative.accept(16000, &vec![-0.1; 4000]).unwrap();
        assert!(negative.finish().unwrap().is_empty());
        let abandoned = backend.stream();
        abandoned.accept(16000, &[0.1; 500]).unwrap();
        drop(abandoned);
        let mut samples = crate::enrollment::SampleSet::create(&paths).unwrap();
        samples.push(&[0.1; 4000]).unwrap();
        assert_eq!(backend.detect_file(&samples.files[0]).unwrap().len(), 2);
        let mismatch = TrainedBackend::load_with(&config, &paths, |_, _, _| {
            Ok(Box::new(FakeEmbeddingSession::new("changed-encoder")))
        });
        assert!(
            mismatch
                .err()
                .unwrap()
                .to_string()
                .contains("existing heads are preserved")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
