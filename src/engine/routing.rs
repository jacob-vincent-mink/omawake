//! Bounded owner threads: native detectors are constructed and used on one
//! thread, while independent groups consume the same audio concurrently.
use super::{
    Detection, Detector, WakeWordBackend, WakeWordStream, audio::read_wave, detect_samples,
};
use crate::{backend::Runtime, config::Config, paths::AppPaths};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::{
    cell::Cell,
    collections::BTreeSet,
    path::Path,
    sync::{
        Arc,
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
const WAIT: Duration = Duration::from_secs(190);
#[derive(Clone, Debug, Serialize)]
pub struct GroupStatus {
    pub profile: String,
    pub backend: String,
    pub runtime: Runtime,
    pub requested_device: String,
    pub fallback_used: bool,
    pub words: Vec<String>,
}
enum Command {
    Start,
    Audio(i32, Arc<[f32]>),
    Finish,
    Cancel,
}
struct Owner {
    tx: Option<SyncSender<Command>>,
    rx: Receiver<Result<Vec<Detection>>>,
    handle: Option<JoinHandle<()>>,
    status: GroupStatus,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.tx.take();
        // Normal shutdown joins immediately. Native IPC is independently
        // bounded; a hung native call cannot hold this thread indefinitely.
        if let Some(handle) = self.handle.take()
            && handle.is_finished()
        {
            let _ = handle.join();
            // Dropping the handle detaches only until its bounded native call
            // completes; closed request/response channels make the owner exit.
        }
    }
}
pub(crate) struct RoutedBackend {
    owners: Vec<Owner>,
    active: Cell<bool>,
    failed: Cell<bool>,
}
pub(crate) fn plans(config: &Config) -> Result<Vec<(String, Config)>> {
    config.validate_engine_references()?;
    let mut result = Vec::new();
    for profile in config.active_engines()? {
        let materialized = config.for_engine(profile.name.as_deref())?;
        for trained in [false, true] {
            let mut sub = materialized.clone();
            sub.wake_words
                .retain(|w| w.enabled && w.uses_trained_head() == trained);
            if !sub.wake_words.is_empty() {
                let label = format!(
                    "{}:{}",
                    profile.name.as_deref().unwrap_or("default"),
                    if trained { "trained" } else { "transcript" }
                );
                result.push((label, sub));
            }
        }
    }
    ensure!(
        result.len() <= 8,
        "at most eight active engine groups are supported"
    );
    Ok(result)
}
impl RoutedBackend {
    pub fn load(groups: Vec<(String, Config)>, paths: &AppPaths) -> Result<Self> {
        Self::load_with(groups, paths, Arc::new(Detector::load_single))
    }
    fn load_with<F>(groups: Vec<(String, Config)>, paths: &AppPaths, loader: Arc<F>) -> Result<Self>
    where
        F: Fn(&Config, &AppPaths) -> Result<Detector> + Send + Sync + 'static,
    {
        ensure!(
            (1..=8).contains(&groups.len()),
            "invalid engine group count"
        );
        // Spawn all before waiting for readiness, so model loads also overlap.
        let mut pending = Vec::new();
        for (label, config) in groups {
            let (tx, commands) = sync_channel(1);
            let (responses, rx) = sync_channel(1);
            let (ready, ready_rx) = sync_channel(1);
            let paths = paths.clone();
            let loader = Arc::clone(&loader);
            let profile = label.clone();
            let handle = thread::Builder::new()
                .name(format!("omawake-{label}"))
                .spawn(move || {
                    let detector = match loader(&config, &paths) {
                        Ok(d) => d,
                        Err(e) => {
                            let _ = ready.send(Err(e));
                            return;
                        }
                    };
                    let status = GroupStatus {
                        profile,
                        backend: detector.backend_kind.into(),
                        runtime: detector.effective_runtime,
                        requested_device: config.backend.device.clone(),
                        fallback_used: detector.fallback_used,
                        words: config
                            .wake_words
                            .iter()
                            .filter(|w| w.enabled)
                            .map(|w| w.id.clone())
                            .collect(),
                    };
                    if ready.send(Ok(status)).is_err() {
                        return;
                    }
                    let mut session = None;
                    while let Ok(command) = commands.recv() {
                        let result = match command {
                            Command::Start => {
                                if session.is_some() {
                                    Err(anyhow::anyhow!("engine stream already active"))
                                } else {
                                    session = Some(detector.session());
                                    Ok(Vec::new())
                                }
                            }
                            Command::Audio(rate, samples) => session
                                .as_ref()
                                .context("engine stream not started")
                                .and_then(|s| s.accept(rate, &samples)),
                            Command::Finish => session
                                .take()
                                .context("engine stream not started")
                                .and_then(|s| s.finish()),
                            Command::Cancel => {
                                session.take();
                                Ok(Vec::new())
                            }
                        };
                        if responses.send(result).is_err() {
                            return;
                        }
                    }
                })?;
            // Owner ensures closing channels even when another group fails.
            pending.push((
                Owner {
                    tx: Some(tx),
                    rx,
                    handle: Some(handle),
                    status: GroupStatus {
                        profile: label,
                        backend: String::new(),
                        runtime: Runtime::Default,
                        requested_device: String::new(),
                        fallback_used: false,
                        words: Vec::new(),
                    },
                },
                ready_rx,
            ));
        }
        let mut owners = Vec::new();
        for (mut owner, ready) in pending {
            owner.status = ready
                .recv_timeout(WAIT)
                .context("engine startup timed out or owner stopped")??;
            owners.push(owner);
        }
        Ok(Self {
            owners,
            active: Cell::new(false),
            failed: Cell::new(false),
        })
    }
    fn exchange(&self, make: impl Fn() -> Command) -> Result<Vec<Detection>> {
        ensure!(
            !self.failed.get(),
            "engine routing failed; reload the detector before sending more audio"
        );
        let result = self.exchange_inner(make);
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }
    fn exchange_inner(&self, make: impl Fn() -> Command) -> Result<Vec<Detection>> {
        for owner in &self.owners {
            owner
                .tx
                .as_ref()
                .context("engine owner stopped")?
                .try_send(make())
                .map_err(|e| {
                    anyhow::anyhow!("engine {} cannot accept request: {e}", owner.status.profile)
                })?;
        }
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        let mut first_error = None;
        // Drain every response even when one group errors, keeping replies
        // aligned with requests and never suppressing inference failures.
        let deadline = std::time::Instant::now() + WAIT;
        for owner in &self.owners {
            match owner
                .rx
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                .context("engine response timed out or owner stopped")
                .and_then(|v| v)
            {
                Ok(detections) => {
                    for d in detections {
                        if seen.insert(d.id.clone()) {
                            result.push(d);
                        }
                    }
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error =
                            Some(error.context(format!("engine {}", owner.status.profile)));
                    }
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(result)
    }
    pub fn statuses(&self) -> Vec<GroupStatus> {
        self.owners.iter().map(|o| o.status.clone()).collect()
    }
}
impl WakeWordBackend for RoutedBackend {
    fn kind(&self) -> &'static str {
        "multi-engine"
    }
    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(Stream {
            backend: self,
            started: Cell::new(false),
            finished: Cell::new(false),
        })
    }
    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (rate, samples) = read_wave(path)?;
        detect_samples(self.stream().as_ref(), rate, &samples)
    }
    fn engine_statuses(&self) -> Vec<GroupStatus> {
        self.statuses()
    }
}
struct Stream<'a> {
    backend: &'a RoutedBackend,
    started: Cell<bool>,
    finished: Cell<bool>,
}
impl Stream<'_> {
    fn start(&self) -> Result<()> {
        ensure!(!self.finished.get(), "routed stream already finished");
        if !self.started.get() {
            ensure!(
                !self.backend.active.get(),
                "another routed stream is active"
            );
            self.backend.exchange(|| Command::Start)?;
            self.started.set(true);
            self.backend.active.set(true);
        }
        Ok(())
    }
}
impl WakeWordStream for Stream<'_> {
    fn accept(&self, rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        ensure!(
            rate > 0 && samples.len() <= 192_000 && samples.iter().all(|s| s.is_finite()),
            "invalid routed audio chunk"
        );
        self.start()?;
        let shared: Arc<[f32]> = samples.into();
        self.backend
            .exchange(|| Command::Audio(rate, Arc::clone(&shared)))
    }
    fn finish(&self) -> Result<Vec<Detection>> {
        self.start()?;
        let result = self.backend.exchange(|| Command::Finish);
        self.finished.set(true);
        self.backend.active.set(false);
        result
    }
}
impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if self.started.get() && !self.finished.get() {
            let _ = self.backend.exchange(|| Command::Cancel);
            self.backend.active.set(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    struct Fake {
        ids: Vec<String>,
        gate: Arc<(Mutex<usize>, Condvar)>,
    }
    struct FakeStream<'a>(&'a Fake);
    impl WakeWordBackend for Fake {
        fn kind(&self) -> &'static str {
            "test-group"
        }
        fn stream(&self) -> Box<dyn WakeWordStream + '_> {
            Box::new(FakeStream(self))
        }
        fn detect_file(&self, _: &Path) -> Result<Vec<Detection>> {
            unreachable!()
        }
    }
    impl WakeWordStream for FakeStream<'_> {
        fn accept(&self, rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
            ensure!(
                rate == 16000 && samples == [0.25, 0.5],
                "PCM was lost or modified"
            );
            let (lock, cv) = &*self.0.gate;
            let mut count = lock.lock().unwrap();
            *count += 1;
            cv.notify_all();
            let (count, wait) = cv
                .wait_timeout_while(count, Duration::from_secs(2), |n| *n < 2)
                .unwrap();
            ensure!(
                !wait.timed_out() && *count >= 2,
                "engine passes did not overlap"
            );
            Ok(self
                .0
                .ids
                .iter()
                .map(|id| Detection {
                    id: id.clone(),
                    tokens: vec!["heard".into()],
                    timestamps: Vec::new(),
                    start_time: 0.0,
                })
                .collect())
        }
        fn finish(&self) -> Result<Vec<Detection>> {
            Ok(Vec::new())
        }
    }
    #[test]
    fn shares_pcm_runs_groups_concurrently_and_reuses_owners() {
        let mut a = Config::default();
        a.wake_words[0].id = "first".into();
        let mut b = a.clone();
        b.wake_words[0].id = "second".into();
        let gate = Arc::new((Mutex::new(0), Condvar::new()));
        let loads = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&loads);
        let shared = Arc::clone(&gate);
        let loader = Arc::new(move |config: &Config, _: &AppPaths| {
            count.fetch_add(1, Ordering::SeqCst);
            Detector::from_backend(
                config,
                Box::new(Fake {
                    ids: config.wake_words.iter().map(|w| w.id.clone()).collect(),
                    gate: Arc::clone(&shared),
                }),
                String::new(),
                Runtime::Default,
                false,
                Duration::ZERO,
            )
        });
        let routed = RoutedBackend::load_with(
            vec![("a".into(), a), ("b".into(), b)],
            &AppPaths::discover(),
            loader,
        )
        .unwrap();
        for _ in 0..2 {
            *gate.0.lock().unwrap() = 0;
            let stream = routed.stream();
            let heard = stream.accept(16000, &[0.25, 0.5]).unwrap();
            assert_eq!(
                heard.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
                ["first", "second"]
            );
            stream.finish().unwrap();
        }
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        let abandoned = routed.stream();
        abandoned.accept(16000, &[0.25, 0.5]).unwrap();
        drop(abandoned);
        routed.stream().finish().unwrap();
        assert_eq!(routed.statuses().len(), 2);
    }
    #[test]
    fn separates_trained_words_and_preserves_profile_ownership() {
        let mut config = Config::default();
        let mut trained = config.wake_words[0].clone();
        trained.id = "unusual".into();
        trained.enrollment = Some(Default::default());
        config.wake_words.push(trained);
        let routes = plans(&config).unwrap();
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].1.wake_words[0].id, "computer");
        assert_eq!(routes[1].1.wake_words[0].id, "unusual");
        assert!(
            routes
                .iter()
                .all(|(_, c)| c.engines.is_empty()
                    && c.wake_words.iter().all(|w| w.engine.is_none()))
        );
    }
}
