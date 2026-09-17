//! Persistent, supervised native encoder. One process owns an encoder and VAD;
//! all compatible phrase heads share its utterance embeddings.
use super::embedding::{Embedding, Encoder};
use super::openvino_genai::{AudioCppVad, ring::ActivityBuffer};
use crate::{config::Config, paths::AppPaths};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs,
    io::{Read, Write},
    os::unix::{
        fs::DirBuilderExt,
        net::{UnixListener, UnixStream},
    },
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

const LIMIT: usize = 8 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(180);
static NEXT: AtomicU64 = AtomicU64::new(0);
#[derive(Serialize, Deserialize)]
enum Request {
    Init {
        config: Box<Config>,
        paths: AppPaths,
    },
    Encode {
        samples: Vec<f32>,
    },
    Start,
    Audio {
        samples: Vec<f32>,
    },
    Finish,
}
#[derive(Serialize, Deserialize)]
pub(crate) struct EncodedUtterance {
    pub embedding: Embedding,
    pub start_sample: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audio: Vec<i16>,
}
#[derive(Serialize, Deserialize)]
enum Response {
    Ready {
        contract: String,
        execution_devices: String,
    },
    Encoded(Embedding),
    Utterances(Vec<EncodedUtterance>),
    Error(String),
}

struct Wire {
    socket: UnixStream,
    cancel: Arc<AtomicBool>,
}
impl Wire {
    fn new(socket: UnixStream, cancel: Arc<AtomicBool>) -> Result<Self> {
        socket.set_read_timeout(Some(Duration::from_millis(50)))?;
        socket.set_write_timeout(Some(Duration::from_millis(50)))?;
        Ok(Self { socket, cancel })
    }
    fn transfer(&mut self, bytes: &mut [u8], write: bool, deadline: Instant) -> Result<()> {
        let mut offset = 0;
        while offset < bytes.len() {
            ensure!(
                !self.cancel.load(Ordering::Relaxed),
                "embedding request cancelled"
            );
            ensure!(Instant::now() < deadline, "embedding worker timed out");
            let operation = if write {
                self.socket.write(&bytes[offset..])
            } else {
                self.socket.read(&mut bytes[offset..])
            };
            match operation {
                Ok(0) => bail!("embedding worker disconnected"),
                Ok(n) => offset += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock
                            | std::io::ErrorKind::TimedOut
                            | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
    fn send<T: Serialize>(&mut self, value: &T) -> Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        ensure!(bytes.len() <= LIMIT, "embedding protocol message too large");
        let deadline = Instant::now() + TIMEOUT;
        self.transfer(&mut (bytes.len() as u32).to_le_bytes(), true, deadline)?;
        self.transfer(&mut bytes, true, deadline)
    }
    fn receive<T: DeserializeOwned>(&mut self) -> Result<T> {
        let deadline = Instant::now() + TIMEOUT;
        let mut length = [0; 4];
        self.transfer(&mut length, false, deadline)?;
        let length = u32::from_le_bytes(length) as usize;
        ensure!(length <= LIMIT, "embedding protocol message too large");
        let mut bytes = vec![0; length];
        self.transfer(&mut bytes, false, deadline)?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

pub(crate) trait EmbeddingSession {
    fn contract(&self) -> &str;
    fn execution_devices(&self) -> &str;
    fn start(&mut self) -> Result<()>;
    fn audio(&mut self, samples: &[f32]) -> Result<Vec<EncodedUtterance>>;
    fn finish(&mut self) -> Result<Vec<EncodedUtterance>>;
}
pub(crate) fn load_session(
    config: &Config,
    paths: &AppPaths,
    cancel: Arc<AtomicBool>,
) -> Result<Box<dyn EmbeddingSession>> {
    Ok(Box::new(Worker::load(config, paths, cancel)?))
}
pub(crate) struct Worker {
    child: Child,
    wire: Wire,
    failed: bool,
    pub contract: String,
    pub execution_devices: String,
}
impl Worker {
    pub fn load(config: &Config, paths: &AppPaths, cancel: Arc<AtomicBool>) -> Result<Self> {
        fs::create_dir_all(&paths.runtime_dir)?;
        let directory = paths.runtime_dir.join(format!(
            "encoder-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&directory)?;
        let result = Self::launch(
            config,
            paths,
            cancel,
            &directory,
            Command::new(std::env::current_exe()?),
        );
        // The accepted socket stays open; unlink only our fresh rendezvous dir.
        let _ = fs::remove_dir_all(&directory);
        result
    }
    fn launch(
        config: &Config,
        paths: &AppPaths,
        cancel: Arc<AtomicBool>,
        directory: &Path,
        mut command: Command,
    ) -> Result<Self> {
        let address = directory.join("ipc");
        let listener = UnixListener::bind(&address)?;
        listener.set_nonblocking(true)?;
        command
            .arg("__embedding-worker")
            .arg(&address)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        let mut dirs = config.backend.library_dirs.clone();
        if let Some(parent) = config
            .backend
            .library
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            dirs.push(parent.to_owned());
        }
        if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
            dirs.extend(std::env::split_paths(&existing));
        }
        command.env("LD_LIBRARY_PATH", std::env::join_paths(dirs)?);
        let mut child = command.spawn().context("start native embedding worker")?;
        let deadline = Instant::now() + TIMEOUT;
        let connected = loop {
            match listener.accept() {
                Ok((socket, _)) => break Ok(socket),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => break Err(e.into()),
            }
            if cancel.load(Ordering::Relaxed) {
                break Err(anyhow::anyhow!("encoder startup cancelled"));
            }
            if Instant::now() >= deadline {
                break Err(anyhow::anyhow!("encoder startup timed out"));
            }
            match child.try_wait() {
                Ok(None) => {}
                Ok(Some(s)) => break Err(anyhow::anyhow!("encoder worker exited: {s}")),
                Err(e) => break Err(e.into()),
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let socket = match connected {
            Ok(s) => s,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let wire = match Wire::new(socket, cancel) {
            Ok(w) => w,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let mut worker = Self {
            child,
            wire,
            failed: false,
            contract: String::new(),
            execution_devices: String::new(),
        };
        match worker.exchange(Request::Init {
            config: Box::new(config.clone()),
            paths: paths.clone(),
        })? {
            Response::Ready {
                contract,
                execution_devices,
            } => {
                ensure!(
                    !contract.is_empty()
                        && contract.len() <= 256
                        && !execution_devices.is_empty()
                        && execution_devices.len() <= 256,
                    "invalid encoder handshake"
                );
                worker.contract = contract;
                worker.execution_devices = execution_devices;
            }
            _ => bail!("unexpected encoder handshake"),
        }
        Ok(worker)
    }
    fn exchange(&mut self, request: Request) -> Result<Response> {
        ensure!(!self.failed, "encoder worker failed; reload the detector");
        let result = (|| {
            self.wire.send(&request)?;
            let response: Response = self.wire.receive()?;
            match (&request, &response) {
                (_, Response::Error(message)) => bail!("{message}"),
                (Request::Init { .. }, Response::Ready { .. }) => {}
                (Request::Start, Response::Utterances(v)) if v.is_empty() => {}
                (Request::Audio { .. } | Request::Finish, Response::Utterances(v)) => {
                    ensure!(v.len() <= 32, "too many encoder utterances");
                    for item in v {
                        item.embedding.validate()?;
                        ensure!(item.audio.len() <= 480_000, "oversized history audio");
                        ensure!(
                            item.embedding.encoder_contract == self.contract,
                            "encoder contract changed in response"
                        );
                    }
                }
                _ => bail!("unexpected encoder protocol response"),
            }
            Ok(response)
        })();
        if result.is_err() {
            self.failed = true;
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        result
    }
    pub fn start(&mut self) -> Result<()> {
        ensure!(
            matches!(self.exchange(Request::Start)?, Response::Utterances(ref v) if v.is_empty()),
            "unexpected encoder stream response"
        );
        Ok(())
    }
    pub fn audio(&mut self, samples: &[f32]) -> Result<Vec<EncodedUtterance>> {
        check_audio(samples, 16_000)?;
        self.utterances(Request::Audio {
            samples: samples.to_vec(),
        })
    }
    pub fn finish(&mut self) -> Result<Vec<EncodedUtterance>> {
        self.utterances(Request::Finish)
    }
    fn utterances(&mut self, request: Request) -> Result<Vec<EncodedUtterance>> {
        match self.exchange(request)? {
            Response::Utterances(v) => Ok(v),
            _ => bail!("unexpected encoder stream response"),
        }
    }
}
impl EmbeddingSession for Worker {
    fn contract(&self) -> &str {
        &self.contract
    }
    fn execution_devices(&self) -> &str {
        &self.execution_devices
    }
    fn start(&mut self) -> Result<()> {
        Worker::start(self)
    }
    fn audio(&mut self, samples: &[f32]) -> Result<Vec<EncodedUtterance>> {
        Worker::audio(self, samples)
    }
    fn finish(&mut self) -> Result<Vec<EncodedUtterance>> {
        Worker::finish(self)
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn check_audio(samples: &[f32], maximum: usize) -> Result<()> {
    ensure!(
        !samples.is_empty()
            && samples.len() <= maximum
            && samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
        "invalid or oversized encoder audio"
    );
    Ok(())
}

pub(crate) fn main(address: &Path) -> Result<()> {
    super::openvino_genai::harden_worker_process()?;
    // This process is owned by one client; a killed client must not leave an
    // accelerator inference process behind. Check the race around prctl.
    let parent = unsafe { libc::getppid() };
    ensure!(
        unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } == 0,
        "set embedding worker parent-death signal: {}",
        std::io::Error::last_os_error()
    );
    ensure!(
        unsafe { libc::getppid() } == parent,
        "embedding client exited during startup"
    );
    let mut wire = Wire::new(
        UnixStream::connect(address)?,
        Arc::new(AtomicBool::new(false)),
    )?;
    let init = wire.receive()?;
    let (config, paths) = match init {
        Request::Init { config, paths } => (config, paths),
        _ => bail!("expected encoder init"),
    };
    let result = worker_loop(&mut wire, &config, &paths);
    if let Err(error) = result {
        let _ = wire.send(&Response::Error(format!("{error:#}")));
    }
    Ok(())
}
trait ActivityDetector {
    fn restart(&mut self) -> Result<()>;
    fn activity(&mut self, samples: &[f32]) -> Result<super::openvino_genai::ring::Activity>;
}
impl ActivityDetector for AudioCppVad {
    fn restart(&mut self) -> Result<()> {
        AudioCppVad::restart(self)
    }
    fn activity(&mut self, samples: &[f32]) -> Result<super::openvino_genai::ring::Activity> {
        AudioCppVad::activity(self, samples)
    }
}
fn worker_loop(wire: &mut Wire, config: &Config, paths: &AppPaths) -> Result<()> {
    let mut encoder = Encoder::load(config, paths)?;
    serve(
        wire,
        config
            .wake_words
            .iter()
            .any(|w| w.enabled && w.enrollment.as_ref().is_some_and(|e| e.history.enabled)),
        |samples| encoder.encode_samples(samples),
        || {
            let library =
                super::audiocpp::resolve_bundled_library(paths, &config.backend.library_dirs)?;
            let model = config.model_directory(paths).join(&config.model.vad);
            Ok(Box::new(AudioCppVad::open(
                &library,
                &model,
                config.backend.threads as i32,
            )?) as Box<dyn ActivityDetector>)
        },
    )
}
fn history_audio(enabled: bool, samples: &[f32]) -> Vec<i16> {
    if enabled {
        samples
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .collect()
    } else {
        Vec::new()
    }
}

fn serve(
    wire: &mut Wire,
    retain_audio: bool,
    mut encode: impl FnMut(&[f32]) -> Result<Embedding>,
    mut load_vad: impl FnMut() -> Result<Box<dyn ActivityDetector>>,
) -> Result<()> {
    // Obtain the contract using the same input path as inference. The dummy
    // embedding is never treated as a training example or a detection.
    let probe = encode(&vec![0.0; 1600])?;
    wire.send(&Response::Ready {
        contract: probe.encoder_contract,
        execution_devices: probe.execution_devices,
    })?;
    let mut vad: Option<Box<dyn ActivityDetector>> = None;
    let mut ring = ActivityBuffer::new();
    let mut pending = Vec::new();
    let mut active = false;
    loop {
        let response = match wire.receive()? {
            Request::Init { .. } => bail!("encoder already initialized"),
            Request::Encode { samples } => {
                check_audio(&samples, 480_000)?;
                Response::Encoded(encode(&samples)?)
            }
            Request::Start => {
                if vad.is_none() {
                    vad = Some(load_vad()?);
                }
                vad.as_mut().context("VAD unavailable")?.restart()?;
                ring = ActivityBuffer::new();
                pending.clear();
                active = true;
                Response::Utterances(Vec::new())
            }
            Request::Audio { samples } => {
                ensure!(active, "encoder stream not started");
                check_audio(&samples, 16_000)?;
                let vad = vad.as_mut().context("encoder stream not started")?;
                pending.extend(samples);
                let mut outputs = Vec::new();
                while pending.len() >= 512 {
                    let frame: Vec<_> = pending.drain(..512).collect();
                    if let Some(utterance) = ring.push(&frame, vad.activity(&frame)?) {
                        outputs.push(EncodedUtterance {
                            embedding: encode(&utterance.samples)?,
                            start_sample: utterance.start_sample,
                            audio: history_audio(retain_audio, &utterance.samples),
                        });
                    }
                }
                Response::Utterances(outputs)
            }
            Request::Finish => {
                ensure!(active, "encoder stream not started");
                active = false;
                let vad = vad.as_mut().context("encoder stream not started")?;
                let mut outputs = Vec::new();
                if !pending.is_empty() {
                    pending.resize(512, 0.0);
                    if let Some(u) = ring.push(&pending, vad.activity(&pending)?) {
                        outputs.push(EncodedUtterance {
                            embedding: encode(&u.samples)?,
                            start_sample: u.start_sample,
                            audio: history_audio(retain_audio, &u.samples),
                        });
                    }
                    pending.clear();
                }
                if let Some(u) = ring.finish() {
                    outputs.push(EncodedUtterance {
                        embedding: encode(&u.samples)?,
                        start_sample: u.start_sample,
                        audio: history_audio(retain_audio, &u.samples),
                    });
                }
                Response::Utterances(outputs)
            }
        };
        wire.send(&response)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{isolated_paths, unique_directory};
    #[test]
    fn controlled_peer_covers_startup_protocol_errors_and_cleanup() {
        for mode in [
            "ok",
            "startup-error",
            "disconnect",
            "error",
            "oversized",
            "bad-embedding",
        ] {
            let root = unique_directory("embedding-peer", mode);
            let paths = isolated_paths(&root);
            let config = Config::default();
            let mut command = Command::new("python3");
            command
                .arg(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/embedding_peer.py"
                ))
                .arg(mode);
            let cancel = Arc::new(AtomicBool::new(false));
            let worker = Worker::launch(&config, &paths, Arc::clone(&cancel), &root, command);
            if mode == "startup-error" {
                assert!(
                    worker
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("controlled unavailable")
                );
            } else {
                let mut worker = worker.unwrap();
                assert_eq!(worker.contract(), "test-encoder");
                assert_eq!(worker.execution_devices(), "TEST");
                worker.start().unwrap();
                assert!(worker.audio(&[]).is_err());
                let result = worker.audio(&[0.1; 512]);
                if mode == "ok" {
                    assert!(result.unwrap().is_empty());
                    worker.finish().unwrap();
                    cancel.store(true, Ordering::Relaxed);
                    assert!(worker.start().is_err());
                } else {
                    assert!(result.is_err(), "mode {mode}");
                }
                let pid = worker.child.id();
                drop(worker);
                assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
            }
            fs::remove_dir_all(root).unwrap();
        }
    }
    struct Vad {
        frames: usize,
    }
    impl ActivityDetector for Vad {
        fn restart(&mut self) -> Result<()> {
            self.frames = 0;
            Ok(())
        }
        fn activity(&mut self, _: &[f32]) -> Result<super::super::openvino_genai::ring::Activity> {
            self.frames += 1;
            Ok(super::super::openvino_genai::ring::Activity {
                start_before_frame_end: (self.frames == 1).then_some(512),
                end_before_frame_end: (self.frames == 3).then_some(0),
            })
        }
    }
    fn embedding() -> Embedding {
        Embedding {
            encoder_contract: "test-encoder".into(),
            values: vec![0.1; 512],
            source_frames: 10,
            inference_ms: 1.0,
            execution_devices: "TEST".into(),
        }
    }
    #[test]
    fn persistent_service_segments_audio_and_restarts_without_reloading_vad() {
        for retain_audio in [false, true] {
            let (parent, child) = UnixStream::pair().unwrap();
            let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let count = Arc::clone(&loads);
            let handle = std::thread::spawn(move || {
                let mut wire = Wire::new(child, Arc::new(AtomicBool::new(false))).unwrap();
                serve(
                    &mut wire,
                    retain_audio,
                    |_| Ok(embedding()),
                    || {
                        count.fetch_add(1, Ordering::Relaxed);
                        Ok(Box::new(Vad { frames: 0 }) as Box<dyn ActivityDetector>)
                    },
                )
            });
            let mut wire = Wire::new(parent, Arc::new(AtomicBool::new(false))).unwrap();
            assert!(matches!(
                wire.receive::<Response>().unwrap(),
                Response::Ready { .. }
            ));
            wire.send(&Request::Encode {
                samples: vec![0.1; 1600],
            })
            .unwrap();
            assert!(matches!(
                wire.receive::<Response>().unwrap(),
                Response::Encoded(_)
            ));
            for samples in [vec![0.1; 8000], vec![0.1; 1000]] {
                wire.send(&Request::Start).unwrap();
                let _: Response = wire.receive().unwrap();
                wire.send(&Request::Audio { samples }).unwrap();
                let a: Response = wire.receive().unwrap();
                wire.send(&Request::Finish).unwrap();
                let b: Response = wire.receive().unwrap();
                let count = match (a, b) {
                    (Response::Utterances(a), Response::Utterances(b)) => {
                        assert!(
                            a.iter()
                                .chain(&b)
                                .all(|u| u.audio.is_empty() != retain_audio)
                        );
                        a.len() + b.len()
                    }
                    _ => 0,
                };
                assert_eq!(count, 1);
            }
            wire.send(&Request::Audio {
                samples: vec![0.1; 512],
            })
            .unwrap();
            assert!(
                handle
                    .join()
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("stream not started")
            );
            assert_eq!(loads.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn startup_failure_removes_owned_rendezvous_directory() {
        let root = unique_directory("ew", "exit");
        let paths = isolated_paths(&root);
        // The unit-test executable does not implement the hidden CLI entry.
        // It exits normally with a CLI error; no native library is loaded.
        let error = load_session(&Config::default(), &paths, Arc::new(AtomicBool::new(false)))
            .err()
            .unwrap();
        assert!(error.to_string().contains("worker exited"), "{error:#}");
        assert_eq!(fs::read_dir(&paths.runtime_dir).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn wire_bounds_and_cancel_are_checked_before_payload_allocation() {
        let (a, mut b) = UnixStream::pair().unwrap();
        let mut wire = Wire::new(a, Arc::new(AtomicBool::new(false))).unwrap();
        b.write_all(&((LIMIT + 1) as u32).to_le_bytes()).unwrap();
        assert!(wire.receive::<Response>().is_err());
        assert!(wire.send(&"x".repeat(LIMIT + 1)).is_err());
        wire.cancel.store(true, Ordering::Relaxed);
        assert!(wire.send(&Request::Start).is_err());
    }
}
