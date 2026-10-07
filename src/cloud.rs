//! Opt-in utterance HTTP transcription with local endpointing and phrase policy.
use crate::{
    cloud_http::{self, CloudConfig},
    config::Config,
    engine::{
        Detection, WakeWordBackend, WakeWordStream,
        audio::{AudioResampler, read_wave},
        detect_samples,
    },
    phrase::{PhraseMatcher, normalize_tokens, record_transcript},
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
    time::Duration,
};

pub fn is_cloud(kind: &str) -> bool {
    matches!(kind, "deepgram" | "openai-compatible")
}
pub fn model_name(config: &Config) -> String {
    if !is_cloud(&config.backend.kind) {
        return config.model.name.clone();
    }
    if !config.backend.cloud.model.is_empty() {
        return config.backend.cloud.model.clone();
    }
    if config.backend.kind == "deepgram" {
        "nova-3".into()
    } else {
        "gpt-4o-mini-transcribe".into()
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Settings {
    kind: String,
    cloud: CloudConfig,
    language: String,
}
impl Settings {
    fn validate(&self) -> Result<()> {
        ensure!(is_cloud(&self.kind), "unsupported cloud ASR provider");
        cloud_http::base_url(&self.cloud, self.defaults().0)?;
        cloud_http::credential(
            &self.cloud,
            self.defaults().1,
            self.kind == "deepgram" || self.cloud.base_url.is_empty(),
        )?;
        ensure!(
            self.language.len() <= 32
                && self
                    .language
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-'),
            "invalid cloud language"
        );
        ensure!(
            self.cloud.model.len() <= 256 && !self.cloud.model.contains(['\r', '\n', '\0']),
            "invalid cloud model"
        );
        Ok(())
    }
    fn defaults(&self) -> (&'static str, &'static str, &'static str) {
        if self.kind == "deepgram" {
            ("https://api.deepgram.com/v1", "DEEPGRAM_API_KEY", "nova-3")
        } else {
            (
                "https://api.openai.com/v1",
                "OPENAI_API_KEY",
                "gpt-4o-mini-transcribe",
            )
        }
    }
    fn transcribe(&self, samples: &[f32]) -> Result<String> {
        ensure!(
            !samples.is_empty()
                && samples.len() <= 16000 * 30
                && samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
            "invalid bounded cloud audio"
        );
        let (base, env, default_model) = self.defaults();
        let mut url = cloud_http::base_url(&self.cloud, base)?;
        let prefix = url.path().trim_end_matches('/').to_owned();
        let model = if self.cloud.model.is_empty() {
            default_model
        } else {
            &self.cloud.model
        };
        let key = cloud_http::credential(
            &self.cloud,
            env,
            self.kind == "deepgram" || self.cloud.base_url.is_empty(),
        )?;
        let agent = cloud_http::agent(&self.cloud);
        let wav = encode_wav(samples)?;
        let (mime, body) = if self.kind == "deepgram" {
            url.set_path(&format!("{prefix}/listen"));
            url.query_pairs_mut().append_pair("model", model);
            if !self.language.is_empty() {
                url.query_pairs_mut()
                    .append_pair("language", &self.language);
            }
            ("audio/wav".to_owned(), wav)
        } else {
            url.set_path(&format!("{prefix}/audio/transcriptions"));
            // Values are separate MIME parts, never interpolated into headers or filenames.
            let boundary = "omawake-cloud-utterance-9f4a91e2";
            let mut body = Vec::new();
            for (name, value) in [
                ("model", model),
                ("language", self.language.as_str()),
                ("response_format", "json"),
            ] {
                if name == "language" && value.is_empty() {
                    continue;
                }
                ensure!(!value.contains(boundary), "invalid multipart parameter");
                body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
            }
            body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"utterance.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
            body.extend_from_slice(&wav);
            body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
            (format!("multipart/form-data; boundary={boundary}"), body)
        };
        let mut request = agent
            .post(url.as_str())
            .set("Content-Type", &mime)
            .set("Accept", "application/json");
        if let Some(key) = key {
            request = request.set(
                "Authorization",
                &format!(
                    "{} {key}",
                    if self.kind == "deepgram" {
                        "Token"
                    } else {
                        "Bearer"
                    }
                ),
            );
        }
        let data = cloud_http::read_json(cloud_http::response(request.send_bytes(&body))?)?;
        let transcript = if self.kind == "deepgram" {
            data.pointer("/results/channels/0/alternatives/0/transcript")
        } else {
            data.get("text")
        };
        let text = transcript
            .and_then(|v| v.as_str())
            .context("cloud ASR response lacks a final transcript")?;
        ensure!(text.len() <= 65536, "cloud transcript exceeds limit");
        Ok(text.trim().to_owned())
    }
}
fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(
            &mut cursor,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for s in samples {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
    }
    Ok(cursor.into_inner())
}
#[derive(Serialize, Deserialize)]
struct Utterance {
    start_sample: u64,
    samples: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
struct Reply {
    start_sample: u64,
    text: Option<String>,
    error: Option<String>,
}
pub fn worker_main(settings: &str) -> Result<()> {
    let settings: Settings = serde_json::from_str(settings)?;
    settings.validate()?;
    let mut input = BufReader::new(std::io::stdin().lock());
    let mut output = std::io::stdout().lock();
    loop {
        let mut frame = Vec::new();
        let n = input
            .by_ref()
            .take(8_000_001)
            .read_until(b'\n', &mut frame)?;
        if n == 0 {
            return Ok(());
        }
        ensure!(
            n <= 8_000_000 && frame.last() == Some(&b'\n'),
            "oversized or truncated cloud worker request"
        );
        let utterance: Utterance = serde_json::from_slice(&frame)?;
        let result = settings.transcribe(&utterance.samples);
        let reply = match result {
            Ok(text) => Reply {
                start_sample: utterance.start_sample,
                text: Some(text),
                error: None,
            },
            Err(e) => Reply {
                start_sample: utterance.start_sample,
                text: None,
                error: Some(format!("{e:#}")),
            },
        };
        serde_json::to_writer(&mut output, &reply)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
}
struct Worker {
    child: Child,
    tx: Option<SyncSender<Utterance>>,
    rx: Receiver<Result<Reply>>,
    handle: Option<thread::JoinHandle<()>>,
    pending: usize,
    timeout: Duration,
}
impl Worker {
    fn start(settings: &Settings, executable: &Path) -> Result<Self> {
        let mut command = Command::new(executable);
        command
            .arg("__cloud-worker")
            .arg(serde_json::to_string(settings)?)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        // Parent death closes cloud requests even on forced daemon termination.
        use std::os::unix::process::CommandExt;
        let parent = unsafe { libc::getpid() };
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::other("cloud worker parent exited"));
                }
                Ok(())
            });
        }
        let mut child = command
            .spawn()
            .context("start cloud transcription worker")?;
        let mut input = child.stdin.take().context("missing cloud worker stdin")?;
        let mut output =
            BufReader::new(child.stdout.take().context("missing cloud worker stdout")?);
        let (tx, requests) = mpsc::sync_channel::<Utterance>(2);
        let (results, rx) = mpsc::sync_channel(3);
        let handle = thread::spawn(move || {
            while let Ok(request) = requests.recv() {
                let result = (|| {
                    serde_json::to_writer(&mut input, &request)?;
                    input.write_all(b"\n")?;
                    input.flush()?;
                    let mut line = Vec::new();
                    let n = output
                        .by_ref()
                        .take(1_048_577)
                        .read_until(b'\n', &mut line)?;
                    ensure!(
                        n > 0 && n <= 1_048_576 && line.last() == Some(&b'\n'),
                        "cloud worker disconnected or oversized reply"
                    );
                    let reply: Reply = serde_json::from_slice(&line)?;
                    ensure!(
                        reply.start_sample == request.start_sample,
                        "cloud worker response order mismatch"
                    );
                    Ok(reply)
                })();
                let failed = result.is_err();
                if results.send(result).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            tx: Some(tx),
            rx,
            handle: Some(handle),
            pending: 0,
            timeout: Duration::from_secs(settings.cloud.timeout_seconds + 2),
        })
    }
    fn submit(&mut self, u: Utterance) -> Result<()> {
        ensure!(
            self.pending < 3,
            "cloud transcription queue is full; detection stopped without dropping or replaying speech"
        );
        self.tx
            .as_ref()
            .context("cloud worker closed")?
            .try_send(u)
            .map_err(|_| anyhow::anyhow!("cloud transcription queue full or worker closed"))?;
        self.pending += 1;
        Ok(())
    }
    fn poll(&mut self, wait: bool) -> Result<Vec<Reply>> {
        let mut out = Vec::new();
        while self.pending > 0 {
            let reply = if wait {
                self.rx
                    .recv_timeout(self.timeout)
                    .map_err(|_| anyhow::anyhow!("cloud worker timed out or disconnected"))?
            } else {
                match self.rx.try_recv() {
                    Ok(v) => v,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(_) => bail!("cloud worker disconnected"),
                }
            }?;
            self.pending -= 1;
            if let Some(error) = reply.error {
                bail!("{error}")
            }
            out.push(reply);
        }
        Ok(out)
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.tx.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
pub struct CloudBackend {
    settings: Settings,
    matcher: PhraseMatcher,
    threshold: f32,
    silence_samples: usize,
    max_samples: usize,
    worker_executable: std::path::PathBuf,
}
impl CloudBackend {
    pub fn load(config: &Config) -> Result<Self> {
        ensure!(
            config.backend.runtime == crate::backend::Runtime::Default
                && config.backend.device_id == 0
                && config.backend.fallback == crate::backend::Fallback::Error,
            "cloud ASR requires runtime=default, device_id=0, fallback=error"
        );
        ensure!(
            matches!(
                config.backend.device.to_ascii_lowercase().as_str(),
                "auto" | "cpu" | "remote"
            ),
            "cloud ASR placement is remote"
        );
        ensure!(
            config.backend.options.is_empty(),
            "cloud ASR does not accept native backend options"
        );
        let settings = Settings {
            kind: config.backend.kind.clone(),
            cloud: config.backend.cloud.clone(),
            language: config.model.language.clone(),
        };
        settings.validate()?;
        ensure!(
            settings.cloud.vad_threshold.is_finite()
                && (0.0001..=0.5).contains(&settings.cloud.vad_threshold),
            "cloud vad_threshold must be 0.0001..0.5 RMS"
        );
        ensure!(
            (100..=2000).contains(&settings.cloud.endpoint_milliseconds),
            "cloud endpoint_milliseconds must be 100..2000"
        );
        Ok(Self {
            threshold: settings.cloud.vad_threshold,
            silence_samples: settings.cloud.endpoint_milliseconds as usize * 16,
            max_samples: settings.cloud.max_audio_seconds.min(30) as usize * 16000,
            worker_executable: std::env::current_exe()?,
            matcher: PhraseMatcher::compile(&config.wake_words)?,
            settings,
        })
    }
    fn make_stream(&self, live: bool) -> Box<dyn WakeWordStream + '_> {
        Box::new(CloudStream {
            backend: self,
            live,
            state: RefCell::new(State {
                resampler: AudioResampler::new(),
                pending: Vec::new(),
                idle: std::collections::VecDeque::new(),
                active: None,
                cursor: 0,
                silence: 0,
                worker: None,
                finished: false,
                failed: false,
            }),
        })
    }
    fn detections(&self, replies: Vec<Reply>) -> Vec<Detection> {
        replies
            .into_iter()
            .flat_map(|r| {
                let text = r.text.unwrap_or_default();
                record_transcript(&text);
                let tokens = normalize_tokens(&text);
                self.matcher
                    .matches(&text)
                    .into_iter()
                    .map(move |m| Detection {
                        id: m.id,
                        tokens: tokens[m.start_token..m.end_token].to_vec(),
                        timestamps: vec![],
                        start_time: r.start_sample as f32 / 16000.0,
                    })
            })
            .collect()
    }
}
impl WakeWordBackend for CloudBackend {
    fn kind(&self) -> &'static str {
        if self.settings.kind == "deepgram" {
            "deepgram"
        } else {
            "openai-compatible"
        }
    }
    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        self.make_stream(false)
    }
    fn live_stream(&self) -> Box<dyn WakeWordStream + '_> {
        self.make_stream(true)
    }
    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (rate, samples) = read_wave(path)?;
        detect_samples(self.stream().as_ref(), rate, &samples)
    }
}
struct State {
    resampler: AudioResampler,
    pending: Vec<f32>,
    idle: std::collections::VecDeque<f32>,
    active: Option<Utterance>,
    cursor: u64,
    silence: usize,
    worker: Option<Worker>,
    finished: bool,
    failed: bool,
}
struct CloudStream<'a> {
    backend: &'a CloudBackend,
    live: bool,
    state: RefCell<State>,
}
impl CloudStream<'_> {
    fn submit(&self, state: &mut State, u: Utterance) -> Result<()> {
        if state.worker.is_none() {
            state.worker = Some(Worker::start(
                &self.backend.settings,
                &self.backend.worker_executable,
            )?)
        }
        state.worker.as_mut().unwrap().submit(u)
    }
    fn feed(&self, state: &mut State, audio: Vec<f32>) -> Result<Vec<Detection>> {
        state.pending.extend(audio);
        let mut out = Vec::new();
        while state.pending.len() >= 320 {
            let frame: Vec<_> = state.pending.drain(..320).collect();
            let voiced =
                (frame.iter().map(|v| v * v).sum::<f32>() / 320.0).sqrt() >= self.backend.threshold;
            if state.active.is_none() && voiced {
                let samples: Vec<_> = state.idle.drain(..).collect();
                state.active = Some(Utterance {
                    start_sample: state.cursor.saturating_sub(samples.len() as u64),
                    samples,
                });
                state.silence = 0;
            }
            state.cursor += 320;
            if let Some(active) = &mut state.active {
                let remaining = self
                    .backend
                    .max_samples
                    .saturating_sub(active.samples.len());
                active
                    .samples
                    .extend_from_slice(&frame[..remaining.min(320)]);
                state.silence = if voiced { 0 } else { state.silence + 320 };
                if state.silence >= self.backend.silence_samples
                    || active.samples.len() >= self.backend.max_samples
                {
                    let u = state.active.take().unwrap();
                    self.submit(state, u)?;
                    out.extend(
                        self.backend
                            .detections(state.worker.as_mut().unwrap().poll(!self.live)?),
                    );
                }
            } else {
                state.idle.extend(frame);
                while state.idle.len() > 3200 {
                    state.idle.pop_front();
                }
            }
        }
        if let Some(w) = &mut state.worker {
            out.extend(self.backend.detections(w.poll(false)?));
        }
        Ok(out)
    }
    fn fail<T>(&self, state: &mut State, result: Result<T>) -> Result<T> {
        if result.is_err() {
            state.failed = true;
            state.worker.take();
            state.active.take();
            state.pending.clear();
        }
        result
    }
}
impl WakeWordStream for CloudStream<'_> {
    fn accept(&self, rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("cloud stream already in use"))?;
        ensure!(
            !state.finished && !state.failed,
            "cloud stream is closed; create a new session"
        );
        ensure!(
            (8000..=192000).contains(&rate)
                && samples.len() <= rate as usize * 30
                && samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
            "invalid cloud stream audio"
        );
        let result = (|| {
            let audio = state.resampler.accept(rate, samples)?;
            self.feed(&mut state, audio)
        })();
        self.fail(&mut state, result)
    }
    fn finish(&self) -> Result<Vec<Detection>> {
        let mut state = self.state.borrow_mut();
        if state.finished {
            return Ok(vec![]);
        }
        ensure!(!state.failed, "cloud stream failed");
        state.finished = true;
        let result = (|| {
            let tail = state.resampler.finish()?;
            let mut out = self.feed(&mut state, tail)?;
            if let Some(mut u) = state.active.take() {
                let remaining = self.backend.max_samples.saturating_sub(u.samples.len());
                u.samples
                    .extend_from_slice(&state.pending[..remaining.min(state.pending.len())]);
                self.submit(&mut state, u)?;
            }
            if let Some(w) = &mut state.worker {
                out.extend(self.backend.detections(w.poll(true)?));
            }
            state.worker.take();
            Ok(out)
        })();
        self.fail(&mut state, result)
    }
}
#[cfg(test)]
#[path = "../tests/unit/cloud.rs"]
mod tests;
