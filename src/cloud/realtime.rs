//! Supervised full-session Deepgram transport. Never reconnects or replays audio.
use super::*;
use std::{
    net::{TcpStream, ToSocketAddrs},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use tungstenite::{
    Message, client::IntoClientRequest, protocol::WebSocketConfig, stream::MaybeTlsStream,
};

#[derive(Serialize, Deserialize)]
enum Input {
    Audio(Vec<f32>),
    Finish,
}
#[derive(Serialize, Deserialize)]
enum Output {
    Final(Reply),
    Done,
}

/// Aggregate final segments until the provider marks an utterance complete.
/// Interim text and repeated/overlapping finalized ranges never trigger actions.
#[derive(Default)]
struct Finals {
    text: String,
    start: Option<f64>,
    end: f64,
    word_end: Option<f64>,
}
impl Finals {
    fn take(&mut self) -> Option<Reply> {
        let start = self.start.take()?;
        let text = std::mem::take(&mut self.text);
        self.word_end = None;
        (!text.trim().is_empty()).then_some(Reply {
            start_sample: (start * 16000.0) as u64,
            text: Some(text),
            error: None,
        })
    }
    fn accept(&mut self, data: &serde_json::Value, max_seconds: u32) -> Result<Option<Reply>> {
        match data.get("type").and_then(|v| v.as_str()) {
            Some("Error") => bail!("Deepgram realtime provider reported an error"),
            Some("UtteranceEnd") => {
                let last = data.get("last_word_end").and_then(|v| v.as_f64());
                return Ok(
                    if last.is_some_and(|last| {
                        last.is_finite() && self.word_end.is_some_and(|end| last + 0.001 >= end)
                    }) {
                        self.take()
                    } else {
                        None
                    },
                );
            }
            Some("Results") => (),
            _ => return Ok(None),
        }
        if data.get("is_final").and_then(|v| v.as_bool()) != Some(true) {
            return Ok(None);
        }
        let start = data
            .get("start")
            .and_then(|v| v.as_f64())
            .context("realtime final lacks start")?;
        let duration = data
            .get("duration")
            .and_then(|v| v.as_f64())
            .context("realtime final lacks duration")?;
        ensure!(
            start.is_finite()
                && duration.is_finite()
                && start >= 0.0
                && duration >= 0.0
                && start + duration < 1e9,
            "invalid realtime final timestamps"
        );
        let end = start + duration;
        // Providers may repeat results after endpointing. Do not append or emit twice.
        if end > self.end && start + 0.001 >= self.end {
            let text = data
                .pointer("/channel/alternatives/0/transcript")
                .and_then(|v| v.as_str())
                .context("realtime final lacks transcript")?;
            if !text.trim().is_empty() {
                self.start.get_or_insert(start);
                if !self.text.is_empty() {
                    self.text.push(' ');
                }
                self.text.push_str(text.trim());
                let word_end = data
                    .pointer("/channel/alternatives/0/words")
                    .and_then(|v| v.as_array())
                    .and_then(|words| words.last())
                    .and_then(|word| word.get("end"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(end);
                ensure!(
                    word_end.is_finite() && word_end >= start && word_end <= end + 0.001,
                    "invalid realtime word timestamps"
                );
                self.word_end = Some(word_end);
                ensure!(
                    self.text.len() <= 65536,
                    "realtime transcript exceeds limit"
                );
            }
            self.end = end;
        }
        ensure!(
            self.start
                .is_none_or(|start| self.end - start <= max_seconds as f64 + 1.0),
            "realtime utterance exceeds max_audio_seconds; session stopped"
        );
        if data.get("speech_final").and_then(|v| v.as_bool()) == Some(true)
            && self
                .start
                .is_some_and(|utterance_start| start >= utterance_start && end + 0.001 >= self.end)
        {
            Ok(self.take())
        } else {
            Ok(None)
        }
    }
}
fn socket(settings: &Settings) -> Result<tungstenite::WebSocket<MaybeTlsStream<TcpStream>>> {
    settings.validate()?;
    let mut url = cloud_http::base_url(&settings.cloud, settings.defaults().0)?;
    let prefix = url.path().trim_end_matches('/').to_owned();
    let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
    url.set_scheme(scheme)
        .map_err(|_| anyhow::anyhow!("invalid realtime URL"))?;
    url.set_path(&format!("{prefix}/listen"));
    let model = if settings.cloud.model.is_empty() {
        settings.defaults().2
    } else {
        &settings.cloud.model
    };
    url.query_pairs_mut()
        .append_pair("model", model)
        .append_pair("encoding", "linear16")
        .append_pair("sample_rate", "16000")
        .append_pair("channels", "1")
        .append_pair("interim_results", "true")
        .append_pair(
            "endpointing",
            &settings.cloud.endpoint_milliseconds.to_string(),
        )
        .append_pair("utterance_end_ms", "1000");
    if !settings.language.is_empty() {
        url.query_pairs_mut()
            .append_pair("language", &settings.language);
    }
    let key = cloud_http::credential(&settings.cloud, settings.defaults().1, true)?.unwrap();
    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|_| anyhow::anyhow!("invalid realtime request"))?;
    request.headers_mut().insert(
        "Authorization",
        format!("Token {key}")
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid realtime credential"))?,
    );
    let address = (
        url.host_str()
            .context("missing realtime host")?
            .trim_matches(['[', ']']),
        url.port_or_known_default()
            .context("missing realtime port")?,
    );
    let mut stream = None;
    for address in address
        .to_socket_addrs()
        .map_err(|_| anyhow::anyhow!("realtime address resolution failed"))?
    {
        if let Ok(socket) = TcpStream::connect_timeout(
            &address,
            Duration::from_secs(10.min(settings.cloud.timeout_seconds)),
        ) {
            stream = Some(socket);
            break;
        }
    }
    let stream = stream.context("realtime connection failed")?;
    let timeout = Some(Duration::from_secs(settings.cloud.timeout_seconds));
    stream.set_read_timeout(timeout)?;
    stream.set_write_timeout(timeout)?;
    let config = WebSocketConfig::default()
        .max_message_size(Some(1_048_576))
        .max_frame_size(Some(1_048_576))
        .max_write_buffer_size(1_048_576);
    let (mut socket, _) = tungstenite::client_tls_with_config(request, stream, Some(config), None)
        .map_err(|_| anyhow::anyhow!("realtime handshake failed"))?;
    tcp(&mut socket).set_read_timeout(Some(Duration::from_millis(20)))?;
    Ok(socket)
}
fn tcp(socket: &mut tungstenite::WebSocket<MaybeTlsStream<TcpStream>>) -> &mut TcpStream {
    match socket.get_mut() {
        MaybeTlsStream::Plain(stream) => stream,
        MaybeTlsStream::Rustls(stream) => &mut stream.sock,
        _ => unreachable!("only rustls and plain loopback transports are enabled"),
    }
}
pub(super) fn worker_main(settings: &str) -> Result<()> {
    let settings: Settings = serde_json::from_str(settings)?;
    let socket = socket(&settings)?;
    let (tx, rx) = mpsc::sync_channel(8);
    thread::spawn(move || {
        let mut input = BufReader::new(std::io::stdin().lock());
        loop {
            let value = (|| {
                let mut line = Vec::new();
                let n = input
                    .by_ref()
                    .take(1_048_577)
                    .read_until(b'\n', &mut line)?;
                if n == 0 {
                    return Ok(Input::Finish);
                }
                ensure!(
                    n <= 1_048_576 && line.last() == Some(&b'\n'),
                    "invalid realtime input frame"
                );
                Ok(serde_json::from_slice::<Input>(&line)?)
            })();
            let finished = matches!(value, Ok(Input::Finish)) || value.is_err();
            if tx.send(value).is_err() || finished {
                break;
            }
        }
    });
    worker_loop(&settings, socket, &rx, &mut std::io::stdout().lock())
}
fn emit(output: &mut impl Write, value: &Output) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}
fn worker_loop(
    settings: &Settings,
    mut socket: tungstenite::WebSocket<MaybeTlsStream<TcpStream>>,
    input: &Receiver<Result<Input>>,
    output: &mut impl Write,
) -> Result<()> {
    let mut finals = Finals::default();
    let mut closing = None;
    let mut last_send = Instant::now();
    let mut last_response = Instant::now();
    let mut last_heartbeat = Instant::now();
    let heartbeat_interval =
        Duration::from_millis((settings.cloud.timeout_seconds * 500).min(3000));
    let timeout = Duration::from_secs(settings.cloud.timeout_seconds);
    loop {
        if closing.is_none() {
            match input.try_recv() {
                Ok(Ok(Input::Audio(samples))) => {
                    ensure!(
                        !samples.is_empty()
                            && samples.len() <= 1600
                            && samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                        "invalid realtime PCM frame"
                    );
                    let bytes: Vec<u8> = samples
                        .iter()
                        .flat_map(|v| ((v * 32767.0) as i16).to_le_bytes())
                        .collect();
                    socket
                        .send(Message::Binary(bytes.into()))
                        .map_err(|_| anyhow::anyhow!("realtime audio send failed"))?;
                    last_send = Instant::now();
                }
                Ok(Ok(Input::Finish)) | Err(mpsc::TryRecvError::Disconnected) => {
                    socket
                        .send(Message::Text("{\"type\":\"CloseStream\"}".into()))
                        .map_err(|_| anyhow::anyhow!("realtime finalize failed"))?;
                    closing = Some(Instant::now());
                }
                Ok(Err(error)) => return Err(error),
                Err(mpsc::TryRecvError::Empty) => {
                    if last_send.elapsed() >= Duration::from_secs(3) {
                        socket
                            .send(Message::Text("{\"type\":\"KeepAlive\"}".into()))
                            .map_err(|_| anyhow::anyhow!("realtime keepalive failed"))?;
                        last_send = Instant::now();
                    }
                }
            }
        }
        if closing.is_none() && last_heartbeat.elapsed() >= heartbeat_interval {
            socket
                .send(Message::Ping(Vec::new().into()))
                .map_err(|_| anyhow::anyhow!("realtime ping failed"))?;
            last_heartbeat = Instant::now();
        }
        match socket.read() {
            Ok(Message::Text(text)) => {
                last_response = Instant::now();
                let data: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|_| anyhow::anyhow!("invalid realtime response JSON"))?;
                if let Some(reply) = finals.accept(&data, settings.cloud.max_audio_seconds)? {
                    emit(output, &Output::Final(reply))?;
                }
                if closing.is_some()
                    && data.get("type").and_then(|v| v.as_str()) == Some("Metadata")
                {
                    if let Some(reply) = finals.take() {
                        emit(output, &Output::Final(reply))?;
                    }
                    emit(output, &Output::Done)?;
                    return Ok(());
                }
            }
            Ok(Message::Close(_)) | Err(tungstenite::Error::ConnectionClosed) => {
                ensure!(
                    closing.is_some(),
                    "realtime provider disconnected; no audio replayed"
                );
                if let Some(reply) = finals.take() {
                    emit(output, &Output::Final(reply))?;
                }
                emit(output, &Output::Done)?;
                return Ok(());
            }
            Ok(_) => {
                last_response = Instant::now();
                socket
                    .flush()
                    .map_err(|_| anyhow::anyhow!("realtime control send failed"))?;
            }
            Err(tungstenite::Error::Io(error))
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => bail!("realtime receive failed; no audio replayed"),
        }
        ensure!(
            !closing.is_some_and(|at| at.elapsed() >= timeout),
            "realtime finalization timed out"
        );
        ensure!(
            closing.is_some() || last_response.elapsed() < timeout,
            "realtime provider response timed out"
        );
    }
}

struct RealtimeWorker {
    child: Child,
    tx: Option<SyncSender<Input>>,
    rx: Receiver<Result<Output>>,
    writer: Option<thread::JoinHandle<()>>,
    reader: Option<thread::JoinHandle<()>>,
    failed: Arc<AtomicBool>,
    timeout: Duration,
}
impl RealtimeWorker {
    fn start(backend: &CloudBackend) -> Result<Self> {
        let mut command = Command::new(&backend.worker_executable);
        command
            .arg("__cloud-stream-worker")
            .arg(serde_json::to_string(&backend.settings)?)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        use std::os::unix::process::CommandExt;
        let parent = unsafe { libc::getpid() };
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::other("realtime parent exited"));
                }
                Ok(())
            });
        }
        let mut child = command.spawn().context("start realtime worker")?;
        let mut stdin = child.stdin.take().context("missing realtime stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().context("missing realtime stdout")?);
        let (tx, requests) = mpsc::sync_channel(32);
        let (results, rx) = mpsc::sync_channel(32);
        let failed = Arc::new(AtomicBool::new(false));
        let fault = failed.clone();
        let writer = thread::spawn(move || {
            while let Ok(input) = requests.recv() {
                if serde_json::to_writer(&mut stdin, &input).is_err()
                    || stdin.write_all(b"\n").is_err()
                    || stdin.flush().is_err()
                {
                    fault.store(true, Ordering::Release);
                    break;
                }
                if matches!(input, Input::Finish) {
                    break;
                }
            }
        });
        let fault = failed.clone();
        let reader = thread::spawn(move || {
            loop {
                let value = (|| {
                    let mut line = Vec::new();
                    let n = stdout
                        .by_ref()
                        .take(1_048_577)
                        .read_until(b'\n', &mut line)?;
                    ensure!(
                        n > 0 && n <= 1_048_576 && line.last() == Some(&b'\n'),
                        "realtime worker disconnected or invalid reply"
                    );
                    Ok::<_, anyhow::Error>(serde_json::from_slice::<Output>(&line)?)
                })();
                let done = matches!(value, Ok(Output::Done));
                let error = value.is_err();
                if results.try_send(value).is_err() {
                    fault.store(true, Ordering::Release);
                    break;
                }
                if error || done {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            tx: Some(tx),
            rx,
            writer: Some(writer),
            reader: Some(reader),
            failed,
            timeout: Duration::from_secs(backend.settings.cloud.timeout_seconds + 2),
        })
    }
    fn send(&self, input: Input, live: bool) -> Result<()> {
        ensure!(
            !self.failed.load(Ordering::Acquire),
            "realtime worker failed"
        );
        let tx = self.tx.as_ref().context("realtime worker closed")?;
        if live {
            tx.try_send(input).map_err(|_| {
                anyhow::anyhow!("realtime audio queue full or closed; session stopped")
            })?;
        } else {
            tx.send(input)
                .map_err(|_| anyhow::anyhow!("realtime worker closed"))?;
        }
        Ok(())
    }
    fn poll(&self, wait: bool) -> Result<Vec<Reply>> {
        ensure!(
            !self.failed.load(Ordering::Acquire),
            "realtime worker failed"
        );
        let mut replies = Vec::new();
        let started = Instant::now();
        loop {
            let value = if wait {
                self.rx
                    .recv_timeout(self.timeout.saturating_sub(started.elapsed()))
                    .map_err(|_| anyhow::anyhow!("realtime worker timed out or disconnected"))?
            } else {
                match self.rx.try_recv() {
                    Ok(v) => v,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(_) => bail!("realtime worker disconnected"),
                }
            }?;
            match value {
                Output::Final(reply) => replies.push(reply),
                Output::Done => {
                    ensure!(wait, "realtime session ended unexpectedly");
                    break;
                }
            }
        }
        Ok(replies)
    }
}
impl Drop for RealtimeWorker {
    fn drop(&mut self) {
        self.tx.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
struct State {
    resampler: AudioResampler,
    pending: Vec<f32>,
    worker: Option<RealtimeWorker>,
    finished: bool,
    failed: bool,
}
pub(super) struct RealtimeStream<'a> {
    backend: &'a CloudBackend,
    live: bool,
    state: RefCell<State>,
}
impl<'a> RealtimeStream<'a> {
    pub(super) fn new(backend: &'a CloudBackend, live: bool) -> Self {
        Self {
            backend,
            live,
            state: RefCell::new(State {
                resampler: AudioResampler::new(),
                pending: Vec::new(),
                worker: None,
                finished: false,
                failed: false,
            }),
        }
    }
    fn feed(&self, state: &mut State, audio: Vec<f32>, flush: bool) -> Result<()> {
        state.pending.extend(audio);
        if state.worker.is_none() && !state.pending.is_empty() {
            state.worker = Some(RealtimeWorker::start(self.backend)?);
        }
        while state.pending.len() >= 1600 || (flush && !state.pending.is_empty()) {
            let n = 1600.min(state.pending.len());
            let frame = state.pending.drain(..n).collect();
            state
                .worker
                .as_ref()
                .unwrap()
                .send(Input::Audio(frame), self.live)?;
        }
        Ok(())
    }
    fn fail<T>(state: &mut State, result: Result<T>) -> Result<T> {
        if result.is_err() {
            state.failed = true;
            state.worker.take();
            state.pending.clear();
        }
        result
    }
}
impl WakeWordStream for RealtimeStream<'_> {
    fn accept(&self, rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("realtime stream in use"))?;
        ensure!(
            !state.finished && !state.failed,
            "realtime session closed; create a new session"
        );
        let result = (|| {
            ensure!(
                (8000..=192000).contains(&rate)
                    && samples.len() <= rate as usize * 30
                    && samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                "invalid realtime audio"
            );
            let audio = state.resampler.accept(rate, samples)?;
            self.feed(&mut state, audio, false)?;
            let replies = if let Some(worker) = &state.worker {
                worker.poll(false)?
            } else {
                Vec::new()
            };
            Ok(self.backend.detections(replies))
        })();
        Self::fail(&mut state, result)
    }
    fn finish(&self) -> Result<Vec<Detection>> {
        let mut state = self.state.borrow_mut();
        if state.finished {
            return Ok(Vec::new());
        }
        ensure!(!state.failed, "realtime stream failed");
        state.finished = true;
        let result = (|| {
            let tail = state.resampler.finish()?;
            self.feed(&mut state, tail, true)?;
            let replies = if let Some(worker) = &state.worker {
                worker.send(Input::Finish, false)?;
                worker.poll(true)?
            } else {
                Vec::new()
            };
            state.worker.take();
            Ok(self.backend.detections(replies))
        })();
        Self::fail(&mut state, result)
    }
}
#[cfg(test)]
#[path = "../../tests/unit/cloud_realtime.rs"]
mod tests;
