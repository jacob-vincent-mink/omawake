mod protocol;
mod ring;

use std::cell::{Cell, RefCell};
use std::env;
#[cfg(omawake_whisper_adapter)]
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};

use self::protocol::{Request, Response, Transcript};
use self::ring::{EndpointBuffer, FRAME_SAMPLES, Utterance};
use super::onnx::resample::AudioResampler;
use super::{Detection, WakeWordBackend, WakeWordStream, detect_samples};
use crate::backend::Runtime;
use crate::config::Config;
use crate::paths::AppPaths;
use crate::phrase::{PhraseMatcher, normalize_tokens};

#[cfg(omawake_whisper_adapter)]
unsafe extern "C" {
    fn oma_whisper_open(
        library_path: *const c_char,
        verifier_path: *const c_char,
        vad_path: *const c_char,
        n_threads: c_int,
        output: *mut *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn oma_whisper_version(provider: *const c_void) -> *const c_char;
    fn oma_whisper_vad_probability(
        provider: *mut c_void,
        samples: *const f32,
        sample_count: c_int,
        probability: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn oma_whisper_vad_reset(provider: *mut c_void);
    fn oma_whisper_transcribe(
        provider: *mut c_void,
        samples: *const f32,
        sample_count: c_int,
        prompt: *const c_char,
        text: *mut c_char,
        text_capacity: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn oma_whisper_close(provider: *mut c_void);
}

pub(super) struct WhisperCppBackend {
    worker: RefCell<Worker>,
    matcher: PhraseMatcher,
    prompt: String,
    next_stream_id: Cell<u64>,
}

struct WhisperCppStream<'a> {
    backend: &'a WhisperCppBackend,
    state: RefCell<ClientStream>,
}

struct ClientStream {
    id: u64,
    started: bool,
    finished: bool,
    resampler: AudioResampler,
    pending: Vec<f32>,
}

impl WhisperCppBackend {
    pub(super) fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        if config.backend.runtime != Runtime::Default {
            bail!("whispercpp currently supports only backend.runtime = default");
        }
        if config.model.sample_rate != 16_000 {
            bail!("whisper.cpp requires model.sample_rate = 16000");
        }
        if !(1..=64).contains(&config.backend.threads) {
            bail!("backend threads must be between 1 and 64");
        }
        let directory = config.model_directory(paths);
        let library = resolve_library(config, paths)?;
        let verifier = resolve_model_asset(&directory, &config.model.verifier, "verifier")?;
        let vad = resolve_model_asset(&directory, &config.model.vad, "VAD")?;
        let matcher = PhraseMatcher::compile(&config.wake_words)?;
        let prompt = config
            .wake_words
            .iter()
            .filter(|word| word.enabled)
            .map(|word| word.phrase.trim())
            .collect::<Vec<_>>()
            .join(", ");
        let worker = Worker::spawn(&library, &verifier, &vad, i32::from(config.backend.threads))?;
        Ok(Self {
            worker: RefCell::new(worker),
            matcher,
            prompt,
            next_stream_id: Cell::new(1),
        })
    }

    fn detections(&self, transcripts: Vec<Transcript>) -> Vec<Detection> {
        transcripts_to_detections(&self.matcher, transcripts)
    }
}

fn transcripts_to_detections(
    matcher: &PhraseMatcher,
    transcripts: Vec<Transcript>,
) -> Vec<Detection> {
    transcripts
        .into_iter()
        .flat_map(|transcript| {
            let tokens = normalize_tokens(&transcript.text);
            matcher
                .matches(&transcript.text)
                .into_iter()
                .map(move |matched| Detection {
                    id: matched.id,
                    tokens: tokens[matched.start_token..matched.end_token].to_vec(),
                    timestamps: Vec::new(),
                    start_time: transcript.start_sample as f32 / 16_000.0,
                })
        })
        .collect()
}

impl WakeWordBackend for WhisperCppBackend {
    fn kind(&self) -> &'static str {
        "whispercpp"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        let id = self.next_stream_id.get();
        self.next_stream_id.set(id.wrapping_add(1).max(1));
        Box::new(WhisperCppStream {
            backend: self,
            state: RefCell::new(ClientStream {
                id,
                started: false,
                finished: false,
                resampler: AudioResampler::new(),
                pending: Vec::new(),
            }),
        })
    }

    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (sample_rate, samples) = super::onnx::read_wave(path)?;
        let stream = self.stream();
        detect_samples(stream.as_ref(), sample_rate, &samples)
    }
}

impl WhisperCppStream<'_> {
    fn ensure_started(&self, state: &mut ClientStream) -> Result<()> {
        if !state.started {
            self.backend
                .worker
                .try_borrow_mut()
                .map_err(|_| anyhow::anyhow!("another whisper.cpp stream is using the worker"))?
                .start(state.id, &self.backend.prompt)?;
            state.started = true;
        }
        Ok(())
    }

    fn submit_ready(&self, state: &mut ClientStream) -> Result<Vec<Detection>> {
        let mut transcripts = Vec::new();
        while state.pending.len() >= FRAME_SAMPLES {
            let frame: Vec<_> = state.pending.drain(..FRAME_SAMPLES).collect();
            transcripts.extend(
                self.backend
                    .worker
                    .try_borrow_mut()
                    .map_err(|_| anyhow::anyhow!("another whisper.cpp stream is using the worker"))?
                    .audio(state.id, &frame)?,
            );
        }
        Ok(self.backend.detections(transcripts))
    }
}

impl WakeWordStream for WhisperCppStream<'_> {
    fn accept(&self, sample_rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("audio contains a non-finite sample");
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("detector stream is already in use"))?;
        if state.finished {
            bail!("audio was supplied after the stream finished");
        }
        self.ensure_started(&mut state)?;
        let normalized = state.resampler.accept(sample_rate, samples)?;
        state.pending.extend(normalized);
        self.submit_ready(&mut state)
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("detector stream is already in use"))?;
        if state.finished {
            return Ok(Vec::new());
        }
        self.ensure_started(&mut state)?;
        let tail = state.resampler.finish()?;
        state.pending.extend(tail);
        let mut detections = self.submit_ready(&mut state)?;
        if !state.pending.is_empty() {
            state.pending.resize(FRAME_SAMPLES, 0.0);
            detections.extend(self.submit_ready(&mut state)?);
        }
        let transcripts = self
            .backend
            .worker
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("another whisper.cpp stream is using the worker"))?
            .finish(state.id)?;
        detections.extend(self.backend.detections(transcripts));
        state.finished = true;
        Ok(detections)
    }
}

impl Drop for WhisperCppStream<'_> {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        if state.started
            && !state.finished
            && let Ok(mut worker) = self.backend.worker.try_borrow_mut()
        {
            let _ = worker.cancel(state.id);
        }
    }
}

fn resolve_model_asset(directory: &Path, configured: &str, label: &str) -> Result<PathBuf> {
    if configured.trim().is_empty() {
        bail!("model.{label} must name a model file");
    }
    let configured = Path::new(configured);
    let path = if configured.is_absolute() {
        configured.to_owned()
    } else {
        directory.join(configured)
    };
    if !path.is_file() {
        bail!("whisper.cpp {label} model is missing: {}", path.display());
    }
    Ok(path)
}

fn resolve_library(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    if !config.backend.library.as_os_str().is_empty() {
        let path = if config.backend.library.is_absolute() {
            config.backend.library.clone()
        } else {
            paths
                .config_file
                .parent()
                .unwrap_or(Path::new("."))
                .join(&config.backend.library)
        };
        if path.is_file() {
            return Ok(path);
        }
        bail!(
            "configured whisper.cpp library is missing: {}",
            path.display()
        );
    }
    if let Some(path) = env::var_os("OMAWAKE_WHISPER_LIBRARY").map(PathBuf::from) {
        if path.is_file() {
            return Ok(path);
        }
        bail!(
            "OMAWAKE_WHISPER_LIBRARY does not name a file: {}",
            path.display()
        );
    }
    for path in [
        "/usr/lib/libwhisper.so.1",
        "/usr/local/lib/libwhisper.so.1",
        "/usr/lib64/libwhisper.so.1",
    ] {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }
    bail!("libwhisper was not found; set backend.library or OMAWAKE_WHISPER_LIBRARY")
}

struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
    active_id: Option<u64>,
}

impl Worker {
    fn spawn(library: &Path, verifier: &Path, vad: &Path, threads: i32) -> Result<Self> {
        let executable = env::current_exe().context("resolve Omawake executable")?;
        let mut command = Command::new(executable);
        command
            .arg("__whisper-worker")
            .arg(library)
            .arg(verifier)
            .arg(vad)
            .arg(threads.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        prepend_library_directory(&mut command, library);
        let mut child = command
            .spawn()
            .context("spawn isolated whisper.cpp worker")?;
        let input = child.stdin.take().context("worker stdin is unavailable")?;
        let output = child
            .stdout
            .take()
            .context("worker stdout is unavailable")?;
        let mut worker = Self {
            child,
            input: Some(input),
            output: BufReader::new(output),
            active_id: None,
        };
        match protocol::read_response(&mut worker.output).context("read worker handshake")? {
            Response::Ready { .. } => Ok(worker),
            Response::Error { message, .. } => {
                let _ = worker.child.wait();
                bail!(message)
            }
            response => bail!("unexpected worker handshake: {response:?}"),
        }
    }

    fn exchange(&mut self, request: &Request, pcm: &[f32]) -> Result<Response> {
        let input = self.input.as_mut().context("worker stdin is closed")?;
        protocol::write_request(input, request, pcm).context("write worker request")?;
        protocol::read_response(&mut self.output).context("read worker response")
    }

    fn start(&mut self, id: u64, prompt: &str) -> Result<()> {
        if let Some(active) = self.active_id {
            bail!("whisper.cpp worker is already serving stream {active}");
        }
        match self.exchange(
            &Request::Start {
                id,
                prompt: prompt.to_owned(),
            },
            &[],
        )? {
            Response::Ack { id: response } if response == id => {
                self.active_id = Some(id);
                Ok(())
            }
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected start response: {response:?}"),
        }
    }

    fn audio(&mut self, id: u64, pcm: &[f32]) -> Result<Vec<Transcript>> {
        match self.exchange(
            &Request::Audio {
                id,
                samples: pcm.len(),
            },
            pcm,
        )? {
            Response::Result {
                id: response,
                transcripts,
            } if response == id => Ok(transcripts),
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected audio response: {response:?}"),
        }
    }

    fn finish(&mut self, id: u64) -> Result<Vec<Transcript>> {
        let response = self.exchange(&Request::Finish { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Result {
                id: response,
                transcripts,
            } if response == id => Ok(transcripts),
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected finish response: {response:?}"),
        }
    }

    fn cancel(&mut self, id: u64) -> Result<()> {
        let response = self.exchange(&Request::Cancel { id }, &[])?;
        self.active_id = None;
        match response {
            Response::Ack { id: response } if response == id => Ok(()),
            Response::Error { message, .. } => bail!(message),
            response => bail!("unexpected cancel response: {response:?}"),
        }
    }

    fn shutdown(&mut self) {
        if let Some(input) = self.input.as_mut() {
            let _ = protocol::write_request(input, &Request::Shutdown, &[]);
        }
        self.input.take();
        let _ = self.child.wait();
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn prepend_library_directory(command: &mut Command, library: &Path) {
    let Some(directory) = library.parent().filter(|path| !path.as_os_str().is_empty()) else {
        return;
    };
    let mut paths = vec![directory.to_owned()];
    if let Some(existing) = env::var_os("LD_LIBRARY_PATH") {
        paths.extend(env::split_paths(&existing));
    }
    if let Ok(joined) = env::join_paths(paths) {
        command.env("LD_LIBRARY_PATH", joined);
    }
}

fn harden_worker_process() -> Result<()> {
    let limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let limit_status = unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) };
    let limit_error = (limit_status != 0).then(io::Error::last_os_error);
    let dump_status = unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) };
    let dump_error = (dump_status != 0).then(io::Error::last_os_error);
    if limit_status == 0 || dump_status == 0 {
        return Ok(());
    }
    bail!(
        "could not disable worker core dumps: setrlimit: {}; prctl: {}",
        limit_error.expect("failed setrlimit has an OS error"),
        dump_error.expect("failed prctl has an OS error")
    )
}

#[cfg(omawake_whisper_adapter)]
struct NativeProvider(*mut c_void);

#[cfg(omawake_whisper_adapter)]
impl NativeProvider {
    fn open(library: &Path, verifier: &Path, vad: &Path, threads: i32) -> Result<Self> {
        let library = path_to_c_string(library)?;
        let verifier = path_to_c_string(verifier)?;
        let vad = path_to_c_string(vad)?;
        let mut provider = std::ptr::null_mut();
        let mut error = vec![0_i8; 2048];
        let status = unsafe {
            oma_whisper_open(
                library.as_ptr(),
                verifier.as_ptr(),
                vad.as_ptr(),
                threads,
                &mut provider,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 || provider.is_null() {
            bail!(c_buffer(&error));
        }
        Ok(Self(provider))
    }

    fn version(&self) -> String {
        let version = unsafe { oma_whisper_version(self.0) };
        if version.is_null() {
            return "unknown".into();
        }
        unsafe { CStr::from_ptr(version) }
            .to_string_lossy()
            .into_owned()
    }

    fn vad_probability(&mut self, samples: &[f32]) -> Result<f32> {
        let mut probability = 0.0;
        let mut error = vec![0_i8; 2048];
        let status = unsafe {
            oma_whisper_vad_probability(
                self.0,
                samples.as_ptr(),
                i32::try_from(samples.len()).context("VAD frame is too large")?,
                &mut probability,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(c_buffer(&error));
        }
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            bail!("whisper VAD returned invalid probability {probability}");
        }
        Ok(probability)
    }

    fn reset_vad(&mut self) {
        unsafe { oma_whisper_vad_reset(self.0) };
    }

    fn transcribe(&mut self, samples: &[f32], prompt: &str) -> Result<String> {
        let prompt = CString::new(prompt).context("wake-word prompt contains NUL")?;
        let mut text = vec![0_i8; 64 * 1024];
        let mut error = vec![0_i8; 2048];
        let status = unsafe {
            oma_whisper_transcribe(
                self.0,
                samples.as_ptr(),
                i32::try_from(samples.len()).context("utterance is too large")?,
                prompt.as_ptr(),
                text.as_mut_ptr(),
                text.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status < 0 {
            bail!(c_buffer(&error));
        }
        Ok(unsafe { CStr::from_ptr(text.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_owned())
    }
}

#[cfg(omawake_whisper_adapter)]
impl Drop for NativeProvider {
    fn drop(&mut self) {
        unsafe { oma_whisper_close(self.0) };
    }
}

#[cfg(not(omawake_whisper_adapter))]
struct NativeProvider;

#[cfg(not(omawake_whisper_adapter))]
impl NativeProvider {
    fn open(_: &Path, _: &Path, _: &Path, _: i32) -> Result<Self> {
        bail!("this Omawake build has no whisper.cpp adapter; build with WHISPER_CPP_ROOT set")
    }

    fn version(&self) -> String {
        "unavailable".into()
    }

    fn vad_probability(&mut self, _: &[f32]) -> Result<f32> {
        bail!("whisper.cpp adapter is unavailable")
    }

    fn reset_vad(&mut self) {}

    fn transcribe(&mut self, _: &[f32], _: &str) -> Result<String> {
        bail!("whisper.cpp adapter is unavailable")
    }
}

#[cfg(omawake_whisper_adapter)]
fn path_to_c_string(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_encoded_bytes()).context("native path contains NUL")
}

#[cfg(omawake_whisper_adapter)]
fn c_buffer(buffer: &[i8]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

struct WorkerStream {
    id: u64,
    prompt: String,
    endpoint: EndpointBuffer,
}

impl WorkerStream {
    fn finish_utterance(
        &self,
        provider: &mut NativeProvider,
        utterance: Utterance,
    ) -> Result<Transcript> {
        let text = provider.transcribe(&utterance.samples, &self.prompt)?;
        Ok(Transcript {
            text,
            start_sample: utterance.start_sample,
            end_sample: utterance.end_sample,
        })
    }
}

pub(crate) fn worker_main(library: &Path, verifier: &Path, vad: &Path, threads: i32) -> Result<()> {
    let mut output = io::stdout().lock();
    let mut input = io::stdin().lock();
    if let Err(error) = harden_worker_process() {
        protocol::write_response(
            &mut output,
            &Response::Error {
                id: None,
                message: format!("{error:#}"),
            },
        )?;
        return Ok(());
    }
    let mut provider = match NativeProvider::open(library, verifier, vad, threads) {
        Ok(provider) => provider,
        Err(error) => {
            protocol::write_response(
                &mut output,
                &Response::Error {
                    id: None,
                    message: format!("{error:#}"),
                },
            )?;
            return Ok(());
        }
    };
    protocol::write_response(
        &mut output,
        &Response::Ready {
            version: provider.version(),
        },
    )?;
    let mut stream: Option<WorkerStream> = None;
    loop {
        let (request, pcm) = protocol::read_request(&mut input)?;
        let response = match request {
            Request::Shutdown => return Ok(()),
            Request::Start { id, prompt } => {
                provider.reset_vad();
                stream = Some(WorkerStream {
                    id,
                    prompt,
                    endpoint: EndpointBuffer::new(),
                });
                Response::Ack { id }
            }
            Request::Audio { id, .. } => match stream.as_mut() {
                Some(active) if active.id == id && pcm.len() == FRAME_SAMPLES => {
                    let result = (|| {
                        if pcm.iter().any(|sample| !sample.is_finite()) {
                            bail!("audio contains a non-finite sample");
                        }
                        let probability = provider.vad_probability(&pcm)?;
                        let utterance = active.endpoint.push(&pcm, probability);
                        let transcript = utterance
                            .map(|utterance| active.finish_utterance(&mut provider, utterance))
                            .transpose()?;
                        if transcript.is_some() {
                            provider.reset_vad();
                        }
                        Ok::<_, anyhow::Error>(transcript.into_iter().collect())
                    })();
                    match result {
                        Ok(transcripts) => Response::Result { id, transcripts },
                        Err(error) => Response::Error {
                            id: Some(id),
                            message: format!("{error:#}"),
                        },
                    }
                }
                Some(_) => Response::Error {
                    id: Some(id),
                    message: "worker stream id does not match".into(),
                },
                None => Response::Error {
                    id: Some(id),
                    message: "worker has no active stream".into(),
                },
            },
            Request::Finish { id } => match stream.take() {
                Some(mut active) if active.id == id => {
                    let result = active
                        .endpoint
                        .finish()
                        .map(|utterance| active.finish_utterance(&mut provider, utterance))
                        .transpose();
                    provider.reset_vad();
                    match result {
                        Ok(transcript) => Response::Result {
                            id,
                            transcripts: transcript.into_iter().collect(),
                        },
                        Err(error) => Response::Error {
                            id: Some(id),
                            message: format!("{error:#}"),
                        },
                    }
                }
                Some(active) => {
                    stream = Some(active);
                    Response::Error {
                        id: Some(id),
                        message: "worker stream id does not match".into(),
                    }
                }
                None => Response::Error {
                    id: Some(id),
                    message: "worker has no active stream".into(),
                },
            },
            Request::Cancel { id } => match stream.as_ref() {
                Some(active) if active.id == id => {
                    stream = None;
                    provider.reset_vad();
                    Response::Ack { id }
                }
                _ => Response::Error {
                    id: Some(id),
                    message: "worker stream id does not match".into(),
                },
            },
        };
        protocol::write_response(&mut output, &response)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WakeWord;

    #[test]
    fn transcript_conversion_uses_whole_phrase_matcher() {
        let matcher = PhraseMatcher::compile(&[WakeWord {
            id: "computer".into(),
            phrase: "hey computer".into(),
            enabled: true,
            command: vec!["true".into()],
        }])
        .unwrap();
        let detections = transcripts_to_detections(
            &matcher,
            vec![Transcript {
                text: "A computerized voice said: hey, computer!".into(),
                start_sample: 8_000,
                end_sample: 16_000,
            }],
        );
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].id, "computer");
        assert_eq!(detections[0].tokens, ["hey", "computer"]);
        assert_eq!(detections[0].start_time, 0.5);
    }

    #[test]
    fn cpu_checkpoint_rejects_accelerator_runtime_without_spawning() {
        let mut config = Config::default();
        config.backend.kind = "whispercpp".into();
        config.backend.runtime = Runtime::Cuda;
        let error = WhisperCppBackend::load(&config, &AppPaths::discover())
            .err()
            .expect("unsupported runtime should fail");
        assert!(error.to_string().contains("only backend.runtime = default"));
    }
}
