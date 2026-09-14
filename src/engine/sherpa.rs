//! Runtime-loaded sherpa-onnx v1.13.8 KWS adapter.

use std::env;
use std::ffi::{CStr, CString, c_char, c_float, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::rc::Rc;
use std::slice;

use anyhow::{Context, Result, bail};
use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LOCAL, RTLD_NOW};

use super::*;

const SHERPA_VERSION: &str = "1.13.8";
const OMA_RUNTIME_ABI: i32 = 1;

pub(crate) fn validate_runtime_libraries(ort_path: &Path, sherpa_path: &Path) -> Result<()> {
    let (sherpa, ort) = open_runtime_libraries(ort_path, sherpa_path)?;
    drop(sherpa);
    drop(ort);
    Ok(())
}

pub(crate) fn validate_runtime_provider(
    ort_path: &Path,
    sherpa_path: &Path,
    provider_path: &Path,
    registration: &str,
    ep_name: &str,
    device: &str,
) -> Result<()> {
    let (sherpa, ort) = open_runtime_libraries(ort_path, sherpa_path)?;
    unsafe {
        let create: CreateOrtRuntime = symbol(&sherpa, b"SherpaOnnxCreateOrtRuntime\0")?;
        let destroy: DestroyOrtRuntime = symbol(&sherpa, b"SherpaOnnxDestroyOrtRuntime\0")?;
        let register: RegisterProvider = symbol(
            &sherpa,
            b"SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary\0",
        )?;
        let has_device: HasProviderDevice =
            symbol(&sherpa, b"SherpaOnnxOrtRuntimeHasExecutionProviderDevice\0")?;
        let runtime_error: RuntimeError = symbol(&sherpa, b"SherpaOnnxOrtRuntimeGetLastError\0")?;
        let runtime = create();
        if runtime.is_null() {
            bail!("create retained sherpa ONNX Runtime environment");
        }
        let guard = RuntimeGuard::new(runtime, destroy);
        let registration = CString::new(registration)?;
        let provider_path = CString::new(provider_path.to_string_lossy().as_bytes())?;
        let ep_name = CString::new(ep_name)?;
        let device = CString::new(device)?;
        if register(runtime, registration.as_ptr(), provider_path.as_ptr()) == 0 {
            bail!(
                "register execution-provider library: {}",
                string(runtime_error(runtime))
            );
        }
        if has_device(runtime, ep_name.as_ptr(), device.as_ptr()) == 0 {
            bail!(
                "requested execution-provider device is unavailable: {}",
                string(runtime_error(runtime))
            );
        }
        drop(guard);
    }
    drop(sherpa);
    drop(ort);
    Ok(())
}

fn open_runtime_libraries(ort_path: &Path, sherpa_path: &Path) -> Result<(Library, Library)> {
    let ort = unsafe { Library::open(Some(ort_path), RTLD_NOW | RTLD_GLOBAL) }
        .with_context(|| format!("load ONNX Runtime {}", ort_path.display()))?;
    let sherpa = unsafe { Library::open(Some(sherpa_path), RTLD_NOW | RTLD_LOCAL) }
        .with_context(|| format!("load sherpa-onnx {}", sherpa_path.display()))?;

    unsafe {
        validate_runtime_identity(&ort, &sherpa)?;
    }
    Ok((sherpa, ort))
}

unsafe fn validate_runtime_identity(ort: &Library, sherpa: &Library) -> Result<()> {
    let version: VersionString = unsafe { symbol(sherpa, b"SherpaOnnxGetVersionStr\0")? };
    let actual = unsafe { string(version()) };
    if actual != SHERPA_VERSION {
        bail!("sherpa-onnx ABI mismatch: expected {SHERPA_VERSION}, loaded {actual}");
    }
    let abi: OmaRuntimeAbi = unsafe {
        symbol(sherpa, b"SherpaOnnxGetOmaRuntimeAbiVersion\0")
            .context("loaded sherpa-onnx is not the required patched Oma runtime")?
    };
    let actual_abi = unsafe { abi() };
    if actual_abi != OMA_RUNTIME_ABI {
        bail!("Oma sherpa runtime ABI mismatch: expected {OMA_RUNTIME_ABI}, loaded {actual_abi}");
    }
    let ort_version: VersionString =
        unsafe { symbol(sherpa, b"SherpaOnnxGetOnnxruntimeVersionStr\0")? };
    let ort_api_base: OrtGetApiBase = unsafe { symbol(ort, b"OrtGetApiBase\0")? };
    let base = unsafe { ort_api_base() };
    if base.is_null() {
        bail!("loaded ONNX Runtime returned a null API base");
    }
    let external_ort_version = unsafe { string(((*base).get_version_string)()) };
    let sherpa_ort_version = unsafe { string(ort_version()) };
    if external_ort_version != sherpa_ort_version {
        bail!(
            "ONNX Runtime ABI mismatch: sherpa expects {sherpa_ort_version}, loaded {external_ort_version}"
        );
    }
    if external_ort_version != "1.29.0" {
        bail!("ONNX Runtime version mismatch: expected 1.29.0, loaded {external_ort_version}");
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub(super) struct FeatureConfig {
    pub sample_rate: i32,
    pub feature_dim: i32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct OnlineTransducerModelConfig {
    pub encoder: Option<String>,
    pub decoder: Option<String>,
    pub joiner: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct OnlineModelConfig {
    pub transducer: OnlineTransducerModelConfig,
    pub tokens: Option<String>,
    pub num_threads: i32,
    pub provider: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct KeywordSpotterConfig {
    pub feat_config: FeatureConfig,
    pub model_config: OnlineModelConfig,
    pub max_active_paths: i32,
    pub num_trailing_blanks: i32,
    pub keywords_score: f32,
    pub keywords_threshold: f32,
    pub keywords_buf: Option<String>,
}

impl Default for KeywordSpotterConfig {
    fn default() -> Self {
        Self {
            feat_config: FeatureConfig {
                sample_rate: 16_000,
                feature_dim: 80,
            },
            model_config: OnlineModelConfig::default(),
            max_active_paths: 4,
            num_trailing_blanks: 1,
            keywords_score: 1.0,
            keywords_threshold: 0.25,
            keywords_buf: None,
        }
    }
}

pub(super) struct SherpaOnnxBackend {
    spotter: Spotter,
}

struct SherpaOnnxStream<'a> {
    backend: &'a SherpaOnnxBackend,
    stream: Stream,
}

impl SherpaOnnxBackend {
    pub(super) fn load(
        config: &Config,
        paths: &AppPaths,
        directory: &Path,
        runtime: Runtime,
        keywords_buffer: &str,
    ) -> Result<Self> {
        let sherpa_config =
            build_sherpa_config(config, paths, directory, runtime, keywords_buffer)?;
        let api = Api::load(config, paths, runtime)?;
        let spotter = Spotter::create(api, &sherpa_config)?
            .context("sherpa-onnx could not create the keyword spotter")?;
        Ok(Self { spotter })
    }

    fn decode_ready(&self, stream: &Stream) -> Vec<Detection> {
        drain_ready(|| {
            if !self.spotter.is_ready(stream) {
                return None;
            }
            self.spotter.decode(stream);
            Some(self.spotter.result(stream).and_then(|result| {
                detection_from_parts(
                    result.keyword,
                    result.tokens,
                    result.timestamps,
                    result.start_time,
                )
            }))
        })
    }
}

impl WakeWordBackend for SherpaOnnxBackend {
    fn kind(&self) -> &'static str {
        "sherpa-onnx"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(SherpaOnnxStream {
            backend: self,
            stream: self.spotter.create_stream(),
        })
    }

    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (sample_rate, samples) = read_wave(path)?;
        let stream = self.stream();
        detect_samples(stream.as_ref(), sample_rate, &samples)
    }
}

impl WakeWordStream for SherpaOnnxStream<'_> {
    fn accept(&self, sample_rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        self.stream.accept_waveform(sample_rate, samples)?;
        Ok(self.backend.decode_ready(&self.stream))
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        self.stream.input_finished()?;
        Ok(self.backend.decode_ready(&self.stream))
    }
}

pub(super) fn read_wave(path: &Path) -> Result<(i32, Vec<f32>)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("read WAV {}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        bail!("WAV sample rate must be positive: {}", path.display());
    }
    if spec.channels != 1 {
        bail!("WAV input must be mono: {}", path.display());
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => {
            let scale = (1_u32 << spec.bits_per_sample.saturating_sub(1)) as f32;
            reader
                .samples::<i16>()
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Int => {
            let scale = (1_u64 << spec.bits_per_sample.saturating_sub(1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    Ok((spec.sample_rate as i32, samples))
}

struct ResultValue {
    keyword: String,
    tokens: Vec<String>,
    timestamps: Vec<f32>,
    start_time: f32,
}

struct Spotter {
    api: Rc<Api>,
    pointer: *const CKeywordSpotter,
}

impl Spotter {
    fn create(api: Rc<Api>, config: &KeywordSpotterConfig) -> Result<Option<Self>> {
        let mut strings = Strings::default();
        let ffi = config.to_ffi(&mut strings)?;
        let pointer = unsafe { (api.create_keyword_spotter)(&ffi) };
        Ok((!pointer.is_null()).then_some(Self { api, pointer }))
    }

    fn create_stream(&self) -> Stream {
        let pointer = unsafe { (self.api.create_keyword_stream)(self.pointer) };
        Stream {
            api: Rc::clone(&self.api),
            pointer,
        }
    }

    fn is_ready(&self, stream: &Stream) -> bool {
        if stream.pointer.is_null() {
            return false;
        }
        unsafe { (self.api.is_keyword_stream_ready)(self.pointer, stream.pointer) != 0 }
    }

    fn decode(&self, stream: &Stream) {
        unsafe { (self.api.decode_keyword_stream)(self.pointer, stream.pointer) }
    }

    fn result(&self, stream: &Stream) -> Option<ResultValue> {
        let pointer = unsafe { (self.api.get_keyword_result)(self.pointer, stream.pointer) };
        if pointer.is_null() {
            return None;
        }
        let raw = unsafe { &*pointer };
        let count = raw.count.max(0) as usize;
        let tokens = if raw.tokens_arr.is_null() {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(raw.tokens_arr, count) }
                .iter()
                .map(|value| string(*value))
                .collect()
        };
        let timestamps = if raw.timestamps.is_null() {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(raw.timestamps, count) }.to_vec()
        };
        let result = ResultValue {
            keyword: string(raw.keyword),
            tokens,
            timestamps,
            start_time: raw.start_time,
        };
        unsafe { (self.api.destroy_keyword_result)(pointer) };
        Some(result)
    }
}

impl Drop for Spotter {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.api.destroy_keyword_spotter)(self.pointer) };
        }
    }
}

struct Stream {
    api: Rc<Api>,
    pointer: *const COnlineStream,
}

impl Stream {
    fn accept_waveform(&self, sample_rate: i32, samples: &[f32]) -> Result<()> {
        if self.pointer.is_null() {
            bail!("sherpa-onnx could not create a keyword stream");
        }
        let count = i32::try_from(samples.len()).context("audio chunk exceeds sherpa ABI limit")?;
        unsafe {
            (self.api.online_stream_accept_waveform)(
                self.pointer,
                sample_rate,
                samples.as_ptr(),
                count,
            )
        };
        Ok(())
    }

    fn input_finished(&self) -> Result<()> {
        if self.pointer.is_null() {
            bail!("sherpa-onnx could not create a keyword stream");
        }
        unsafe { (self.api.online_stream_input_finished)(self.pointer) }
        Ok(())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.api.destroy_online_stream)(self.pointer) };
        }
    }
}

struct Api {
    // Drop sherpa before ORT because sherpa's unload path may still use ORT.
    _sherpa: Library,
    _ort: Library,
    create_keyword_spotter: CreateKeywordSpotter,
    destroy_keyword_spotter: DestroyKeywordSpotter,
    create_keyword_stream: CreateKeywordStream,
    is_keyword_stream_ready: IsKeywordStreamReady,
    decode_keyword_stream: DecodeKeywordStream,
    get_keyword_result: GetKeywordResult,
    destroy_keyword_result: DestroyKeywordResult,
    destroy_online_stream: DestroyOnlineStream,
    online_stream_accept_waveform: OnlineStreamAcceptWaveform,
    online_stream_input_finished: OnlineStreamInputFinished,
    runtime: *mut COrtRuntime,
    destroy_ort_runtime: DestroyOrtRuntime,
}

impl Api {
    fn load(config: &Config, paths: &AppPaths, runtime: Runtime) -> Result<Rc<Self>> {
        let ort_path = resolve_library(
            &config.backend.onnxruntime_library,
            crate::runtime_paths::ONNXRUNTIME_LIBRARY_ENV,
            "libonnxruntime.so",
            config,
            paths,
        )?;
        let sherpa_path = resolve_library(
            &config.backend.sherpa_library,
            crate::runtime_paths::SHERPA_LIBRARY_ENV,
            "libsherpa-onnx-c-api.so",
            config,
            paths,
        )?;
        let (sherpa, ort) = open_runtime_libraries(&ort_path, &sherpa_path)?;

        unsafe {
            let create_ort_runtime: CreateOrtRuntime =
                symbol(&sherpa, b"SherpaOnnxCreateOrtRuntime\0")?;
            let destroy_ort_runtime: DestroyOrtRuntime =
                symbol(&sherpa, b"SherpaOnnxDestroyOrtRuntime\0")?;
            let register_provider: RegisterProvider = symbol(
                &sherpa,
                b"SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary\0",
            )?;
            let has_provider_device: HasProviderDevice =
                symbol(&sherpa, b"SherpaOnnxOrtRuntimeHasExecutionProviderDevice\0")?;
            let runtime_error: RuntimeError =
                symbol(&sherpa, b"SherpaOnnxOrtRuntimeGetLastError\0")?;
            let create_keyword_spotter = symbol(&sherpa, b"SherpaOnnxCreateKeywordSpotter\0")?;
            let destroy_keyword_spotter = symbol(&sherpa, b"SherpaOnnxDestroyKeywordSpotter\0")?;
            let create_keyword_stream = symbol(&sherpa, b"SherpaOnnxCreateKeywordStream\0")?;
            let is_keyword_stream_ready = symbol(&sherpa, b"SherpaOnnxIsKeywordStreamReady\0")?;
            let decode_keyword_stream = symbol(&sherpa, b"SherpaOnnxDecodeKeywordStream\0")?;
            let get_keyword_result = symbol(&sherpa, b"SherpaOnnxGetKeywordResult\0")?;
            let destroy_keyword_result = symbol(&sherpa, b"SherpaOnnxDestroyKeywordResult\0")?;
            let destroy_online_stream = symbol(&sherpa, b"SherpaOnnxDestroyOnlineStream\0")?;
            let online_stream_accept_waveform =
                symbol(&sherpa, b"SherpaOnnxOnlineStreamAcceptWaveform\0")?;
            let online_stream_input_finished =
                symbol(&sherpa, b"SherpaOnnxOnlineStreamInputFinished\0")?;

            let provider = if runtime == Runtime::Default {
                None
            } else {
                let (prefix, registration, ep_name, device) = provider_request(config, runtime)?;
                let provider_path = resolve_library(
                    &config.backend.provider_library,
                    crate::runtime_paths::PROVIDER_LIBRARY_ENV,
                    prefix,
                    config,
                    paths,
                )?;
                Some((
                    CString::new(registration)?,
                    CString::new(provider_path.to_string_lossy().as_bytes())?,
                    CString::new(ep_name)?,
                    CString::new(device)?,
                ))
            };
            let runtime_handle = create_ort_runtime();
            if runtime_handle.is_null() {
                bail!("create retained sherpa ONNX Runtime environment");
            }
            let mut guard = RuntimeGuard::new(runtime_handle, destroy_ort_runtime);
            if let Some((registration, provider_path, ep_name, device)) = provider {
                if register_provider(
                    runtime_handle,
                    registration.as_ptr(),
                    provider_path.as_ptr(),
                ) == 0
                {
                    let message = string(runtime_error(runtime_handle));
                    bail!("register execution-provider library: {message}");
                }
                if has_provider_device(runtime_handle, ep_name.as_ptr(), device.as_ptr()) == 0 {
                    let message = string(runtime_error(runtime_handle));
                    bail!("requested execution-provider device is unavailable: {message}");
                }
            }
            let runtime_handle = guard.release();
            Ok(Rc::new(Self {
                create_keyword_spotter,
                destroy_keyword_spotter,
                create_keyword_stream,
                is_keyword_stream_ready,
                decode_keyword_stream,
                get_keyword_result,
                destroy_keyword_result,
                destroy_online_stream,
                online_stream_accept_waveform,
                online_stream_input_finished,
                runtime: runtime_handle,
                destroy_ort_runtime,
                _sherpa: sherpa,
                _ort: ort,
            }))
        }
    }
}

struct RuntimeGuard {
    pointer: *mut COrtRuntime,
    destroy: DestroyOrtRuntime,
}

impl RuntimeGuard {
    fn new(pointer: *mut COrtRuntime, destroy: DestroyOrtRuntime) -> Self {
        Self { pointer, destroy }
    }

    fn release(&mut self) -> *mut COrtRuntime {
        let pointer = self.pointer;
        self.pointer = ptr::null_mut();
        pointer
    }
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.destroy)(self.pointer) };
        }
    }
}

impl Drop for Api {
    fn drop(&mut self) {
        if !self.runtime.is_null() {
            unsafe { (self.destroy_ort_runtime)(self.runtime) };
            self.runtime = ptr::null_mut();
        }
    }
}

fn provider_request(
    config: &Config,
    runtime: Runtime,
) -> Result<(&'static str, &'static str, &'static str, &'static str)> {
    match runtime {
        Runtime::Default => bail!("CPU runtime does not use a provider plugin"),
        Runtime::Cuda => Ok((
            "libonnxruntime_providers_cuda.so",
            "omawake-cuda",
            "CUDAExecutionProvider",
            "gpu",
        )),
        Runtime::Openvino => {
            let canonical = config.backend.canonical_device()?;
            let (execution_provider, device) = match canonical.as_str() {
                "auto" => ("OpenVINOExecutionProvider.AUTO", ""),
                "cpu" => ("OpenVINOExecutionProvider", "cpu"),
                "gpu" => ("OpenVINOExecutionProvider", "gpu"),
                "npu" => ("OpenVINOExecutionProvider", "npu"),
                _ => bail!(
                    "OpenVINO plugin runtime supports auto, cpu, gpu, or npu selection; got {canonical}"
                ),
            };
            Ok((
                "libonnxruntime_providers_openvino.so",
                "omawake-openvino",
                execution_provider,
                device,
            ))
        }
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    Ok(*unsafe { library.get::<T>(name) }
        .with_context(|| format!("resolve native symbol {}", String::from_utf8_lossy(name)))?)
}

fn resolve_library(
    configured: &Path,
    variable: &str,
    prefix: &str,
    config: &Config,
    paths: &AppPaths,
) -> Result<PathBuf> {
    if !configured.as_os_str().is_empty() {
        return validate_explicit_library(resolve_config_path(configured, &paths.config_file));
    }
    if let Some(environment) = env::var_os(variable).filter(|value| !value.is_empty()) {
        return validate_explicit_library(PathBuf::from(environment));
    }
    let report = crate::runtime_paths::report(&config.backend, &paths.config_file);
    let ambient = env::var_os("LD_LIBRARY_PATH")
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();
    report
        .effective_library_dirs
        .iter()
        .chain(&ambient)
        .find_map(|directory| library_in(directory, prefix))
        .with_context(|| {
            format!(
                "locate {prefix}; set backend.library_dirs, backend.{}_library, or {variable}",
                if prefix.contains("sherpa") {
                    "sherpa"
                } else {
                    "onnxruntime"
                }
            )
        })
}

fn resolve_config_path(path: &Path, config_path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

fn validate_explicit_library(path: PathBuf) -> Result<PathBuf> {
    if !path.is_absolute() || !path.is_file() {
        bail!(
            "native library must be an absolute existing file: {}",
            path.display()
        );
    }
    Ok(path)
}

fn library_in(directory: &Path, prefix: &str) -> Option<PathBuf> {
    let mut matches = fs::read_dir(directory)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            (entry
                .file_type()
                .is_ok_and(|kind| kind.is_file() || kind.is_symlink())
                && (name == prefix || name.starts_with(&format!("{prefix}."))))
            .then(|| entry.path())
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches.into_iter().next()
}

fn string(pointer: *const c_char) -> String {
    if pointer.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

#[derive(Default)]
struct Strings(Vec<CString>);

impl Strings {
    fn pointer(&mut self, value: Option<&str>) -> Result<*const c_char> {
        let Some(value) = value else {
            return Ok(ptr::null());
        };
        let value = CString::new(value).context("sherpa configuration contains an interior NUL")?;
        self.0.push(value);
        Ok(self.0.last().unwrap().as_ptr())
    }
}

impl KeywordSpotterConfig {
    fn to_ffi(&self, strings: &mut Strings) -> Result<CKeywordSpotterConfig> {
        Ok(CKeywordSpotterConfig {
            feat_config: CFeatureConfig {
                sample_rate: self.feat_config.sample_rate,
                feature_dim: self.feat_config.feature_dim,
            },
            model_config: COnlineModelConfig {
                transducer: COnlineTransducerModelConfig {
                    encoder: strings.pointer(self.model_config.transducer.encoder.as_deref())?,
                    decoder: strings.pointer(self.model_config.transducer.decoder.as_deref())?,
                    joiner: strings.pointer(self.model_config.transducer.joiner.as_deref())?,
                },
                paraformer: COnlineParaformerModelConfig::default(),
                zipformer2_ctc: COnlineSingleModelConfig::default(),
                tokens: strings.pointer(self.model_config.tokens.as_deref())?,
                num_threads: self.model_config.num_threads,
                provider: strings.pointer(self.model_config.provider.as_deref())?,
                debug: 0,
                model_type: ptr::null(),
                modeling_unit: ptr::null(),
                bpe_vocab: ptr::null(),
                tokens_buf: ptr::null(),
                tokens_buf_size: 0,
                nemo_ctc: COnlineSingleModelConfig::default(),
                t_one_ctc: COnlineSingleModelConfig::default(),
            },
            max_active_paths: self.max_active_paths,
            num_trailing_blanks: self.num_trailing_blanks,
            keywords_score: self.keywords_score,
            keywords_threshold: self.keywords_threshold,
            keywords_file: ptr::null(),
            keywords_buf: strings.pointer(self.keywords_buf.as_deref())?,
            keywords_buf_size: self
                .keywords_buf
                .as_ref()
                .map_or(0, |value| value.len() as i32),
        })
    }
}

// Layouts and signatures below are pinned to sherpa-onnx v1.13.8
// `c-api/c-api.h`. Keep `ffi_layout_matches_pinned_sherpa_header` in sync when
// changing the supported native ABI.
#[repr(C)]
#[derive(Clone, Copy)]
struct CFeatureConfig {
    sample_rate: i32,
    feature_dim: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct COnlineTransducerModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
    joiner: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct COnlineParaformerModelConfig {
    encoder: *const c_char,
    decoder: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct COnlineSingleModelConfig {
    model: *const c_char,
}

#[repr(C)]
struct COnlineModelConfig {
    transducer: COnlineTransducerModelConfig,
    paraformer: COnlineParaformerModelConfig,
    zipformer2_ctc: COnlineSingleModelConfig,
    tokens: *const c_char,
    num_threads: i32,
    provider: *const c_char,
    debug: i32,
    model_type: *const c_char,
    modeling_unit: *const c_char,
    bpe_vocab: *const c_char,
    tokens_buf: *const u8,
    tokens_buf_size: i32,
    nemo_ctc: COnlineSingleModelConfig,
    t_one_ctc: COnlineSingleModelConfig,
}

#[repr(C)]
struct CKeywordSpotterConfig {
    feat_config: CFeatureConfig,
    model_config: COnlineModelConfig,
    max_active_paths: i32,
    num_trailing_blanks: i32,
    keywords_score: c_float,
    keywords_threshold: c_float,
    keywords_file: *const c_char,
    keywords_buf: *const c_char,
    keywords_buf_size: i32,
}

#[repr(C)]
struct CKeywordResult {
    keyword: *const c_char,
    _tokens: *const c_char,
    tokens_arr: *const *const c_char,
    count: i32,
    timestamps: *mut c_float,
    start_time: c_float,
    _json: *const c_char,
}

#[repr(C)]
struct CKeywordSpotter {
    _private: [u8; 0],
}

#[repr(C)]
struct COnlineStream {
    _private: [u8; 0],
}

#[repr(C)]
struct COrtRuntime {
    _private: [u8; 0],
}

#[repr(C)]
struct COrtApiBase {
    get_api: unsafe extern "C" fn(u32) -> *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

type VersionString = unsafe extern "C" fn() -> *const c_char;
type OmaRuntimeAbi = unsafe extern "C" fn() -> i32;
type OrtGetApiBase = unsafe extern "C" fn() -> *const COrtApiBase;
type CreateOrtRuntime = unsafe extern "C" fn() -> *mut COrtRuntime;
type DestroyOrtRuntime = unsafe extern "C" fn(*mut COrtRuntime);
type RegisterProvider = unsafe extern "C" fn(*mut COrtRuntime, *const c_char, *const c_char) -> i32;
type HasProviderDevice =
    unsafe extern "C" fn(*mut COrtRuntime, *const c_char, *const c_char) -> i32;
type RuntimeError = unsafe extern "C" fn(*const COrtRuntime) -> *const c_char;
type CreateKeywordSpotter =
    unsafe extern "C" fn(*const CKeywordSpotterConfig) -> *const CKeywordSpotter;
type DestroyKeywordSpotter = unsafe extern "C" fn(*const CKeywordSpotter);
type CreateKeywordStream = unsafe extern "C" fn(*const CKeywordSpotter) -> *const COnlineStream;
type IsKeywordStreamReady =
    unsafe extern "C" fn(*const CKeywordSpotter, *const COnlineStream) -> i32;
type DecodeKeywordStream = unsafe extern "C" fn(*const CKeywordSpotter, *const COnlineStream);
type GetKeywordResult =
    unsafe extern "C" fn(*const CKeywordSpotter, *const COnlineStream) -> *const CKeywordResult;
type DestroyKeywordResult = unsafe extern "C" fn(*const CKeywordResult);
type DestroyOnlineStream = unsafe extern "C" fn(*const COnlineStream);
type OnlineStreamAcceptWaveform = unsafe extern "C" fn(*const COnlineStream, i32, *const f32, i32);
type OnlineStreamInputFinished = unsafe extern "C" fn(*const COnlineStream);

#[cfg(test)]
mod ffi_tests {
    use super::*;
    use std::mem::{offset_of, size_of};
    use std::process::Command;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn ffi_layout_matches_pinned_sherpa_header() {
        assert_eq!(size_of::<CFeatureConfig>(), 8);
        assert_eq!(size_of::<COnlineModelConfig>(), 136);
        assert_eq!(offset_of!(COnlineModelConfig, provider), 64);
        assert_eq!(offset_of!(COnlineModelConfig, nemo_ctc), 120);
        assert_eq!(size_of::<CKeywordSpotterConfig>(), 184);
        assert_eq!(offset_of!(CKeywordSpotterConfig, max_active_paths), 144);
        assert_eq!(offset_of!(CKeywordSpotterConfig, keywords_buf), 168);
        assert_eq!(size_of::<CKeywordResult>(), 56);
        assert_eq!(offset_of!(CKeywordResult, count), 24);
        assert_eq!(offset_of!(CKeywordResult, timestamps), 32);
        assert_eq!(offset_of!(CKeywordResult, _json), 48);
    }

    #[test]
    fn ffi_conversion_preserves_strings_and_rejects_nul() {
        let mut config = KeywordSpotterConfig::default();
        config.model_config.transducer.encoder = Some("encoder.onnx".into());
        config.keywords_buf = Some("TOKENS @wake".into());
        let mut strings = Strings::default();
        let ffi = config.to_ffi(&mut strings).unwrap();
        assert_eq!(string(ffi.model_config.transducer.encoder), "encoder.onnx");
        assert_eq!(string(ffi.keywords_buf), "TOKENS @wake");

        config.keywords_buf = Some("bad\0keyword".into());
        assert!(config.to_ffi(&mut Strings::default()).is_err());
    }

    #[test]
    fn plugin_requests_are_exact_and_unknown_devices_fail_closed() {
        let mut config = Config::default();
        config.backend.runtime = Runtime::Cuda;
        config.backend.device = "gpu".into();
        assert_eq!(provider_request(&config, Runtime::Cuda).unwrap().3, "gpu");

        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "npu".into();
        assert_eq!(
            provider_request(&config, Runtime::Openvino).unwrap().3,
            "npu"
        );
        config.backend.device = "auto".into();
        let auto = provider_request(&config, Runtime::Openvino).unwrap();
        assert_eq!(auto.2, "OpenVINOExecutionProvider.AUTO");
        assert_eq!(auto.3, "");
        config.backend.device = "tpu".into();
        assert!(provider_request(&config, Runtime::Openvino).is_err());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn loads_pinned_fake_stack_and_runs_keyword_stream() {
        let root = env::temp_dir().join(format!("omawake-ffi-stack-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let compile = |name: &str, source: &str| {
            let source_path = root.join(format!("{name}.c"));
            let library_path = root.join(format!("lib{name}.so"));
            fs::write(&source_path, source).unwrap();
            let output = Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source_path)
                .arg("-o")
                .arg(&library_path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            library_path
        };
        let ort = compile(
            "onnxruntime",
            r#"
#include <stdint.h>
typedef struct { const void *(*GetApi)(uint32_t); const char *(*GetVersionString)(void); } OrtApiBase;
static const char *version(void) { return "1.29.0"; }
static const OrtApiBase base = {0, version};
const OrtApiBase *OrtGetApiBase(void) { return &base; }
"#,
        );
        let sherpa = compile(
            "sherpa-onnx-c-api",
            r#"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { const char *keyword; const char *tokens; const char *const *tokens_arr; int32_t count; float *timestamps; float start_time; const char *json; } Result;
static int runtime, spotter, stream, ready;
static const char *tokens[] = {"WAKE", "WORD"};
static float timestamps[] = {0.25f, 0.50f};
static Result result = {"wake-word", "WAKE WORD", tokens, 2, timestamps, 0.25f, "{}"};
const char *SherpaOnnxGetVersionStr(void) { return "1.13.8"; }
const char *SherpaOnnxGetOnnxruntimeVersionStr(void) { return "1.29.0"; }
int32_t SherpaOnnxGetOmaRuntimeAbiVersion(void) { return 1; }
void *SherpaOnnxCreateOrtRuntime(void) { return &runtime; }
void SherpaOnnxDestroyOrtRuntime(void *p) { (void)p; }
int32_t SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(void *r, const char *n, const char *p) { return r && n && p; }
int32_t SherpaOnnxOrtRuntimeHasExecutionProviderDevice(void *r, const char *ep, const char *device) { return r && ep && strcmp(device, "gpu") == 0; }
const char *SherpaOnnxOrtRuntimeGetLastError(const void *r) { (void)r; return "fake error"; }
const void *SherpaOnnxCreateKeywordSpotter(const void *config) { return config ? &spotter : 0; }
void SherpaOnnxDestroyKeywordSpotter(const void *p) { (void)p; }
const void *SherpaOnnxCreateKeywordStream(const void *p) { return p ? &stream : 0; }
int32_t SherpaOnnxIsKeywordStreamReady(const void *p, const void *s) { return p && s && ready; }
void SherpaOnnxDecodeKeywordStream(const void *p, const void *s) { (void)p; (void)s; ready = 0; }
const Result *SherpaOnnxGetKeywordResult(const void *p, const void *s) { return p && s ? &result : 0; }
void SherpaOnnxDestroyKeywordResult(const Result *r) { (void)r; }
void SherpaOnnxDestroyOnlineStream(const void *s) { (void)s; }
void SherpaOnnxOnlineStreamAcceptWaveform(const void *s, int32_t rate, const float *samples, int32_t count) { ready = s && rate > 0 && samples && count > 0; }
void SherpaOnnxOnlineStreamInputFinished(const void *s) { ready = s != 0; }
"#,
        );
        let provider = compile(
            "onnxruntime_providers_cuda",
            "int omawake_provider(void) { return 1; }",
        );
        let paths = AppPaths {
            config_file: root.join("config.toml"),
            data_dir: root.join("data"),
            state_dir: root.join("state"),
            runtime_dir: root.join("run"),
        };
        let mut app = Config::default();
        app.backend.onnxruntime_library = ort;
        app.backend.sherpa_library = sherpa;
        app.backend.provider_library = provider;
        app.backend.runtime = Runtime::Cuda;
        app.backend.device = "gpu".into();

        let api = Api::load(&app, &paths, Runtime::Cuda).unwrap();
        let config = KeywordSpotterConfig {
            keywords_buf: Some("WAKE WORD @wake-word".into()),
            ..Default::default()
        };
        let spotter = Spotter::create(api, &config).unwrap().unwrap();
        let stream = spotter.create_stream();
        stream.accept_waveform(16_000, &[0.1, 0.2]).unwrap();
        assert!(spotter.is_ready(&stream));
        spotter.decode(&stream);
        let result = spotter.result(&stream).unwrap();
        assert_eq!(result.keyword, "wake-word");
        assert_eq!(result.tokens, ["WAKE", "WORD"]);
        assert_eq!(result.timestamps, [0.25, 0.5]);
        stream.input_finished().unwrap();

        for model in ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"] {
            fs::write(root.join(model), b"fixture").unwrap();
        }
        app.model.directory = root.display().to_string();
        app.model.encoder = "encoder.onnx".into();
        app.model.decoder = "decoder.onnx".into();
        app.model.joiner = "joiner.onnx".into();
        app.model.tokens = "tokens.txt".into();
        let backend =
            SherpaOnnxBackend::load(&app, &paths, &root, Runtime::Cuda, "WAKE @wake-word").unwrap();
        assert_eq!(backend.kind(), "sherpa-onnx");
        let backend_stream = backend.stream();
        assert_eq!(
            backend_stream.accept(16_000, &[0.1, 0.2]).unwrap()[0].id,
            "wake-word"
        );
        assert_eq!(backend_stream.finish().unwrap()[0].id, "wake-word");
        drop(backend_stream);

        let wav = root.join("input.wav");
        let mut writer = hound::WavWriter::create(
            &wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(0.1_f32).unwrap();
        writer.finalize().unwrap();
        assert!(!backend.detect_file(&wav).unwrap().is_empty());
        drop(backend);

        let stereo = root.join("stereo.wav");
        let writer = hound::WavWriter::create(
            &stereo,
            hound::WavSpec {
                channels: 2,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.finalize().unwrap();
        assert!(read_wave(&stereo).unwrap_err().to_string().contains("mono"));

        let int24 = root.join("int24.wav");
        let mut writer = hound::WavWriter::create(
            &int24,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 24,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.write_sample(1_i32 << 22).unwrap();
        writer.finalize().unwrap();
        assert_eq!(read_wave(&int24).unwrap().1, [0.5]);

        let mut discovered = app.clone();
        discovered.backend.onnxruntime_library = PathBuf::new();
        discovered.backend.sherpa_library = PathBuf::new();
        discovered.backend.provider_library = PathBuf::new();
        discovered.backend.library_dirs = vec![root.clone()];
        drop(Api::load(&discovered, &paths, Runtime::Default).unwrap());

        let openvino = compile(
            "onnxruntime_providers_openvino",
            "int omawake_openvino_provider(void) { return 1; }",
        );
        let mut unavailable = app.clone();
        unavailable.backend.runtime = Runtime::Openvino;
        unavailable.backend.device = "npu".into();
        unavailable.backend.provider_library = openvino;
        let error = match Api::load(&unavailable, &paths, Runtime::Openvino) {
            Ok(_) => panic!("fake runtime unexpectedly exposed an NPU"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("device is unavailable"));

        assert!(
            resolve_library(
                Path::new("missing.so"),
                "OMAWAKE_ONNXRUNTIME_LIBRARY",
                "libonnxruntime.so",
                &app,
                &paths,
            )
            .is_err()
        );
        assert_eq!(string(ptr::null()), "");
        let _ = fs::remove_dir_all(root);
    }
}
