use anyhow::{Context, Result, bail};
use ort::{
    ep::ExecutionProviderLibrary,
    logging::LogLevel,
    memory::{Allocator, DeviceType},
    session::{OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel},
    value::{DynValue, Tensor},
};
use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{backend::Runtime, config::Config, paths::AppPaths};

const ENCODER_FRAMES: usize = 45;
const ENCODER_OUTPUT_FRAMES: usize = 8;
const ENCODER_DIMENSION: usize = 320;
const VOCABULARY_SIZE: usize = 500;
const NPU_MIN_BEAM_WIDTH: usize = 8;

struct ProviderSelection {
    ep: &'static str,
    device_type: Option<DeviceType>,
    device_id: Option<u32>,
    options: Vec<(String, String)>,
    specialize_shapes: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct InferencePlan {
    encoder: String,
    decoder: String,
    joiner: String,
    pub beam_width: usize,
    fixed_batch: Option<usize>,
}

pub struct Model {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    encoder_state_shapes: Vec<Vec<i64>>,
    token_symbols: Vec<String>,
    fixed_batch: Option<usize>,
    // Registration is environment-wide. Retain the handle to document and
    // enforce that the plugin remains registered for the session lifetime.
    _provider: Option<ExecutionProviderLibrary>,
}

pub struct EncoderState {
    values: Vec<DynValue>,
}

impl Model {
    pub(super) fn plan(config: &Config, runtime: Runtime) -> Result<InferencePlan> {
        inference_plan(config, runtime)
    }

    pub fn load(
        config: &Config,
        paths: &AppPaths,
        directory: &Path,
        runtime: Runtime,
        plan: &InferencePlan,
    ) -> Result<Self> {
        let libraries = crate::runtime_paths::discover(&config.backend, &paths.config_file);
        let core = libraries
            .onnxruntime_library
            .context("ONNX Runtime 1.30.0 was not found; reinstall the release package or configure an exact core library path")?;
        validate_runtime_version(&core)?;
        ort::init_from(&core)
            .with_context(|| format!("load ONNX Runtime {}", core.display()))?
            .with_name("omawake")
            .commit();
        let environment = ort::environment::Environment::current()?;

        let (provider, selection) = if runtime == Runtime::Default {
            (None, None)
        } else {
            let library = libraries.provider_library.with_context(|| {
                format!(
                    "{} execution-provider plugin was not found; install a compatible official plugin package and select it with `omawake setup runtime`",
                    runtime_name(runtime)
                )
            })?;
            let handle = environment
                .register_ep_library(format!("omawake-{}", runtime_name(runtime)), &library)
                .with_context(|| format!("register provider plugin {}", library.display()))?;
            (
                Some(handle),
                Some(provider_selection(config, paths, runtime)?),
            )
        };

        let required = |name: &str| -> Result<PathBuf> {
            let path = directory.join(name);
            if !path.is_file() {
                bail!("required model asset is missing: {}", path.display());
            }
            Ok(path)
        };
        let encoder_path = required(&plan.encoder)?;
        let decoder_path = required(&plan.decoder)?;
        let joiner_path = required(&plan.joiner)?;
        let tokens_path = required(&config.model.tokens)?;

        let encoder = build_session(
            &encoder_path,
            config.backend.threads,
            selection.as_ref(),
            plan.fixed_batch.map(|_| 1),
        )
        .context("load streaming encoder")?;
        let decoder = build_session(
            &decoder_path,
            config.backend.threads,
            selection.as_ref(),
            plan.fixed_batch,
        )
        .context("load transducer decoder")?;
        let joiner = build_session(
            &joiner_path,
            config.backend.threads,
            selection.as_ref(),
            plan.fixed_batch,
        )
        .context("load transducer joiner")?;
        validate_graph_contract(&encoder, &decoder, &joiner)?;
        let encoder_state_shapes = encoder
            .inputs()
            .iter()
            .skip(1)
            .map(|input| tensor_shape(input.dtype()).map(concrete_shape))
            .collect::<Result<Vec<_>>>()?;
        let token_symbols = load_tokens(&tokens_path)?;
        if token_symbols.len() != VOCABULARY_SIZE {
            bail!(
                "tokens.txt has {} symbols; model expects {VOCABULARY_SIZE}",
                token_symbols.len()
            );
        }
        Ok(Self {
            encoder,
            decoder,
            joiner,
            encoder_state_shapes,
            token_symbols,
            fixed_batch: plan.fixed_batch,
            _provider: provider,
        })
    }

    pub fn initial_state(&self) -> Result<EncoderState> {
        let final_index = self.encoder_state_shapes.len().saturating_sub(1);
        let values = self
            .encoder_state_shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                let elements = element_count(shape)?;
                if index == final_index {
                    Ok(Tensor::from_array((shape.clone(), vec![0i64; elements]))?.into_dyn())
                } else {
                    Ok(Tensor::from_array((shape.clone(), vec![0.0f32; elements]))?.into_dyn())
                }
            })
            .collect::<Result<_>>()?;
        Ok(EncoderState { values })
    }

    pub fn encode(
        &mut self,
        features: Vec<f32>,
        state: EncoderState,
    ) -> Result<(Vec<f32>, EncoderState)> {
        if features.len() != ENCODER_FRAMES * 80 {
            bail!(
                "encoder requires {ENCODER_FRAMES}x80 features, got {} values",
                features.len()
            );
        }
        let names: Vec<String> = self
            .encoder
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect();
        let mut inputs = Vec::with_capacity(names.len());
        inputs.push((
            names[0].clone(),
            Tensor::from_array((vec![1, ENCODER_FRAMES as i64, 80], features))?.into_dyn(),
        ));
        inputs.extend(names.into_iter().skip(1).zip(state.values));
        let mut selector = OutputSelector::default();
        for (index, output) in self.encoder.outputs().iter().enumerate() {
            let shape = if index == 0 {
                vec![1, ENCODER_OUTPUT_FRAMES as i64, ENCODER_DIMENSION as i64]
            } else {
                self.encoder_state_shapes[index - 1].clone()
            };
            selector = if index + 1 == self.encoder.outputs().len() {
                selector.preallocate(
                    output.name().to_owned(),
                    Tensor::<i64>::new(&Allocator::default(), shape)?,
                )
            } else {
                selector.preallocate(
                    output.name().to_owned(),
                    Tensor::<f32>::new(&Allocator::default(), shape)?,
                )
            };
        }
        let options = RunOptions::new()?.with_outputs(selector);
        let outputs = self.encoder.run_with_options(inputs, &options)?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, ENCODER_OUTPUT_FRAMES as i64, ENCODER_DIMENSION as i64] {
            bail!("unexpected encoder output shape {shape:?}");
        }
        let encoded = values.to_vec();
        let mut next_state = Vec::with_capacity(self.encoder_state_shapes.len());
        for (index, state_shape) in self.encoder_state_shapes.iter().enumerate() {
            let output = &outputs[index + 1];
            let value = if index + 1 == self.encoder_state_shapes.len() {
                let (_, values) = output.try_extract_tensor::<i64>()?;
                Tensor::from_array((state_shape.clone(), values.to_vec()))?.into_dyn()
            } else {
                let (_, values) = output.try_extract_tensor::<f32>()?;
                Tensor::from_array((state_shape.clone(), values.to_vec()))?.into_dyn()
            };
            next_state.push(value);
        }
        Ok((encoded, EncoderState { values: next_state }))
    }

    pub fn decode_join(
        &mut self,
        encoded: &[f32],
        contexts: Vec<i64>,
        paths: usize,
    ) -> Result<Vec<f32>> {
        if contexts.len() != paths * 2 {
            bail!("decoder context count does not match beam size");
        }
        if self.fixed_batch.is_none() {
            let outputs = self.decoder.run(ort::inputs![Tensor::from_array((
                vec![paths as i64, 2],
                contexts
            ))?])?;
            let (shape, decoded) = outputs[0].try_extract_tensor::<f32>()?;
            if shape.as_ref() != [paths as i64, ENCODER_DIMENSION as i64] {
                bail!("unexpected decoder output shape {shape:?}");
            }
            let mut repeated = Vec::with_capacity(paths * ENCODER_DIMENSION);
            for _ in 0..paths {
                repeated.extend_from_slice(encoded);
            }
            let outputs = self.joiner.run(ort::inputs![
                Tensor::from_array((vec![paths as i64, ENCODER_DIMENSION as i64], repeated))?,
                Tensor::from_array((
                    vec![paths as i64, ENCODER_DIMENSION as i64],
                    decoded.to_vec()
                ))?,
            ])?;
            let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
            if shape.as_ref() != [paths as i64, VOCABULARY_SIZE as i64] {
                bail!("unexpected joiner output shape {shape:?}");
            }
            return Ok(logits.to_vec());
        }
        let batch = self.fixed_batch.context("fixed NPU batch is unavailable")?;
        if paths > batch {
            bail!("beam has {paths} paths but the NPU graph was specialized for {batch}");
        }
        let mut padded_contexts = contexts;
        while padded_contexts.len() < batch * 2 {
            padded_contexts.extend_from_slice(&[-1, 0]);
        }
        let decoder_options =
            RunOptions::new()?.with_outputs(OutputSelector::default().preallocate(
                self.decoder.outputs()[0].name().to_owned(),
                Tensor::<f32>::new(&Allocator::default(), [batch, ENCODER_DIMENSION])?,
            ));
        let outputs = self.decoder.run_with_options(
            ort::inputs![Tensor::from_array((
                vec![batch as i64, 2],
                padded_contexts
            ))?],
            &decoder_options,
        )?;
        let (shape, decoded) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.as_ref() != [batch as i64, ENCODER_DIMENSION as i64] {
            bail!("unexpected decoder output shape {shape:?}");
        }
        // Ignore decoder values for padded hypotheses. Zeroing those rows
        // before the quantized joiner keeps its activation range independent
        // of placeholder contexts and matches the fixed-batch reference run.
        let decoded = joiner_decoded_batch(decoded, paths, batch)?;
        let mut repeated = Vec::with_capacity(batch * ENCODER_DIMENSION);
        for _ in 0..batch {
            repeated.extend_from_slice(encoded);
        }
        let joiner_options =
            RunOptions::new()?.with_outputs(OutputSelector::default().preallocate(
                self.joiner.outputs()[0].name().to_owned(),
                Tensor::<f32>::new(&Allocator::default(), [batch, VOCABULARY_SIZE])?,
            ));
        let outputs = self.joiner.run_with_options(
            ort::inputs![
                Tensor::from_array((vec![batch as i64, ENCODER_DIMENSION as i64], repeated))?,
                Tensor::from_array((vec![batch as i64, ENCODER_DIMENSION as i64], decoded))?,
            ],
            &joiner_options,
        )?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.as_ref() != [batch as i64, VOCABULARY_SIZE as i64] {
            bail!("unexpected joiner output shape {shape:?}");
        }
        Ok(logits[..paths * VOCABULARY_SIZE].to_vec())
    }

    pub fn token(&self, id: i64) -> &str {
        self.token_symbols
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("<invalid>")
    }
}

fn build_session(
    path: &Path,
    threads: u16,
    selection: Option<&ProviderSelection>,
    fixed_batch: Option<usize>,
) -> Result<Session> {
    let optimization = if selection.is_some() {
        GraphOptimizationLevel::Disable
    } else {
        GraphOptimizationLevel::Level3
    };
    let mut builder = Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(optimization)
        .map_err(ort_error)?
        .with_intra_threads(threads as usize)
        .map_err(ort_error)?;
    if let Some(selection) = selection {
        if selection.specialize_shapes {
            builder = builder.with_log_level(LogLevel::Fatal).map_err(ort_error)?;
        }
        let environment = ort::environment::Environment::current()?;
        let devices: Vec<_> = environment
            .devices()
            .filter(|device| {
                device.ep().ok() == Some(selection.ep)
                    && selection
                        .device_type
                        .is_none_or(|kind| device.hardware_device().ty() == kind)
                    && selection
                        .device_id
                        .is_none_or(|id| device.hardware_device().id() == id)
            })
            .collect();
        if devices.is_empty() {
            bail!("{} plugin exposed no matching device", selection.ep);
        }
        let mut options = selection.options.clone();
        if selection.specialize_shapes {
            options.push((
                format!("{}.reshape_input", selection.ep),
                fixed_input_shapes(path, threads, fixed_batch.unwrap_or(1))?,
            ));
        }
        builder = builder
            .with_devices(devices, Some(&options))
            .map_err(ort_error)?;
    }
    Ok(builder.commit_from_file(path)?)
}

fn ort_error<R>(error: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn fixed_input_shapes(path: &Path, threads: u16, batch: usize) -> Result<String> {
    let inspection = Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(ort_error)?
        .with_intra_threads(threads as usize)
        .map_err(ort_error)?
        .commit_from_file(path)?;
    inspection
        .inputs()
        .iter()
        .map(|input| {
            let dimensions = tensor_shape(input.dtype())?
                .into_iter()
                .enumerate()
                .map(|(index, dimension)| {
                    if dimension < 0 && index == 0 {
                        batch.to_string()
                    } else {
                        dimension.max(1).to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            Ok(format!("{}[{dimensions}]", input.name()))
        })
        .collect::<Result<Vec<_>>>()
        .map(|inputs| inputs.join(","))
}

fn provider_selection(
    config: &Config,
    paths: &AppPaths,
    runtime: Runtime,
) -> Result<ProviderSelection> {
    let device = config.backend.canonical_device()?;
    let (ep, device_type, device_id) = match (runtime, device.as_str()) {
        (Runtime::Openvino, "auto") => ("OpenVINOExecutionProvider.AUTO", None, None),
        (Runtime::Openvino, "cpu") => ("OpenVINOExecutionProvider", Some(DeviceType::CPU), None),
        (Runtime::Openvino, "gpu") => ("OpenVINOExecutionProvider", Some(DeviceType::GPU), None),
        (Runtime::Openvino, "npu") => ("OpenVINOExecutionProvider", Some(DeviceType::NPU), None),
        (Runtime::Cuda, _) => (
            "CUDAExecutionProvider",
            Some(DeviceType::GPU),
            Some(config.backend.device_id),
        ),
        _ => bail!("invalid provider selection"),
    };
    let mut options = Vec::new();
    if runtime == Runtime::Openvino {
        let cache = paths
            .cache_dir
            .join("openvino")
            .join(&device)
            .join("compiled");
        fs::create_dir_all(&cache)?;
        let mut load_config = serde_json::Map::new();
        let mut properties = serde_json::Map::new();
        properties.insert(
            "CACHE_DIR".into(),
            cache.to_string_lossy().into_owned().into(),
        );
        if device == "npu" {
            properties.insert("EXECUTION_MODE_HINT".into(), "ACCURACY".into());
            properties.insert("LOG_LEVEL".into(), "LOG_NONE".into());
        }
        load_config.insert(device.to_ascii_uppercase(), properties.into());
        options.push((
            format!("{ep}.load_config"),
            serde_json::Value::Object(load_config).to_string(),
        ));
    }
    for (key, value) in &config.backend.options {
        validate_option(key, value)?;
        if runtime == Runtime::Openvino && matches!(key.as_str(), "load_config" | "reshape_input") {
            bail!(
                "backend option {key} is managed by Omawake for OpenVINO; remove it from backend.options"
            );
        }
        options.push((format!("{ep}.{key}"), value.clone()));
    }
    Ok(ProviderSelection {
        ep,
        device_type,
        device_id,
        options,
        specialize_shapes: runtime == Runtime::Openvino && device == "npu",
    })
}

fn inference_plan(config: &Config, runtime: Runtime) -> Result<InferencePlan> {
    let configured_width = usize::try_from(config.model.max_active_paths)
        .ok()
        .filter(|width| *width > 0)
        .context("model.max_active_paths must be positive")?;
    let explicit_npu =
        runtime == Runtime::Openvino && config.backend.canonical_device()?.as_str() == "npu";
    let beam_width = if explicit_npu {
        configured_width.max(NPU_MIN_BEAM_WIDTH)
    } else {
        configured_width
    };
    let mut plan = InferencePlan {
        encoder: config.model.encoder.clone(),
        decoder: config.model.decoder.clone(),
        joiner: config.model.joiner.clone(),
        beam_width,
        fixed_batch: explicit_npu.then_some(beam_width),
    };
    if let Some(spec) = crate::catalog::model(&config.model.name) {
        let known_graphs = [spec.encoder, spec.openvino_npu_encoder, spec.cuda_encoder]
            .contains(&config.model.encoder.as_str())
            && [spec.decoder, spec.openvino_npu_decoder, spec.cuda_decoder]
                .contains(&config.model.decoder.as_str())
            && [spec.joiner, spec.openvino_npu_joiner, spec.cuda_joiner]
                .contains(&config.model.joiner.as_str());
        if known_graphs {
            plan.encoder = match runtime {
                Runtime::Openvino if explicit_npu => spec.openvino_npu_encoder,
                Runtime::Cuda => spec.cuda_encoder,
                _ => spec.encoder,
            }
            .into();
            plan.decoder = match runtime {
                Runtime::Openvino if explicit_npu => spec.openvino_npu_decoder,
                Runtime::Cuda => spec.cuda_decoder,
                _ => spec.decoder,
            }
            .into();
            plan.joiner = match runtime {
                Runtime::Openvino if explicit_npu => spec.openvino_npu_joiner,
                Runtime::Cuda => spec.cuda_joiner,
                _ => spec.joiner,
            }
            .into();
        }
    }
    Ok(plan)
}

fn joiner_decoded_batch(decoded: &[f32], paths: usize, batch: usize) -> Result<Vec<f32>> {
    let active = paths
        .checked_mul(ENCODER_DIMENSION)
        .context("decoder output size overflow")?;
    let total = batch
        .checked_mul(ENCODER_DIMENSION)
        .context("decoder batch size overflow")?;
    if active > decoded.len() || active > total {
        bail!("decoder output does not cover the active fixed batch");
    }
    let mut values = decoded[..active].to_vec();
    values.resize(total, 0.0);
    Ok(values)
}

fn validate_graph_contract(encoder: &Session, decoder: &Session, joiner: &Session) -> Result<()> {
    if encoder.inputs().len() != 39 || encoder.outputs().len() != 39 {
        bail!(
            "unsupported encoder state contract: {} inputs and {} outputs",
            encoder.inputs().len(),
            encoder.outputs().len()
        );
    }
    if decoder.inputs().len() != 1 || decoder.outputs().len() != 1 {
        bail!("unsupported decoder graph contract");
    }
    if joiner.inputs().len() != 2 || joiner.outputs().len() != 1 {
        bail!("unsupported joiner graph contract");
    }
    Ok(())
}

fn tensor_shape(value_type: &ort::value::ValueType) -> Result<Vec<i64>> {
    match value_type {
        ort::value::ValueType::Tensor { shape, .. } => Ok(shape.iter().copied().collect()),
        other => bail!("expected tensor input, got {other}"),
    }
}

fn concrete_shape(mut shape: Vec<i64>) -> Vec<i64> {
    for dimension in &mut shape {
        if *dimension < 0 {
            *dimension = 1;
        }
    }
    shape
}

fn element_count(shape: &[i64]) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, &dimension| {
        total
            .checked_mul(dimension as usize)
            .context("state shape overflow")
    })
}

fn load_tokens(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut tokens = Vec::new();
    for line in contents.lines() {
        let (token, id) = line.rsplit_once(' ').context("malformed tokens.txt line")?;
        let id: usize = id.parse()?;
        if tokens.len() <= id {
            tokens.resize(id + 1, String::new());
        }
        if !tokens[id].is_empty() {
            bail!("duplicate token id {id}");
        }
        tokens[id] = token.to_owned();
    }
    Ok(tokens)
}

fn validate_option(key: &str, value: &str) -> Result<()> {
    if key.is_empty()
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        bail!("backend option key {key:?} is invalid");
    }
    if value.contains(['\0', '\r', '\n']) {
        bail!("backend option {key} contains an invalid character");
    }
    Ok(())
}

fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "cpu",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
    }
}

fn validate_runtime_version(path: &Path) -> Result<()> {
    #[repr(C)]
    struct OrtApiBase {
        get_api: unsafe extern "system" fn(u32) -> *const std::ffi::c_void,
        get_version_string: unsafe extern "system" fn() -> *const std::ffi::c_char,
    }
    type GetApiBase = unsafe extern "system" fn() -> *const OrtApiBase;
    unsafe {
        let library = libloading::Library::new(path)
            .with_context(|| format!("load ONNX Runtime {}", path.display()))?;
        let get_base: libloading::Symbol<GetApiBase> = library
            .get(b"OrtGetApiBase\0")
            .context("resolve OrtGetApiBase")?;
        let base = get_base();
        if base.is_null() {
            bail!("OrtGetApiBase returned null");
        }
        let raw = ((*base).get_version_string)();
        if raw.is_null() {
            bail!("ONNX Runtime returned a null version string");
        }
        let version = std::ffi::CStr::from_ptr(raw).to_string_lossy();
        if version != "1.30.0" {
            bail!("Omawake requires ONNX Runtime 1.30.0, found {version}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_plan_selects_the_calibrated_graphs_and_beam_shape() {
        let mut config = Config::default();
        let spec = crate::catalog::models()[0];

        let cpu = inference_plan(&config, Runtime::Default).unwrap();
        assert_eq!(cpu.encoder, spec.encoder);
        assert_eq!(cpu.decoder, spec.decoder);
        assert_eq!(cpu.joiner, spec.joiner);
        assert_eq!(cpu.beam_width, 4);
        assert_eq!(cpu.fixed_batch, None);

        config.backend.runtime = Runtime::Openvino;
        for device in ["auto", "cpu", "gpu"] {
            config.backend.device = device.into();
            let plan = inference_plan(&config, Runtime::Openvino).unwrap();
            assert_eq!(plan.encoder, spec.encoder);
            assert_eq!(plan.beam_width, 4);
            assert_eq!(plan.fixed_batch, None);
        }

        config.backend.device = "npu".into();
        let npu = inference_plan(&config, Runtime::Openvino).unwrap();
        assert_eq!(npu.encoder, spec.openvino_npu_encoder);
        assert_eq!(npu.decoder, spec.openvino_npu_decoder);
        assert_eq!(npu.joiner, spec.openvino_npu_joiner);
        assert_eq!(npu.beam_width, NPU_MIN_BEAM_WIDTH);
        assert_eq!(npu.fixed_batch, Some(NPU_MIN_BEAM_WIDTH));

        config.model.max_active_paths = 12;
        let wider_npu = inference_plan(&config, Runtime::Openvino).unwrap();
        assert_eq!(wider_npu.beam_width, 12);
        assert_eq!(wider_npu.fixed_batch, Some(12));
    }

    #[test]
    fn runtime_plan_normalizes_catalog_graphs_for_cpu_fallback() {
        let mut config = Config::default();
        let spec = crate::catalog::models()[0];
        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "npu".into();
        config.model.encoder = spec.openvino_npu_encoder.into();
        config.model.decoder = spec.openvino_npu_decoder.into();
        config.model.joiner = spec.openvino_npu_joiner.into();

        let fallback = inference_plan(&config, Runtime::Default).unwrap();
        assert_eq!(fallback.encoder, spec.encoder);
        assert_eq!(fallback.decoder, spec.decoder);
        assert_eq!(fallback.joiner, spec.joiner);
        assert_eq!(fallback.fixed_batch, None);

        config.model.name = "custom".into();
        config.model.encoder = "custom-encoder.onnx".into();
        let custom = inference_plan(&config, Runtime::Openvino).unwrap();
        assert_eq!(custom.encoder, "custom-encoder.onnx");
        assert_eq!(custom.beam_width, NPU_MIN_BEAM_WIDTH);
    }

    #[test]
    fn fixed_batch_zeros_decoder_rows_for_padded_hypotheses() {
        let decoded = vec![3.0; NPU_MIN_BEAM_WIDTH * ENCODER_DIMENSION];
        let padded = joiner_decoded_batch(&decoded, 2, NPU_MIN_BEAM_WIDTH).unwrap();
        assert!(
            padded[..2 * ENCODER_DIMENSION]
                .iter()
                .all(|value| *value == 3.0)
        );
        assert!(
            padded[2 * ENCODER_DIMENSION..]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            joiner_decoded_batch(&decoded, NPU_MIN_BEAM_WIDTH + 1, NPU_MIN_BEAM_WIDTH).is_err()
        );
    }
}
