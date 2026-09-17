//! Frozen Whisper encoder via the public, dynamically loaded OpenVINO Rust
//! wrapper. No decoder, ONNX Runtime, private API, or copied KWS implementation.
use crate::{config::Config, paths::AppPaths};
use anyhow::{Context, Result, ensure};
use openvino::{
    CompiledModel, Core, DeviceType, ElementType, InferRequest, PartialShape, PropertyKey,
    RwPropertyKey, Shape, Tensor,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Embedding {
    pub encoder_contract: String,
    pub values: Vec<f32>,
    pub source_frames: usize,
    pub inference_ms: f64,
    pub execution_devices: String,
}
impl Embedding {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.values.len() == 512 && self.values.iter().all(|v| v.is_finite()),
            "invalid Whisper embedding"
        );
        ensure!(
            (1..=1500).contains(&self.source_frames)
                && self.inference_ms.is_finite()
                && self.inference_ms >= 0.0,
            "invalid embedding metrics"
        );
        ensure!(
            !self.encoder_contract.is_empty() && self.encoder_contract.len() <= 256,
            "invalid encoder contract"
        );
        Ok(())
    }
}
pub(crate) struct Encoder {
    request: InferRequest,
    _compiled: CompiledModel,
    _core: Core,
    contract: String,
    execution_devices: String,
}
impl Encoder {
    pub fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        ensure!(
            config.backend.runtime == crate::backend::Runtime::Openvino,
            "trained Whisper encoder requires OpenVINO"
        );
        let device = match config.backend.device.to_ascii_uppercase().as_str() {
            "CPU" => "CPU",
            "GPU" | "IGPU" => "GPU",
            "NPU" => "NPU",
            _ => anyhow::bail!("unsupported encoder device {}", config.backend.device),
        };
        let base = config.model_directory(paths);
        let xml = base.join("openvino_encoder_model.xml");
        let bin = base.join("openvino_encoder_model.bin");
        let mut hash = Sha256::new();
        hash.update(b"omawake-whisper-base.en-slaney80-hann400-hop160-reflect200-pool-real-l2-v1;silero-roll320-v1");
        for path in [&xml, &bin, &base.join(&config.model.vad)] {
            let mut file = fs::File::open(path)
                .with_context(|| format!("open encoder asset {}", path.display()))?;
            hash.update(file.metadata()?.len().to_le_bytes());
            let mut buffer = [0; 64 * 1024];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
        }
        let contract = format!("whisper-base-en-v1-{:x}", hash.finalize());
        let library = library(config, paths)?;
        openvino_sys::library::load_from(&library)
            .map_err(|e| anyhow::anyhow!("load OpenVINO {}: {e}", library.display()))?;
        let plugins = library
            .parent()
            .context("OpenVINO library has no parent")?
            .join("plugins.xml");
        let mut core = if plugins.is_file() {
            Core::new_with_config(plugins.to_str().context("non-UTF8 plugin path")?)?
        } else {
            Core::new()?
        };
        ensure!(
            core.available_devices()?
                .iter()
                .any(|d| d.as_ref().split('.').next() == Some(device)),
            "selected OpenVINO device {device} is unavailable"
        );
        let cache = paths
            .cache_dir
            .join("enrollment-encoder")
            .join(&contract)
            .join(device.to_ascii_lowercase());
        fs::create_dir_all(&cache)?;
        core.set_property(
            &DeviceType::from(device),
            &RwPropertyKey::CacheDir,
            cache.to_str().context("non-UTF8 cache path")?,
        )?;
        if device == "CPU" {
            core.set_property(
                &DeviceType::from(device),
                &RwPropertyKey::InferenceNumThreads,
                &config.backend.threads.to_string(),
            )?;
            // Match training precision across devices as far as this runtime supports.
            core.set_property(
                &DeviceType::from(device),
                &RwPropertyKey::HintInferencePrecision,
                "f32",
            )?;
        }
        let mut model = core.read_model_from_file(
            xml.to_str().context("non-UTF8 model path")?,
            bin.to_str().context("non-UTF8 weights path")?,
        )?;
        ensure!(
            model.get_inputs_len()? == 1 && model.get_outputs_len()? == 1,
            "expected a single-input, single-output Whisper encoder"
        );
        model.reshape_single_input(&PartialShape::new_static(3, &[1, 80, 3000])?)?;
        let mut compiled = core.compile_model(&model, DeviceType::from(device))?;
        let execution_devices = compiled
            .get_property(&PropertyKey::Other(Cow::Borrowed("EXECUTION_DEVICES")))?
            .into_owned();
        ensure!(
            execution_devices
                .split([' ', ',', ';'])
                .filter(|s| !s.is_empty())
                .all(|s| s.split('.').next() == Some(device))
                && !execution_devices.is_empty(),
            "encoder placement differs from requested {device}: {execution_devices}"
        );
        let request = compiled.create_infer_request()?;
        Ok(Self {
            request,
            _compiled: compiled,
            _core: core,
            contract,
            execution_devices,
        })
    }
    pub fn encode_samples(&mut self, samples: &[f32]) -> Result<Embedding> {
        ensure!(
            !samples.is_empty()
                && samples.len() <= 480_000
                && samples.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
            "invalid encoder audio"
        );
        let started = Instant::now();
        let mel = super::whisper_features::log_mel(samples)?;
        ensure!(mel.len() == 80 * 3000, "invalid Whisper feature dimensions");
        let mut input = Tensor::new(ElementType::F32, &Shape::new(&[1, 80, 3000])?)?;
        input.get_data_mut::<f32>()?.copy_from_slice(&mel);
        self.request.set_input_tensor(&input)?;
        self.request.infer()?;
        let output = self.request.get_output_tensor()?;
        ensure!(
            output.get_element_type()? == ElementType::F32
                && output.get_shape()?.get_dimensions() == [1, 1500, 512],
            "unsupported Whisper encoder output type/shape"
        );
        let output = output.get_data::<f32>()?;
        let (frames, values) = pool(output, samples.len())?;
        let result = Embedding {
            encoder_contract: self.contract.clone(),
            values,
            source_frames: frames,
            inference_ms: started.elapsed().as_secs_f64() * 1000.0,
            execution_devices: self.execution_devices.clone(),
        };
        result.validate()?;
        Ok(result)
    }
}
fn pool(output: &[f32], samples: usize) -> Result<(usize, Vec<f32>)> {
    ensure!(
        output.len() == 1500 * 512 && (1..=480_000).contains(&samples),
        "invalid encoder output or source length"
    );
    ensure!(
        output.iter().all(|v| v.is_finite()),
        "encoder returned non-finite values"
    );
    let frames = samples.div_ceil(320);
    let mut pooled = vec![0_f64; 512];
    for frame in output.as_chunks::<512>().0.iter().take(frames) {
        for (sum, value) in pooled.iter_mut().zip(frame) {
            *sum += *value as f64 / frames as f64;
        }
    }
    let norm = pooled.iter().map(|v| v * v).sum::<f64>().sqrt();
    ensure!(
        norm.is_finite() && norm > 1e-12,
        "encoder embedding has no finite content"
    );
    Ok((
        frames,
        pooled.into_iter().map(|v| (v / norm) as f32).collect(),
    ))
}

fn library(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    let mut dirs = config.backend.library_dirs.clone();
    if let Some(parent) = config
        .backend
        .library
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        dirs.insert(0, parent.to_owned());
    }
    if let Some(var) = std::env::var_os("OMAWAKE_OPENVINO_LIBRARY") {
        return fs::canonicalize(var).context("configured OpenVINO C library");
    }
    if let Some(var) = std::env::var_os("LD_LIBRARY_PATH") {
        dirs.extend(std::env::split_paths(&var));
    }
    dirs.extend([PathBuf::from("/usr/lib"), PathBuf::from("/usr/local/lib")]);
    let base = paths.config_file.parent().unwrap_or(Path::new("."));
    for dir in dirs {
        let dir = if dir.is_absolute() {
            dir
        } else {
            base.join(dir)
        };
        let path = dir.join("libopenvino_c.so");
        if path.is_file() {
            return fs::canonicalize(path).context("resolve OpenVINO C library");
        }
    }
    anyhow::bail!("libopenvino_c.so not found in selected runtime; configure backend.library_dirs")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pooling_ignores_padding_and_enforces_encoder_shape() {
        let mut output = vec![0.0; 1500 * 512];
        output[0] = 2.0;
        output[513] = 4.0;
        for frame in output.as_chunks_mut::<512>().0.iter_mut().skip(2) {
            frame[2] = 100.0;
        }
        let (frames, v) = pool(&output, 321).unwrap();
        assert_eq!(frames, 2);
        assert!((v[0] - 0.4472136).abs() < 1e-6);
        assert!((v[1] - 0.8944272).abs() < 1e-6);
        assert_eq!(v[2], 0.0);
        assert_eq!(pool(&output, 320).unwrap().1[0], 1.0);
        assert!(pool(&output, 0).is_err());
        assert!(pool(&output, 480001).is_err());
        assert!(pool(&output[..512], 320).is_err());
        output[700] = f32::NAN;
        assert!(pool(&output, 320).is_err());
        assert!(pool(&vec![0.0; 1500 * 512], 320).is_err());
    }
    #[test]
    fn invalid_models_fail_before_loading_native_code() {
        let root = crate::test_support::unique_directory("embedding-validation", "missing-model");
        let paths = crate::test_support::isolated_paths(&root);
        let mut config = Config::default();
        assert!(Encoder::load(&config, &paths).is_err());
        config.backend.runtime = crate::backend::Runtime::Openvino;
        config.backend.device = "unavailable".into();
        assert!(Encoder::load(&config, &paths).is_err());
        config.backend.device = "cpu".into();
        assert!(
            Encoder::load(&config, &paths)
                .err()
                .unwrap()
                .to_string()
                .contains("encoder asset")
        );
        let mut e = Embedding {
            encoder_contract: "encoder".into(),
            values: vec![0.1; 512],
            source_frames: 1,
            inference_ms: 1.0,
            execution_devices: "TEST".into(),
        };
        e.validate().unwrap();
        e.source_frames = 0;
        assert!(e.validate().is_err());
        e.source_frames = 1;
        e.values[0] = f32::NAN;
        assert!(e.validate().is_err());
        let model = paths.data_dir.join("models").join(&config.model.name);
        fs::create_dir_all(&model).unwrap();
        for asset in [
            "openvino_encoder_model.xml",
            "openvino_encoder_model.bin",
            &config.model.vad,
        ] {
            fs::write(
                model.join(asset),
                b"fixture bytes for hashing; never passed to a native model reader",
            )
            .unwrap();
        }
        let libdir = root.join("invalid-library");
        fs::create_dir_all(&libdir).unwrap();
        fs::write(
            libdir.join("libopenvino_c.so"),
            b"not an ELF shared library",
        )
        .unwrap();
        config.backend.library_dirs = vec![libdir];
        assert!(
            Encoder::load(&config, &paths)
                .err()
                .unwrap()
                .to_string()
                .contains("load OpenVINO")
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod library_tests {
    use super::*;
    #[test]
    fn selects_explicit_and_config_relative_runtime_directories() {
        let root = crate::test_support::unique_directory("encoder-lib", "paths");
        let paths = crate::test_support::isolated_paths(&root);
        let dir = root.join("default/runtime");
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("libopenvino_c.so");
        fs::write(&file, b"lookup-only fixture, never loaded").unwrap();
        let mut config = Config::default();
        config.backend.library_dirs = vec!["runtime".into()];
        assert_eq!(
            library(&config, &paths).unwrap(),
            fs::canonicalize(&file).unwrap()
        );
        config.backend.library_dirs.clear();
        config.backend.library = dir.join("libopenvino_genai_c.so");
        assert_eq!(
            library(&config, &paths).unwrap(),
            fs::canonicalize(&file).unwrap()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
