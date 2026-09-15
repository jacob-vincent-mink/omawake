//! Direct execution of the bundled icefall GigaSpeech keyword model.
//!
//! The graph contract and decoding behavior are specified by icefall PR #1428,
//! its `keywords_search` reference decoder, and its modified Aho-Corasick
//! context graph. The implementation here is independent Rust and uses only the
//! model's ONNX files plus ONNX Runtime; there is no sherpa runtime or ABI.

mod fbank;
mod keyword;
mod model;
mod resample;

use anyhow::{Context, Result, bail};
use std::{cell::RefCell, fs, path::Path};

use self::{
    fbank::OnlineFbank,
    keyword::{KeywordBeam, KeywordGraph},
    model::{EncoderState, Model},
    resample::AudioResampler,
};
use super::{Detection, WakeWordBackend, WakeWordStream, detect_samples};
use crate::{backend::Runtime, config::Config, paths::AppPaths};

const INPUT_FRAMES: usize = 45;
const FRAME_ADVANCE: usize = 32;
const OUTPUT_FRAMES: usize = 8;
const ENCODER_DIMENSION: usize = 320;

pub(super) struct OmaOnnxBackend {
    model: RefCell<Model>,
    graph: KeywordGraph,
    beam_width: usize,
    trailing_blanks: usize,
}

struct OmaOnnxStream<'a> {
    backend: &'a OmaOnnxBackend,
    state: RefCell<StreamState>,
}

struct StreamState {
    resampler: AudioResampler,
    fbank: OnlineFbank,
    encoder: Option<EncoderState>,
    beam: KeywordBeam,
    encoder_frame: usize,
    finished: bool,
}

impl OmaOnnxBackend {
    pub(super) fn load(
        config: &Config,
        paths: &AppPaths,
        directory: &Path,
        runtime: Runtime,
        keywords_buffer: &str,
    ) -> Result<Self> {
        if config.model.sample_rate != fbank::SAMPLE_RATE as i32 {
            bail!("the bundled GigaSpeech model requires model.sample_rate = 16000");
        }
        let token_path = directory.join(&config.model.tokens);
        let token_contents = fs::read_to_string(&token_path)
            .with_context(|| format!("read model tokens {}", token_path.display()))?;
        let graph = KeywordGraph::from_buffer(
            keywords_buffer,
            &token_contents,
            config.model.keywords_score,
            config.model.keywords_threshold,
        )?;
        let trailing_blanks = usize::try_from(config.model.num_trailing_blanks)
            .context("model.num_trailing_blanks must not be negative")?;
        let plan = Model::plan(config, runtime)?;
        let beam_width = plan.beam_width;
        let model = Model::load(config, paths, directory, runtime, &plan)?;
        Ok(Self {
            model: RefCell::new(model),
            graph,
            beam_width,
            trailing_blanks,
        })
    }

    fn decode_ready(&self, stream: &mut StreamState) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        let mut model = self.model.try_borrow_mut().map_err(|_| {
            anyhow::anyhow!("a second detector stream attempted inference concurrently")
        })?;
        if stream.encoder.is_none() {
            stream.encoder = Some(model.initial_state()?);
        }
        while stream.fbank.frames_ready() > INPUT_FRAMES {
            let features = stream.fbank.chunk(INPUT_FRAMES)?;
            let state = stream
                .encoder
                .take()
                .context("encoder state is unavailable")?;
            let (encoded, next_state) = model.encode(features, state)?;
            stream.encoder = Some(next_state);
            for output_frame in 0..OUTPUT_FRAMES {
                let paths = stream.beam.len();
                let logits = model.decode_join(
                    &encoded
                        [output_frame * ENCODER_DIMENSION..(output_frame + 1) * ENCODER_DIMENSION],
                    stream.beam.contexts(),
                    paths,
                )?;
                if let Some(matched) =
                    stream
                        .beam
                        .advance(&logits, stream.encoder_frame, &self.graph)?
                {
                    let tokens = matched
                        .tokens
                        .iter()
                        .map(|&token| model.token(token).to_owned())
                        .collect();
                    let timestamps = matched
                        .timestamps
                        .iter()
                        .map(|&frame| frame as f32 * 0.04)
                        .collect();
                    detections.push(Detection {
                        id: matched.id,
                        tokens,
                        timestamps,
                        start_time: 0.0,
                    });
                }
                stream.encoder_frame += 1;
            }
            stream.fbank.advance(FRAME_ADVANCE);
            if stream.beam.trailing_blanks() as f32 * 0.04 > 1.5 {
                stream.beam.reset();
            }
        }
        Ok(detections)
    }
}

impl WakeWordBackend for OmaOnnxBackend {
    fn kind(&self) -> &'static str {
        "omawake-onnx"
    }

    fn stream(&self) -> Box<dyn WakeWordStream + '_> {
        Box::new(OmaOnnxStream {
            backend: self,
            state: RefCell::new(StreamState {
                resampler: AudioResampler::new(),
                fbank: OnlineFbank::new(),
                encoder: None,
                beam: KeywordBeam::new(self.beam_width, self.trailing_blanks),
                encoder_frame: 0,
                finished: false,
            }),
        })
    }

    fn detect_file(&self, path: &Path) -> Result<Vec<Detection>> {
        let (sample_rate, samples) = read_wave(path)?;
        let stream = self.stream();
        detect_samples(stream.as_ref(), sample_rate, &samples)
    }
}

impl WakeWordStream for OmaOnnxStream<'_> {
    fn accept(&self, sample_rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        let mut stream = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("detector stream is already in use"))?;
        if stream.finished {
            bail!("audio was supplied after the stream finished");
        }
        let normalized = stream.resampler.accept(sample_rate, samples)?;
        stream.fbank.accept(&normalized)?;
        self.backend.decode_ready(&mut stream)
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        let mut stream = self
            .state
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("detector stream is already in use"))?;
        if stream.finished {
            return Ok(Vec::new());
        }
        stream.finished = true;
        let final_samples = stream.resampler.finish()?;
        stream.fbank.accept(&final_samples)?;
        stream.fbank.finish();
        self.backend.decode_ready(&mut stream)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::WakeWord, engine::Detector};
    use std::{env, path::PathBuf};

    fn fixture_paths(name: &str) -> AppPaths {
        let root = env::temp_dir().join(format!("omawake-onnx-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        AppPaths {
            config_file: root.join("config.toml"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
            state_dir: root.join("state"),
            runtime_dir: root.join("run"),
        }
    }

    #[test]
    fn wav_reader_covers_float_wide_integer_and_rejection_paths() {
        let paths = fixture_paths("wav-formats");
        fs::create_dir_all(&paths.data_dir).unwrap();

        let float_path = paths.data_dir.join("float.wav");
        let mut writer = hound::WavWriter::create(
            &float_path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 8_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(-0.25f32).unwrap();
        writer.write_sample(0.5f32).unwrap();
        writer.finalize().unwrap();
        let (rate, samples) = read_wave(&float_path).unwrap();
        assert_eq!(rate, 8_000);
        assert_eq!(samples, [-0.25, 0.5]);

        let wide_path = paths.data_dir.join("wide.wav");
        let mut writer = hound::WavWriter::create(
            &wide_path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 24,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.write_sample(-(1i32 << 22)).unwrap();
        writer.write_sample(1i32 << 21).unwrap();
        writer.finalize().unwrap();
        let (_, samples) = read_wave(&wide_path).unwrap();
        assert_eq!(samples, [-0.5, 0.25]);

        let stereo_path = paths.data_dir.join("stereo.wav");
        let mut writer = hound::WavWriter::create(
            &stereo_path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.write_sample(0i16).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.finalize().unwrap();
        assert!(
            read_wave(&stereo_path)
                .unwrap_err()
                .to_string()
                .contains("mono")
        );
        assert!(read_wave(&paths.data_dir.join("missing.wav")).is_err());
    }

    #[test]
    fn real_model_fixtures_preserve_direct_streaming_parity() {
        let Some((ort, model)) = real_fixture() else {
            return;
        };
        let paths = fixture_paths("real-model");
        let mut config = Config::default();
        config.backend.onnxruntime_library = ort;
        config.model.directory = model.to_string_lossy().into_owned();
        config.wake_words = [
            ("light-up", "Light up"),
            ("lovely-child", "Lovely child"),
            ("forever", "Forever"),
        ]
        .into_iter()
        .map(|(id, phrase)| WakeWord {
            id: id.into(),
            phrase: phrase.into(),
            enabled: true,
            command: vec!["true".into()],
        })
        .collect();

        let detector = Detector::load(&config, &paths).unwrap();
        assert_eq!(detector.backend_kind, "omawake-onnx");
        let zero = detector
            .detect_file(&model.join("test_wavs/0.wav"))
            .unwrap();
        assert_eq!(
            zero.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            ["light-up"]
        );
        let one = detector
            .detect_file(&model.join("test_wavs/1.wav"))
            .unwrap();
        assert_eq!(
            one.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            ["lovely-child", "forever"]
        );

        let stream = detector.session();
        assert!(stream.accept(16_000, &[f32::NAN]).is_err());
        assert!(stream.finish().unwrap().is_empty());
        assert!(stream.finish().unwrap().is_empty());
        assert!(stream.accept(16_000, &[0.0]).is_err());
    }

    #[test]
    fn real_openvino_cpu_plugin_preserves_detection_when_available() {
        let Some((ort, model)) = real_fixture() else {
            return;
        };
        let Some(provider) = env::var_os("OMAWAKE_TEST_OPENVINO_PROVIDER").map(PathBuf::from)
        else {
            return;
        };
        assert!(
            provider.is_file(),
            "OMAWAKE_TEST_OPENVINO_PROVIDER is not a file"
        );
        let paths = fixture_paths("real-openvino");
        let mut config = Config::default();
        config.backend.runtime = Runtime::Openvino;
        config.backend.device = "cpu".into();
        config.backend.fallback = crate::backend::Fallback::Error;
        config.backend.onnxruntime_library = ort;
        config.backend.provider_library = provider.clone();
        config.backend.library_dirs = vec![provider.parent().unwrap().to_owned()];
        config.model.directory = model.to_string_lossy().into_owned();
        config.wake_words = vec![WakeWord {
            id: "light-up".into(),
            phrase: "Light up".into(),
            enabled: true,
            command: vec!["true".into()],
        }];

        let detector = Detector::load(&config, &paths).unwrap();
        assert_eq!(detector.effective_runtime, Runtime::Openvino);
        assert!(!detector.fallback_used);
        assert_eq!(
            detector
                .detect_file(&model.join("test_wavs/0.wav"))
                .unwrap()[0]
                .id,
            "light-up"
        );
    }

    fn real_fixture() -> Option<(PathBuf, PathBuf)> {
        let ort = env::var_os("OMAWAKE_TEST_ONNXRUNTIME").map(PathBuf::from);
        let model = env::var_os("OMAWAKE_TEST_MODEL").map(PathBuf::from);
        match (ort, model) {
            (None, None) => None,
            (Some(ort), Some(model)) => {
                assert!(ort.is_file(), "OMAWAKE_TEST_ONNXRUNTIME is not a file");
                assert!(model.is_dir(), "OMAWAKE_TEST_MODEL is not a directory");
                Some((ort, model))
            }
            _ => panic!("set both OMAWAKE_TEST_ONNXRUNTIME and OMAWAKE_TEST_MODEL"),
        }
    }
}
