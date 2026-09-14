//! Thin sherpa-onnx adapter. Model validation, keyword compilation, runtime
//! selection, chunking, and detection validation remain in the parent module.

use super::*;
use sherpa_onnx::{KeywordSpotter, Wave};

pub(super) struct SherpaOnnxBackend {
    spotter: KeywordSpotter,
}

struct SherpaOnnxStream<'a> {
    backend: &'a SherpaOnnxBackend,
    stream: sherpa_onnx::OnlineStream,
}

impl SherpaOnnxBackend {
    pub(super) fn load(
        config: &Config,
        directory: &Path,
        runtime: Runtime,
        keywords_buffer: &str,
    ) -> Result<Self> {
        let sherpa_config = build_sherpa_config(config, directory, runtime, keywords_buffer)?;
        let spotter = KeywordSpotter::create(&sherpa_config)
            .context("sherpa-onnx could not create the keyword spotter")?;
        Ok(Self { spotter })
    }

    fn decode_ready(&self, stream: &sherpa_onnx::OnlineStream) -> Vec<Detection> {
        drain_ready(|| {
            if !self.spotter.is_ready(stream) {
                return None;
            }
            self.spotter.decode(stream);
            Some(self.spotter.get_result(stream).and_then(|result| {
                detection_from_parts(
                    result.keyword,
                    result.tokens_arr,
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
        let wave = Wave::read(path.to_string_lossy().as_ref())
            .with_context(|| format!("read audio fixture {}", path.display()))?;
        let stream = self.stream();
        detect_samples(stream.as_ref(), wave.sample_rate(), wave.samples())
    }
}

impl WakeWordStream for SherpaOnnxStream<'_> {
    fn accept(&self, sample_rate: i32, samples: &[f32]) -> Result<Vec<Detection>> {
        self.stream.accept_waveform(sample_rate, samples);
        Ok(self.backend.decode_ready(&self.stream))
    }

    fn finish(&self) -> Result<Vec<Detection>> {
        self.stream.input_finished();
        Ok(self.backend.decode_ready(&self.stream))
    }
}
