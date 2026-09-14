//! Thin CPAL adapter. Device-dependent calls live here so the audio policy and
//! sample processing in the parent module remain deterministic to test.

use super::*;
use ::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ::cpal::{Device, Stream};

pub(super) struct CpalCaptureFactory;

impl CaptureStream for Stream {
    fn play(&self) -> Result<()> {
        StreamTrait::play(self).context("start input stream")
    }
}

impl CaptureFactory for CpalCaptureFactory {
    fn open(&self, requested_device: &str, queue_capacity: usize) -> Result<OpenedCapture> {
        let host = ::cpal::default_host();
        open_capture(
            &CpalAudioSource { host: &host },
            requested_device,
            queue_capacity,
        )
    }
}

pub(super) fn input_devices() -> Result<Vec<String>> {
    let host = ::cpal::default_host();
    input_device_names(&CpalAudioSource { host: &host })
}

struct CpalAudioSource<'a> {
    host: &'a ::cpal::Host,
}

impl DeviceSource for CpalAudioSource<'_> {
    type Device = Device;

    fn default_device(&self) -> Result<Option<Self::Device>> {
        Ok(self.host.default_input_device())
    }

    fn input_devices(&self) -> Result<Vec<Self::Device>> {
        Ok(self
            .host
            .input_devices()
            .context("enumerate input devices")?
            .collect())
    }

    fn device_name(&self, device: &Self::Device) -> Result<String> {
        device.name().context("query input device name")
    }
}

impl AudioSource for CpalAudioSource<'_> {
    fn input_config(&self, device: &Self::Device) -> Result<(SampleEncoding, StreamConfig)> {
        let supported = device
            .default_input_config()
            .context("query input configuration")?;
        let encoding = supported_sample_encoding(supported.sample_format())?;
        Ok((encoding, supported.into()))
    }

    fn build_stream(
        &self,
        device: &Self::Device,
        config: &StreamConfig,
        encoding: SampleEncoding,
        sender: SyncSender<AudioEvent>,
    ) -> Result<Box<dyn CaptureStream>> {
        let stream = match encoding {
            SampleEncoding::F32 => build_stream::<f32, _>(device, config, sender, f32_to_f32),
            SampleEncoding::I16 => build_stream::<i16, _>(device, config, sender, i16_to_f32),
            SampleEncoding::U16 => build_stream::<u16, _>(device, config, sender, u16_to_f32),
        }?;
        Ok(Box::new(stream))
    }
}

fn build_stream<T, F>(
    device: &Device,
    config: &StreamConfig,
    sender: SyncSender<AudioEvent>,
    convert: F,
) -> Result<Stream, ::cpal::BuildStreamError>
where
    T: ::cpal::SizedSample,
    F: Fn(T) -> f32 + Send + Sync + 'static,
{
    let channels = usize::from(config.channels);
    let sample_rate = config.sample_rate.0 as i32;
    let error_sender = sender.clone();
    device.build_input_stream(
        config,
        move |input: &[T], _| {
            send_samples(&sender, sample_rate, mix_to_mono(input, channels, &convert));
        },
        move |error| send_error(&error_sender, error.to_string()),
        None,
    )
}
