use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use ::cpal::{SampleFormat, StreamConfig};
use anyhow::{Context, Result, bail};

mod cpal;
use self::cpal::CpalCaptureFactory;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampleEncoding {
    F32,
    I16,
    U16,
}

pub enum AudioEvent {
    Samples { sample_rate: i32, samples: Vec<f32> },
    Error(String),
}

pub struct Capture {
    _stream: Box<dyn CaptureStream>,
    receiver: Receiver<AudioEvent>,
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

trait CaptureStream: Send {
    fn play(&self) -> Result<()>;
}

struct OpenedCapture {
    stream: Box<dyn CaptureStream>,
    receiver: Receiver<AudioEvent>,
    device_name: String,
    sample_rate: u32,
    channels: u16,
}

trait CaptureFactory {
    fn open(&self, requested_device: &str, queue_capacity: usize) -> Result<OpenedCapture>;
}

impl Capture {
    pub fn start(requested_device: &str, queue_capacity: usize) -> Result<Self> {
        Self::start_with(&CpalCaptureFactory, requested_device, queue_capacity)
    }

    fn start_with(
        factory: &impl CaptureFactory,
        requested_device: &str,
        queue_capacity: usize,
    ) -> Result<Self> {
        let opened = factory.open(requested_device, queue_capacity)?;
        opened
            .stream
            .play()
            .with_context(|| format!("start input device {}", opened.device_name))?;
        Ok(Self {
            _stream: opened.stream,
            receiver: opened.receiver,
            device_name: opened.device_name,
            sample_rate: opened.sample_rate,
            channels: opened.channels,
        })
    }

    pub fn receiver(&self) -> &Receiver<AudioEvent> {
        &self.receiver
    }
}

pub fn input_devices() -> Result<Vec<String>> {
    cpal::input_devices()
}

fn input_device_names<S: DeviceSource>(source: &S) -> Result<Vec<String>> {
    let names = source
        .input_devices()?
        .iter()
        .filter_map(|device| source.device_name(device).ok())
        .collect::<Vec<_>>();
    Ok(normalize_device_names(names))
}

fn normalize_device_names(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut names = names.into_iter().collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn wants_default_device(requested: &str) -> bool {
    requested.is_empty() || requested.eq_ignore_ascii_case("default")
}

fn supported_sample_encoding(format: SampleFormat) -> Result<SampleEncoding> {
    match format {
        SampleFormat::F32 => Ok(SampleEncoding::F32),
        SampleFormat::I16 => Ok(SampleEncoding::I16),
        SampleFormat::U16 => Ok(SampleEncoding::U16),
        other => bail!("unsupported input sample format {other:?}"),
    }
}

fn f32_to_f32(value: f32) -> f32 {
    value
}

fn i16_to_f32(value: i16) -> f32 {
    value as f32 / 32768.0
}

fn u16_to_f32(value: u16) -> f32 {
    (value as f32 - 32768.0) / 32768.0
}

trait DeviceSource {
    type Device;

    fn default_device(&self) -> Result<Option<Self::Device>>;
    fn input_devices(&self) -> Result<Vec<Self::Device>>;
    fn device_name(&self, device: &Self::Device) -> Result<String>;
}

trait AudioSource: DeviceSource {
    fn input_config(&self, device: &Self::Device) -> Result<(SampleEncoding, StreamConfig)>;
    fn build_stream(
        &self,
        device: &Self::Device,
        config: &StreamConfig,
        encoding: SampleEncoding,
        sender: SyncSender<AudioEvent>,
    ) -> Result<Box<dyn CaptureStream>>;
}

fn choose_device<S: DeviceSource>(requested: &str, source: &S) -> Result<S::Device> {
    if wants_default_device(requested) {
        return source.default_device()?.context("no default input device");
    }
    let needle = requested.to_lowercase();
    source
        .input_devices()?
        .into_iter()
        .find(|device| {
            source
                .device_name(device)
                .map(|name| name.to_lowercase() == needle)
                .unwrap_or(false)
        })
        .with_context(|| format!("input device not found: {requested}"))
}

fn open_capture<S: AudioSource>(
    source: &S,
    requested_device: &str,
    queue_capacity: usize,
) -> Result<OpenedCapture> {
    let device = choose_device(requested_device, source)?;
    let device_name = source
        .device_name(&device)
        .unwrap_or_else(|_| "unknown".into());
    let (encoding, config) = source
        .input_config(&device)
        .with_context(|| format!("query input configuration for {device_name}"))?;
    let sample_rate = config.sample_rate.0;
    let channels = config.channels;
    let (sender, receiver) = sync_channel(queue_capacity.max(1));
    let stream = source
        .build_stream(&device, &config, encoding, sender)
        .with_context(|| format!("open input device {device_name}"))?;
    Ok(OpenedCapture {
        stream,
        receiver,
        device_name,
        sample_rate,
        channels,
    })
}

fn send_samples(sender: &SyncSender<AudioEvent>, sample_rate: i32, samples: Vec<f32>) {
    let _ = sender.try_send(AudioEvent::Samples {
        sample_rate,
        samples,
    });
}

fn send_error(sender: &SyncSender<AudioEvent>, error: String) {
    let _ = sender.try_send(AudioEvent::Error(error));
}

fn mix_to_mono<T: Copy>(input: &[T], channels: usize, convert: &impl Fn(T) -> f32) -> Vec<f32> {
    input
        .chunks(channels.max(1))
        .map(|frame| frame.iter().copied().map(convert).sum::<f32>() / frame.len() as f32)
        .collect()
}

#[cfg(test)]
#[path = "../tests/unit/audio.rs"]
mod tests;
