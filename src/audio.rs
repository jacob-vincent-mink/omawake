use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};

pub enum AudioEvent {
    Samples { sample_rate: i32, samples: Vec<f32> },
    Error(String),
}

pub struct Capture {
    _stream: Stream,
    receiver: Receiver<AudioEvent>,
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Capture {
    pub fn start(requested_device: &str, queue_capacity: usize) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_device(&host, requested_device)?;
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        let supported = device
            .default_input_config()
            .with_context(|| format!("query input configuration for {device_name}"))?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let sample_rate = config.sample_rate.0;
        let channels = config.channels;
        let (sender, receiver) = sync_channel(queue_capacity.max(1));

        let stream = match sample_format {
            SampleFormat::F32 => build_stream::<f32, _>(&device, &config, sender, |v| v),
            SampleFormat::I16 => {
                build_stream::<i16, _>(&device, &config, sender, |v| v as f32 / 32768.0)
            }
            SampleFormat::U16 => {
                build_stream::<u16, _>(&device, &config, sender, |v| (v as f32 - 32768.0) / 32768.0)
            }
            other => bail!("unsupported input sample format {other:?}"),
        }
        .with_context(|| format!("open input device {device_name}"))?;
        stream
            .play()
            .with_context(|| format!("start input device {device_name}"))?;

        Ok(Self {
            _stream: stream,
            receiver,
            device_name,
            sample_rate,
            channels,
        })
    }

    pub fn receiver(&self) -> &Receiver<AudioEvent> {
        &self.receiver
    }
}

pub fn input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let mut names = host
        .input_devices()
        .context("enumerate input devices")?
        .filter_map(|device| device.name().ok())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    Ok(names)
}

fn select_device(host: &cpal::Host, requested: &str) -> Result<Device> {
    if requested.is_empty() || requested.eq_ignore_ascii_case("default") {
        return host
            .default_input_device()
            .context("no default input device");
    }
    let needle = requested.to_lowercase();
    host.input_devices()
        .context("enumerate input devices")?
        .find(|device| {
            device
                .name()
                .map(|name| name.to_lowercase() == needle)
                .unwrap_or(false)
        })
        .with_context(|| format!("input device not found: {requested}"))
}

fn build_stream<T, F>(
    device: &Device,
    config: &StreamConfig,
    sender: SyncSender<AudioEvent>,
    convert: F,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample,
    F: Fn(T) -> f32 + Send + Sync + 'static,
{
    let channels = usize::from(config.channels);
    let sample_rate = config.sample_rate.0 as i32;
    let error_sender = sender.clone();
    device.build_input_stream(
        config,
        move |input: &[T], _| {
            let mono = input
                .chunks(channels)
                .map(|frame| frame.iter().copied().map(&convert).sum::<f32>() / channels as f32)
                .collect();
            let _ = sender.try_send(AudioEvent::Samples {
                sample_rate,
                samples: mono,
            });
        },
        move |error| {
            let _ = error_sender.try_send(AudioEvent::Error(error.to_string()));
        },
        None,
    )
}
