use super::*;
use anyhow::anyhow;
use std::cell::Cell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct FakeDeviceSource {
    default: Option<String>,
    devices: Vec<String>,
    bad_name: Option<String>,
    fail_default: bool,
    fail_enumeration: bool,
    default_calls: Cell<usize>,
    enumeration_calls: Cell<usize>,
}

impl FakeDeviceSource {
    fn new(default: Option<&str>, devices: &[&str]) -> Self {
        Self {
            default: default.map(str::to_owned),
            devices: devices.iter().map(|value| (*value).to_owned()).collect(),
            bad_name: None,
            fail_default: false,
            fail_enumeration: false,
            default_calls: Cell::new(0),
            enumeration_calls: Cell::new(0),
        }
    }
}

impl DeviceSource for FakeDeviceSource {
    type Device = String;

    fn default_device(&self) -> Result<Option<Self::Device>> {
        self.default_calls.set(self.default_calls.get() + 1);
        if self.fail_default {
            Err(anyhow!("default query failed"))
        } else {
            Ok(self.default.clone())
        }
    }

    fn input_devices(&self) -> Result<Vec<Self::Device>> {
        self.enumeration_calls.set(self.enumeration_calls.get() + 1);
        if self.fail_enumeration {
            Err(anyhow!("enumeration failed"))
        } else {
            Ok(self.devices.clone())
        }
    }

    fn device_name(&self, device: &Self::Device) -> Result<String> {
        if self.bad_name.as_ref() == Some(device) {
            Err(anyhow!("disconnected"))
        } else {
            Ok(device.clone())
        }
    }
}

struct FakeCaptureStream {
    played: Arc<AtomicBool>,
    fail: bool,
}

impl CaptureStream for FakeCaptureStream {
    fn play(&self) -> Result<()> {
        self.played.store(true, Ordering::SeqCst);
        if self.fail {
            Err(anyhow!("play failed"))
        } else {
            Ok(())
        }
    }
}

struct FakeCaptureFactory {
    played: Arc<AtomicBool>,
    fail_open: bool,
    fail_play: bool,
    open_calls: Cell<usize>,
    observed_capacity: Cell<usize>,
}

struct FakeAudioSource {
    devices: FakeDeviceSource,
    encoding: SampleEncoding,
    fail_config: bool,
    fail_build: bool,
    played: Arc<AtomicBool>,
}

impl FakeAudioSource {
    fn working() -> Self {
        Self {
            devices: FakeDeviceSource::new(Some("Built-in"), &["Built-in", "USB Mic"]),
            encoding: SampleEncoding::F32,
            fail_config: false,
            fail_build: false,
            played: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl DeviceSource for FakeAudioSource {
    type Device = String;

    fn default_device(&self) -> Result<Option<Self::Device>> {
        self.devices.default_device()
    }

    fn input_devices(&self) -> Result<Vec<Self::Device>> {
        self.devices.input_devices()
    }

    fn device_name(&self, device: &Self::Device) -> Result<String> {
        self.devices.device_name(device)
    }
}

impl AudioSource for FakeAudioSource {
    fn input_config(&self, _: &Self::Device) -> Result<(SampleEncoding, StreamConfig)> {
        if self.fail_config {
            return Err(anyhow!("config failed"));
        }
        Ok((
            self.encoding,
            StreamConfig {
                channels: 2,
                sample_rate: ::cpal::SampleRate(22_050),
                buffer_size: ::cpal::BufferSize::Default,
            },
        ))
    }

    fn build_stream(
        &self,
        _: &Self::Device,
        _: &StreamConfig,
        _: SampleEncoding,
        sender: SyncSender<AudioEvent>,
    ) -> Result<Box<dyn CaptureStream>> {
        if self.fail_build {
            return Err(anyhow!("build failed"));
        }
        send_samples(&sender, 22_050, vec![0.5]);
        Ok(Box::new(FakeCaptureStream {
            played: Arc::clone(&self.played),
            fail: false,
        }))
    }
}

impl FakeCaptureFactory {
    fn working() -> Self {
        Self {
            played: Arc::new(AtomicBool::new(false)),
            fail_open: false,
            fail_play: false,
            open_calls: Cell::new(0),
            observed_capacity: Cell::new(0),
        }
    }
}

impl CaptureFactory for FakeCaptureFactory {
    fn open(&self, requested_device: &str, queue_capacity: usize) -> Result<OpenedCapture> {
        self.open_calls.set(self.open_calls.get() + 1);
        self.observed_capacity.set(queue_capacity);
        if self.fail_open {
            return Err(anyhow!("open failed"));
        }
        assert_eq!(requested_device, "test microphone");
        let (sender, receiver) = sync_channel(queue_capacity.max(1));
        send_samples(&sender, 22_050, vec![0.125, -0.125]);
        Ok(OpenedCapture {
            stream: Box::new(FakeCaptureStream {
                played: Arc::clone(&self.played),
                fail: self.fail_play,
            }),
            receiver,
            device_name: "Fake Input".into(),
            sample_rate: 22_050,
            channels: 2,
        })
    }
}

#[test]
fn mixes_interleaved_samples_to_mono() {
    assert_eq!(
        mix_to_mono(&[1.0_f32, -1.0, 0.5, 0.5], 2, &f32_to_f32),
        vec![0.0, 0.5]
    );
    assert_eq!(
        mix_to_mono(&[i16::MIN, i16::MAX], 1, &i16_to_f32),
        vec![-1.0, 32767.0 / 32768.0]
    );
    assert_eq!(mix_to_mono(&[0_u16, 65535], 2, &u16_to_f32).len(), 1);
    assert_eq!(
        mix_to_mono(&[2.0_f32, 4.0, 9.0], 2, &f32_to_f32),
        vec![3.0, 9.0]
    );
    assert_eq!(mix_to_mono(&[0.75_f32], 0, &f32_to_f32), vec![0.75]);
    assert!(mix_to_mono::<f32>(&[], 2, &f32_to_f32).is_empty());
}

#[test]
fn sample_formats_and_normalization_boundaries_are_explicit() {
    assert_eq!(
        supported_sample_encoding(SampleFormat::F32).unwrap(),
        SampleEncoding::F32
    );
    assert_eq!(
        supported_sample_encoding(SampleFormat::I16).unwrap(),
        SampleEncoding::I16
    );
    assert_eq!(
        supported_sample_encoding(SampleFormat::U16).unwrap(),
        SampleEncoding::U16
    );
    assert!(supported_sample_encoding(SampleFormat::I8).is_err());

    assert_eq!(f32_to_f32(-0.25), -0.25);
    assert_eq!(i16_to_f32(i16::MIN), -1.0);
    assert_eq!(i16_to_f32(0), 0.0);
    assert_eq!(u16_to_f32(0), -1.0);
    assert_eq!(u16_to_f32(32768), 0.0);
}

#[test]
fn normalizes_names_and_recognizes_default_requests() {
    assert_eq!(
        normalize_device_names(["Mic B".into(), "Mic A".into(), "Mic B".into()]),
        vec!["Mic A", "Mic B"]
    );
    assert!(wants_default_device(""));
    assert!(wants_default_device("DEFAULT"));
    assert!(!wants_default_device("microphone"));
}

#[test]
fn generic_device_listing_sorts_deduplicates_and_skips_stale_devices() {
    let mut source = FakeDeviceSource::new(None, &["Mic B", "stale", "Mic A", "Mic B"]);
    source.bad_name = Some("stale".into());
    assert_eq!(input_device_names(&source).unwrap(), vec!["Mic A", "Mic B"]);
    source.fail_enumeration = true;
    assert_eq!(
        input_device_names(&source).unwrap_err().to_string(),
        "enumeration failed"
    );
}

#[test]
fn generic_capture_open_uses_selected_device_config_and_bounded_queue() {
    let source = FakeAudioSource::working();
    let opened = open_capture(&source, "usb mic", 0).unwrap();
    assert_eq!(opened.device_name, "USB Mic");
    assert_eq!(opened.sample_rate, 22_050);
    assert_eq!(opened.channels, 2);
    assert!(matches!(
        opened.receiver.recv().unwrap(),
        AudioEvent::Samples { sample_rate: 22_050, samples } if samples == vec![0.5]
    ));
}

#[test]
fn generic_capture_open_contextualizes_selection_config_and_build_failures() {
    let source = FakeAudioSource::working();
    assert_eq!(
        open_capture(&source, "missing", 1)
            .err()
            .unwrap()
            .to_string(),
        "input device not found: missing"
    );

    let mut source = FakeAudioSource::working();
    source.fail_config = true;
    let error = open_capture(&source, "default", 1).err().unwrap();
    assert_eq!(error.to_string(), "query input configuration for Built-in");
    assert!(format!("{error:#}").contains("config failed"));

    let mut source = FakeAudioSource::working();
    source.fail_build = true;
    let error = open_capture(&source, "default", 1).err().unwrap();
    assert_eq!(error.to_string(), "open input device Built-in");
    assert!(format!("{error:#}").contains("build failed"));

    let mut source = FakeAudioSource::working();
    source.devices.bad_name = Some("Built-in".into());
    let opened = open_capture(&source, "default", 1).unwrap();
    assert_eq!(opened.device_name, "unknown");
}

#[test]
fn device_selection_uses_default_without_enumerating() {
    let source = FakeDeviceSource::new(Some("Built-in"), &["USB Mic"]);
    let selected = choose_device("default", &source).unwrap();
    assert_eq!(selected, "Built-in");
    assert_eq!(source.default_calls.get(), 1);
    assert_eq!(source.enumeration_calls.get(), 0);

    let selected = choose_device("", &source).unwrap();
    assert_eq!(selected, "Built-in");
    assert_eq!(source.default_calls.get(), 2);
    assert_eq!(source.enumeration_calls.get(), 0);
}

#[test]
fn device_selection_matches_names_case_insensitively() {
    let source = FakeDeviceSource::new(None, &["Built-in", "USB Mic", "Webcam"]);
    let selected = choose_device("usb MIC", &source).unwrap();
    assert_eq!(selected, "USB Mic");
    assert_eq!(source.default_calls.get(), 0);
    assert_eq!(source.enumeration_calls.get(), 1);

    // A device whose name cannot be queried is skipped, as CPAL device
    // lists can contain a stale entry while another valid device remains.
    let mut source = FakeDeviceSource::new(None, &["stale", "working"]);
    source.bad_name = Some("stale".into());
    let selected = choose_device("working", &source).unwrap();
    assert_eq!(selected, "working");
}

#[test]
fn device_selection_reports_default_enumeration_and_match_failures() {
    let source = FakeDeviceSource::new(None, &[]);
    let no_default = choose_device("DEFAULT", &source).unwrap_err();
    assert_eq!(no_default.to_string(), "no default input device");

    let mut source = FakeDeviceSource::new(None, &[]);
    source.fail_enumeration = true;
    let enumeration = choose_device("mic", &source).unwrap_err();
    assert_eq!(enumeration.to_string(), "enumeration failed");

    let source = FakeDeviceSource::new(None, &["Built-in"]);
    let missing = choose_device("Missing Mic", &source).unwrap_err();
    assert_eq!(missing.to_string(), "input device not found: Missing Mic");

    let mut source = FakeDeviceSource::new(None, &[]);
    source.fail_default = true;
    let default_error = choose_device("default", &source).unwrap_err();
    assert_eq!(default_error.to_string(), "default query failed");
}

#[test]
fn callback_delivery_is_non_blocking_and_preserves_payloads() {
    let (sender, receiver) = sync_channel(1);
    send_samples(&sender, 8_000, vec![0.25, -0.5]);
    // A full bounded queue drops the next callback value instead of blocking
    // the realtime audio thread.
    send_error(&sender, "dropped".into());
    match receiver.recv().unwrap() {
        AudioEvent::Samples {
            sample_rate,
            samples,
        } => {
            assert_eq!(sample_rate, 8_000);
            assert_eq!(samples, vec![0.25, -0.5]);
        }
        AudioEvent::Error(_) => unreachable!(),
    }
    send_error(&sender, "device disconnected".into());
    assert!(matches!(
        receiver.recv().unwrap(),
        AudioEvent::Error(message) if message == "device disconnected"
    ));

    drop(receiver);
    send_samples(&sender, 16_000, vec![]);
    send_error(&sender, "ignored".into());
}

#[test]
fn injected_capture_factory_assembles_and_starts_capture() {
    let factory = FakeCaptureFactory::working();
    let capture = Capture::start_with(&factory, "test microphone", 3).unwrap();
    assert_eq!(factory.open_calls.get(), 1);
    assert_eq!(factory.observed_capacity.get(), 3);
    assert!(factory.played.load(Ordering::SeqCst));
    assert_eq!(capture.device_name, "Fake Input");
    assert_eq!(capture.sample_rate, 22_050);
    assert_eq!(capture.channels, 2);
    match capture.receiver().recv().unwrap() {
        AudioEvent::Samples {
            sample_rate,
            samples,
        } => {
            assert_eq!(sample_rate, 22_050);
            assert_eq!(samples, vec![0.125, -0.125]);
        }
        AudioEvent::Error(message) => panic!("unexpected capture error: {message}"),
    }
}

#[test]
fn injected_capture_factory_propagates_open_and_start_errors() {
    let mut factory = FakeCaptureFactory::working();
    factory.fail_open = true;
    assert_eq!(
        Capture::start_with(&factory, "test microphone", 1)
            .err()
            .unwrap()
            .to_string(),
        "open failed"
    );
    assert!(!factory.played.load(Ordering::SeqCst));

    let mut factory = FakeCaptureFactory::working();
    factory.fail_play = true;
    let error = Capture::start_with(&factory, "test microphone", 0)
        .err()
        .unwrap();
    assert!(factory.played.load(Ordering::SeqCst));
    assert_eq!(factory.observed_capacity.get(), 0);
    assert!(error.to_string().contains("start input device Fake Input"));
    assert!(format!("{error:#}").contains("play failed"));
}

#[test]
fn audio_events_retain_samples_and_errors() {
    let samples = AudioEvent::Samples {
        sample_rate: 16_000,
        samples: vec![0.25],
    };
    match samples {
        AudioEvent::Samples {
            sample_rate,
            samples,
        } => {
            assert_eq!(sample_rate, 16_000);
            assert_eq!(samples, vec![0.25]);
        }
        AudioEvent::Error(_) => unreachable!(),
    }
    assert!(
        matches!(AudioEvent::Error("bad".into()), AudioEvent::Error(message) if message == "bad")
    );
}

#[test]
fn duplicate_legacy_names_require_an_unambiguous_selector() {
    let source = FakeDeviceSource::new(None, &["USB", "usb"]);
    assert!(
        choose_device("USB", &source)
            .unwrap_err()
            .to_string()
            .contains("ambiguous")
    );
}
