//! Explicit PipeWire capture; CPAL remains the default and legacy-name backend.
use super::*;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

struct PipeWireStream {
    child: Arc<Mutex<Child>>,
    stopped: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}
impl CaptureStream for PipeWireStream {
    fn play(&self) -> Result<()> {
        Ok(())
    }
}
impl Drop for PipeWireStream {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

pub(super) fn start(selector: &str, capacity: usize) -> Result<Capture> {
    let device = crate::audio_devices::resolve(selector, "input")?;
    let target = selector
        .strip_prefix("pipewire:")
        .context("expected PipeWire selector")?;
    let mut command = Command::new("pw-record");
    command
        .args([
            "--target",
            target,
            "--properties",
            crate::audio_devices::PINNED_PROPERTIES,
            "--raw",
            "--format",
            "f32",
            "--rate",
            "16000",
            "--channels",
            "1",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // The recorder must not survive a killed daemon and leave a microphone open.
    unsafe {
        command.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() == 1 {
                return Err(std::io::Error::other("capture parent exited"));
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .context("start pw-record; install PipeWire tools")?;
    let mut stdout = child.stdout.take().context("pw-record has no stdout")?;
    let child = Arc::new(Mutex::new(child));
    let stopped = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let (sender, receiver) = sync_channel(capacity.max(1));
    let read_sender = sender.clone();
    let read_progress = progress.clone();
    let read_stop = stopped.clone();
    let reader = thread::spawn(move || {
        let mut block = [0_u8; 1024];
        while !read_stop.load(Ordering::Relaxed) {
            if let Err(error) = stdout.read_exact(&mut block) {
                let _ = read_sender.try_send(AudioEvent::Error(format!(
                    "PipeWire capture ended: {error}"
                )));
                break;
            }
            read_progress.store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
            let samples = block
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect();
            send_samples(&read_sender, 16_000, samples);
        }
    });
    let watch_child = child.clone();
    let watch_stop = stopped.clone();
    let watcher = thread::spawn(move || {
        while !watch_stop.load(Ordering::Relaxed) {
            let finished = watch_child
                .lock()
                .ok()
                .is_none_or(|mut c| c.try_wait().ok().flatten().is_some());
            if finished
                || start.elapsed().as_millis() as u64 > progress.load(Ordering::Relaxed) + 3000
            {
                let _ = sender.try_send(AudioEvent::Error(
                    "PipeWire microphone disconnected or stopped delivering audio".into(),
                ));
                if let Ok(mut c) = watch_child.lock() {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
    });
    Ok(Capture {
        _stream: Box::new(PipeWireStream {
            child,
            stopped,
            workers: vec![reader, watcher],
        }),
        receiver,
        device_name: device.selector,
        sample_rate: 16_000,
        channels: 1,
    })
}
