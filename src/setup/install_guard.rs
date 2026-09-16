//! Shared installation boundaries: one writer per profile, space and interruption.
use anyhow::{Context, Result, bail};
use std::cell::RefCell;
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

thread_local! {
    static CANCEL: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

// signal-hook intentionally leaves its handler installed after unregister.
// Keep one conditional default action so a later Ctrl-C/SIGTERM still terminates
// the CLI normally once the last installer has finished.
static SIGNAL_USERS: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
static SIGNAL_IDLE: std::sync::OnceLock<Arc<AtomicBool>> = std::sync::OnceLock::new();
struct SignalScope;
impl SignalScope {
    fn enter() -> Result<Self> {
        let mut users = SIGNAL_USERS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let idle = match SIGNAL_IDLE.get() {
            Some(idle) => idle,
            None => {
                let idle = Arc::new(AtomicBool::new(false));
                let first = signal_hook::flag::register_conditional_default(
                    signal_hook::consts::SIGINT,
                    idle.clone(),
                )?;
                if let Err(error) = signal_hook::flag::register_conditional_default(
                    signal_hook::consts::SIGTERM,
                    idle.clone(),
                ) {
                    signal_hook::low_level::unregister(first);
                    return Err(error.into());
                }
                SIGNAL_IDLE.get_or_init(|| idle)
            }
        };
        idle.store(false, Ordering::SeqCst);
        *users += 1;
        Ok(Self)
    }
}
impl Drop for SignalScope {
    fn drop(&mut self) {
        let mut users = SIGNAL_USERS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *users -= 1;
        if *users == 0 {
            SIGNAL_IDLE.get().unwrap().store(true, Ordering::SeqCst);
        }
    }
}

pub(super) struct InstallGuard {
    _lock: File,
    _signals: SignalScope,
    signals: Vec<signal_hook::SigId>,
    previous: Option<Arc<AtomicBool>>,
    staging: Option<PathBuf>,
}

impl InstallGuard {
    pub(super) fn acquire(data: &Path, id: &str) -> Result<Self> {
        anyhow::ensure!(
            !id.is_empty()
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                && id != "."
                && id != "..",
            "invalid model profile ID"
        );
        let locks = data.join("models/.locks");
        fs::create_dir_all(&locks)?;
        // Keep the inode after unlock: unlinking a lock file permits two owners.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(locks.join(format!("{id}.lock")))?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(std::io::Error::last_os_error())
                .context("model installation is busy; retry after the other installer finishes");
        }
        let signals = SignalScope::enter()?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let previous = CANCEL.with(|slot| slot.replace(Some(cancelled.clone())));
        let mut guard = Self {
            _lock: file,
            _signals: signals,
            signals: Vec::new(),
            previous,
            staging: None,
        };
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            guard
                .signals
                .push(signal_hook::flag::register(signal, cancelled.clone())?);
        }
        Ok(guard)
    }

    pub(super) fn staging(&mut self, path: &Path) {
        self.staging = Some(path.to_owned());
    }
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        if let Some(path) = &self.staging {
            let _ = fs::remove_dir_all(path);
        }
        for id in self.signals.drain(..) {
            signal_hook::low_level::unregister(id);
        }
        CANCEL.with(|slot| slot.replace(self.previous.take()));
        // Explicit unlock also releases any transient descriptor inherited by a concurrent fork.
        unsafe {
            libc::flock(self._lock.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

pub(super) fn check_cancelled() -> Result<()> {
    if CANCEL.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }) {
        bail!("model installation cancelled; active model was not replaced");
    }
    Ok(())
}

pub(super) fn copy(source: &Path, target: &Path) -> Result<()> {
    let mut input = File::open(source)?;
    let mut output = File::create(target)?;
    let mut buffer = [0; 128 * 1024];
    loop {
        check_cancelled()?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        output.write_all(&buffer[..count])?;
    }
    output.sync_all()?;
    Ok(())
}

/// Existing installations and caches already consume the reported free space.
/// Budget the additional staging copy, unverified download objects and metadata.
pub(super) fn preflight(
    models: &Path,
    downloads: &Path,
    staging_bytes: u64,
    download_bytes: u64,
) -> Result<()> {
    fs::create_dir_all(downloads)?;
    let shared = fs::metadata(models)?.dev() == fs::metadata(downloads)?.dev();
    let required = if shared {
        staging_bytes
            .checked_add(download_bytes)
            .context("installation size overflow")?
    } else {
        staging_bytes
    };
    require_space(models, required, available_bytes(models)?)?;
    if !shared && download_bytes > 0 {
        require_space(downloads, download_bytes, available_bytes(downloads)?)?;
    }
    check_cancelled()
}

fn require_space(path: &Path, bytes: u64, available: u64) -> Result<()> {
    const RESERVE: u64 = 16 * 1024 * 1024;
    let required = bytes
        .checked_add(RESERVE)
        .context("installation size overflow")?;
    anyhow::ensure!(
        available >= required,
        "insufficient disk space at {}: need {required} additional bytes (including 16 MiB metadata reserve), have {available}; active model retained",
        path.display()
    );
    Ok(())
}

#[allow(clippy::unnecessary_cast)] // statvfs field widths differ across targets.
fn available_bytes(path: &Path) -> Result<u64> {
    let path = CString::new(path.as_os_str().as_bytes())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error()).context("check installation disk space");
    }
    let stats = unsafe { stats.assume_init() };
    (stats.f_bavail as u64)
        .checked_mul(stats.f_frsize as u64)
        .context("filesystem free-space overflow")
}

#[cfg(test)]
pub(super) fn cancel_current() {
    CANCEL.with(|slot| {
        slot.borrow()
            .as_ref()
            .unwrap()
            .store(true, Ordering::Relaxed)
    });
}

#[cfg(test)]
#[path = "../../tests/unit/install_guard.rs"]
mod tests;
