use super::*;
fn root() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "install-guard-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}
#[test]
fn profile_lock_excludes_other_writers_and_releases_without_unlinking() {
    let root = root();
    let guard = InstallGuard::acquire(&root, "profile").unwrap();
    assert!(
        InstallGuard::acquire(&root, "profile")
            .err()
            .unwrap()
            .to_string()
            .contains("busy")
    );
    let other = InstallGuard::acquire(&root, "other").unwrap();
    drop(other);
    drop(guard);
    assert!(root.join("models/.locks/profile.lock").exists());
    drop(InstallGuard::acquire(&root, "profile").unwrap());
    for id in ["", "../bad", "..", ".", "a/b"] {
        assert!(InstallGuard::acquire(&root, id).is_err());
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn cancellation_stops_copy_and_cleans_only_owned_staging() {
    let root = root();
    let target = root.join("active");
    fs::write(&target, b"old").unwrap();
    let mut guard = InstallGuard::acquire(&root, "profile").unwrap();
    let staging = root.join("staging");
    fs::create_dir(&staging).unwrap();
    guard.staging(&staging);
    copy(&target, &staging.join("copy")).unwrap();
    cancel_current();
    assert!(
        copy(&target, &staging.join("cancelled"))
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    assert!(check_cancelled().is_err());
    drop(guard);
    assert!(check_cancelled().is_ok());
    assert!(!staging.exists());
    assert_eq!(fs::read(&target).unwrap(), b"old");
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn disk_budget_rejects_exhaustion_and_overflow_and_checks_real_filesystem() {
    let root = root();
    assert!(
        require_space(&root, 100, 100)
            .unwrap_err()
            .to_string()
            .contains("insufficient disk")
    );
    require_space(&root, 100, 100 + 16 * 1024 * 1024).unwrap();
    assert!(require_space(&root, u64::MAX, u64::MAX).is_err());
    assert!(available_bytes(&root.join("absent")).is_err());
    preflight(&root, &root.join("downloads"), 0, 0).unwrap();
    assert!(preflight(&root, &root.join("downloads"), u64::MAX, 1).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn interrupt_cancels_an_install_and_default_signals_work_afterwards() {
    use std::os::unix::process::ExitStatusExt;
    if let Ok(mode) = std::env::var("OMA_INSTALL_SIGNAL_TEST") {
        let guard = InstallGuard::acquire(&root(), "signals").unwrap();
        if mode == "active" {
            unsafe {
                libc::raise(libc::SIGINT);
            }
            assert!(check_cancelled().is_err());
            drop(guard);
            return;
        }
        drop(guard);
        unsafe {
            libc::raise(libc::SIGTERM);
        }
        panic!("SIGTERM was swallowed after installation");
    }
    for mode in ["active", "finished"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "setup::install_guard::tests::interrupt_cancels_an_install_and_default_signals_work_afterwards"])
            .env("OMA_INSTALL_SIGNAL_TEST",mode).output().unwrap();
        if mode == "active" {
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        } else {
            assert_eq!(result.status.signal(), Some(libc::SIGTERM));
        }
    }
}
