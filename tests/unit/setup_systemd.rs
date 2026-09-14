use super::*;

#[test]
fn unit_uses_absolute_binary_and_config() {
    let unit = generate(Path::new("/opt/oma speak"), Path::new("/tmp/config.toml"));
    assert!(unit.contains("ExecStart=\"/opt/oma speak\" --config \"/tmp/config.toml\" daemon"));
    assert!(unit.contains("Restart=on-failure"));
    assert!(unit.contains("XDG_RUNTIME_DIR=%t"));
}

#[test]
fn unit_preserves_and_escapes_the_native_library_path() {
    let unit = generate_with_library_path(
        Path::new("/opt/omawake"),
        Path::new("/tmp/config.toml"),
        Some(OsStr::new("/opt/oma lib:/opt/%t/openvino\\runtime\"quoted")),
    );
    assert!(unit.contains(
        "Environment=\"LD_LIBRARY_PATH=/opt/oma lib:/opt/%%t/openvino\\\\runtime\\\"quoted\""
    ));
}

#[test]
fn setup_reload_only_restarts_a_service_that_was_active() {
    use std::cell::Cell;

    let restarted = Cell::new(false);
    assert!(
        !reload_if_was_active_with(
            false,
            || {
                restarted.set(true);
                Ok(())
            },
            || true
        )
        .unwrap()
    );
    assert!(!restarted.get());

    assert!(reload_if_was_active_with(true, || Ok(()), || true).unwrap());
    assert!(reload_if_was_active_with(true, || Ok(()), || false).is_err());
}
