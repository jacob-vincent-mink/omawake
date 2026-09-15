use super::*;

#[test]
fn generated_unit_owns_only_its_exact_config_argument() {
    let config = Path::new("/home/user/.config/omawake/config.toml");
    let unit = generate(Path::new("/usr/bin/omawake"), config);
    assert!(unit_targets_config(&unit, config));
    assert!(!unit_targets_config(
        &unit,
        Path::new("/home/user/.config/omawake/other.toml")
    ));
    assert!(!unit_targets_config(
        "ExecStart=/usr/bin/omawake daemon\n# --config \"/home/user/.config/omawake/config.toml\"",
        config
    ));
    let alternate = Path::new("/home/user/alternate.toml");
    let alternate_unit = generate(Path::new("/usr/bin/omawake"), alternate);
    assert!(targets_config_with_unit(config, config, None));
    assert!(!targets_config_with_unit(alternate, config, None));
    assert!(!targets_config_with_unit(
        config,
        config,
        Some(&alternate_unit)
    ));
    assert!(targets_config_with_unit(
        alternate,
        config,
        Some(&alternate_unit)
    ));
}

#[test]
fn unit_uses_absolute_binary_and_config() {
    let unit = generate(Path::new("/opt/oma speak"), Path::new("/tmp/config.toml"));
    assert!(unit.contains("ExecStart=\"/opt/oma speak\" --config \"/tmp/config.toml\" daemon"));
    assert!(unit.contains("Restart=on-failure"));
    assert!(unit.contains("XDG_RUNTIME_DIR=%t"));
}

#[test]
fn unit_escapes_systemd_specifiers_quotes_backslashes_and_controls() {
    let unit = generate(
        Path::new("/opt/oma%wake/quote\"back\\slash"),
        Path::new("/tmp/config\nnext.toml"),
    );
    assert!(unit.contains(r#"ExecStart="/opt/oma%%wake/quote\"back\\slash""#));
    assert!(unit.contains(r#"--config "/tmp/config\nnext.toml" daemon"#));
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

#[test]
fn atomic_unit_write_creates_missing_parent_directories() {
    let root = std::env::temp_dir().join(format!("omawake-systemd-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let unit = root.join("nested/omawake.service");
    write_atomic(&unit, b"[Unit]\n").unwrap();
    assert_eq!(fs::read(unit).unwrap(), b"[Unit]\n");
}
