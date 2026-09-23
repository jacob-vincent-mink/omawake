use super::*;

#[test]
fn generated_unit_owns_only_its_exact_config_argument() {
    let config = Path::new("/home/user/.config/omawake/config.toml");
    let unit = generate(Path::new("/usr/bin/omawake"), config);
    assert!(has_managed_template(&unit, Some(config)));
    assert!(!has_managed_template(
        &unit,
        Some(Path::new("/home/user/.config/omawake/other.toml"))
    ));
    assert!(!has_managed_template(
        "ExecStart=/usr/bin/omawake daemon\n# --config \"/home/user/.config/omawake/config.toml\"",
        Some(config)
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
    let handwritten = format!(
        "[Service]\nExecStart=/usr/bin/omawake --config {} daemon\n",
        quote(config)
    );
    assert!(!targets_config_with_unit(
        config,
        config,
        Some(&handwritten)
    ));
}

#[test]
fn generated_unit_recognizes_equivalent_existing_config_spellings() {
    let paths = service_test_paths();
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    fs::write(&paths.config_file, b"config").unwrap();
    let with_dot = paths.config_file.parent().unwrap().join("./config.toml");
    let unit = generate(Path::new("/usr/bin/omawake"), &with_dot);
    assert!(has_managed_template(&unit, Some(&paths.config_file)));
    let other = paths.config_file.with_file_name("other.toml");
    fs::write(&other, b"other").unwrap();
    assert!(!has_managed_template(&unit, Some(&other)));
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

fn service_test_paths() -> AppPaths {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "omawake-service-control-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    AppPaths {
        config_file: root.join("config/omawake/config.toml"),
        config_home: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("runtime"),
    }
}

#[test]
fn service_controls_require_an_installed_local_unit() {
    use std::cell::Cell;

    let paths = service_test_paths();
    let called = Cell::new(false);
    let error = set_active_with(
        &paths,
        "start",
        true,
        |_| {
            called.set(true);
            Ok(())
        },
        || Ok(true),
    )
    .unwrap_err();
    assert!(error.to_string().contains("not installed"));
    assert!(!called.get());

    let path = service_path(&paths);
    fs::create_dir_all(&path).unwrap();
    let error = set_active_with(&paths, "stop", false, |_| Ok(()), || Ok(false)).unwrap_err();
    assert!(error.to_string().contains("not a file"));
    fs::remove_dir_all(
        paths
            .config_file
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    )
    .unwrap();
}

#[test]
fn service_controls_issue_the_action_and_check_the_resulting_state() {
    let paths = service_test_paths();
    let path = service_path(&paths);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        generate(Path::new("/usr/bin/omawake"), &paths.config_file),
    )
    .unwrap();

    for (action, expected) in [("start", true), ("stop", false)] {
        set_active_with(
            &paths,
            action,
            expected,
            |actual| {
                assert_eq!(actual, action);
                Ok(())
            },
            || Ok(expected),
        )
        .unwrap();

        let error =
            set_active_with(&paths, action, expected, |_| Ok(()), || Ok(!expected)).unwrap_err();
        assert!(error.to_string().contains(&format!("did not {action}")));
    }

    let error = set_active_with(
        &paths,
        "start",
        true,
        |_| bail!("systemctl failed"),
        || panic!("active state must not be queried after command failure"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("systemctl failed"));
    let error =
        set_active_with(&paths, "stop", false, |_| Ok(()), || bail!("probe failed")).unwrap_err();
    assert!(error.to_string().contains("probe failed"));
    fs::remove_dir_all(
        paths
            .config_file
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    )
    .unwrap();
}

#[test]
fn service_ownership_requires_the_generated_unit_shape_and_exact_config() {
    let paths = service_test_paths();
    let path = service_path(&paths);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let generated = generate(Path::new("/usr/bin/omawake"), &paths.config_file);
    fs::write(&path, &generated).unwrap();
    assert!(is_managed(&paths, &paths.config_file));
    assert!(!is_managed(&paths, Path::new("/other/config.toml")));

    let escaped_config = Path::new("/tmp/percent%/quote\"back\\slash.toml");
    fs::write(
        &path,
        generate(Path::new("/opt/percent%/omawake"), escaped_config),
    )
    .unwrap();
    assert!(is_managed(&paths, escaped_config));

    // The original CLI generated this exact unit without a managed marker.
    fs::write(&path, &generated[MANAGED_MARKER.len()..]).unwrap();
    assert!(is_managed(&paths, &paths.config_file));
    let renamed_binary = generate(Path::new("/opt/omawake-0.0.3"), &paths.config_file);
    fs::write(&path, &renamed_binary[MANAGED_MARKER.len()..]).unwrap();
    assert!(is_managed(&paths, &paths.config_file));

    let handwritten = generated.replace("Restart=on-failure", "Restart=always");
    fs::write(&path, handwritten).unwrap();
    assert!(!is_managed(&paths, &paths.config_file));
    let error = install(&paths, &paths.config_file, false).unwrap_err();
    assert!(error.to_string().contains("unrecognized"));
    assert!(
        fs::read_to_string(&path)
            .unwrap()
            .contains("Restart=always")
    );
    let called = std::cell::Cell::new(false);
    let error = set_active_with(
        &paths,
        "stop",
        false,
        |_| {
            called.set(true);
            Ok(())
        },
        || Ok(false),
    )
    .unwrap_err();
    assert!(error.to_string().contains("unrecognized"));
    assert!(!called.get());

    fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink("/usr/bin/omawake", &path).unwrap();
    assert!(!is_managed(&paths, &paths.config_file));
    let error = install(&paths, &paths.config_file, false).unwrap_err();
    assert!(error.to_string().contains("not a regular file"));
    fs::remove_dir_all(
        paths
            .config_file
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    )
    .unwrap();
}

#[test]
fn uninstall_preserves_unrecognized_or_running_units_and_command_failures() {
    let paths = service_test_paths();
    let path = service_path(&paths);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let handwritten = "[Unit]\nDescription=My personal service\n";
    fs::write(&path, handwritten).unwrap();
    let error = uninstall_with(
        &paths,
        || panic!("must not disable unrecognized unit"),
        || Ok(false),
        || Ok(()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("unrecognized"));
    assert_eq!(fs::read_to_string(&path).unwrap(), handwritten);

    let generated = generate(Path::new("/usr/bin/omawake"), &paths.config_file);
    fs::write(&path, &generated).unwrap();
    let error =
        uninstall_with(&paths, || bail!("disable failed"), || Ok(false), || Ok(())).unwrap_err();
    assert!(error.to_string().contains("disable failed"));
    assert_eq!(fs::read_to_string(&path).unwrap(), generated);

    let error = uninstall_with(&paths, || Ok(()), || Ok(true), || Ok(())).unwrap_err();
    assert!(error.to_string().contains("remains active"));
    assert_eq!(fs::read_to_string(&path).unwrap(), generated);

    let error = uninstall_with(&paths, || Ok(()), || bail!("probe failed"), || Ok(())).unwrap_err();
    assert!(error.to_string().contains("probe failed"));
    assert_eq!(fs::read_to_string(&path).unwrap(), generated);

    let reloaded = std::cell::Cell::new(false);
    uninstall_with(
        &paths,
        || Ok(()),
        || Ok(false),
        || {
            reloaded.set(true);
            Ok(())
        },
    )
    .unwrap();
    assert!(reloaded.get());
    assert!(!path.exists());
    fs::remove_dir_all(
        paths
            .config_file
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    )
    .unwrap();
}
