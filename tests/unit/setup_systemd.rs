use super::*;

#[test]
fn unit_uses_absolute_binary_and_config() {
    let unit = generate(Path::new("/opt/oma speak"), Path::new("/tmp/config.toml"));
    assert!(unit.contains("ExecStart=\"/opt/oma speak\" --config \"/tmp/config.toml\" daemon"));
    assert!(unit.contains("Restart=on-failure"));
    assert!(unit.contains("XDG_RUNTIME_DIR=%t"));
}
