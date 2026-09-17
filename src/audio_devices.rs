//! PipeWire discovery and persistent routing identifiers. Kept identical in
//! Omawake and Omaspeak so their independent binaries share a device contract.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const PINNED_PROPERTIES: &str =
    "{ node.dont-fallback = true node.dont-reconnect = true node.dont-move = true }";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Device {
    pub selector: String,
    pub label: String,
    pub direction: String,
    pub backend: String,
    pub is_default: bool,
    pub available: bool,
}

impl Device {
    pub fn system_default(direction: &str) -> Self {
        Self {
            selector: "default".into(),
            label: "System default".into(),
            direction: direction.into(),
            backend: "default".into(),
            is_default: true,
            available: true,
        }
    }
}

pub fn is_default(selector: &str) -> bool {
    selector.is_empty() || selector.eq_ignore_ascii_case("default")
}

/// Legacy CPAL names are accepted only for input. PipeWire selectors always
/// use node.name, never a transient object id or serial.
pub fn validate(selector: &str, direction: &str) -> Result<()> {
    if is_default(selector) {
        return Ok(());
    }
    if selector.chars().any(char::is_control) {
        bail!("audio device contains control characters");
    }
    if let Some(name) = selector.strip_prefix("pipewire:") {
        if name.is_empty() || name.trim() != name || name.parse::<u64>().is_ok() {
            bail!("use pipewire:<node.name>, not an empty name or numeric object ID");
        }
        return Ok(());
    }
    if direction == "input" && !selector.trim().is_empty() {
        return Ok(());
    }
    bail!("output device must be default or pipewire:<node.name>")
}

pub fn discover(direction: &str) -> Result<Vec<Device>> {
    let mut child = Command::new("pw-dump")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("start pw-dump; install PipeWire tools for device discovery")?;
    let stdout = child.stdout.take().context("pw-dump has no stdout")?;
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(8 * 1024 * 1024)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            Ok(None) => break Err(anyhow::anyhow!("PipeWire discovery timed out")),
            Err(error) => break Err(error.into()),
        }
    };
    if outcome.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let bytes = reader
        .join()
        .map_err(|_| anyhow::anyhow!("PipeWire reader failed"))??;
    if !outcome?.success() {
        bail!("cannot connect to PipeWire; check the user audio service");
    }
    parse_inventory(&bytes, direction)
}

pub fn parse_inventory(bytes: &[u8], direction: &str) -> Result<Vec<Device>> {
    let objects: Vec<Value> = serde_json::from_slice(bytes).context("parse PipeWire inventory")?;
    let class = if direction == "input" {
        "Audio/Source"
    } else {
        "Audio/Sink"
    };
    let default_key = if direction == "input" {
        "default.audio.source"
    } else {
        "default.audio.sink"
    };
    let default_name = objects
        .iter()
        .filter_map(|o| o.get("metadata").and_then(Value::as_array))
        .flatten()
        .filter(|m| m["key"] == default_key)
        .find_map(|m| {
            let value = &m["value"];
            let parsed;
            let value = if let Some(s) = value.as_str() {
                parsed = serde_json::from_str::<Value>(s).ok()?;
                &parsed
            } else {
                value
            };
            value["name"].as_str().map(str::to_owned)
        });
    let mut devices = Vec::new();
    for object in &objects {
        if object["type"] != "PipeWire:Interface:Node" {
            continue;
        }
        let props = &object["info"]["props"];
        if props["media.class"] != class {
            continue;
        }
        let Some(name) = props["node.name"].as_str() else {
            continue;
        };
        let selector = format!("pipewire:{name}");
        validate(&selector, direction)?;
        devices.push(Device {
            selector,
            label: props["node.description"]
                .as_str()
                .or(props["node.nick"].as_str())
                .unwrap_or(name)
                .into(),
            direction: direction.into(),
            backend: "pipewire".into(),
            is_default: default_name.as_deref() == Some(name),
            available: true,
        });
    }
    devices.sort_by(|a, b| a.label.cmp(&b.label).then(a.selector.cmp(&b.selector)));
    Ok(devices)
}

pub fn resolve(selector: &str, direction: &str) -> Result<Device> {
    validate(selector, direction)?;
    resolve_from(selector, discover(direction)?)
}

fn resolve_from(selector: &str, devices: Vec<Device>) -> Result<Device> {
    let mut matches = devices.into_iter().filter(|d| d.selector == selector);
    let device = matches.next().with_context(|| {
        format!("audio device unavailable: {selector}; refresh audio-devices or select default")
    })?;
    if matches.next().is_some() {
        bail!("ambiguous audio device: {selector}");
    }
    Ok(device)
}

/// Discovery failure is data in settings, rather than a broken settings page.
pub fn inventory(direction: &str, selected: &str) -> Value {
    inventory_from(direction, selected, discover(direction))
}

pub fn inventory_from(direction: &str, selected: &str, found: Result<Vec<Device>>) -> Value {
    let (mut devices, error) = match found {
        Ok(devices) => (devices, None),
        Err(error) => (Vec::new(), Some(format!("{error:#}"))),
    };
    devices.insert(0, Device::system_default(direction));
    if !is_default(selected) && !devices.iter().any(|d| d.selector == selected) {
        devices.push(Device {
            selector: selected.into(),
            label: format!("{selected} (unavailable)"),
            direction: direction.into(),
            backend: if selected.starts_with("pipewire:") {
                "pipewire"
            } else {
                "cpal"
            }
            .into(),
            is_default: false,
            available: false,
        });
    }
    json!({"schema_version":1, "direction":direction, "selected":selected, "devices":devices, "error":error})
}

pub fn status(requested: &str, effective: Option<&str>, error: Option<&str>) -> Value {
    json!({"requested":requested, "effective":effective, "error":error})
}

/// Enum choices retain availability and labels for schema-driven settings.
pub fn schema_choices(inventory: &Value) -> Vec<Value> {
    inventory["devices"].as_array().into_iter().flatten().map(|d| json!({
        "value":d["selector"], "label":d["label"], "available":d["available"], "backend":d["backend"]
    })).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inventory_filters_direction_and_uses_names() {
        let dump = br#"[{"type":"PipeWire:Interface:Node","id":17,"info":{"props":{"media.class":"Audio/Source","node.name":"mic-a","node.description":"USB"}}},{"type":"PipeWire:Interface:Node","info":{"props":{"media.class":"Audio/Sink","node.name":"sink-a"}}},{"metadata":[{"key":"default.audio.source","value":{"name":"mic-a"}}]}]"#;
        let inputs = parse_inventory(dump, "input").unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].selector, "pipewire:mic-a");
        assert!(inputs[0].is_default);
        assert_eq!(inputs[0].label, "USB");
        assert_eq!(
            parse_inventory(dump, "output").unwrap()[0].selector,
            "pipewire:sink-a"
        );
        assert!(parse_inventory(b"bad", "input").is_err());
    }
    #[test]
    fn invalid_offline_and_ambiguous_devices() {
        for s in ["pipewire:", "pipewire:42", "pipewire: x", "x\n"] {
            assert!(validate(s, "input").is_err());
        }
        assert!(validate("legacy name", "input").is_ok());
        assert!(validate("legacy name", "output").is_err());
        assert!(validate("pipewire:disconnected", "output").is_ok());
        let d = Device::system_default("input");
        assert!(resolve_from("default", vec![d.clone(), d]).is_err());
        assert!(resolve_from("missing", vec![]).is_err());
        let report = inventory_from(
            "output",
            "pipewire:offline",
            Err(anyhow::anyhow!("offline")),
        );
        assert_eq!(report["devices"][0]["selector"], "default");
        assert_eq!(report["devices"][1]["available"], false);
        assert_eq!(report["error"], "offline");
    }
}
