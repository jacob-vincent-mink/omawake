//! Audio setup uses the same inventory contract as external settings clients.
use super::wizard::{MenuItem, select, select_horizontal};
use anyhow::{Context, Result};
use serde_json::Value;

pub fn choose(current: &str, inventory: impl Fn(&str) -> Value) -> Result<Option<String>> {
    choose_with(current, inventory, select)
}

pub fn choose_horizontal(
    current: &str,
    inventory: impl Fn(&str) -> Value,
) -> Result<Option<String>> {
    choose_with(current, inventory, select_horizontal)
}

fn choose_with(
    current: &str,
    inventory: impl Fn(&str) -> Value,
    mut select: impl FnMut(&str, &str, &[MenuItem], usize) -> Result<Option<usize>>,
) -> Result<Option<String>> {
    loop {
        let report = inventory(current);
        let devices = report["devices"]
            .as_array()
            .context("device inventory has no devices")?;
        let mut items = Vec::new();
        for d in devices {
            let label = d["label"].as_str().unwrap_or("Unknown device");
            let selector = d["selector"].as_str().context("device has no selector")?;
            let detail = format!(
                "{}{}",
                selector,
                if selector == current { " · Saved" } else { "" }
            );
            items.push(MenuItem::available(label, detail));
        }
        items.push(MenuItem::available(
            "Refresh devices",
            "Scan again after connecting a device.",
        ));
        let preferred = devices
            .iter()
            .position(|d| d["selector"] == current)
            .unwrap_or(0);
        let help = report["error"].as_str().unwrap_or("Choose a device for this application. System defaults are unchanged. Disconnected saved devices can be kept.");
        let Some(index) = select("Audio device", help, &items, preferred)? else {
            return Ok(None);
        };
        if index == devices.len() {
            continue;
        }
        return Ok(Some(
            devices.get(index).context("invalid device choice")?["selector"]
                .as_str()
                .context("device has no selector")?
                .into(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refresh_cancel_and_offline_selection() {
        let inventory = |_: &str| serde_json::json!({"devices":[{"selector":"default","label":"System default"},{"selector":"pipewire:offline","label":"Offline","available":false}]});
        let mut choices = [Some(2), Some(1)].into_iter();
        assert_eq!(
            choose_with("pipewire:offline", inventory, |_, _, _, preferred| {
                assert_eq!(preferred, 1);
                Ok(choices.next().unwrap())
            })
            .unwrap()
            .as_deref(),
            Some("pipewire:offline")
        );
        assert_eq!(
            choose_with("default", inventory, |_, _, _, _| Ok(None)).unwrap(),
            None
        );
    }
}
