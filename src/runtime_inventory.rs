//! Provider-neutral native runtime proof records.
//!
//! Discovery and validation live with each provider because a loadable shared
//! library alone cannot prove that the selected device can create a session.

use std::path::Path;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::backend::{BackendConfig, Runtime};
use crate::config::Config;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Evidence {
    pub versions: Vec<String>,
    pub provider_registration: bool,
    pub available_devices: Vec<String>,
    pub selected_device: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Probe {
    pub loadable: bool,
    pub device_accessible: bool,
    pub ready: bool,
    pub evidence: Evidence,
    pub errors: Vec<String>,
}

pub const fn name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
        Runtime::Vulkan => "vulkan",
        Runtime::Hip => "hip",
    }
}

pub fn apply_with(
    config: &Config,
    path: &Path,
    apply: bool,
    probe: impl FnOnce(&BackendConfig, &Path) -> Probe,
) -> Result<Probe> {
    let result = probe(&config.backend, path);
    if !result.ready {
        bail!(
            "runtime candidate rejected; config unchanged: {}",
            result.errors.join("; ")
        );
    }
    if apply {
        config.save(path)?;
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../tests/unit/runtime_inventory.rs"]
mod tests;
