//! Provider-neutral native runtime proof records.
//!
//! Discovery and validation live with each provider because a loadable shared
//! library alone cannot prove that the selected device can create a session.

use std::path::Path;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::backend::Runtime;
use crate::config::Config;
use crate::paths::AppPaths;

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

pub fn probe(config: &Config, config_path: &Path) -> Probe {
    let backend = &config.backend;
    let mut paths = AppPaths::discover();
    paths.config_file = config_path.to_owned();
    if backend.kind == "openvino-genai" && backend.runtime == Runtime::Openvino {
        return match crate::engine::openvino_genai::probe_runtime(config, &paths) {
            Ok(evidence) => Probe {
                loadable: true,
                device_accessible: true,
                ready: true,
                evidence: Evidence {
                    versions: vec![format!(
                        "OpenVINO {} · GenAI C {} · {}",
                        evidence.runtime_build, evidence.genai_library, evidence.full_device_name
                    )],
                    provider_registration: true,
                    available_devices: vec![evidence.available_device.to_ascii_lowercase()],
                    selected_device: Some(evidence.requested_device.to_ascii_lowercase()),
                },
                errors: Vec::new(),
            },
            Err(error) => Probe {
                errors: vec![format!("{error:#}")],
                ..Default::default()
            },
        };
    }
    match crate::engine::audiocpp::probe_provider(config, &paths) {
        Ok((library, version)) => Probe {
            loadable: true,
            device_accessible: true,
            ready: true,
            evidence: Evidence {
                versions: vec![format!("{version} · {}", library.display())],
                provider_registration: true,
                available_devices: vec![
                    backend
                        .canonical_device()
                        .unwrap_or_else(|_| backend.device.clone()),
                ],
                selected_device: Some(
                    backend
                        .canonical_device()
                        .unwrap_or_else(|_| backend.device.clone()),
                ),
            },
            errors: Vec::new(),
        },
        Err(error) => Probe {
            errors: vec![format!("{error:#}")],
            ..Default::default()
        },
    }
}

pub fn apply_with(
    config: &Config,
    path: &Path,
    apply: bool,
    probe: impl FnOnce(&Config, &Path) -> Probe,
) -> Result<Probe> {
    let result = probe(config, path);
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
