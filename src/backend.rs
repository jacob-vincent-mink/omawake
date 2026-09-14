use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    #[default]
    Default,
    Openvino,
    Cuda,
}

impl Runtime {
    pub const fn capability(self) -> &'static str {
        match self {
            Self::Default => "cpu",
            Self::Openvino => "openvino",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fallback {
    #[default]
    Error,
    Cpu,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    pub kind: String,
    pub runtime: Runtime,
    pub device: String,
    pub threads: u16,
    pub fallback: Fallback,
    pub device_id: u32,
    pub provider_config: String,
    pub options: BTreeMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            kind: "sherpa-onnx".into(),
            runtime: Runtime::Default,
            device: "auto".into(),
            threads: 2,
            fallback: Fallback::Error,
            device_id: 0,
            provider_config: String::new(),
            options: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BackendError {
    #[error("backend threads must be between 1 and 64")]
    InvalidThreads,
    #[error("backend device {device} is invalid for runtime {runtime:?}")]
    InvalidDevice { runtime: Runtime, device: String },
    #[error("backend device_id is only valid with runtime cuda")]
    InvalidDeviceId,
    #[error("backend runtime {runtime:?} is unavailable in this build (requires {capability})")]
    CapabilityUnavailable {
        runtime: Runtime,
        capability: &'static str,
    },
}

impl BackendConfig {
    pub fn canonical_device(&self) -> Result<String, BackendError> {
        canonical_device(self.runtime, &self.device)
    }

    pub fn validate_shape(&self) -> Result<(), BackendError> {
        if !(1..=64).contains(&self.threads) {
            return Err(BackendError::InvalidThreads);
        }
        if self.runtime != Runtime::Cuda && self.device_id != 0 {
            return Err(BackendError::InvalidDeviceId);
        }
        self.canonical_device()?;
        Ok(())
    }

    pub fn validate_capabilities(&self, compiled: &[&str]) -> Result<(), BackendError> {
        self.validate_shape()?;
        let required = self.runtime.capability();
        if compiled.contains(&required) {
            Ok(())
        } else {
            Err(BackendError::CapabilityUnavailable {
                runtime: self.runtime,
                capability: required,
            })
        }
    }
}

pub const fn compiled_capabilities() -> &'static [&'static str] {
    #[cfg(all(feature = "openvino", feature = "cuda"))]
    return &["cpu", "openvino", "cuda"];
    #[cfg(all(feature = "openvino", not(feature = "cuda")))]
    return &["cpu", "openvino"];
    #[cfg(all(not(feature = "openvino"), feature = "cuda"))]
    return &["cpu", "cuda"];
    #[cfg(not(any(feature = "openvino", feature = "cuda")))]
    &["cpu"]
}

pub fn canonical_device(runtime: Runtime, raw: &str) -> Result<String, BackendError> {
    let trimmed = raw.trim();
    let invalid = || BackendError::InvalidDevice {
        runtime,
        device: raw.to_owned(),
    };

    match runtime {
        Runtime::Default => match trimmed.to_ascii_lowercase().as_str() {
            "auto" => Ok("auto".into()),
            "cpu" => Ok("cpu".into()),
            _ => Err(invalid()),
        },
        Runtime::Cuda => match trimmed.to_ascii_lowercase().as_str() {
            "auto" => Ok("auto".into()),
            "gpu" => Ok("gpu".into()),
            _ => Err(invalid()),
        },
        Runtime::Openvino => canonical_openvino_device(trimmed).ok_or_else(invalid),
    }
}

fn canonical_openvino_device(raw: &str) -> Option<String> {
    let upper = raw.to_ascii_uppercase();
    if matches!(upper.as_str(), "AUTO" | "NPU" | "GPU" | "CPU") {
        return Some(upper.to_ascii_lowercase());
    }

    let (mode, entries) = upper.split_once(':')?;
    if !matches!(mode, "AUTO" | "HETERO" | "MULTI") {
        return None;
    }
    let devices: Vec<_> = entries.split(',').map(str::trim).collect();
    let minimum = if mode == "AUTO" { 1 } else { 2 };
    if devices.len() < minimum
        || devices
            .iter()
            .any(|item| !matches!(*item, "CPU" | "GPU" | "NPU"))
    {
        return None;
    }
    Some(format!("{mode}:{}", devices.join(",")))
}

#[cfg(test)]
#[path = "../tests/unit/backend.rs"]
mod tests;
