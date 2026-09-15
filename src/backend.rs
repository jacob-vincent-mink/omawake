use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    #[default]
    Default,
    Openvino,
    Cuda,
    Vulkan,
    Hip,
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
    pub library: PathBuf,
    pub library_dirs: Vec<PathBuf>,
    pub options: BTreeMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            kind: "audiocpp".into(),
            runtime: Runtime::Default,
            device: "cpu".into(),
            threads: 2,
            fallback: Fallback::Error,
            device_id: 0,
            library: PathBuf::new(),
            library_dirs: Vec::new(),
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
    #[error("backend device_id is only valid with an accelerated audio.cpp runtime")]
    InvalidDeviceId,
}

impl BackendConfig {
    pub fn canonical_device(&self) -> Result<String, BackendError> {
        canonical_device(self.runtime, &self.device)
    }

    pub fn validate_shape(&self) -> Result<(), BackendError> {
        if !(1..=64).contains(&self.threads) {
            return Err(BackendError::InvalidThreads);
        }
        if matches!(self.runtime, Runtime::Default | Runtime::Openvino) && self.device_id != 0 {
            return Err(BackendError::InvalidDeviceId);
        }
        self.canonical_device()?;
        Ok(())
    }
}

pub const fn supported_capabilities() -> &'static [&'static str] {
    &["cpu", "openvino", "cuda", "vulkan", "hip"]
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
        Runtime::Cuda | Runtime::Vulkan | Runtime::Hip => {
            match trimmed.to_ascii_lowercase().as_str() {
                "auto" => Ok("auto".into()),
                "gpu" => Ok("gpu".into()),
                _ => Err(invalid()),
            }
        }
        Runtime::Openvino => match trimmed.to_ascii_lowercase().as_str() {
            "auto" => Ok("auto".into()),
            "cpu" => Ok("cpu".into()),
            "gpu" => Ok("gpu".into()),
            "npu" => Ok("npu".into()),
            _ => Err(invalid()),
        },
    }
}

#[cfg(test)]
#[path = "../tests/unit/backend.rs"]
mod tests;
