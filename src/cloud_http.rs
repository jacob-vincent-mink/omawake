//! Shared bounded HTTP transport. Credentials are resolved only at request time.
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{io::Read, time::Duration};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CloudConfig {
    /// HTTPS API root; HTTP is permitted only for explicit loopback development servers.
    pub base_url: String,
    pub vad_threshold: f32,
    pub endpoint_milliseconds: u32,
    /// Environment variable name, never the API key itself. Empty permits keyless compatible servers.
    pub api_key_env: String,
    pub model: String,
    pub timeout_seconds: u64,
    pub max_audio_seconds: u32,
}
impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.01,
            endpoint_milliseconds: 500,
            base_url: String::new(),
            api_key_env: String::new(),
            model: String::new(),
            timeout_seconds: 60,
            max_audio_seconds: 30,
        }
    }
}
pub fn base_url(config: &CloudConfig, default: &str) -> Result<url::Url> {
    ensure!(
        (1..=300).contains(&config.timeout_seconds),
        "cloud timeout_seconds must be 1..300"
    );
    ensure!(
        (1..=300).contains(&config.max_audio_seconds),
        "cloud max_audio_seconds must be 1..300"
    );
    let raw = if config.base_url.is_empty() {
        default
    } else {
        &config.base_url
    };
    let url = url::Url::parse(raw).map_err(|_| anyhow::anyhow!("invalid cloud base_url"))?;
    ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "cloud base_url must not contain credentials, query parameters or fragments"
    );
    let loopback = url.host_str().is_some_and(|h| {
        h == "localhost"
            || h == "[::1]"
            || h.parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    ensure!(
        url.scheme() == "https" || (url.scheme() == "http" && loopback),
        "cloud endpoints require HTTPS (HTTP allowed only on loopback)"
    );
    Ok(url)
}
pub fn credential(
    config: &CloudConfig,
    default_env: &str,
    required: bool,
) -> Result<Option<String>> {
    let name = if config.api_key_env.is_empty() {
        default_env
    } else {
        &config.api_key_env
    };
    ensure!(
        name.is_empty()
            || (name.len() <= 128
                && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                && !name.as_bytes()[0].is_ascii_digit()),
        "invalid cloud api_key_env name"
    );
    let value = (!name.is_empty())
        .then(|| std::env::var(name).ok())
        .flatten();
    if let Some(value) = &value {
        ensure!(
            !value.is_empty() && value.len() <= 4096 && !value.contains(['\r', '\n', '\0']),
            "cloud credential is empty or invalid"
        );
    }
    ensure!(
        !required || value.is_some(),
        "cloud credential unavailable; set environment variable {name}"
    );
    Ok(value)
}
pub fn agent(config: &CloudConfig) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .redirects(0)
        .timeout_connect(Duration::from_secs(10.min(config.timeout_seconds)))
        .timeout_read(Duration::from_secs(config.timeout_seconds))
        .timeout_write(Duration::from_secs(config.timeout_seconds))
        .timeout(Duration::from_secs(config.timeout_seconds))
        .build()
}
pub fn response(
    result: std::result::Result<ureq::Response, ureq::Error>,
) -> Result<ureq::Response> {
    // Deliberately omit response bodies and transport URLs: they can echo text or credentials.
    match result {
        Ok(response) if response.status() == 200 => Ok(response),
        Ok(response) => bail!("cloud request failed (HTTP {})", response.status()),
        Err(ureq::Error::Status(status, _)) => bail!("cloud request failed (HTTP {status})"),
        Err(ureq::Error::Transport(_)) => bail!("cloud request transport failed or timed out"),
    }
}
pub fn read_json(response: ureq::Response) -> Result<serde_json::Value> {
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(1_048_577)
        .read_to_end(&mut bytes)
        .map_err(|_| anyhow::anyhow!("cloud response read failed or timed out"))?;
    ensure!(bytes.len() <= 1_048_576, "cloud response exceeds 1 MiB");
    serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("cloud response is not valid JSON"))
}

#[cfg(test)]
#[path = "../tests/unit/cloud_http.rs"]
mod tests;
