use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    Status,
    Pause,
    Resume,
    Shutdown,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Request {
    pub protocol: u32,
    pub id: String,
    #[serde(flatten)]
    pub command: Command,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Response {
    pub protocol: u32,
    pub id: String,
    #[serde(flatten)]
    pub result: ResultPayload,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResultPayload {
    State {
        state: String,
        details: serde_json::Value,
    },
    Error {
        code: String,
        message: String,
    },
}

impl Response {
    pub fn error(id: impl Into<String>, code: &str, error: impl std::fmt::Display) -> Self {
        Self {
            protocol: 1,
            id: id.into(),
            result: ResultPayload::Error {
                code: code.into(),
                message: error.to_string(),
            },
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/protocol.rs"]
mod tests;
