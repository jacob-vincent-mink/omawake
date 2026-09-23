use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::paths::AppPaths;
use crate::protocol::{Command, Request, Response, ResultPayload};

pub(crate) fn socket_targets_config(paths: &AppPaths) -> Result<Option<bool>> {
    let mut stream = match UnixStream::connect(paths.socket()) {
        Ok(stream) => stream,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound
                    | std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::ConnectionReset
            ) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error).context("connect to the Omawake control socket"),
    };
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let request = Request {
        protocol: 1,
        id: "config-owner".into(),
        command: Command::Status,
    };
    serde_json::to_writer(&mut stream, &request)?;
    stream.write_all(b"\n")?;
    let mut line = String::new();
    if BufReader::new(&mut stream).read_line(&mut line).is_err() {
        return Ok(Some(true)); // A legacy or unresponsive daemon is safest to treat as the target.
    }
    let Ok(response) = serde_json::from_str::<Response>(&line) else {
        return Ok(Some(true));
    };
    if response.protocol != 1 || response.id != request.id {
        return Ok(Some(true));
    }
    Ok(Some(
        response_targets_config(&response, paths).unwrap_or(true),
    ))
}

pub(crate) fn response_targets_config(response: &Response, paths: &AppPaths) -> Result<bool> {
    let ResultPayload::State { details, .. } = &response.result else {
        bail!("daemon could not identify its configuration");
    };
    let Some(owner) = details
        .get("config_path")
        .and_then(serde_json::Value::as_str)
    else {
        bail!("daemon status did not identify its configuration");
    };
    let owner = Path::new(owner);
    if owner == paths.config_file {
        return Ok(true);
    }
    Ok(fs::canonicalize(owner)
        .ok()
        .zip(fs::canonicalize(&paths.config_file).ok())
        .is_some_and(|(owner, requested)| owner == requested))
}

pub(crate) fn reserve(paths: &AppPaths) -> Result<Vec<UnixListener>> {
    // Keep the spelling of the config path locked if an atomic save replaces a
    // symlink. Lock its resolved target too, so aliases share daemon ownership.
    let path = if paths.config_file.is_absolute() {
        paths.config_file.clone()
    } else {
        std::env::current_dir()?.join(&paths.config_file)
    };
    let mut identities = vec![path.clone()];
    if let Ok(resolved) = fs::canonicalize(&path) {
        identities.push(resolved);
    }
    identities.sort();
    identities.dedup();
    let user = unsafe { libc::geteuid() };
    let mut listeners = Vec::with_capacity(identities.len());
    for identity in identities {
        let digest = Sha256::digest(identity.as_os_str().as_bytes());
        let suffix: String = digest
            .iter()
            .take(16)
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let name = format!("omawake-daemon-{user}-{suffix}");
        let address = SocketAddr::from_abstract_name(name.as_bytes())?;
        listeners.push(UnixListener::bind_addr(&address).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AddrInUse {
                anyhow::anyhow!(
                    "another Omawake daemon is already running for {}",
                    path.display()
                )
            } else {
                error.into()
            }
        })?);
    }
    Ok(listeners)
}
