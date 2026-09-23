use std::fs;
use std::os::linux::net::SocketAddrExt;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::{SocketAddr, UnixListener};

use anyhow::Result;
use sha2::{Digest, Sha256};

use crate::paths::AppPaths;

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
