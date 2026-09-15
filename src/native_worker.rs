use std::fs::{self, OpenOptions};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};

static NEXT_RESPONSE: AtomicU64 = AtomicU64::new(0);

pub(crate) struct ResponseFile {
    path: PathBuf,
}

impl ResponseFile {
    pub(crate) fn create(directory: &Path, purpose: &str) -> Result<Self> {
        fs::create_dir_all(directory)
            .with_context(|| format!("create native worker directory {}", directory.display()))?;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("secure native worker directory {}", directory.display()))?;
        for _ in 0..32 {
            let sequence = NEXT_RESPONSE.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(".{purpose}-{}-{sequence}.json", std::process::id()));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
            {
                Ok(_) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("create native worker response {}", path.display())
                    });
                }
            }
        }
        bail!("could not allocate a unique native worker response file")
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn read_json<T: DeserializeOwned>(&self) -> Result<T> {
        let bytes = fs::read(&self.path)
            .with_context(|| format!("read native worker response {}", self.path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parse native worker response {}", self.path.display()))
    }
}

impl Drop for ResponseFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(crate) fn write_json<T: Serialize>(
    path: &Path,
    expected_directory: &Path,
    value: &T,
) -> Result<()> {
    if path.parent() != Some(expected_directory) {
        bail!("native worker response must be inside the application runtime directory");
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if !name.starts_with('.') || !name.ends_with(".json") {
        bail!("invalid native worker response filename");
    }
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect native worker response {}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.permissions().mode() & 0o077 != 0 {
        bail!("native worker response must be a private regular file");
    }
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("open native worker response {}", path.display()))?;
    serde_json::to_writer(&mut file, value)
        .with_context(|| format!("write native worker response {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync native worker response {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    fn directory(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "omawake-native-worker-{}-{name}",
            std::process::id()
        ))
    }

    #[test]
    fn response_files_are_private_validated_and_removed() {
        let response_directory = directory("response");
        let response = ResponseFile::create(&response_directory, "test").unwrap();
        assert_eq!(
            fs::metadata(response.path()).unwrap().permissions().mode() & 0o777,
            0o600
        );
        write_json(
            response.path(),
            &response_directory,
            &serde_json::json!({"ok": true}),
        )
        .unwrap();
        assert_eq!(
            response.read_json::<serde_json::Value>().unwrap()["ok"],
            true
        );
        let path = response.path().to_owned();
        drop(response);
        assert!(!path.exists());

        let outside = directory("outside");
        fs::create_dir_all(&outside).unwrap();
        let file = outside.join("response.json");
        File::create(&file).unwrap();
        assert!(write_json(&file, &response_directory, &serde_json::json!({})).is_err());
    }
}
