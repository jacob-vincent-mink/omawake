use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn unique_directory(scope: &str, name: &str) -> PathBuf {
    for _ in 0..1_024 {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "omawake-{scope}-{}-{sequence}-{name}",
            std::process::id()
        ));
        match fs::create_dir(&directory) {
            Ok(()) => return directory,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!(
                "create exclusive Omawake test directory {}: {error}",
                directory.display()
            ),
        }
    }
    panic!("could not allocate an exclusive Omawake test directory for {scope}/{name}");
}

#[test]
fn repeated_fixture_names_reserve_distinct_directories_without_deleting_peers() {
    let first = unique_directory("fixture-allocation", "same-name");
    let marker = first.join("still-owned");
    fs::write(&marker, b"first fixture").unwrap();

    let second = unique_directory("fixture-allocation", "same-name");

    assert_ne!(first, second);
    assert!(first.is_dir());
    assert!(second.is_dir());
    assert_eq!(fs::read(marker).unwrap(), b"first fixture");
    fs::remove_dir_all(first).unwrap();
    fs::remove_dir_all(second).unwrap();
}

pub(crate) struct FakeEmbeddingSession {
    pub contract: String,
    pub inverted: bool,
    audio: Vec<f32>,
}
impl FakeEmbeddingSession {
    pub fn new(contract: &str) -> Self {
        Self {
            contract: contract.into(),
            inverted: false,
            audio: Vec::new(),
        }
    }
}
impl crate::engine::embedding_worker::EmbeddingSession for FakeEmbeddingSession {
    fn contract(&self) -> &str {
        &self.contract
    }
    fn execution_devices(&self) -> &str {
        "TEST"
    }
    fn start(&mut self) -> anyhow::Result<()> {
        self.audio.clear();
        Ok(())
    }
    fn audio(
        &mut self,
        samples: &[f32],
    ) -> anyhow::Result<Vec<crate::engine::embedding_worker::EncodedUtterance>> {
        self.audio.extend_from_slice(samples);
        Ok(Vec::new())
    }
    fn finish(&mut self) -> anyhow::Result<Vec<crate::engine::embedding_worker::EncodedUtterance>> {
        if self.audio.is_empty() {
            return Ok(Vec::new());
        }
        let sign = if self.audio.iter().sum::<f32>() > 0.0 {
            1.0
        } else {
            -1.0
        };
        let mut values = vec![0.0; 512];
        values[0] = if self.inverted { -sign } else { sign };
        values[1] = 0.1;
        self.audio.clear();
        Ok(vec![crate::engine::embedding_worker::EncodedUtterance {
            embedding: crate::engine::embedding::Embedding {
                encoder_contract: self.contract.clone(),
                values,
                source_frames: 10,
                inference_ms: 1.0,
                execution_devices: "TEST".into(),
            },
            start_sample: 0,
        }])
    }
}
pub(crate) fn isolated_paths(root: &std::path::Path) -> crate::paths::AppPaths {
    crate::paths::AppPaths {
        config_file: root.join("default/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("runtime"),
    }
}
