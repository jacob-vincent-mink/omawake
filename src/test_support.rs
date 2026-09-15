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
