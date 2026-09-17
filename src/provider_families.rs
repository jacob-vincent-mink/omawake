//! Query the pinned audio.cpp registry inside the existing native probe boundary.
use anyhow::{Context, Result};
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};

type Create = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> i32;
type Free = unsafe extern "C" fn(*mut c_void);
type Count = unsafe extern "C" fn(*const c_void) -> usize;
type Family = unsafe extern "C" fn(*const c_void, usize, *mut *const c_char) -> i32;

pub(crate) fn require(library: &Library, required: &[&str]) -> Result<Vec<String>> {
    // Symbols and returned strings belong to the loaded provider; the registry
    // is always freed before that library can be dropped.
    unsafe {
        let create: Create = *library
            .get(b"audiocpp_registry_create\0")
            .context("provider lacks registry creation")?;
        let free: Free = *library
            .get(b"audiocpp_registry_free\0")
            .context("provider lacks registry cleanup")?;
        let count: Count = *library
            .get(b"audiocpp_registry_family_count\0")
            .context("provider lacks model-family discovery")?;
        let family: Family = *library
            .get(b"audiocpp_registry_family\0")
            .context("provider lacks model-family discovery")?;
        let mut registry = std::ptr::null_mut();
        let status = create(std::ptr::null(), &mut registry);
        if status != 0 || registry.is_null() {
            if !registry.is_null() {
                free(registry);
            }
            anyhow::bail!("audio.cpp registry creation failed ({status})");
        }
        let result = (|| {
            let count = count(registry);
            anyhow::ensure!(
                count <= 256,
                "audio.cpp registry exceeds 256 model families"
            );
            let mut families = Vec::new();
            for index in 0..count {
                let mut name = std::ptr::null();
                let status = family(registry, index, &mut name);
                anyhow::ensure!(
                    status == 0 && !name.is_null(),
                    "audio.cpp family discovery failed at index {index}"
                );
                families.push(
                    CStr::from_ptr(name)
                        .to_str()
                        .context("invalid provider family name")?
                        .to_owned(),
                );
            }
            for name in required {
                anyhow::ensure!(
                    families.iter().any(|found| found == name),
                    "audio.cpp provider is missing required model family {name}; install a provider built with that family (registered: {})",
                    families.join(", ")
                );
            }
            Ok(families)
        })();
        free(registry);
        result
    }
}

#[cfg(test)]
#[path = "../tests/unit/provider_families.rs"]
mod tests;
