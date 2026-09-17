use super::*;
#[test]
fn registry_reports_missing_families_and_releases_handles_on_all_failures() {
    let root = std::env::temp_dir().join(format!(
        "provider-families-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("stub.c");
    std::fs::write(&source,r#"
#include <stddef.h>
static int mode, freed;
void set_mode(int value) { mode=value; freed=0; }
int free_count(void) { return freed; }
int audiocpp_registry_create(const char *config, void **out) { (void)config; *out=mode==1?NULL:(void*)1; return 0; }
void audiocpp_registry_free(void *registry) { (void)registry; freed++; }
size_t audiocpp_registry_family_count(const void *registry) { (void)registry; return mode==2?257:2; }
int audiocpp_registry_family(const void *registry, size_t i, const char **out) {
(void)registry; *out=mode==3?NULL:mode==5?"\xff":i==0?"supertonic":"moonshine_asr"; return mode==4?1:0;
}
"#).unwrap();
    let path = root.join("stub.so");
    assert!(
        std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&source)
            .arg("-o")
            .arg(&path)
            .status()
            .unwrap()
            .success()
    );
    let library = unsafe { Library::new(&path) }.unwrap();
    let set = unsafe { library.get::<unsafe extern "C" fn(i32)>(b"set_mode\0") }.unwrap();
    let freed = unsafe { library.get::<unsafe extern "C" fn() -> i32>(b"free_count\0") }.unwrap();
    assert_eq!(
        require(&library, &["supertonic"]).unwrap(),
        ["supertonic", "moonshine_asr"]
    );
    assert_eq!(unsafe { freed() }, 1);
    assert!(
        require(&library, &["silero_vad"])
            .unwrap_err()
            .to_string()
            .contains("missing required model family silero_vad")
    );
    for mode in 1..=5 {
        unsafe { set(mode) };
        assert!(require(&library, &[]).is_err());
        assert_eq!(unsafe { freed() }, if mode == 1 { 0 } else { 1 });
    }
    drop(library);
    std::fs::remove_dir_all(root).unwrap();
}
