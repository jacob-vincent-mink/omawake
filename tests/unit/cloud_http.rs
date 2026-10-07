use super::*;
#[test]
fn endpoint_limits_loopback_and_auth_policy_are_validated_without_requests() {
    for host in [
        "http://localhost:1234",
        "http://127.0.0.2:1234",
        "http://[::1]:1234",
        "https://example.com",
    ] {
        let c = CloudConfig {
            base_url: host.into(),
            ..Default::default()
        };
        assert!(base_url(&c, "").is_ok());
    }
    for timeout_seconds in [0, 301] {
        let c = CloudConfig {
            timeout_seconds,
            ..Default::default()
        };
        assert!(base_url(&c, "https://example.com").is_err());
    }
    for max_audio_seconds in [0, 301] {
        let c = CloudConfig {
            max_audio_seconds,
            ..Default::default()
        };
        assert!(base_url(&c, "https://example.com").is_err());
    }
    assert!(base_url(&CloudConfig::default(), "malformed secret").is_err());
    let c = CloudConfig::default();
    assert!(credential(&c, "", false).unwrap().is_none());
    assert!(credential(&c, "", true).is_err());
    for name in ["9INVALID", "INVALID-NAME"] {
        let c = CloudConfig {
            api_key_env: name.into(),
            ..Default::default()
        };
        assert!(credential(&c, "", false).is_err());
    }
    let name = format!("AUDIO_TEST_CREDENTIAL_{}", std::process::id());
    let c = CloudConfig {
        api_key_env: name.clone(),
        ..Default::default()
    };
    for secret in ["", "bad\nkey"] {
        unsafe {
            std::env::set_var(&name, secret);
        }
        assert!(credential(&c, "", true).is_err());
    }
    unsafe {
        std::env::set_var(&name, "test-key");
    }
    assert_eq!(
        credential(&c, "", true).unwrap().as_deref(),
        Some("test-key")
    );
    unsafe {
        std::env::remove_var(name);
    }
}
#[test]
fn status_and_json_failures_never_reveal_vendor_bodies() {
    assert_eq!(
        response(Ok(ureq::Response::new(200, "OK", "{}").unwrap()))
            .unwrap()
            .status(),
        200
    );
    for status in [201, 302, 401, 500] {
        let err = response(Ok(ureq::Response::new(status, "status", "secret").unwrap()))
            .unwrap_err()
            .to_string();
        assert!(!err.contains("secret"));
        assert!(
            response(Err(ureq::Error::Status(
                status,
                ureq::Response::new(status, "status", "secret").unwrap()
            )))
            .is_err()
        );
    }
    assert_eq!(
        read_json(ureq::Response::new(200, "OK", "{\"text\":\"hello\"}").unwrap()).unwrap()["text"],
        "hello"
    );
    assert!(
        !read_json(ureq::Response::new(200, "OK", "secret").unwrap())
            .unwrap_err()
            .to_string()
            .contains("secret")
    );
    assert!(read_json(ureq::Response::new(200, "OK", &"x".repeat(1_048_577)).unwrap()).is_err());
}
