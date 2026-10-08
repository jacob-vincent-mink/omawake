use super::*;
#[test]
fn prompts_support_defaults_and_abort_on_closed_input() {
    let mut output = Vec::new();
    assert_eq!(
        prompt_io(&mut &b"\n"[..], &mut output, "Model", "default").unwrap(),
        "default"
    );
    assert_eq!(
        prompt_io(&mut &b" chosen \n"[..], &mut output, "Model", "default").unwrap(),
        "chosen"
    );
    assert!(prompt_io(&mut &b""[..], &mut output, "Model", "default").is_err());
    assert!(
        String::from_utf8(output)
            .unwrap()
            .contains("Model [default]:")
    );
}
#[test]
fn guided_cloud_setup_selects_provider_model_andrealtime() {
    let root = std::env::temp_dir().join(format!("omawake-cloud-guided-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    let options = CloudOptions {
        provider: None,
        model: None,
        base_url: None,
        api_key_env: None,
        realtime: None,
    };
    let mut answers = ["deepgram", "nova-3", "yes"].into_iter();
    configure_with(options, &path, &mut |_, _| {
        Ok(answers.next().unwrap().to_owned())
    })
    .unwrap();
    let config = Config::load(&path).unwrap();
    assert_eq!(config.backend.kind, "deepgram");
    assert_eq!(config.backend.cloud.model, "nova-3");
    assert!(config.backend.cloud.realtime);
    let before = fs::read(&path).unwrap();
    let options = CloudOptions {
        provider: None,
        model: None,
        base_url: None,
        api_key_env: None,
        realtime: None,
    };
    assert!(
        configure_with(options, &path, &mut |_, _| Err(anyhow::anyhow!(
            "closed input"
        )))
        .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::remove_dir_all(root).unwrap();
}
