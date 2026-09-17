use super::*;

fn connection(command: Command) -> (UnixStream, UnixStream) {
    let (mut client, server) = UnixStream::pair().unwrap();
    client
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    server
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    let request = Request {
        protocol: 1,
        id: "test".into(),
        command,
    };
    serde_json::to_writer(&mut client, &request).unwrap();
    client.write_all(b"\n").unwrap();
    (client, server)
}

fn poll(control: &mut OwnedControl, state: &str, stream: Option<UnixStream>) -> Option<Command> {
    let mut stream = stream;
    control
        .poll(state, &json!({}), || Ok(stream.take()))
        .unwrap()
}

fn response(client: &mut UnixStream) -> String {
    let mut text = String::new();
    BufReader::new(client).read_line(&mut text).unwrap();
    let response: Response = serde_json::from_str(&text).unwrap();
    match response.result {
        ResultPayload::State { state, .. } => state,
        other => panic!("unexpected response: {other:?}"),
    }
}

#[test]
fn hold_acknowledges_only_after_capture_has_returned_and_disconnect_resumes() {
    let mut control = OwnedControl::default();
    let (mut client, server) = connection(Command::HoldPause);
    assert!(matches!(
        poll(&mut control, "armed", Some(server)),
        Some(Command::Pause)
    ));
    let mut byte = [0];
    assert!(matches!(
        client.read(&mut byte).unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    ));
    assert!(poll(&mut control, "paused", None).is_none());
    assert_eq!(response(&mut client), "paused");
    drop(client);
    assert!(matches!(
        poll(&mut control, "paused", None),
        Some(Command::Resume)
    ));
}

#[test]
fn independent_owners_and_manual_pause_do_not_resume_each_other() {
    let mut control = OwnedControl::default();
    let (mut first, server) = connection(Command::HoldPause);
    poll(&mut control, "armed", Some(server));
    poll(&mut control, "paused", None);
    assert_eq!(response(&mut first), "paused");
    let (mut second, server) = connection(Command::HoldPause);
    assert!(poll(&mut control, "paused", Some(server)).is_none());
    assert_eq!(response(&mut second), "paused");
    let (mut manual, server) = connection(Command::Pause);
    assert!(matches!(
        poll(&mut control, "paused", Some(server)),
        Some(Command::Pause)
    ));
    assert_eq!(response(&mut manual), "paused");
    drop(first);
    assert!(poll(&mut control, "paused", None).is_none());
    let (mut resume, server) = connection(Command::Resume);
    assert!(poll(&mut control, "paused", Some(server)).is_none());
    assert_eq!(response(&mut resume), "paused");
    drop(second);
    assert!(matches!(
        poll(&mut control, "paused", None),
        Some(Command::Resume)
    ));
    let (_manual, server) = connection(Command::Pause);
    poll(&mut control, "armed", Some(server));
    assert!(poll(&mut control, "paused", None).is_none());
}

#[test]
fn disconnect_before_acknowledgement_and_owner_limit_are_bounded() {
    let mut control = OwnedControl::default();
    let (client, server) = connection(Command::HoldPause);
    poll(&mut control, "armed", Some(server));
    drop(client);
    assert!(matches!(
        poll(&mut control, "paused", None),
        Some(Command::Resume)
    ));
    let mut owners = Vec::new();
    for _ in 0..MAX_HOLDS {
        let (mut client, server) = connection(Command::HoldPause);
        poll(&mut control, "paused", Some(server));
        assert_eq!(response(&mut client), "paused");
        owners.push(client);
    }
    let (mut extra, server) = connection(Command::HoldPause);
    assert!(poll(&mut control, "paused", Some(server)).is_none());
    let mut text = String::new();
    BufReader::new(&mut extra).read_line(&mut text).unwrap();
    assert!(
        matches!(serde_json::from_str::<Response>(&text).unwrap().result, ResultPayload::Error { code, .. } if code == "busy")
    );
    drop(owners);
    assert!(matches!(
        poll(&mut control, "paused", None),
        Some(Command::Resume)
    ));
}

#[test]
fn status_errors_and_shutdown_use_the_owned_controller() {
    let mut control = OwnedControl::default();
    for (payload, expected) in [
        (b"bad-json\n".as_slice(), "invalid_request"),
        (
            b"{\"protocol\":2,\"id\":\"old\",\"type\":\"status\"}\n".as_slice(),
            "protocol_mismatch",
        ),
    ] {
        let (mut client, server) = UnixStream::pair().unwrap();
        client.write_all(payload).unwrap();
        assert!(poll(&mut control, "armed", Some(server)).is_none());
        let mut text = String::new();
        BufReader::new(client).read_line(&mut text).unwrap();
        assert!(
            matches!(serde_json::from_str::<Response>(&text).unwrap().result, ResultPayload::Error { code, .. } if code == expected)
        );
    }
    let (mut client, server) = connection(Command::Status);
    assert!(poll(&mut control, "armed", Some(server)).is_none());
    let mut text = String::new();
    BufReader::new(&mut client).read_line(&mut text).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        value["details"]["pause"],
        json!({"manual":false,"owners":0})
    );
    let (mut client, server) = connection(Command::Shutdown);
    assert!(matches!(
        poll(&mut control, "armed", Some(server)),
        Some(Command::Shutdown)
    ));
    assert_eq!(response(&mut client), "stopping");
    assert!(
        control
            .poll("armed", &json!({}), || anyhow::bail!("accept failed"))
            .is_err()
    );
}
