//! Connection-owned pauses; acknowledgement follows microphone release.
use super::*;
use std::os::fd::AsRawFd;

const MAX_HOLDS: usize = 32;
const MAX_COMMANDS_PER_POLL: usize = 32;

#[derive(Default)]
pub(super) struct OwnedControl {
    manual: bool,
    holds: Vec<Hold>,
}

struct Hold {
    stream: UnixStream,
    id: String,
    acknowledged: bool,
}

impl OwnedControl {
    pub(super) fn poll(
        &mut self,
        state: &str,
        details: &Value,
        mut accept: impl FnMut() -> Result<Option<UnixStream>>,
    ) -> Result<Option<Command>> {
        self.holds.retain(|hold| connected(&hold.stream));
        if state == "paused" {
            for hold in &mut self.holds {
                if !hold.acknowledged {
                    let response = state_response(&hold.id, "paused", details);
                    hold.acknowledged = send(&mut hold.stream, &response).is_ok();
                }
            }
            self.holds.retain(|hold| hold.acknowledged);
        }
        // Bound control traffic per audio poll so status floods cannot starve capture.
        for _ in 0..MAX_COMMANDS_PER_POLL {
            let Some(mut stream) = accept()? else { break };
            stream.set_write_timeout(Some(Duration::from_millis(500)))?;
            let request = match read_request(&mut stream) {
                Ok(request) if request.protocol == 1 => request,
                Ok(request) => {
                    write_response(
                        &mut stream,
                        &Response::error(
                            request.id,
                            "protocol_mismatch",
                            "unsupported protocol version",
                        ),
                    );
                    continue;
                }
                Err(error) => {
                    write_response(
                        &mut stream,
                        &Response::error("unknown", "invalid_request", error),
                    );
                    continue;
                }
            };
            match request.command {
                Command::HoldPause => {
                    if self.holds.len() == MAX_HOLDS {
                        write_response(
                            &mut stream,
                            &Response::error(request.id, "busy", "pause owner limit reached"),
                        );
                        continue;
                    }
                    let mut hold = Hold {
                        stream,
                        id: request.id,
                        acknowledged: false,
                    };
                    if state == "paused" {
                        hold.acknowledged = send(
                            &mut hold.stream,
                            &state_response(&hold.id, "paused", details),
                        )
                        .is_ok();
                        if !hold.acknowledged {
                            continue;
                        }
                    }
                    self.holds.push(hold);
                    if state != "paused" {
                        // Return to the capture owner first. Its next paused poll
                        // acknowledges only after the Capture value was dropped.
                        return Ok(Some(Command::Pause));
                    }
                }
                Command::Pause => {
                    self.manual = true;
                    write_response(&mut stream, &state_response(&request.id, "paused", details));
                    return Ok(Some(Command::Pause));
                }
                Command::Resume => {
                    self.manual = false;
                    let paused = !self.holds.is_empty();
                    write_response(
                        &mut stream,
                        &state_response(
                            &request.id,
                            if paused { "paused" } else { "armed" },
                            details,
                        ),
                    );
                    if !paused {
                        return Ok(Some(Command::Resume));
                    }
                }
                Command::Status => {
                    let mut details = details.clone();
                    details["pause"] = json!({"manual": self.manual, "owners": self.holds.len()});
                    write_response(&mut stream, &state_response(&request.id, state, &details));
                }
                Command::Shutdown => {
                    write_response(
                        &mut stream,
                        &state_response(&request.id, "stopping", details),
                    );
                    return Ok(Some(Command::Shutdown));
                }
            }
        }
        if state == "paused" && !self.manual && self.holds.is_empty() {
            Ok(Some(Command::Resume))
        } else {
            Ok(None)
        }
    }
}

fn state_response(id: &str, state: &str, details: &Value) -> Response {
    Response {
        protocol: 1,
        id: id.into(),
        result: ResultPayload::State {
            state: state.into(),
            details: details.clone(),
        },
    }
}

fn send(stream: &mut UnixStream, response: &Response) -> Result<()> {
    let mut bytes = serde_json::to_vec(response)?;
    bytes.push(b'\n');
    stream.write_all(&bytes)?;
    Ok(())
}

fn connected(stream: &UnixStream) -> bool {
    let mut byte = 0_u8;
    // A hold is a one-request connection. Extra data violates that protocol;
    // EOF, reset or extra bytes all release it. EAGAIN means the owner is alive.
    let result = unsafe {
        libc::recv(
            stream.as_raw_fd(),
            (&mut byte as *mut u8).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    result < 0
        && matches!(
            std::io::Error::last_os_error().kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
        )
}

#[cfg(test)]
#[path = "../../tests/unit/pause_ownership.rs"]
mod tests;
