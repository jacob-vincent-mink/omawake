use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

const MAX_HEADER_BYTES: usize = 64 * 1024;
pub(super) const FRAME_SAMPLES: usize = 512;

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum Request {
    Start { id: u64 },
    Audio { id: u64, samples: usize },
    Finish { id: u64 },
    Cancel { id: u64 },
    Shutdown,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Transcript {
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum Response {
    Ready {
        version: String,
    },
    Ack {
        id: u64,
    },
    Result {
        id: u64,
        transcripts: Vec<Transcript>,
    },
    Error {
        id: Option<u64>,
        message: String,
    },
}

fn write_json<T: Serialize>(output: &mut impl Write, value: &T) -> io::Result<()> {
    let encoded = serde_json::to_vec(value).map_err(io::Error::other)?;
    if encoded.len() > MAX_HEADER_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "protocol header is too large",
        ));
    }
    let length = u32::try_from(encoded.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "protocol header is too large"))?;
    output.write_all(&length.to_le_bytes())?;
    output.write_all(&encoded)?;
    output.flush()
}

fn read_json<T: DeserializeOwned>(input: &mut impl Read) -> io::Result<T> {
    let mut length = [0_u8; 4];
    input.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_HEADER_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "protocol header is too large",
        ));
    }
    let mut encoded = vec![0; length];
    input.read_exact(&mut encoded)?;
    serde_json::from_slice(&encoded).map_err(io::Error::other)
}

pub(super) fn write_request(
    output: &mut impl Write,
    request: &Request,
    pcm: &[f32],
) -> io::Result<()> {
    let expected = match request {
        Request::Audio { samples, .. } => *samples,
        Request::Start { .. }
        | Request::Finish { .. }
        | Request::Cancel { .. }
        | Request::Shutdown => 0,
    };
    if pcm.len() != expected || pcm.len() > FRAME_SAMPLES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PCM length does not match bounded request",
        ));
    }
    write_json(output, request)?;
    for sample in pcm {
        output.write_all(&sample.to_le_bytes())?;
    }
    output.flush()
}

pub(super) fn read_request(input: &mut impl Read) -> io::Result<(Request, Vec<f32>)> {
    let request: Request = read_json(input)?;
    let samples = match request {
        Request::Audio { samples, .. } if samples == 0 || samples > FRAME_SAMPLES => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "audio request is empty or too large",
            ));
        }
        Request::Audio { samples, .. } => samples,
        _ => 0,
    };
    let mut bytes = vec![0_u8; samples * size_of::<f32>()];
    input.read_exact(&mut bytes)?;
    let (samples, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    Ok((
        request,
        samples.iter().copied().map(f32::from_le_bytes).collect(),
    ))
}

pub(super) fn write_response(output: &mut impl Write, response: &Response) -> io::Result<()> {
    write_json(output, response)
}

pub(super) fn read_response(input: &mut impl Read) -> io::Result<Response> {
    read_json(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_round_trips_one_exact_frame_and_bounds_allocations() {
        let request = Request::Audio {
            id: 7,
            samples: FRAME_SAMPLES,
        };
        let pcm = vec![0.25; FRAME_SAMPLES];
        let mut encoded = Vec::new();
        write_request(&mut encoded, &request, &pcm).unwrap();
        let (decoded, output) = read_request(&mut encoded.as_slice()).unwrap();
        assert!(matches!(decoded, Request::Audio { id: 7, .. }));
        assert_eq!(output, pcm);

        let oversized = Request::Audio {
            id: 8,
            samples: FRAME_SAMPLES + 1,
        };
        assert!(write_request(&mut Vec::new(), &oversized, &[]).is_err());
    }
}
