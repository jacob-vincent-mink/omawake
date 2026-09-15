# Native providers

Omawake owns capture, resampling, a bounded pre-roll and 320 ms post-roll ring,
phrase matching, cooldowns, actions, setup, and evaluation in Rust. Native
libraries provide VAD and speech transcription behind public C APIs.

## Provider matrix

| Provider | Models | Devices | Delivery |
|---|---|---|---|
| `audiocpp` | Silero VAD plus Moonshine Streaming Tiny | CPU, CUDA, Vulkan, HIP | CPU provider in releases; complete accelerator builds supplied by users |
| `openvino-genai` | Silero VAD plus Whisper Base.en INT8 | Intel CPU, GPU, NPU | Complete OpenVINO installation supplied by users |
| `whispercpp` | whisper.cpp Silero VAD plus Whisper | CPU comparison provider | Complete compatible installation supplied by users |

The Rust executable dynamically loads provider libraries and never invokes a
provider CLI. Each native session lives in a supervised Omawake worker process,
which keeps models warm and contains native crashes. Workers receive exact
library and dependency directories; the parent executable and optional systemd
unit remain runtime-neutral.

The OpenVINO path uses the packaged audio.cpp CPU library only for Silero VAD;
Whisper inference and placement use the external OpenVINO GenAI installation.
This keeps one VAD implementation across providers without asking an OpenVINO
installation to supply audio.cpp.

A configured external provider is one complete native installation. Setup never
assembles a vendor runtime from partial packages and never installs vendor
software. It probes the provider ABI and requested device, runs a file-only
model proof, and saves configuration only after the candidate succeeds.
OpenVINO GPU/NPU setup also requires a persistent compiled cache artifact before
activation.

## Default pipeline

The release profile uses the exact Silero VAD 6.2.1 safetensors file from the
original model repository and Moonshine Streaming Tiny Q8_0 from the pinned
audio.cpp GGUF conversion. Silero gates compute; Rust retains bounded context;
Moonshine transcribes a completed utterance; the provider-neutral matcher
normalizes and matches complete configured phrases without fuzzy matching.
Configured transcript aliases use that same exact whole-phrase matcher. The
whisper.cpp provider additionally passes phrases and aliases as its initial
decoder prompt; Moonshine and OpenVINO apply them after transcription because
their current provider APIs expose no equivalent qualified biasing control.

The current English profile requires no per-phrase training for arbitrary text
phrases within its language coverage. It must not be described as supporting
languages outside the selected verifier's declared coverage.

The pinned stripped CPU provider is about 5.1 MB and exports only the versioned
audio.cpp public C ABI. Its static closure and notices cover audio.cpp
(Apache-2.0), ggml (MIT), cJSON (MIT), libyaml (MIT), PocketFFT-derived code
(BSD-3-Clause), and conservatively attributed llama tokenizer code (MIT).
Deployment specs and provider CLI/server/test targets are disabled.

## Source and license boundaries

Omawake source is MIT. audio.cpp is Apache-2.0. Moonshine and Silero are MIT.
The optional OpenAI Whisper weights and OpenVINO conversion are Apache-2.0.
The optional whisper.cpp ABI declarations remain under the upstream MIT terms;
its implementation and model weights are external.

Every downloadable asset is pinned by original source revision, converted
source revision where applicable, exact byte size, and SHA-256. Model setup
writes provenance and full license terms beside installed assets. Release
archives include the exact notices for the statically retained provider closure.

## Acceptance gates

The qualification method and current phrase-verifier evidence live in
[`EVALUATION.md`](EVALUATION.md). A releasable provider must pass positive,
never-seen phrase, noisy/competing/far-field negative, false activation,
latency, placement, parity, and idle duty/power gates. Tests use file input and
must return ordinary errors rather than signals or core dumps.
