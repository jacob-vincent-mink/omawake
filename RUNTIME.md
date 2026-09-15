# Runtime contract

Omawake integrates native providers as libraries. The release executable has
no inference runtime in `DT_NEEDED` and does not launch provider CLIs. Its
hidden workers are re-executions of Omawake itself for crash isolation and warm
sessions.

| Runtime | Provider | Devices | Delivery |
|---|---|---|---|
| `default` | audio.cpp public C ABI | CPU | Compact provider in release |
| `cuda` | audio.cpp public C ABI | NVIDIA GPU | User-supplied complete build |
| `vulkan` | audio.cpp public C ABI | Vulkan GPU | User-supplied complete build |
| `hip` | audio.cpp public C ABI | AMD GPU | User-supplied complete build |
| `openvino` | OpenVINO GenAI C API | Intel CPU/GPU/NPU | User-supplied complete install |

Discovery checks an exact `backend.library`, configured `library_dirs`, and
package-relative `lib/` directories. `omawake setup runtime --dir DIR` records
a complete provider only after its isolated ABI probe and silent file-only
provider/model proof both succeed. A focused runtime apply therefore requires
the matching catalog model to be installed already; use the full guided setup
to select and install both together. Setup reports choices and remediation
without installing vendor software.

The default model is the English Moonshine Streaming Tiny Q8_0 verifier plus
Silero VAD 6.2.1. The OpenVINO profile uses Whisper Base.en INT8 plus the same
Silero model. Phrase verification uses provider-neutral, normalized whole-text
matching with no per-phrase training and no fuzzy matching.

An optional `whispercpp` backend can load a compatible external whisper.cpp
library through its public C ABI. It is a comparison/development provider and
is not included in release archives.
