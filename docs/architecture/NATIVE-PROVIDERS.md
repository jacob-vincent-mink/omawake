# Native provider migration

Omawake will replace its current keyword model and remove ONNX Runtime before
the first stable release. The new pipeline is built from independently sourced
components:

1. The selected engine's native Silero VAD gates speech and bounds compute.
2. A Rust-owned ring buffer preserves pre-roll and complete utterances.
3. A native verifier transcribes each utterance. audio.cpp with Moonshine
   Streaming Tiny is the leading default candidate; upstream whisper.cpp is an
   independent selectable engine and comparison baseline.
4. Omawake normalizes and matches configured whole wake phrases, then dispatches
   the associated action.

This avoids copying sherpa-onnx implementation code or shipping a model archive
published by that project. The behavior belongs to Omawake: stream state,
debouncing, phrase matching, action mapping, process supervision, setup, and
evaluation are implemented and tested here.

## Provider matrix

| Provider | Model form | Devices | Delivery |
| --- | --- | --- | --- |
| `whispercpp` | Whisper GGML/GGUF plus whisper.cpp's Silero VAD | CPU, CUDA, Vulkan | Proven independent verifier baseline and selectable engine. Setup selects one complete compatible installation. |
| `audiocpp` | Native Silero VAD plus a selected audio.cpp ASR family; Moonshine Streaming Tiny is the first qualified model | CPU, CUDA, Vulkan; later HIP and Metal | Leading default candidate for one-library VAD and verification. Other audio.cpp STT families remain addressable through the same engine contract after their own model and corpus gates pass. |
| `openvino` | Intel-compatible VAD and verifier models | Intel CPU, GPU, NPU | Separate optional installation. Only public, unpatched runtime behavior is supported. |
| future enrollment verifier | Independently trained frozen speech encoder and small enrolled phrase head | Determined by the chosen engine | Research path after the verifier baseline and corpus are stable. |

A configured provider path names one complete native installation. Omawake will
not combine a base inference runtime with separately sourced execution-provider
plugins. Setup discovers and probes optional providers without installing
vendor software, and saves configuration only after a real file-only request
succeeds.

## Evidence for the decision

The whisper baseline spike dynamically loaded unmodified upstream
`libwhisper.so.1.9.3` from an 803,352-byte Rust controller/worker. Two requests
used one worker PID and one model load. It transcribed the target phrases in the
test clips, rejected missing or incomplete installations with normal errors,
and had no ONNX Runtime, sherpa, whisper, or ggml load-time dependency.

The tested installation consisted of a 638,496-byte `libwhisper`, 2,060,888
bytes of CPU ggml libraries, a 77,704,715-byte `tiny.en` model, and an 885,098
byte Silero VAD model. These are development measurements rather than release
asset promises.

audio.cpp has a first-class `silero_vad` family with offline and streaming C
ABI sessions. It loads the 16 kHz checkpoint from safetensors and constructs the
inference graph in ggml; GGUF is not required for this small bundled model. The
same audio.cpp backend abstraction can execute VAD and its Moonshine ASR family,
which can reduce Omawake to one external C ABI. whisper.cpp also publishes a
Silero VAD C API, so the Whisper engine does not need to mix in an audio.cpp
library. Each configured engine owns a coherent VAD and verifier installation.

The CPU C ABI proof used 16.07 seconds of file-only audio with one second of
silence on each side. Offline inference took 41.49 ms cold and 38.78 ms hot;
streaming inference took 34.24 ms in 512-frame chunks and emitted speech
start/end events. Model load took 0.14 ms and session creation took 2.10 ms. The
stripped provider was 4.12 MB and the F32 safetensors checkpoint was 1,239,748
bytes with SHA-256
`c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1`.
The tested file was downloaded from the original Silero repository and was
byte-identical to audio.cpp's bundled checkpoint. No audio device or ONNX
Runtime library was opened.

The combined audio.cpp Silero and Moonshine proof recovered all 12 expected
wake-phrase events in 11 positive clips, the same result as Whisper Tiny. Across
those clips, warm two-thread CPU inference averaged 245 ms for Silero plus
Moonshine versus 662 ms for Whisper Tiny ASR alone. A 210-second, 20-speaker
negative smoke set produced no whole-phrase matches, including explicit near
misses. This is enough to proceed with production integration, but the
ten-hour negative corpus remains the release gate. The stripped combined
provider was 4.71 MB and the Moonshine Tiny Q8 GGUF was 60.4 MB.

Moonshine's current audio.cpp streaming mode buffers audio until finalization.
That is suitable for an endpointed wake verifier behind streaming VAD, but it
must not be described as incremental ASR. The worker keeps residual samples and
the bounded utterance ring in Rust.

## Source and license boundaries

Omawake source remains MIT. whisper.cpp and the original Whisper code and
weights are MIT. Silero VAD code and published model are MIT. Release packaging
must retain the copyright and license text for every bundled component and
pin every downloaded artifact by upstream revision, byte size, and SHA-256.

Omawake will not copy sherpa-onnx code, translate its keyword-search
implementation, or use its hosted KWS model archive. Design work starts from
the original component publications and their public interfaces. A short code
comment may identify general inspiration where useful, but provenance lives in
third-party notices and architecture documentation.

## Migration sequence

1. Integrate the proven upstream whisper.cpp verifier and its Silero VAD C API
   through a supervised worker and bounded IPC. Keep resampling, ring-buffer
   limits, pre-roll, hangover, utterance timeouts, and reset behavior in Rust.
2. Integrate audio.cpp Silero plus Moonshine Streaming Tiny as a complete
   engine, then compare it with whisper.cpp on the full positive and negative
   corpus. Keep audio.cpp's engine contract model-family aware so another STT
   family can be qualified without changing Omawake's daemon or phrase matcher.
   Pin each engine's exact C ABI and library version and validate every symbol
   before creating native handles.
3. Add Unicode-aware phrase normalization, word-boundary matching, per-phrase
   thresholds and cooldowns, and multiple phrase-to-action mappings.
4. Replace the model catalog and guided setup. Default setup downloads the
   pinned models for the selected default engine and uses its packaged CPU
   provider. Optional setup selects a complete external CUDA, Vulkan, or
   OpenVINO installation and shows only compatible model combinations.
5. Run the checked-in evaluation manifests against both the current baseline
   and the replacement, then remove the old keyword model, `ort`, plugin
   discovery, and ORT configuration. There is no legacy config migration before
   a stable release.
6. Research an enrollment verifier only after the corpus exposes the Whisper
   verifier's miss rate, false activations, and latency limits.

## Acceptance gates

- Default setup works from a clean user account with the packaged CPU provider
  and guided, license-aware model downloads.
- On-demand file and microphone detection work without a systemd unit.
- The daemon keeps one worker and model warm, restarts a failed worker with a
  bounded policy, and reports ordinary errors without generating core dumps.
- Positive recall and F1 do not materially regress from the checked-in
  baseline. At least ten hours of varied negative speech and background audio
  establish false activations per hour for every threshold under consideration.
- End-to-end latency includes VAD gate, utterance close, verification, matching,
  and action dispatch. Cold and hot latency, peak RSS, and compute placement are
  recorded for each supported device.
- CPU, CUDA, Vulkan, and any OpenVINO path produce equivalent normalized phrase
  decisions on the same file-only corpus within the declared tolerance.
- The release executable has no load-time dependency on whisper.cpp, OpenVINO,
  CUDA, or ONNX Runtime. The default installation includes one known-good CPU
  provider; optional providers are used only after explicit setup.
