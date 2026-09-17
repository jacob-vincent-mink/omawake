# whisper.cpp default smoke evidence

Recorded 2026-09-15 on the development host with isolated XDG directories,
fallback disabled and two CPU threads. No microphone, playback, command actions
or live configuration were used. This is one file-only smoke test, not a
continuous-listening or hardware qualification result.

The actual installer downloaded `whisper-base.en-ggml-silero-v6.2.0` from the
catalog's revision-pinned canonical sources and verified both SHA-256 hashes.
`setup runtime --runtime default --device cpu` reported whisper.cpp 1.9.3,
`loadable: true`, `ready: false`, and `model_inference_verified: false`, without
applying configuration. A subsequent `test --audio jfk.wav --json` detected the
configured phrase `ask not` once at 3.008 seconds. There were no actions or
fallback; observed model load was 71 ms. See [raw detection](detection.json).
Timing is a single observation, not a performance baseline.

| Artifact | Revision or SHA-256 |
|---|---|
| whisper.cpp source | `371b5a7561823ab2bb32142d2751e35e7534727b` (1.9.3) |
| Built libwhisper.so.1.9.3 | `b3c706c77af76e37c44b9b91b0b59765ff854983cf120a4496239ca7b677bd68` |
| Upstream samples/jfk.wav | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| Whisper Base.en GGML | `a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002` |
| Silero 6.2.0 GGML | `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987` |

The test configuration selected `backend.kind = "whispercpp"`, runtime
`default`, device `cpu`, threads `2`, fallback `error`, an explicit provider
library, verifier `ggml-base.en.bin`, VAD `ggml-silero-v6.2.0.bin`, and 16 kHz
input. The only enabled wake phrase was `ask not` (ID `ask-not`).

To reproduce, build whisper.cpp at the revision above, install the catalog
bundle into an isolated XDG data directory, configure those fields, preview the
CPU runtime, then run the file-only test against upstream `samples/jfk.wav`.
The temporary absolute library path is intentionally not a portable setup
recommendation. Real positive/near-match/negative corpora, repeated resource
measurements and device qualification remain outstanding.
