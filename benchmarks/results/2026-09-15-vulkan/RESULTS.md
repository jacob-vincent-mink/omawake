# Intel Vulkan validation

These measurements use the published `v0.0.1-rc.3` binary, from source
`d3da01b482f0d518d8cd96a748cbad13ed562f83`. Binary SHA-256:
`e21e2b06dd2a354035123ee3649b815e6d0dbc832317c46320ee4baf9e580f04`.
They are the Vulkan baseline for the follow-up release, not measurements of
its later setup and service changes.

The local host is an Intel Core Ultra X7 358H with an Arc B390 integrated GPU.
The combined audio.cpp provider was built from pinned revision
`e9ff20042ec85af960a720368c6927cda19ad65f` with Vulkan enabled and
`AUDIOCPP_MODELS=moonshine_asr,supertonic`. The provider enumerated
`Vulkan:0 Intel Arc B390 (PTL) [IGPU]`. Worker command lines confirmed the
exact provider, model, `vulkan` backend, and device `0`; setup also passed
model-backed validation with fallback disabled. These establish provider
selection and successful execution; hardware utilization counters were not
collected.

All runs used isolated configuration directories and files. No microphone,
audio playback, or wake-word actions were used. SHA-256 hashes of the local
raw reports and their summarized metrics are in [metrics.json](metrics.json).

## Same-model comparison

Both devices used Moonshine Streaming Tiny Q8_0 with Silero VAD 6.2.1. The
positive evaluation used the same 60 generated Hey Jarvis clips as the
[rc.3 evaluation](../2026-09-15-rc3/RESULTS.md).

| Device | Benchmark model load | Warm p50 / p95 | Warm p50 RTF | Positive recall |
|---|---:|---:|---:|---:|
| CPU | 177 ms | 72 / 93 ms | 0.0342 | 59/60 (98.3%) |
| Vulkan iGPU | 181 ms | 119 / 147 ms | 0.0544 | 59/60 (98.3%) |

Benchmark percentiles cover ten measured file inferences after warmup.
Evaluation loads were 209 ms on CPU and 163 ms on Vulkan. Model load is a
fresh process measurement with potentially warm filesystem/driver caches,
not a first-boot or first-ever shader compilation measurement.

Vulkan preserved recall on this positive set but was slower than CPU for this
small model. This pass does not measure long-form false activations, power,
or uncommon-name accuracy; it does not establish that Vulkan is universally
faster. Hardware setup recommendations follow the requested device preference
and still require a successful model probe.
