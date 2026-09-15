# Accelerator setup

Omawake never downloads or installs CUDA, Vulkan, HIP, OpenVINO, or device
drivers. Install the stack appropriate for the machine, then give setup a
complete provider directory. The isolated probe must load the library, create
the requested device session, and report placement before config is saved.

```bash
omawake setup runtime --json
omawake setup runtime --runtime cuda --device gpu --dir /opt/audiocpp-cuda --apply
omawake setup runtime --runtime vulkan --device gpu --dir /opt/audiocpp-vulkan --apply
omawake setup runtime --runtime hip --device gpu --dir /opt/audiocpp-hip --apply
```

These three runtimes load audio.cpp through its public C ABI. The provider must
be built with its corresponding backend and include or locate every vendor
dependency. `backend.device_id` chooses an accelerator ordinal. Omawake never
invokes an audio.cpp executable.

Focused `setup runtime --apply` expects the compatible catalog model to be
installed already because it proves a real silent inference before committing
the configuration. Use `omawake setup` for a fresh machine so runtime and model
are selected and proved as one transaction.

## Intel CPU, integrated GPU, and NPU

The OpenVINO path uses the OpenVINO GenAI C API directly. Install a complete
OpenVINO distribution containing `libopenvino_genai_c.so`, `libopenvino_c.so`,
the selected CPU/GPU/NPU plugin, and its transitive libraries. Then run one of:

```bash
omawake setup runtime --runtime openvino --device cpu --dir /opt/intel/openvino --apply
omawake setup runtime --runtime openvino --device gpu --dir /opt/intel/openvino --apply
omawake setup runtime --runtime openvino --device npu --dir /opt/intel/openvino --apply
```

Install the matching Intel GPU or NPU driver separately. For GPU and NPU,
setup downloads the OpenVINO model only when model setup is requested, compiles
its persistent cache synchronously, and refuses to save the candidate if the
cache or file-only proof is incomplete. Cached blobs live below
`${XDG_CACHE_HOME:-$HOME/.cache}/omawake/openvino/<device>`.

Use `omawake setup check` after any provider change. Add extra dependency
directories with `backend.library_dirs`; workers construct their own loader
environment, so the parent process and optional systemd unit stay
runtime-neutral.
