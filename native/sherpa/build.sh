#!/usr/bin/env bash
set -euo pipefail

SHERPA_TAG=v1.13.8
SHERPA_COMMIT=11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf
EXTENDED_RUNTIME_PATCH_SHA256=b462ad2f88bbd5811df8180d40e4bd798569b0a1aa58fd9aaa4a38394e293f86
EXCEPTION_SAFETY_PATCH_SHA256=ad895ce231ec7ce2d1e9575450b6493c88d2272b8d3bbded8ad4a24a26d8944a

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work_dir=${OMA_NATIVE_ROOT:-/tmp/oma-native-sherpa}
jobs=${OMA_BUILD_JOBS:-10}
ort_root=${OMA_ORT_ROOT:-}
ort_include=${OMA_ORT_INCLUDE_DIR:-}
ort_lib=${OMA_ORT_LIB_DIR:-}
sherpa_source=${work_dir}/sherpa-onnx
sherpa_build=${work_dir}/sherpa-build
runtime_dir=${work_dir}/runtime

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

verify_checkout() {
  local actual
  actual=$(git -C "$1" rev-parse HEAD)
  if [[ ${actual} != "$2" ]]; then
    echo "$1 is at ${actual}; expected $2" >&2
    exit 1
  fi
}

apply_patches() {
  local patch
  for patch in "${patches[@]}"; do
    git -C "${sherpa_source}" apply --check "${patch}"
    git -C "${sherpa_source}" apply "${patch}"
  done
}

restore_patches() {
  local patch
  for patch in "${patches[@]}"; do
    git -C "${sherpa_source}" apply "${patch}" >/dev/null 2>&1 || true
  done
}

verify_or_apply_exact_patch_state() {
  if [[ -z $(git -C "${sherpa_source}" status --short) ]]; then
    apply_patches
    return
  fi

  local index patch
  trap restore_patches EXIT
  for ((index = ${#patches[@]} - 1; index >= 0; --index)); do
    patch=${patches[index]}
    git -C "${sherpa_source}" apply --reverse --check "${patch}" || {
      echo "${sherpa_source} contains changes outside the reviewed patch set" >&2
      exit 1
    }
    git -C "${sherpa_source}" apply --reverse "${patch}"
  done

  if [[ -n $(git -C "${sherpa_source}" status --short) ]]; then
    echo "${sherpa_source} contains changes outside the reviewed patch set" >&2
    exit 1
  fi
  apply_patches
  trap - EXIT HUP INT TERM
}

for command in cc cmake cp git mkdir patchelf sha256sum; do
  require_command "${command}"
done

[[ ${work_dir} == /* ]] || {
  echo "OMA_NATIVE_ROOT must be an absolute path" >&2
  exit 1
}
[[ ${jobs} =~ ^[1-9][0-9]*$ ]] || {
  echo "OMA_BUILD_JOBS must be a positive integer" >&2
  exit 1
}

if [[ -z ${ort_include} ]]; then
  [[ -n ${ort_root} ]] || {
    echo "set OMA_ORT_ROOT, or both OMA_ORT_INCLUDE_DIR and OMA_ORT_LIB_DIR" >&2
    exit 1
  }
  for candidate in \
    "${ort_root}/include" \
    "${ort_root}/include/onnxruntime" \
    "${ort_root}/include/onnxruntime/core/session"; do
    if [[ -f ${candidate}/onnxruntime_cxx_api.h ]]; then
      ort_include=${candidate}
      break
    fi
  done
fi

if [[ -z ${ort_lib} ]]; then
  [[ -n ${ort_root} ]] || {
    echo "set OMA_ORT_ROOT, or both OMA_ORT_INCLUDE_DIR and OMA_ORT_LIB_DIR" >&2
    exit 1
  }
  for candidate in \
    "${ort_root}/lib" \
    "${ort_root}/lib64" \
    "${ort_root}/build/Release" \
    "${ort_root}/Release"; do
    if [[ -f ${candidate}/libonnxruntime.so ]]; then
      ort_lib=${candidate}
      break
    fi
  done
fi

[[ ${ort_include} == /* && -f ${ort_include}/onnxruntime_cxx_api.h ]] || {
  echo "ONNX Runtime headers not found; OMA_ORT_INCLUDE_DIR must contain onnxruntime_cxx_api.h" >&2
  exit 1
}
[[ ${ort_lib} == /* && -f ${ort_lib}/libonnxruntime.so ]] || {
  echo "ONNX Runtime shared SDK not found; OMA_ORT_LIB_DIR must contain libonnxruntime.so" >&2
  exit 1
}

extended_runtime_patch=${script_dir}/patches/0001-sherpa-onnx-extended-ort-runtime-v1.13.8.patch
exception_safety_patch=${script_dir}/patches/0002-sherpa-onnx-keyword-spotter-exception-safety-v1.13.8.patch
patches=("${extended_runtime_patch}" "${exception_safety_patch}")
printf '%s  %s\n' "${EXTENDED_RUNTIME_PATCH_SHA256}" "${extended_runtime_patch}" | sha256sum -c -
printf '%s  %s\n' "${EXCEPTION_SAFETY_PATCH_SHA256}" "${exception_safety_patch}" | sha256sum -c -

mkdir -p "${work_dir}"
if [[ ! -e ${sherpa_source}/.git ]]; then
  git clone --branch "${SHERPA_TAG}" --depth 1 \
    https://github.com/k2-fsa/sherpa-onnx.git "${sherpa_source}"
fi
verify_checkout "${sherpa_source}" "${SHERPA_COMMIT}"
verify_or_apply_exact_patch_state

env \
  SHERPA_ONNXRUNTIME_INCLUDE_DIR="${ort_include}" \
  SHERPA_ONNXRUNTIME_LIB_DIR="${ort_lib}" \
cmake -S "${sherpa_source}" -B "${sherpa_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=ON \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF

cmake --build "${sherpa_build}" --target sherpa-onnx-c-api -- -j"${jobs}"

mkdir -p "${runtime_dir}/lib" "${runtime_dir}/include/sherpa-onnx/c-api" \
  "${runtime_dir}/tests"
rm -f \
  "${runtime_dir}/lib/libsherpa-onnx-cxx-api.so" \
  "${runtime_dir}/include/sherpa-onnx/c-api/cxx-api.h"
cp "${sherpa_build}/lib/libsherpa-onnx-c-api.so" "${runtime_dir}/lib/"
patchelf --set-rpath "\$ORIGIN" "${runtime_dir}/lib/libsherpa-onnx-c-api.so"
cp "${sherpa_source}/sherpa-onnx/c-api/c-api.h" \
  "${runtime_dir}/include/sherpa-onnx/c-api/"

cc -std=c11 -Wall -Wextra -Werror -pthread \
  -I"${runtime_dir}/include" \
  "${script_dir}/tests/runtime-contract.c" \
  -L"${runtime_dir}/lib" -lsherpa-onnx-c-api \
  "-Wl,-rpath,\$ORIGIN/../lib" -Wl,-rpath-link,"${ort_lib}" \
  -o "${runtime_dir}/tests/runtime-contract"
env ORT_DISABLE_TELEMETRY=1 \
  LD_LIBRARY_PATH="${runtime_dir}/lib:${ort_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${runtime_dir}/tests/runtime-contract"

cat >"${runtime_dir}/env.sh" <<EOF
export SHERPA_ONNX_LIB_DIR='${runtime_dir}/lib'
export OMA_ORT_INCLUDE_DIR='${ort_include}'
export OMA_ORT_LIB_DIR='${ort_lib}'
export LD_LIBRARY_PATH='${runtime_dir}/lib:${ort_lib}'\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOF

cat <<EOF
Patched sherpa-onnx assembled at ${runtime_dir}
The output contains only the C API shared library and header. ONNX Runtime was
supplied externally from ${ort_lib}; it was not built or copied.

Load the build environment with:
  source '${runtime_dir}/env.sh'
EOF
