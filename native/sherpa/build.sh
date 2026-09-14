#!/usr/bin/env bash
set -euo pipefail

SHERPA_TAG=v1.13.8
SHERPA_COMMIT=11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf
OMA_RUNTIME_PATCH_SHA256=3ffc93d211fe02696e6cf3a4213bf0a758bf0d93c03ebf01e25ef68213bf9f29

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

apply_once() {
  if git -C "$1" apply --check "$2"; then
    git -C "$1" apply "$2"
  elif git -C "$1" apply --reverse --check "$2"; then
    echo "patch already applied: $2"
  else
    echo "patch does not apply cleanly: $2" >&2
    exit 1
  fi
}

for command in cmake cp git mkdir patchelf sha256sum; do
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

runtime_patch=${script_dir}/patches/sherpa-onnx-oma-runtime-v1.13.8.patch
printf '%s  %s\n' "${OMA_RUNTIME_PATCH_SHA256}" "${runtime_patch}" | sha256sum -c -

mkdir -p "${work_dir}"
if [[ ! -e ${sherpa_source}/.git ]]; then
  git clone --branch "${SHERPA_TAG}" --depth 1 \
    https://github.com/k2-fsa/sherpa-onnx.git "${sherpa_source}"
fi
verify_checkout "${sherpa_source}" "${SHERPA_COMMIT}"
apply_once "${sherpa_source}" "${runtime_patch}"

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

mkdir -p "${runtime_dir}/lib" "${runtime_dir}/include/sherpa-onnx/c-api"
rm -f \
  "${runtime_dir}/lib/libsherpa-onnx-cxx-api.so" \
  "${runtime_dir}/include/sherpa-onnx/c-api/cxx-api.h"
cp "${sherpa_build}/lib/libsherpa-onnx-c-api.so" "${runtime_dir}/lib/"
patchelf --set-rpath "\$ORIGIN" "${runtime_dir}/lib/libsherpa-onnx-c-api.so"
cp "${sherpa_source}/sherpa-onnx/c-api/c-api.h" \
  "${runtime_dir}/include/sherpa-onnx/c-api/"

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
