#!/usr/bin/env bash
set -euo pipefail

ORT_TAG=v1.29.0
ORT_COMMIT=2e2543fbe9fae542f921d47a72d21d5a4ef0b710
SHERPA_TAG=v1.13.8
SHERPA_COMMIT=11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf
OPENVINO_VERSION=2026.2.1.21919.ede283a88e3
OPENVINO_ARCHIVE=openvino_toolkit_rhel8_${OPENVINO_VERSION}_x86_64.tgz
OPENVINO_SHA256=cf7a3eb84a1edbd852f719a4ba8c15dbf02a3744ab61488b6cae44747d90bf78
OPENVINO_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2.1/linux/${OPENVINO_ARCHIVE}
ORT_PATCH_SHA256=6d6ec445dc761208aded9c7d786e9112d2920a1509e2683c86c6d2bca8fa499f
SHERPA_PATCH_SHA256=b265953742a4d6a131e3c32e58ae648d108b4ca7e5e7bb5d26c60467981f18f0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work_dir=${OMA_NATIVE_ROOT:-/tmp/oma-native-unified}
jobs=${OMA_BUILD_JOBS:-10}
cuda_home=${CUDA_HOME:-/usr/local/cuda}
cudnn_home=${CUDNN_HOME:-/usr}
cuda_architectures=${OMA_CUDA_ARCHITECTURES:-75;80;86;89;90;100;120}
ort_source=${work_dir}/onnxruntime
ort_build=${work_dir}/ort-build
sherpa_source=${work_dir}/sherpa-onnx
sherpa_build=${work_dir}/sherpa-build
sherpa_install=${work_dir}/sherpa-install
openvino_dir=${work_dir}/openvino-2026.2.1
runtime_dir=${work_dir}/runtime

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

verify_checkout() {
  local directory=$1
  local expected=$2
  local actual
  actual=$(git -C "${directory}" rev-parse HEAD)
  if [[ ${actual} != "${expected}" ]]; then
    echo "${directory} is at ${actual}; expected ${expected}" >&2
    exit 1
  fi
}

apply_once() {
  local directory=$1
  local patch=$2
  if git -C "${directory}" apply --check "${patch}"; then
    git -C "${directory}" apply "${patch}"
  elif git -C "${directory}" apply --reverse --check "${patch}"; then
    echo "patch already applied: ${patch}"
  else
    echo "patch does not apply cleanly: ${patch}" >&2
    exit 1
  fi
}

restore_reviewed_patch() {
  git -C "${verify_restore_directory}" apply "${verify_restore_patch}" >/dev/null 2>&1 || true
}

interrupt_patch_verification() {
  local status=$1
  restore_reviewed_patch
  trap - EXIT HUP INT TERM
  exit "${status}"
}

verify_patch_state() {
  local directory=$1
  local patch=$2
  git -C "${directory}" apply --reverse --check "${patch}" || {
    echo "${directory} does not contain the reviewed patch" >&2
    exit 1
  }
  verify_restore_directory=${directory}
  verify_restore_patch=${patch}
  trap restore_reviewed_patch EXIT
  trap 'interrupt_patch_verification 129' HUP
  trap 'interrupt_patch_verification 130' INT
  trap 'interrupt_patch_verification 143' TERM
  git -C "${directory}" apply --reverse "${patch}"
  if [[ -n $(git -C "${directory}" status --short) ]]; then
    git -C "${directory}" apply "${patch}"
    trap - EXIT HUP INT TERM
    unset verify_restore_directory verify_restore_patch
    echo "${directory} contains changes other than the reviewed patch" >&2
    exit 1
  fi
  git -C "${directory}" apply "${patch}"
  trap - EXIT HUP INT TERM
  unset verify_restore_directory verify_restore_patch
}

for command in awk c++ cc cmake cp curl env git make mkdir mv python3 sed sha256sum tar uname; do
  require_command "${command}"
done

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo "the pinned OpenVINO archive supports Linux x86_64 only" >&2
  exit 1
}
[[ ${work_dir} == /* ]] || {
  echo "OMA_NATIVE_ROOT must be an absolute path" >&2
  exit 1
}
[[ ${jobs} =~ ^[1-9][0-9]*$ ]] || {
  echo "OMA_BUILD_JOBS must be a positive integer" >&2
  exit 1
}
[[ ${cuda_architectures} =~ ^([0-9]+|native)(;([0-9]+|native))*$ ]] || {
  echo "OMA_CUDA_ARCHITECTURES must be a semicolon-separated list of CMake CUDA architectures" >&2
  exit 1
}
[[ -x ${cuda_home}/bin/nvcc ]] || {
  echo "CUDA_HOME does not contain bin/nvcc: ${cuda_home}" >&2
  exit 1
}
[[ -d ${cudnn_home} ]] || {
  echo "CUDNN_HOME is not a directory: ${cudnn_home}" >&2
  exit 1
}
cudnn_include=
for candidate in \
  "${cudnn_home}/include" \
  "${cudnn_home}/include/$(uname -m)-linux-gnu"; do
  if [[ -f ${candidate}/cudnn.h ]]; then
    cudnn_include=${candidate}
    break
  fi
done
[[ -n ${cudnn_include} ]] || {
  echo "CUDNN_HOME does not contain cudnn.h: ${cudnn_home}" >&2
  exit 1
}
cudnn_lib_dir=
for candidate in \
  "${cudnn_home}/lib64" \
  "${cudnn_home}/lib" \
  "${cudnn_home}/lib/$(uname -m)-linux-gnu"; do
  if [[ -f ${candidate}/libcudnn.so ]]; then
    cudnn_lib_dir=${candidate}
    break
  fi
done
[[ -n ${cudnn_lib_dir} ]] || {
  echo "CUDNN_HOME does not contain libcudnn.so: ${cudnn_home}" >&2
  exit 1
}

mkdir -p "${work_dir}"

ort_patch=${script_dir}/../openvino/patches/onnxruntime-openvino-zero-element-tensors-v1.29.0.patch
sherpa_patch=${script_dir}/../openvino/patches/sherpa-onnx-supertonic-component-providers-v1.13.8.patch
printf '%s  %s\n' "${ORT_PATCH_SHA256}" "${ort_patch}" | sha256sum -c -
printf '%s  %s\n' "${SHERPA_PATCH_SHA256}" "${sherpa_patch}" | sha256sum -c -

if [[ ! -d ${ort_source}/.git ]]; then
  git clone --branch "${ORT_TAG}" --depth 1 \
    https://github.com/microsoft/onnxruntime.git "${ort_source}"
fi
verify_checkout "${ort_source}" "${ORT_COMMIT}"
apply_once "${ort_source}" "${ort_patch}"
verify_patch_state "${ort_source}" "${ort_patch}"

openvino_archive_path=${work_dir}/${OPENVINO_ARCHIVE}
if [[ ! -f ${openvino_archive_path} ]]; then
  openvino_download=${openvino_archive_path}.download
  curl -fL "${OPENVINO_URL}" -o "${openvino_download}"
  printf '%s  %s\n' "${OPENVINO_SHA256}" "${openvino_download}" | sha256sum -c -
  mv "${openvino_download}" "${openvino_archive_path}"
fi
printf '%s  %s\n' "${OPENVINO_SHA256}" "${openvino_archive_path}" | sha256sum -c -

if [[ -e ${openvino_dir} && ! -d ${openvino_dir}/runtime ]]; then
  echo "${openvino_dir} exists but is not a complete OpenVINO runtime" >&2
  exit 1
fi
if [[ ! -d ${openvino_dir}/runtime ]]; then
  tar -xzf "${openvino_archive_path}" -C "${work_dir}"
  extracted=${work_dir}/openvino_toolkit_rhel8_${OPENVINO_VERSION}_x86_64
  [[ -d ${extracted}/runtime ]] || {
    echo "OpenVINO archive did not contain the expected runtime directory" >&2
    exit 1
  }
  mv "${extracted}" "${openvino_dir}"
fi

"${ort_source}/build.sh" \
  --config Release \
  --update \
  --build_shared_lib \
  --use_openvino NPU \
  --use_cuda \
  --cuda_home "${cuda_home}" \
  --cudnn_home "${cudnn_home}" \
  --skip_tests \
  --parallel "${jobs}" \
  --compile_no_warning_as_error \
  --cmake_generator 'Unix Makefiles' \
  --build_dir "${ort_build}" \
  --cmake_extra_defines \
    FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
    CMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    OpenVINO_DIR="${openvino_dir}/runtime/cmake"

cmake --build "${ort_build}/Release" \
  --target onnxruntime onnxruntime_providers_openvino onnxruntime_providers_cuda -- -j"${jobs}"

if [[ ! -d ${sherpa_source}/.git ]]; then
  git clone --branch "${SHERPA_TAG}" --depth 1 \
    https://github.com/k2-fsa/sherpa-onnx.git "${sherpa_source}"
fi
verify_checkout "${sherpa_source}" "${SHERPA_COMMIT}"
apply_once "${sherpa_source}" "${sherpa_patch}"
verify_patch_state "${sherpa_source}" "${sherpa_patch}"

env \
  SHERPA_ONNXRUNTIME_INCLUDE_DIR="${ort_source}/include/onnxruntime/core/session" \
  SHERPA_ONNXRUNTIME_LIB_DIR="${ort_build}/Release" \
cmake -S "${sherpa_source}" -B "${sherpa_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${sherpa_install}" \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=ON \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=ON \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=ON \
  -DSHERPA_ONNX_ENABLE_TTS=ON \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF

cmake --build "${sherpa_build}" -- -j"${jobs}"
cmake --install "${sherpa_build}"

sherpa_lib_dir=${sherpa_install}/lib
[[ -d ${sherpa_lib_dir} ]] || sherpa_lib_dir=${sherpa_install}/lib64
mkdir -p "${runtime_dir}/lib" \
  "${runtime_dir}/include/sherpa-onnx" \
  "${runtime_dir}/include/onnxruntime"
cp -a \
  "${sherpa_lib_dir}/libcargs.so" \
  "${sherpa_lib_dir}/libsherpa-onnx-c-api.so" \
  "${sherpa_lib_dir}/libsherpa-onnx-cxx-api.so" \
  "${runtime_dir}/lib/"
cp -a \
  "${ort_build}"/Release/libonnxruntime.so* \
  "${ort_build}/Release/libonnxruntime_providers_shared.so" \
  "${ort_build}/Release/libonnxruntime_providers_openvino.so" \
  "${ort_build}/Release/libonnxruntime_providers_cuda.so" \
  "${runtime_dir}/lib/"
cp -a "${sherpa_install}/include/sherpa-onnx/." \
  "${runtime_dir}/include/sherpa-onnx/"
cp -a "${ort_source}/include/onnxruntime/core/session/." \
  "${runtime_dir}/include/onnxruntime/"

cat >"${runtime_dir}/env.sh" <<EOF
export SHERPA_ONNX_LIB_DIR='${runtime_dir}/lib'
export LD_LIBRARY_PATH='${runtime_dir}/lib:${openvino_dir}/runtime/lib/intel64:${openvino_dir}/runtime/3rdparty/tbb/lib:${cuda_home}/lib64:${cudnn_lib_dir}'\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOF

cat <<EOF
Native runtime assembled at ${runtime_dir}

Load it before building an all-runtime app:
  source '${runtime_dir}/env.sh'
  cargo build --release --all-features
EOF
