#!/usr/bin/env bash
set -euo pipefail

readonly AUDIOCPP_COMMIT="e9ff20042ec85af960a720368c6927cda19ad65f"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 AUDIOCPP_SOURCE BUILD_DIRECTORY" >&2
  exit 2
fi

source_directory=$1
build_directory=$2
actual_commit=$(git -C "${source_directory}" rev-parse HEAD)
if [[ "${actual_commit}" != "${AUDIOCPP_COMMIT}" ]]; then
  echo "audio.cpp source is ${actual_commit}; expected ${AUDIOCPP_COMMIT}" >&2
  exit 1
fi

cmake -S "${source_directory}" -B "${build_directory}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_VERSION="${AUDIOCPP_COMMIT:0:7}" \
  -DAUDIOCPP_BUILD_C_API=ON \
  -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
  -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=moonshine_asr \
  -DENGINE_ENABLE_NATIVE_CPU=OFF \
  -DENGINE_ENABLE_LLAMAFILE=OFF \
  -DENGINE_ENABLE_OPENMP=ON \
  -DENGINE_BUILD_EXAMPLES=OFF \
  -DENGINE_BUILD_TESTS=OFF \
  -DENGINE_BUILD_EXTENDED_TESTS=OFF \
  -DENGINE_BUILD_MODEL_TESTS=OFF
cmake --build "${build_directory}" \
  --parallel "${AUDIOCPP_BUILD_JOBS:-4}" \
  --target audiocpp

test -f "${build_directory}/bin/libaudiocpp.so.0.1.0"
printf '%s\n' "${build_directory}/bin/libaudiocpp.so.0.1.0"
