#!/usr/bin/env bash
set -euo pipefail
export XDG_CONFIG_HOME=/home/jacob/src/github.com/jacob-vincent-mink/.omawake-rc2-release-openvino-final/run-20260915T065138Z-391864/default-cpu/config
export XDG_DATA_HOME=/home/jacob/src/github.com/jacob-vincent-mink/.omawake-rc2-release-openvino-final/run-20260915T065138Z-391864/default-cpu/data
export XDG_STATE_HOME=/home/jacob/src/github.com/jacob-vincent-mink/.omawake-rc2-release-openvino-final/run-20260915T065138Z-391864/default-cpu/state
export XDG_RUNTIME_DIR=/home/jacob/src/github.com/jacob-vincent-mink/.omawake-rc2-release-openvino-final/run-20260915T065138Z-391864/default-cpu/runtime
export XDG_CACHE_HOME=/home/jacob/src/github.com/jacob-vincent-mink/.omawake-rc2-release-openvino-final/run-20260915T065138Z-391864/default-cpu/cache
cd -- /home/jacob/src/github.com/jacob-vincent-mink/.omawake-rc2-release-openvino-final/run-20260915T065138Z-391864/default-cpu
exec /tmp/omawake-rc2-final-stage-output-clean/omawake --config /home/jacob/src/github.com/jacob-vincent-mink/.omawake-rc2-release-openvino-final/run-20260915T065138Z-391864/default-cpu/config/omawake/cold.toml benchmark --warmup 0 --iterations 1 /home/jacob/.local/share/omawake/models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/0.wav /home/jacob/.local/share/omawake/models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/test_wavs/1.wav
