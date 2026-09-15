#!/usr/bin/env bash
set -euo pipefail

binary=${1:-target/debug/omawake}
ort_library=${2:-/tmp/onnxruntime-linux-x64-1.30.0/lib/libonnxruntime.so.1.30.0}
model=${3:-${XDG_DATA_HOME:-$HOME/.local/share}/omawake/models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cat >"$work/config.toml" <<EOF
[backend]
kind = "omawake-onnx"
runtime = "default"
device = "auto"
threads = 2
fallback = "error"
device_id = 0
onnxruntime_library = "$ort_library"

[backend.options]

[model]
name = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
directory = "$model"
encoder = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
decoder = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
joiner = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
tokens = "tokens.txt"
bpe_model = "bpe.model"
sample_rate = 16000
keywords_score = 1.5
keywords_threshold = 0.25
max_active_paths = 4
num_trailing_blanks = 1

[[wake_words]]
id = "light-up"
phrase = "Light up"
enabled = true
command = ["true"]

[[wake_words]]
id = "lovely-child"
phrase = "Lovely child"
enabled = true
command = ["true"]

[[wake_words]]
id = "forever"
phrase = "Forever"
enabled = true
command = ["true"]
EOF

"$binary" --config "$work/config.toml" test --audio "$model/test_wavs/0.wav" --json >"$work/0.json"
"$binary" --config "$work/config.toml" test --audio "$model/test_wavs/1.wav" --json >"$work/1.json"
python3 - "$work/0.json" "$work/1.json" <<'PY'
import json, sys

zero, one = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
assert [(d["id"], d["timestamps"]) for d in zero["detections"]] == [
    ("light-up", [3.0, 3.0399999618530273, 3.0799999237060547, 3.1599998474121094])
]
assert [(d["id"], d["timestamps"]) for d in one["detections"]] == [
    ("lovely-child", [5.359999656677246, 5.559999942779541, 5.839999675750732, 6.0, 6.039999961853027]),
    ("forever", [10.880000114440918, 10.960000038146973, 11.0]),
]
print("direct KWS parity: ok")
PY
