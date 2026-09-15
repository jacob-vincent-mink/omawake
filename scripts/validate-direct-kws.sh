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
assert [(d["id"], d["tokens"]) for d in zero["detections"]] == [
    ("light-up", ["▁", "L", "IGHT", "▁UP"])
]
assert [(d["id"], d["tokens"]) for d in one["detections"]] == [
    ("lovely-child", ["▁LOVE", "LY", "▁CHI", "L", "D"]),
    ("forever", ["▁FOR", "E", "VER"]),
]
for report in (zero, one):
    for detection in report["detections"]:
        timestamps = detection["timestamps"]
        assert len(timestamps) == len(detection["tokens"])
        assert timestamps == sorted(timestamps)
        assert all(timestamp >= 0 for timestamp in timestamps)
print("direct KWS parity: ok")
PY
