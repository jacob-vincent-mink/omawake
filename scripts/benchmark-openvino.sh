#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/benchmark-openvino.sh [--dry-run]

Prepare and optionally run isolated Omawake benchmarks for default CPU and
OpenVINO CPU, GPU, and NPU. The command never records from a microphone.

Environment:
  OMAWAKE_BIN            runtime-neutral Omawake binary
  OMAWAKE_OPENVINO_LIBRARY_PATH  colon-separated external OpenVINO stack path
  OMAWAKE_MODEL_DIR     installed official KWS model directory
  OMAWAKE_BENCH_ROOT    parent for preserved run artifacts
  OMAWAKE_WARMUP        warmup iterations per WAV (default: 2)
  OMAWAKE_ITERATIONS    measured iterations per WAV (default: 10)
  OMAWAKE_THREADS       backend threads (default: 2)
  OMAWAKE_LANES         space-separated lane names (default: all four)
  OMAWAKE_ACCELERATOR_MODEL_VARIANT  int8, fp32-encoder, or fp32 (default: fp32-encoder)
  OMAWAKE_NPU_QDQ_OPTIMIZER  True or False (default: True)
  OMAWAKE_NPU_BUSY_COUNTER  optional readable npu_busy_time_us sysfs path

The external OpenVINO runtime stack and its native loader environment must
already exist. The same Omawake executable is used for every lane.
The harness also requires jq to validate every measured detection set.
EOF
}

dry_run=false
case "${1:-}" in
  "") ;;
  --dry-run) dry_run=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (($# > 1)); then
  usage >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)
model_id=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
omawake_bin=${OMAWAKE_BIN:-$repo_dir/target/release/omawake}
if [[ $omawake_bin != /* ]]; then
  omawake_bin=$PWD/$omawake_bin
fi
openvino_library_path=${OMAWAKE_OPENVINO_LIBRARY_PATH:-$(dirname -- "$omawake_bin")}
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  openvino_library_path=$openvino_library_path:$LD_LIBRARY_PATH
fi
model_dir=${OMAWAKE_MODEL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/omawake/models/$model_id}
artifact_parent=${OMAWAKE_BENCH_ROOT:-$repo_dir/benchmark-artifacts}
warmup=${OMAWAKE_WARMUP:-2}
iterations=${OMAWAKE_ITERATIONS:-10}
threads=${OMAWAKE_THREADS:-2}
accelerator_model_variant=${OMAWAKE_ACCELERATOR_MODEL_VARIANT:-fp32-encoder}
npu_qdq_optimizer=${OMAWAKE_NPU_QDQ_OPTIMIZER:-True}
read -r -a lanes <<<"${OMAWAKE_LANES:-default-cpu openvino-cpu openvino-gpu openvino-npu}"

for numeric in warmup iterations threads; do
  value=${!numeric}
  if [[ ! $value =~ ^[0-9]+$ ]]; then
    printf 'omawake benchmark: %s must be an unsigned integer, got %q\n' "$numeric" "$value" >&2
    exit 2
  fi
done
if ((iterations < 1)); then
  printf 'omawake benchmark: iterations must be at least 1\n' >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  printf 'omawake benchmark: jq is required to validate benchmark JSON\n' >&2
  exit 1
fi
if ((threads < 1 || threads > 64)); then
  printf 'omawake benchmark: threads must be between 1 and 64\n' >&2
  exit 2
fi
if [[ $npu_qdq_optimizer != True && $npu_qdq_optimizer != False ]]; then
  printf 'omawake benchmark: OMAWAKE_NPU_QDQ_OPTIMIZER must be True or False\n' >&2
  exit 2
fi
case "$accelerator_model_variant" in
  int8)
    accelerator_encoder="encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    accelerator_decoder="decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    accelerator_joiner="joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    ;;
  fp32-encoder)
    accelerator_encoder="encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    accelerator_decoder="decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    accelerator_joiner="joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    ;;
  fp32)
    accelerator_encoder="encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    accelerator_decoder="decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    accelerator_joiner="joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
    ;;
  *)
    printf 'omawake benchmark: unknown accelerator model variant: %s\n' "$accelerator_model_variant" >&2
    exit 2
    ;;
esac

declare -A seen_lanes=()
for lane in "${lanes[@]}"; do
  case "$lane" in
    default-cpu|openvino-cpu|openvino-gpu|openvino-npu) ;;
    *) printf 'omawake benchmark: unknown lane: %s\n' "$lane" >&2; exit 2 ;;
  esac
  if [[ -n ${seen_lanes[$lane]:-} ]]; then
    printf 'omawake benchmark: duplicate lane: %s\n' "$lane" >&2
    exit 2
  fi
  seen_lanes[$lane]=1
done
if ((${#lanes[@]} == 0)); then
  printf 'omawake benchmark: OMAWAKE_LANES selected no lanes\n' >&2
  exit 2
fi

required_model_files=(
  encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
  decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx
  joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx
  "$accelerator_encoder"
  "$accelerator_decoder"
  "$accelerator_joiner"
  tokens.txt
  bpe.model
  test_wavs/0.wav
  test_wavs/1.wav
  test_wavs/test_keywords.txt
)
for relative in "${required_model_files[@]}"; do
  if [[ ! -f $model_dir/$relative ]]; then
    printf 'omawake benchmark: required model file is missing: %s\n' "$model_dir/$relative" >&2
    exit 1
  fi
done
model_dir=$(cd -- "$model_dir" && pwd -P)
wav_files=("$model_dir/test_wavs/0.wav" "$model_dir/test_wavs/1.wav")

if ! $dry_run && [[ ! -x $omawake_bin ]]; then
  printf 'omawake benchmark: binary is not executable: %s\n' "$omawake_bin" >&2
  exit 1
fi

toml_escape() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\t'/\\t}
  value=${value//$'\r'/\\r}
  value=${value//$'\n'/\\n}
  printf '%s' "$value"
}

shell_assign() {
  printf 'export %s=%q\n' "$1" "$2"
}

find_npu_busy_counter() {
  if [[ -n ${OMAWAKE_NPU_BUSY_COUNTER:-} ]]; then
    if [[ -r $OMAWAKE_NPU_BUSY_COUNTER ]]; then
      printf '%s\n' "$OMAWAKE_NPU_BUSY_COUNTER"
    else
      printf 'omawake benchmark: NPU busy counter is not readable: %s\n' \
        "$OMAWAKE_NPU_BUSY_COUNTER" >&2
      return 1
    fi
    return 0
  fi
  find /sys/devices -type f -name npu_busy_time_us -readable -print -quit 2>/dev/null || true
}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir=$artifact_parent/run-$timestamp-$$
if ! mkdir -p -- "$artifact_parent"; then
  printf 'omawake benchmark: cannot create artifact parent: %s\n' "$artifact_parent" >&2
  exit 1
fi
if ! mkdir -m 0700 -- "$run_dir"; then
  printf 'omawake benchmark: cannot create artifact directory: %s\n' "$run_dir" >&2
  exit 1
fi
run_dir=$(cd -- "$run_dir" && pwd -P)
npu_busy_counter=$(find_npu_busy_counter)

{
  printf 'created_utc=%s\n' "$timestamp"
  printf 'repo=%s\nmodel=%s\nbinary=%s\n' \
    "$repo_dir" "$model_dir" "$omawake_bin"
  printf 'openvino_library_path=%s\n' "$openvino_library_path"
  if [[ -f $omawake_bin ]]; then
    sha256sum "$omawake_bin"
  fi
  printf 'cold_warmup=0\ncold_iterations=1\nhot_warmup=%s\nhot_iterations=%s\nthreads=%s\n' \
    "$warmup" "$iterations" "$threads"
  printf 'lanes=%s\naccelerator_model_variant=%s\nnpu_qdq_optimizer=%s\n' \
    "${lanes[*]}" "$accelerator_model_variant" "$npu_qdq_optimizer"
  printf 'npu_busy_counter=%s\n' "${npu_busy_counter:-unavailable}"
  printf 'test_keywords_sha256='
  sha256sum "$model_dir/test_wavs/test_keywords.txt" | awk '{print $1}'
  for wav in "${wav_files[@]}"; do
    sha256sum "$wav"
  done
} >"$run_dir/manifest.txt"
cp -- "$model_dir/test_wavs/test_keywords.txt" "$run_dir/test_keywords.txt"

write_config() {
  local path=$1 runtime=$2 device=$3 profiling_prefix=$4
  local escaped_model escaped_profile encoder decoder joiner
  escaped_model=$(toml_escape "$model_dir")
  escaped_profile=$(toml_escape "$profiling_prefix")
  encoder="encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
  decoder="decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
  joiner="joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
  if [[ $runtime == openvino && $device != cpu ]]; then
    encoder="$accelerator_encoder"
    decoder="$accelerator_decoder"
    joiner="$accelerator_joiner"
  fi
  {
    printf '[backend]\n'
    printf 'kind = "sherpa-onnx"\nruntime = "%s"\ndevice = "%s"\n' "$runtime" "$device"
    printf 'threads = %s\nfallback = "error"\ndevice_id = 0\nprovider_config = ""\n\n' "$threads"
    printf '[backend.options]\n'
    if [[ $runtime == openvino ]]; then
      printf 'ProfilingFilePrefix = "%s"\n' "$escaped_profile"
      if [[ $device == npu ]]; then
        printf 'load_config = "{\\"NPU\\":{\\"NPU_PLATFORM\\":\\"5010\\",\\"NPU_QDQ_OPTIMIZATION\\":\\"%s\\"}}"\n' "$npu_qdq_optimizer"
      fi
    fi
    printf '\n[model]\nname = "%s"\ndirectory = "%s"\n' "$model_id" "$escaped_model"
    printf 'encoder = "%s"\ndecoder = "%s"\njoiner = "%s"\n' "$encoder" "$decoder" "$joiner"
    printf 'tokens = "tokens.txt"\nbpe_model = "bpe.model"\nsample_rate = 16000\n'
    printf 'keywords_score = 1.5\nkeywords_threshold = 0.25\nmax_active_paths = 4\nnum_trailing_blanks = 1\n'
    printf '\n[audio]\ndevice = "default"\nchannels = "mono"\nbuffer_milliseconds = 200\n'
    printf '\n[daemon]\ncooldown_milliseconds = 1500\nqueue_capacity = 8\n'
    printf '\n[[wake_words]]\nid = "light-up"\nphrase = "Light Up"\nenabled = true\ncommand = ["true"]\n'
    printf '\n[[wake_words]]\nid = "lovely-child"\nphrase = "Lovely Child"\nenabled = true\ncommand = ["true"]\n'
    printf '\n[[wake_words]]\nid = "forever"\nphrase = "Forever"\nenabled = true\ncommand = ["true"]\n'
  } >"$path"
  chmod 0600 "$path"
}

write_command() {
  local path=$1 config=$2 binary=$3 runtime=$4 command_warmup=$5 command_iterations=$6
  local config_home=$7 data_home=$8 state_home=$9 runtime_home=${10} working_dir=${11}
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    shell_assign XDG_CONFIG_HOME "$config_home"
    shell_assign XDG_DATA_HOME "$data_home"
    shell_assign XDG_STATE_HOME "$state_home"
    shell_assign XDG_RUNTIME_DIR "$runtime_home"
    if [[ $runtime == openvino ]]; then
      shell_assign LD_LIBRARY_PATH "$openvino_library_path"
    fi
    printf 'cd -- %q\n' "$working_dir"
    printf 'exec %q --config %q benchmark --warmup %q --iterations %q' \
      "$binary" "$config" "$command_warmup" "$command_iterations"
    printf ' %q' "${wav_files[@]}"
    printf '\n'
  } >"$path"
  chmod 0700 "$path"
}

prepare_lane() {
  local lane=$1 runtime=$2 device=$3 binary=$4
  local lane_dir=$run_dir/$lane
  local config_home=$lane_dir/config data_home=$lane_dir/data state_home=$lane_dir/state
  local runtime_home=$lane_dir/runtime profile_dir=$lane_dir/profiles
  local cold_config=$config_home/omawake/cold.toml hot_config=$config_home/omawake/hot.toml
  mkdir -p -- "$config_home/omawake" "$data_home" "$state_home" "$runtime_home" \
    "$profile_dir/cold" "$profile_dir/hot"
  chmod 0700 "$lane_dir" "$config_home" "$config_home/omawake" "$data_home" \
    "$state_home" "$runtime_home" "$profile_dir" "$profile_dir/cold" "$profile_dir/hot"
  write_config "$cold_config" "$runtime" "$device" "$profile_dir/cold/$lane"
  write_config "$hot_config" "$runtime" "$device" "$profile_dir/hot/$lane"
  write_command "$lane_dir/cold.command.sh" "$cold_config" "$binary" "$runtime" 0 1 \
    "$config_home" "$data_home" "$state_home" "$runtime_home" "$lane_dir"
  write_command "$lane_dir/hot.command.sh" "$hot_config" "$binary" "$runtime" \
    "$warmup" "$iterations" "$config_home" "$data_home" "$state_home" "$runtime_home" "$lane_dir"
}

for lane in "${lanes[@]}"; do
  case "$lane" in
    default-cpu) prepare_lane "$lane" default cpu "$omawake_bin" ;;
    openvino-cpu) prepare_lane "$lane" openvino cpu "$omawake_bin" ;;
    openvino-gpu) prepare_lane "$lane" openvino gpu "$omawake_bin" ;;
    openvino-npu) prepare_lane "$lane" openvino npu "$omawake_bin" ;;
  esac
done

if $dry_run; then
  printf 'Prepared benchmark artifacts without running a binary: %s\n' "$run_dir"
  exit 0
fi

failed=0
for lane in "${lanes[@]}"; do
  lane_dir=$run_dir/$lane
  for phase in cold hot; do
    if [[ $lane == openvino-npu && -n $npu_busy_counter ]]; then
      cat -- "$npu_busy_counter" >"$lane_dir/$phase.npu_busy_time_us.before"
    fi

    set +e
    "$lane_dir/$phase.command.sh" >"$lane_dir/$phase.benchmark.json" \
      2>"$lane_dir/$phase.benchmark.stderr.log"
    status=$?
    set -e

    printf '%s\n' "$status" >"$lane_dir/$phase.exit-status.txt"
    provider_config=$(find "$lane_dir/state/omawake/cache/openvino" -type f \
      -name provider.config -print -quit 2>/dev/null || true)
    if [[ -n $provider_config ]]; then
      cp -- "$provider_config" "$lane_dir/$phase.provider.config"
    fi
    validation_status=$status
    if [[ $lane == openvino-* ]]; then
      if grep -aE 'Failed to enable OpenVINO Execution Provider|Fallback to cpu|Device (CPU|GPU|NPU) is not available' \
        "$lane_dir/$phase.benchmark.stderr.log" >"$lane_dir/$phase.provider-fallback.txt"; then
        validation_status=1
      fi
      : >"$lane_dir/$phase.provider-evidence.txt"
      for profile in "$lane_dir/profiles/$phase/"*; do
        [[ -f $profile ]] || continue
        if grep -aFqm1 'OpenVINOExecutionProvider' "$profile"; then
          printf '%s\n' "$profile" >>"$lane_dir/$phase.provider-evidence.txt"
        fi
      done
      if [[ ! -s $lane_dir/$phase.provider-evidence.txt ]]; then
        validation_status=1
      fi
    fi
    expected_iterations=$iterations
    [[ $phase == cold ]] && expected_iterations=1
    : >"$lane_dir/$phase.detection-evidence.txt"
    if jq -e --argjson count "$expected_iterations" '
      (.files | length) == 2 and
      (.files[0].path | endswith("/test_wavs/0.wav")) and
      (.files[1].path | endswith("/test_wavs/1.wav")) and
      (.files[0].iterations | length) == $count and
      (.files[1].iterations | length) == $count and
      all(.files[0].iterations[]; [.detections[].id] == ["light-up"]) and
      all(.files[1].iterations[]; [.detections[].id] == ["lovely-child", "forever"])
    ' "$lane_dir/$phase.benchmark.json" >/dev/null; then
      jq -r '.files[] | .iterations[] | [.iteration, ([.detections[].id] | join(","))] | @tsv' \
        "$lane_dir/$phase.benchmark.json" >"$lane_dir/$phase.detection-evidence.txt"
    else
      validation_status=1
    fi
    if [[ $lane == openvino-npu && -n $npu_busy_counter ]]; then
      cat -- "$npu_busy_counter" >"$lane_dir/$phase.npu_busy_time_us.after"
      before=$(<"$lane_dir/$phase.npu_busy_time_us.before")
      after=$(<"$lane_dir/$phase.npu_busy_time_us.after")
      if [[ $before =~ ^[0-9]+$ && $after =~ ^[0-9]+$ ]]; then
        delta=$((after - before))
        printf '%s\n' "$delta" >"$lane_dir/$phase.npu_busy_time_us.delta"
        if ((delta <= 0)); then
          validation_status=1
        fi
      else
        validation_status=1
      fi
    fi
    printf '%s\n' "$validation_status" >"$lane_dir/$phase.validation-status.txt"
    if ((validation_status != 0)); then
      printf 'omawake benchmark: %s %s failed validation (process status %s); artifacts preserved\n' \
        "$lane" "$phase" "$status" >&2
      failed=1
    fi
  done
  find "$lane_dir/profiles" -type f -print | sort >"$lane_dir/profile-files.txt"
done

printf 'Benchmark artifacts: %s\n' "$run_dir"
exit "$failed"
