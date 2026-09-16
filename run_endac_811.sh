#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$ROOT/configs/model_configs_endac_811.yaml"
OUT="$ROOT/outputs/coco4000_512_endac_811"
STAGE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$OUT/logs"

run_pair() {
  local script="$1" model_a="$2" gpu_a="$3" model_b="$4" gpu_b="$5" suffix="$6"
  "$PYTHON_BIN" "$script" --model "$model_a" --config "$CONFIG" --device "cuda:$gpu_a" >"$OUT/logs/${model_a}_${suffix}.log" 2>&1 &
  local pid_a=$!
  "$PYTHON_BIN" "$script" --model "$model_b" --config "$CONFIG" --device "cuda:$gpu_b" >"$OUT/logs/${model_b}_${suffix}.log" 2>&1 &
  local pid_b=$!
  local status=0
  wait "$pid_a" || status=$?
  wait "$pid_b" || status=$?
  return "$status"
}

if [[ "$STAGE" == "label" || "$STAGE" == "all" ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/prepare_endac_811.py" --config "$CONFIG"
fi

if [[ "$STAGE" == "extract" || "$STAGE" == "all" ]]; then
  run_pair "$ROOT/scripts/extract_endac_811.py" qwen2_5_vl_7b 0 llava_1_5_7b 1 extract
  run_pair "$ROOT/scripts/extract_endac_811.py" qwen3_vl_8b 0 internvl_2_5_8b 1 extract
fi

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  run_pair "$ROOT/scripts/train_endac_811.py" qwen2_5_vl_7b 0 llava_1_5_7b 1 train
  run_pair "$ROOT/scripts/train_endac_811.py" qwen3_vl_8b 0 internvl_2_5_8b 1 train
  "$PYTHON_BIN" "$ROOT/scripts/train_endac_811.py" --model all --config "$CONFIG" --stage summary
fi

if [[ "$STAGE" != "label" && "$STAGE" != "extract" && "$STAGE" != "train" && "$STAGE" != "all" ]]; then
  echo "usage: $0 [label|extract|train|all]" >&2
  exit 2
fi
