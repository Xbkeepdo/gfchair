#!/bin/bash
# Full four-model JFFN-P comparison.  Extraction is image-sharded across two GPUs.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/private/envs/vicr/bin/python}"
CONFIG="${CONFIG:-configs/model_configs_inslen_official_target.yaml}"
GPU0="${GPU0:-cuda:0}"
GPU1="${GPU1:-cuda:1}"
TRAINING_DEVICE="${TRAINING_DEVICE:-$GPU0}"
export DGST_COST_VARIANT_EMD_WORKERS="${DGST_COST_VARIANT_EMD_WORKERS:-8}"
export DGST_FOUR_GATE_PREP_CACHE="${DGST_FOUR_GATE_PREP_CACHE:-1}"
export DGST_FOUR_GATE_EMD_DEDUP="${DGST_FOUR_GATE_EMD_DEDUP:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODELS=(
  llava_1_5_7b
  internvl_2_5_8b
  qwen2_5_vl_7b
  qwen3_vl_8b
)

for model_name in "${MODELS[@]}"; do
  result_dir="outputs/${model_name}/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_p_comparison"
  mkdir -p "$result_dir/logs"

  "$PYTHON_BIN" scripts/run_jffn_p_comparison.py \
    --model "$model_name" --config "$CONFIG" --stage smoke --device "$GPU0" \
    2>&1 | tee "$result_dir/logs/smoke.log"

  "$PYTHON_BIN" scripts/run_jffn_p_comparison.py \
    --model "$model_name" --config "$CONFIG" --stage calibrate --device "$GPU0" \
    2>&1 | tee "$result_dir/logs/calibrate.log"

  "$PYTHON_BIN" scripts/run_jffn_p_comparison.py \
    --model "$model_name" --config "$CONFIG" --stage extract \
    --device "$GPU0" --rank 0 --world-size 2 \
    >"$result_dir/logs/extract_rank00.log" 2>&1 &
  rank0_pid=$!
  "$PYTHON_BIN" scripts/run_jffn_p_comparison.py \
    --model "$model_name" --config "$CONFIG" --stage extract \
    --device "$GPU1" --rank 1 --world-size 2 \
    >"$result_dir/logs/extract_rank01.log" 2>&1 &
  rank1_pid=$!

  rank0_status=0
  rank1_status=0
  wait "$rank0_pid" || rank0_status=$?
  wait "$rank1_pid" || rank1_status=$?
  if (( rank0_status != 0 || rank1_status != 0 )); then
    echo "${model_name}: extraction failed (rank0=${rank0_status}, rank1=${rank1_status})" >&2
    exit 1
  fi

  "$PYTHON_BIN" scripts/run_jffn_p_comparison.py \
    --model "$model_name" --config "$CONFIG" --stage train \
    --training-device "$TRAINING_DEVICE" \
    2>&1 | tee "$result_dir/logs/train.log"
  "$PYTHON_BIN" scripts/run_jffn_p_comparison.py \
    --model "$model_name" --config "$CONFIG" --stage analyze \
    2>&1 | tee "$result_dir/logs/analyze.log"
done

"$PYTHON_BIN" scripts/run_jffn_p_comparison.py \
  --model all --config "$CONFIG" --stage summarize
