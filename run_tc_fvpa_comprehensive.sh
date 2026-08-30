#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-llava_1_5_7b}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
NUM_SHARDS="${NUM_SHARDS:-2}"

if ! "${PYTHON_BIN}" -c 'import torch, yaml' >/dev/null 2>&1; then
  echo "PYTHON_BIN=${PYTHON_BIN} lacks torch/PyYAML; set PYTHON_BIN to the experiment environment." >&2
  exit 2
fi

exec "${PYTHON_BIN}" scripts/run_tc_fvpa_comprehensive.py \
  --model "${MODEL}" \
  --device "${DEVICES%%,*}" \
  --devices "${DEVICES}" \
  --num-shards "${NUM_SHARDS}" \
  --layers 8,16,24,32 \
  --target-scalars log_probability,margin,logit \
  --integration-points 1,4,8,16,32 \
  --formal \
  --resume \
  "$@"
