# TC-FVPA runbook

Use the repository root and the existing experiment environment:

```bash
cd /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
export PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python
```

## Inspect the frozen formal plan

```bash
$PYTHON_BIN scripts/run_tc_fvpa_comprehensive.py \
  --model llava_1_5_7b \
  --device cuda:0 --devices cuda:0,cuda:1 \
  --num-shards 2 --layers 8,16,24,32 \
  --target-scalars log_probability,margin,logit \
  --integration-points 1,4,8,16,32 \
  --formal --resume --dry-run
```

## Development smoke

Keep smoke artifacts in a root whose name says `smoke`; do not merge them into a formal root.

```bash
OUT=outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke
$PYTHON_BIN scripts/run_tc_fvpa_path_attribution.py \
  --model llava_1_5_7b --device cuda:0 --devices cuda:0 \
  --output-dir "$OUT" --smoke --num-images 1 \
  --max-targets-per-image 1 --layers 32 \
  --target-scalars log_probability,margin,logit \
  --integration-points 1,4 --quadrature trapezoid,gauss_legendre \
  --attribution-mode path
$PYTHON_BIN scripts/run_tc_fvpa_path_attribution.py \
  --model llava_1_5_7b --device cuda:0 --devices cuda:0 \
  --output-dir "$OUT" --smoke --num-images 1 \
  --max-targets-per-image 1 --layers 32 \
  --target-scalars log_probability,margin,logit \
  --integration-points 1,4 --attribution-mode local \
  --audit-vector-cases 0
$PYTHON_BIN scripts/run_tc_fvpa_fp32_causal.py \
  --model llava_1_5_7b --device cuda:0 --devices cuda:0 \
  --output-dir "$OUT" --path-result-dir "$OUT" --smoke \
  --num-images 1 --max-targets-per-image 1 --layers 32 \
  --target-scalars log_probability,margin,logit --integration-points 1,4
```

## Formal execution and resume

The launcher is idempotent at completed shard boundaries. It runs a 500-image local-only cohort and a separate 200-image path/LOO cohort; do not replace these with one 500-image path run. Formal low-precision path execution keeps `path_batch_size=1` so BF16/FP16 GEMM shapes remain comparable with the completed Qwen2 run. Batch 2 is an explicit exploratory throughput option, not the current formal default.

```bash
MODEL=llava_1_5_7b DEVICES=cuda:0,cuda:1 NUM_SHARDS=2 \
PYTHON_BIN="$PYTHON_BIN" bash run_tc_fvpa_comprehensive.sh

MODEL=internvl_2_5_8b DEVICES=cuda:0,cuda:1 NUM_SHARDS=2 \
PYTHON_BIN="$PYTHON_BIN" bash run_tc_fvpa_comprehensive.sh

MODEL=qwen2_5_vl_7b DEVICES=cuda:0,cuda:1 NUM_SHARDS=2 \
PYTHON_BIN="$PYTHON_BIN" bash run_tc_fvpa_comprehensive.sh

MODEL=qwen3_vl_8b DEVICES=cuda:0,cuda:1 NUM_SHARDS=2 \
PYTHON_BIN="$PYTHON_BIN" bash run_tc_fvpa_comprehensive.sh
```

The Qwen2.5 and Qwen3 causal-prefix adapters now pass their real-model gates. Qwen2 retains a per-case suffix-gradient gate because acceptance varies with the prefix. Qwen3 layers 9/18/27 use the preregistered, prevalidated full-row suffix route while retaining per-case clean-logit, selected-logit, scalar, and argmax checks; layer 36 uses the exact final-block suffix. A failed gate must fall back to exact replay or persist a failure, never silently accept an approximate route.

As of 2026-08-30, Qwen2 has completed the currently implemented formal stages. Qwen3 local Riesz is complete and its path stage is running; LLaVA and InternVL remain queued. The current launcher does not run POPE, CLEVR, or AMBER. Fixed-QK, full activation patching, pixel counterfactuals, formal detection/bootstrap, real neuron intervention, and several geometry controls remain explicitly `BLOCKED` or `NOT RUN` until implemented.

## Analyze and load

```bash
$PYTHON_BIN scripts/analyze_tc_fvpa_comprehensive.py \
  --model llava_1_5_7b --device cuda:0 --devices cuda:0,cuda:1 \
  --output-dir "$OUT" --layers 32 \
  --target-scalars log_probability,margin,logit --integration-points 1,4
$PYTHON_BIN scripts/build_tc_fvpa_reports.py \
  --model llava_1_5_7b --device cuda:0 --devices cuda:0 \
  --output-dir "$OUT" --layers 32 \
  --target-scalars log_probability,margin,logit --integration-points 1,4
$PYTHON_BIN scripts/load_tc_fvpa_results.py "$OUT" \
  --model llava_1_5_7b --layer 32 \
  --method PATH_LOG_PROBABILITY_GAUSS_LEGENDRE_K4
```

The verified loader requires `manifests/output_checksums.json`, which the analysis/report phases generate. Status and exact resume commands live in `manifests/run_status.json`. PT/CSV.GZ are authoritative when Parquet dependencies are unavailable.
