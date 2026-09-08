# Formal scope and blocker audit

Status timestamp: 2026-08-29 UTC. This audit distinguishes measured development smoke evidence from the preregistered formal study. Smoke rows are never counted toward a formal conclusion.

## Measured in this execution

| Model | Path scope | Path status | Path elapsed | True-FP32 rows | Shapley |
|---|---:|---|---:|---:|---:|
| LLaVA-1.5-7B | 1 image, 1 HALL target, layer 32, both quadratures, K=1/4, 3 scalars | PASS, smoke only | 12.725 s | 336, final block, PASS | 3 scalar rows, 8 regions x 8 permutations, smoke only |
| InternVL2.5-8B | 1 image, 1 HALL target, layer 32, both quadratures, K=1/4, 3 scalars | PASS, smoke only | 248.028 s | 336, final block, PASS | NOT RUN |
| Qwen2.5-VL-7B | no measured case | BLOCKED before model load | n/a | NOT RUN | NOT RUN |
| Qwen3-VL-8B | no measured case | BLOCKED before model load | n/a | NOT RUN | NOT RUN |

The two Qwen commands both exited 1 and persisted `BLOCKED`: their dynamic visual-token / causal-prefix branch-replacement adapters have not passed real-model parity. LLaVA/InternVL evidence must not be extrapolated to them.

## Formal minimum versus completion

| Family, per model | Required minimum | Formally completed here | Status |
|---|---:|---:|---|
| Local Riesz | 500 images, >=2,000 target-layer cases, layers 8/16/24/32 | 0 | NOT RUN |
| Path | 200 images, >=800 target-layer cases, K=1/4/8/16/32 | 0 | NOT RUN |
| High-precision causal | 100 images, >=400 target-layer cases | 0 | NOT RUN; final-layer smoke only |
| Region full counterfactual | 100 images, >=400 cases | 0 | BLOCKED/NOT RUN; only frozen-write conditional estimand exists |
| Shapley | >=50 target-layer cases, 8/16 regions, >=128 permutations | 0 | NOT RUN; one LLaVA smoke case only |
| Detection / spatial bootstrap | full cohort, image splits, 10,000 cluster resamples | 0 | NOT RUN |
| POPE/CLEVR/AMBER | specified QA protocols | 0 | NOT RUN |

## Runtime evidence

Even the reduced final-layer K=1/4 smoke took 248.028 seconds per InternVL case without FlashAttention2. A lower bound that unrealistically assumes no extra cost for four layers or K=8/16/32 is `800 x 248.028 = 55.1 GPU-hours` for InternVL path alone. The formal grid has many more unique downstream-gradient points, so the actual cost is materially higher. LLaVA's corresponding lower bound is 2.83 GPU-hours before that multiplier. No reduced run is relabeled formal.

## Other blockers

- True-FP32 Route A is validated only for the final decoder block; lower-layer downstream adapters are blocked.
- Fixed-QK zeroing, full activation patching, and end-to-end pixel counterfactuals are not implemented/measured.
- Real neuron interventions, train-only geometry calibration, hallucination detectors, paired 10,000-replicate bootstrap, and external QA are not measured.
- `pandas`, `pyarrow`, and `fastparquet` are absent in both available Python environments. CSV.GZ and PT artifacts exist; Parquet outputs are explicitly `BLOCKED`, not silently omitted.

## Test audit

- Directed pre-existing JFFN/Jacobian regression set: 28/28 PASS.
- New TC-FVPA/path/artifact/loader/Shapley/SwiGLU/geometry/analysis tests: 34/34 PASS.
- Full repository discovery: 333 total, 309 passed, 12 failed, 12 errored. None of the 24 failures is in a new TC-FVPA module. The errors/failures reproduce unrelated active-config/manifest expectations and missing NLTK `punkt` data (including 7 InsLen tokenization errors), as already separated from the directed scientific regression set. They remain visible rather than being relabeled PASS.

## Exact formal resume entry point

```bash
cd /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python bash run_tc_fvpa_comprehensive.sh
```

This launcher freezes models/layers/scalars/integration points, uses two shards/devices, supports resume, and writes per-model formal roots. It should be run only after budgeting the measured runtime and completing Qwen/lower-layer parity work.
