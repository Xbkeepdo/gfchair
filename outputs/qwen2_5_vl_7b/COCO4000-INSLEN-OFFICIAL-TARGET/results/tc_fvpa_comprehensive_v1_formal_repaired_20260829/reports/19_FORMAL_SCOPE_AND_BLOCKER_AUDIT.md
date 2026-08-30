# Formal scope and blocker audit

Status timestamp: 2026-08-29 UTC. This audit distinguishes measured development smoke evidence from the preregistered formal study. Smoke rows are never counted toward a formal conclusion.

## Measured in this execution

| Model | Path scope | Path status | Measured timing | True-FP32 rows | Shapley |
|---|---:|---|---:|---:|---:|
| LLaVA-1.5-7B | 1 image, 1 HALL target, layer 32, both quadratures, K=1/4, 3 scalars | PASS, smoke only | legacy stage 12.725 s / case 4.888 s; optimized stage 7.902 s / case 0.361 s | 336, final block, PASS | 3 scalar rows, 8 regions x 8 permutations, smoke only |
| InternVL2.5-8B | 1 image, 1 HALL target, layer 32, both quadratures, K=1/4, 3 scalars | PASS, smoke only | legacy stage 248.028 s / case 3.828 s; optimized stage 11.196 s / case 1.041 s | 336, final block, PASS | NOT RUN |
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

## Runtime evidence and correction

The earlier audit incorrectly treated the InternVL stage elapsed time (248.028 s, dominated by one-time initialization/shared-storage I/O) as a per-case path runtime. The persisted legacy case runtime was 3.828 s, so the former `55.1 GPU-hours` lower bound is invalid and is retracted.

The runner now separates the 500-image local-only cohort from the 200-image finite-path cohort, caches the already-computed alpha=0/1 endpoints, and uses the exact final-block suffix `z + replacement -> final norm -> LM head` after real-model parity checks. On the same one-case K=1/4 smoke shape, optimized case times were 0.361 s for LLaVA and 1.041 s for InternVL. LLaVA FP16 full-vocabulary parity error was 0.0078125. InternVL BF16 parity had full-vocabulary/selected-logit/scalar maximum errors `0.125/0/0.001483` with matching argmax, within its recorded 0.25 BF16 tolerance.

These final-layer smoke timings still do not determine the formal four-layer K=1/4/8/16/32 runtime: lower layers retain parity-safe full multimodal replay, storage I/O varies sharply, and no formal-scale benchmark has run. Consequently this audit reports formal runtime as unmeasured instead of replacing the bad estimate with another extrapolation. No reduced run is relabeled formal.

## Other blockers

- True-FP32 Route A is validated only for the final decoder block; lower-layer downstream adapters are blocked.
- Fixed-QK zeroing, full activation patching, and end-to-end pixel counterfactuals are not implemented/measured.
- Real neuron interventions, train-only geometry calibration, hallucination detectors, paired 10,000-replicate bootstrap, and external QA are not measured.
- `pandas`, `pyarrow`, and `fastparquet` are absent in both available Python environments. CSV.GZ and PT artifacts exist; Parquet outputs are explicitly `BLOCKED`, not silently omitted.

## Test audit

- Directed pre-existing JFFN/Jacobian regression set: 28/28 PASS.
- New TC-FVPA/path/artifact/loader/Shapley/SwiGLU/geometry/analysis tests: 37/37 PASS.
- Full repository discovery: 336 total, 312 passed, 12 failed, 12 errored. None of the 24 failures is in a new TC-FVPA module. The errors/failures reproduce unrelated active-config/manifest expectations and missing NLTK `punkt` data (including 7 InsLen tokenization errors), as already separated from the directed scientific regression set. They remain visible rather than being relabeled PASS.

## Exact formal resume entry point

```bash
cd /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python bash run_tc_fvpa_comprehensive.sh
```

This launcher freezes models/layers/scalars/integration points, schedules independent 500-image local and 200-image path cohorts across two shards/devices, supports resume, and writes per-model formal roots. It should be run only after a representative four-layer benchmark and completion of Qwen/lower-layer parity work.
