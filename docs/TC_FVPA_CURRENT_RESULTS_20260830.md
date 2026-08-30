# TC-FVPA current results handoff — 2026-08-30

This is a deliberately partial handoff for independent review. It separates completed measurements from queued work and from experiment families that are not implemented. It must not be read as a four-model final verdict.

## Executive status

- Qwen2.5-VL-7B has completed all currently runnable coordinator phases: local Riesz, path/LOO, true-FP32 causal intervention, final-layer Shapley, analysis, frozen-write counterfactual export, and reports.
- Qwen3-VL-8B local Riesz is complete; formal path is running on two RTX 3090 GPUs.
- The new LLaVA-1.5-7B and InternVL2.5-8B formal TC-FVPA runs are queued after Qwen3.
- POPE/CLEVR/AMBER VQA generalization is intentionally not in the active queue.
- Fixed-QK value zeroing, full activation patching, end-to-end pixel counterfactuals, real neuron interventions, formal hallucination detectors, 10,000-replicate detection/spatial bootstrap, and several geometry controls remain `BLOCKED` or `NOT RUN`.

The strongest current scientific statement is therefore within-model and provisional: Qwen2 path integration improves current-block completeness and true-FP32 interventions support local linear predictions, while token-level spatial localization still does not improve over the simpler WRITE baseline.

## Completed Qwen2 formal scope

| Family | Persisted scope | Status |
|---|---:|---|
| Local Riesz | 500 images, 3,496 target-layer cases, layers 7/14/21/28 | PASS, 0 failures |
| Path/LOO | 200 images, 1,316 target-layer cases, K=1/4/8/16/32, two quadratures | PASS, 0 failures |
| True-FP32 intervention | 216,384 rows from 100 images | PASS, 0 failures |
| Frozen-write counterfactual export | 43,428 rows | PASS for this conditional estimand |
| Final-layer Shapley | 300 scalar rows, 128 permutations | PASS, 0 failures |
| Formal hallucination detector | no persisted result | NOT RUN |
| External VQA | no persisted result | NOT RUN |

The local cohort contains 2,348 REAL and 1,148 HALL target-layer rows. The path cohort is a separate preregistered 200-image subset and must not be confused with the 500-image local cohort.

## Qwen2 numerical findings

### Path completeness

K=32 lowers the median absolute completeness error relative to the clean-endpoint K=1 approximation:

| Scalar | K=1 median absolute error | K=32 median absolute error | Reduction |
|---|---:|---:|---:|
| target log-probability | 0.022123 | 0.017965 | 18.79% |
| fixed-competitor margin | 0.095032 | 0.082031 | 13.68% |
| target logit | 0.142151 | 0.126892 | 10.73% |

The table uses trapezoid K=32; Gauss-Legendre gives nearly identical reductions. Residual error remains nonzero, so the result supports an improvement over a point Jacobian rather than perfect finite-effect recovery.

For frozen-write finite effects, aggregate visual WRITE is strongly associated with the observed conditional intervention: Pearson is 0.875 for log-probability, 0.906 for margin, and 0.732 for logit. Top-32 positive path attribution gives Pearson 0.792/0.662/0.415 respectively; a random-token control gives only 0.098/0.078/0.045. These are descriptive correlations on 1,316 cases, not a substitute for full image/pixel causality.

### True-FP32 local causal validation

At symmetric intervention scale `eta=0.025`, aggregate visual-response interventions give:

| Scalar | Sign agreement | Median relative error | Zero-observed fraction |
|---|---:|---:|---:|
| target log-probability | 97.56% | 1.06% | 1.14% |
| target logit | 99.35% | 0.64% | 0% |
| margin | 99.35% | 0.99% | 0% |

This materially improves on the earlier FP16/BF16 intervention study, where most small effects were quantized to zero. The result validates local prediction for the measured FFN-output directions; it does not establish a complete image-removal causal effect.

### Spatial localization

Descriptive REAL-box averages show no Qwen2 spatial advantage over WRITE:

| Method | Top-1 pointing | Patch AUPRC | BBox mass |
|---|---:|---:|---:|
| WRITE | 0.54330 | 0.46846 | 0.39058 |
| JFFN | 0.52098 | 0.46356 | 0.38731 |
| signed Q | 0.51295 | 0.47000 | 0.40933 |
| path log-probability, GL K=32 | 0.50687 | 0.42877 | 0.38625 |

Signed Q has a mixed profile (higher AUPRC/BBox mass but lower Top-1). Formal paired image-cluster bootstrap has not been run for this Qwen2 cohort, so no significance claim is made.

## Relationship to the completed earlier JFFN study

The already published LLaVA/InternVL second-round study concluded Outcome C:

- WRITE explains most JFFN token ranking; JFFN does not improve paired spatial localization over WRITE.
- Adding Jacobian sensitivity S conditional on WRITE magnitude I does not give a stable hallucination-detection gain: LLaVA AUROC delta is +0.02317 with 95% CI [-0.00300,+0.04917], and InternVL is +0.00150 with CI [-0.00043,+0.00330].
- Signed and target-aligned quantities remain useful mechanistic diagnostics, but their detection and localization gains do not replicate consistently.

The current Qwen2 spatial result is qualitatively consistent with Outcome C. A four-model TC-FVPA conclusion must wait for Qwen3/LLaVA/InternVL completion and formal paired statistics.

## Four-model safe optimization validation

On the same two-image, 12-case, four-layer, full-K workload with conservative `path_batch_size=1`, optimized versus legacy case runtime is:

| Model | Legacy | Optimized | Reduction | Token-map equivalence |
|---|---:|---:|---:|---|
| Qwen3-VL-8B | 313.393 s | 293.604 s | 6.31% | 528/528 exact |
| Qwen2.5-VL-7B | 288.991 s | 274.670 s | 4.96% | 528/528 exact |
| InternVL2.5-8B | 422.307 s | 405.685 s | 3.94% | 528/528 exact |
| LLaVA-1.5-7B | 579.272 s | 563.347 s | 2.75% | 528/528 exact |

All four runs measured 12/12 cases with zero failures, and frozen-write LOO effects were exactly preserved. Exploratory low-precision path batching is not used for the formal runs because it changes BF16 GEMM shapes and produces small numerical differences.

## Interpretation boundaries and known gaps

1. The source-wise write is conditional on the observed clean attention decomposition. Frozen-write removal is not the same estimand as fixed-QK value zeroing, activation patching, or a pixel counterfactual.
2. Qwen2 reports are `PARTIAL` at the study level because cross-model, detection, external QA, and several causal families remain incomplete even though the runnable Qwen2 shards passed.
3. The generated `02_EXISTING_PIPELINE_AUDIT.md` and `19_FORMAL_SCOPE_AND_BLOCKER_AUDIT.md` are timestamped pre-run audits. Their statements that Qwen had no measured formal cases describe the state before the repaired execution; `manifests/run_status.json`, the metrics files, and this handoff describe the later persisted state.
4. CSV.GZ and PT are authoritative. Parquet export is explicitly blocked by absent optional pandas/pyarrow dependencies and is not interpreted as missing measurement.
5. Large token-map, audit-vector, and FP32 shard files remain outside Git. The public snapshot contains compact case tables, convergence/intervention tables, manifests, metrics, schemas, and reports sufficient to inspect the stated results.

## Review entry points

- `jffn_second_round_incremental_validation_report.md`: completed earlier LLaVA/InternVL scientific report.
- `docs/TC_FVPA_RUNBOOK.md`: execution and interpretation rules.
- `outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829/manifests/run_status.json`: authoritative stage status.
- The same Qwen2 root's `metrics/` and `tables/`: compact summaries and reanalysis tables.
- `docs/CURRENT_TASK.md`: chronological implementation, failures, fixes, and active queue history.
