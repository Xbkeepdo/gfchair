# HANDOFF TO CHATGPT

Repository commit: `dd3c1ecec02c37bac23687959a0508b0b53a452f`

Modified/untracked files at final report build:

```text
## main...origin/main
 M README.md
 M docs/CURRENT_TASK.md
?? docs/TC_FVPA_RUNBOOK.md
?? features/ffn_visual_interactions.py
?? features/ffn_visual_path_attribution.py
?? features/representation_geometry_controls.py
?? features/swiglu_visual_mechanism.py
?? features/tc_fvpa_artifacts.py
?? run_tc_fvpa_comprehensive.sh
?? scripts/analyze_tc_fvpa_comprehensive.py
?? scripts/build_tc_fvpa_reports.py
?? scripts/load_tc_fvpa_results.py
?? scripts/run_tc_fvpa_comprehensive.py
?? scripts/run_tc_fvpa_counterfactuals.py
?? scripts/run_tc_fvpa_fp32_causal.py
?? scripts/run_tc_fvpa_path_attribution.py
?? scripts/run_tc_fvpa_shapley.py
?? scripts/tc_fvpa_common.py
?? tests/test_ffn_visual_interactions.py
?? tests/test_ffn_visual_path_attribution.py
?? tests/test_load_tc_fvpa_results.py
?? tests/test_representation_geometry_controls.py
?? tests/test_swiglu_visual_mechanism.py
?? tests/test_tc_fvpa_analysis.py
?? tests/test_tc_fvpa_artifacts.py
```

Model: `qwen2_5_vl_7b`

Scientific verdict: **NOT RUN**

## Core equations and exact meanings

- `a_m = sum_h W_O^h(alpha_hm v_hm)`: clean-attention conditional source write; output bias is added once only to the full attention reconstruction.
- `delta_m = J_G(z) a_m`, where `G=FFN(Norm(z))` excludes the residual identity.
- `g = grad_m S(m_0)` is the Euclidean Riesz representative of the downstream scalar differential; `r=J_G(z)^T g` is its pre-FFN pullback.
- `C_m^local = g^T delta_m = r^T a_m` may be positive or negative.
- `e_m^path = integral J_G(z0+alpha A)a_m d alpha`, with `z0=z-A`; this baseline is not no-image.
- `ATTN`, `WRITE=||a_m||`, `JFFN=||J_G a_m||`, and `GAIN=JFFN/WRITE` are non-target baselines. `SIGNED_Q`, local Riesz, and path Riesz retain their sign in raw storage.

## Denominators

```json
{
  "fp32_intervention_rows": 0,
  "frozen_write_rows": 0,
  "images": 0,
  "labels": {},
  "layers": [],
  "model": "qwen2_5_vl_7b",
  "path_convergence_rows": 0,
  "shapley_rows": 0,
  "spatial_metric_rows": 0,
  "target_layer_cases": 0
}
```

## Persisted numerical summaries

```json
{
  "comprehensive": {},
  "counterfactuals": {},
  "fp32": {},
  "shapley": {},
  "validation": {}
}
```

## Experiment status

```json
{
  "completed_shards": [],
  "created_unix": 1787960511.6825135,
  "failed_cases": [],
  "failed_shards": [],
  "resume_commands": [
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_path_attribution.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0 --smoke --num-images 1 --max-targets-per-image 1 --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4 --quadrature trapezoid,gauss_legendre",
    "/opt/conda/private/envs/vicr/bin/python scripts/build_tc_fvpa_reports.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0 --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4"
  ],
  "schema_version": "tc-fvpa-comprehensive-v1",
  "stages": {
    "path_attribution:qwen2_5_vl_7b:shard0": {
      "details": {
        "error": "qwen2_5_vl_7b path extraction is BLOCKED: the Qwen causal-prefix branch-replacement adapter has not passed real-model parity."
      },
      "started_unix": 1787960512.030787,
      "status": "BLOCKED",
      "updated_unix": 1787960512.6211154
    },
    "reports:qwen2_5_vl_7b": {
      "details": {
        "reports": 22
      },
      "started_unix": 1787960861.0983613,
      "status": "PASS",
      "updated_unix": 1787961366.8861136
    }
  },
  "updated_unix": 1787961366.8861136
}
```

## Negative results and limitations

Existing pre-TC-FVPA evidence shows that WRITE explains most JFFN token ranking and JFFN does not exceed WRITE spatially on LLaVA/InternVL. This handoff does not extrapolate to Qwen. True-FP32 downstream is currently valid only for the final block; lower-layer FP32, fixed-QK, activation/pixel counterfactual, Shapley, neuron intervention, geometry, formal detection and external QA remain explicitly incomplete unless their reports say otherwise.

No confidence interval or detector metric is reported for this development cohort: it has too few independent images/classes for the preregistered 10,000-replicate image-cluster bootstrap. No scientific figure is promoted from this underpowered cohort; inspect the numerical smoke artifacts in `tables/` instead.

## Fixed and unresolved implementation issues

Fixed in this implementation: the target never enters its causal prefix; the margin competitor is frozen from clean logits; output-projection bias is assigned once; local JVP/VJP duality and path completeness are test-covered; negative contributions are persisted; the final-block causal route executes FFN, final norm, and LM head in actual FP32; shards and checksums are atomic and deduplicated.

Unresolved: Qwen adapter parity, lower-layer true-FP32 downstream execution, fixed-QK/full-activation/pixel estimands, real neuron intervention, train-only cohort geometry calibration, formal spatial/detection statistics, and external QA. Parquet is blocked by missing optional dependencies; PT and CSV.GZ remain authoritative.

## Data paths and loader

- Case table: `tables/case_layer.csv.gz` (Parquet only when the environment supports it).
- Token maps: `shards/token_maps_rankXX_shard_XXXXX.pt`.
- Interventions: `tables/interventions.csv.gz` when measured.
- Loader: repository `scripts/load_tc_fvpa_results.py`; run `python scripts/load_tc_fvpa_results.py /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1 --model qwen2_5_vl_7b --method WRITE` after output checksums exist.
- Complete status and resume commands: `manifests/run_status.json`.

No missing measurement is replaced by zero or inference.
