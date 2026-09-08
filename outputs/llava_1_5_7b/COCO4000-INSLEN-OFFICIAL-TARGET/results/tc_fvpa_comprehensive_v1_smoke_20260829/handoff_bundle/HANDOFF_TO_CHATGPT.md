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

Model: `llava_1_5_7b`

Scientific verdict: **PARTIAL: mechanistic local/path measurements exist, but the complete four-model causal/spatial/detection decision is not available.**

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
  "fp32_intervention_rows": 336,
  "frozen_write_rows": 33,
  "images": 1,
  "labels": {
    "HALL": 1,
    "REAL": 0
  },
  "layers": [
    32
  ],
  "model": "llava_1_5_7b",
  "path_convergence_rows": 12,
  "shapley_rows": 3,
  "spatial_metric_rows": 0,
  "target_layer_cases": 1
}
```

## Persisted numerical summaries

```json
{
  "comprehensive": {
    "case_rows": 1,
    "counterfactual_rows": 33,
    "images": 1,
    "labels": {
      "HALL": 1,
      "REAL": 0
    },
    "layers": [
      32
    ],
    "model": "llava_1_5_7b",
    "parquet": {
      "case_layer": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 1,
        "status": "BLOCKED"
      },
      "path_convergence": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 12,
        "status": "BLOCKED"
      },
      "spatial_metrics": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 0,
        "status": "BLOCKED"
      },
      "token_scores_sample": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 12096,
        "status": "BLOCKED"
      }
    },
    "path_convergence_rows": 12,
    "path_vs_frozen_write": {
      "log_probability:aggregate_visual_write": {
        "count": 1,
        "mae": 0.0041245222091674805,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:highest_attention": {
        "count": 1,
        "mae": 0.0030167698860168457,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:highest_jffn": {
        "count": 1,
        "mae": 0.0010988116264343262,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:highest_local_riesz": {
        "count": 1,
        "mae": 0.0030167698860168457,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:highest_path": {
        "count": 1,
        "mae": 0.0030167698860168457,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:highest_write": {
        "count": 1,
        "mae": 0.0010988116264343262,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:most_negative_path": {
        "count": 1,
        "mae": 0.0037689208984375,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:random_token": {
        "count": 1,
        "mae": 0.00010353326797485352,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:top16_positive_path": {
        "count": 1,
        "mae": 0.0005618929862976074,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:top32_positive_path": {
        "count": 1,
        "mae": 0.0015391111373901367,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:top4_positive_path": {
        "count": 1,
        "mae": 0.001337885856628418,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "logit:aggregate_visual_write": {
        "count": 1,
        "mae": 0.009460806846618652,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "logit:highest_attention": {
        "count": 1,
        "mae": 0.0020389556884765625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_jffn": {
        "count": 1,
        "mae": 0.005374908447265625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_local_riesz": {
        "count": 1,
        "mae": 0.0020389556884765625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_path": {
        "count": 1,
        "mae": 0.0020389556884765625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_write": {
        "count": 1,
        "mae": 0.005374908447265625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:most_negative_path": {
        "count": 1,
        "mae": 0.004150390625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:random_token": {
        "count": 1,
        "mae": 0.00012373924255371094,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:top16_positive_path": {
        "count": 1,
        "mae": 0.0011481642723083496,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:top32_positive_path": {
        "count": 1,
        "mae": 0.002160012722015381,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:top4_positive_path": {
        "count": 1,
        "mae": 0.0037050247192382812,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:aggregate_visual_write": {
        "count": 1,
        "mae": 0.006161689758300781,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "margin:highest_attention": {
        "count": 1,
        "mae": 0.006918430328369141,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "margin:highest_jffn": {
        "count": 1,
        "mae": 0.001251220703125,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:highest_local_riesz": {
        "count": 1,
        "mae": 0.006918430328369141,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "margin:highest_path": {
        "count": 1,
        "mae": 0.006918430328369141,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "margin:highest_write": {
        "count": 1,
        "mae": 0.001251220703125,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:most_negative_path": {
        "count": 1,
        "mae": 0.008835792541503906,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:random_token": {
        "count": 1,
        "mae": 7.510185241699219e-05,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:top16_positive_path": {
        "count": 1,
        "mae": 0.0046694278717041016,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "margin:top32_positive_path": {
        "count": 1,
        "mae": 0.0041214823722839355,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "margin:top4_positive_path": {
        "count": 1,
        "mae": 0.0019168853759765625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      }
    },
    "shard_audit": {
      "failures": [],
      "schema_version": "tc-fvpa-comprehensive-v1",
      "shards": [
        {
          "failures": 0,
          "path": "shards/case_rank00_shard_00000.pt",
          "rows": 1,
          "sha256": "036f0c87d90ae6fc9b6adc12bc19fda686cacbdfe6eb27939efcaa759037dc8c"
        }
      ],
      "unique_cases": 1
    },
    "spatial_metric_rows": 0,
    "token_score_sample_rows": 12096
  },
  "counterfactuals": {
    "blocked_or_not_run": [
      "fixed_qk",
      "activation_patching",
      "pixel_counterfactual"
    ],
    "family_status": {
      "activation_patching": "NOT_RUN",
      "fixed_qk": "NOT_RUN",
      "frozen_write": "PASS",
      "pixel_counterfactual": "NOT_RUN"
    },
    "important_estimand_note": "Frozen-write effects condition on the observed clean attention decomposition and are not complete image-patch effects.",
    "model": "llava_1_5_7b",
    "resume_command": "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
    "rows": 33
  },
  "fp32": {
    "blocked_layer_rows": 0,
    "elapsed_seconds": 7.853467264038045,
    "failures": [],
    "measured_rows": 336,
    "model": "llava_1_5_7b",
    "protocol": "tc_fvpa_true_fp32_final_block_v1"
  },
  "shapley": {
    "blocked_layers": [],
    "failures": [],
    "measured_rows": 3,
    "model": "llava_1_5_7b",
    "permutations": 8
  },
  "validation": {
    "artifact_loader_analysis_tests_passed": 11,
    "geometry_tests_passed": 6,
    "path_riesz_tests_passed": 10,
    "repository_suite": {
      "errors": 12,
      "failures": 12,
      "note": "Failures are pre-existing configuration/manifest/NLTK-data issues outside TC-FVPA; see formal scope audit.",
      "passed": 309,
      "tc_fvpa_regressions": 0,
      "total": 333
    },
    "scope": "synthetic/unit tests; not formal real-model evidence",
    "shapley_interaction_tests_passed": 4,
    "status": "PASS",
    "swiglu_tests_passed": 3,
    "tests_failed": 0,
    "tests_passed": 34
  }
}
```

## Experiment status

```json
{
  "completed_shards": [],
  "created_unix": 1787959208.1028955,
  "failed_cases": [],
  "failed_shards": [],
  "resume_commands": [
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_path_attribution.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --num-shards 1 --shard-id 0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --num-images 1 --max-targets-per-image 1 --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4 --quadrature trapezoid,gauss_legendre --audit-vector-cases 1",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_fp32_causal.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --num-shards 1 --shard-id 0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --path-result-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --num-images 1 --max-targets-per-image 1 --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4 --logit-batch-size 64",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_shapley.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --num-shards 1 --shard-id 0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --num-images 1 --max-targets-per-image 1 --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4 --region-counts 8 --permutations 8 --coalition-batch-size 64",
    "/opt/conda/private/envs/vicr/bin/python scripts/analyze_tc_fvpa_comprehensive.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
    "/opt/conda/private/envs/vicr/bin/python scripts/build_tc_fvpa_reports.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4"
  ],
  "schema_version": "tc-fvpa-comprehensive-v1",
  "stages": {
    "analysis:llava_1_5_7b": {
      "details": {
        "case_rows": 1,
        "parquet": {
          "case_layer": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 1,
            "status": "BLOCKED"
          },
          "path_convergence": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 12,
            "status": "BLOCKED"
          },
          "spatial_metrics": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 0,
            "status": "BLOCKED"
          },
          "token_scores_sample": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 12096,
            "status": "BLOCKED"
          }
        },
        "required_case_rows": 1
      },
      "started_unix": 1787959831.8021502,
      "status": "PASS",
      "updated_unix": 1787960223.3247836
    },
    "counterfactuals:llava_1_5_7b": {
      "details": {
        "blocked_or_not_run": [
          "fixed_qk",
          "activation_patching",
          "pixel_counterfactual"
        ],
        "family_status": {
          "activation_patching": "NOT_RUN",
          "fixed_qk": "NOT_RUN",
          "frozen_write": "PASS",
          "pixel_counterfactual": "NOT_RUN"
        },
        "important_estimand_note": "Frozen-write effects condition on the observed clean attention decomposition and are not complete image-patch effects.",
        "model": "llava_1_5_7b",
        "resume_command": "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model llava_1_5_7b --device cuda:0 --devices cuda:0 --output-dir outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
        "rows": 33
      },
      "started_unix": 1787959836.0091236,
      "status": "BLOCKED",
      "updated_unix": 1787960226.178228
    },
    "fp32_causal:llava_1_5_7b:shard0": {
      "details": {
        "blocked_layers": [],
        "failures": 0,
        "measured_rows": 336,
        "route": "FP32 final-block FFN + final norm + LM head"
      },
      "started_unix": 1787959367.678229,
      "status": "PASS",
      "updated_unix": 1787959376.6340435
    },
    "path_attribution:llava_1_5_7b:shard0": {
      "details": {
        "elapsed_seconds": 12.72509107994847,
        "failed_cases": 0,
        "images": 1,
        "measured_cases": 1
      },
      "started_unix": 1787959208.1907563,
      "status": "PASS",
      "updated_unix": 1787959221.8926182
    },
    "reports:llava_1_5_7b": {
      "details": {
        "reports": 22
      },
      "started_unix": 1787959838.95797,
      "status": "PASS",
      "updated_unix": 1787961346.4351583
    },
    "shapley:llava_1_5_7b:shard0": {
      "details": {
        "blocked_layers": [],
        "failures": 0,
        "measured_rows": 3
      },
      "started_unix": 1787959458.8659253,
      "status": "PASS",
      "updated_unix": 1787959467.116395
    }
  },
  "updated_unix": 1787961346.4351583
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
- Loader: repository `scripts/load_tc_fvpa_results.py`; run `python scripts/load_tc_fvpa_results.py /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --model llava_1_5_7b --method WRITE` after output checksums exist.
- Complete status and resume commands: `manifests/run_status.json`.

No missing measurement is replaced by zero or inference.
