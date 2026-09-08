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

Model: `internvl_2_5_8b`

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
  "model": "internvl_2_5_8b",
  "path_convergence_rows": 12,
  "shapley_rows": 0,
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
    "model": "internvl_2_5_8b",
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
        "rows": 5376,
        "status": "BLOCKED"
      }
    },
    "path_convergence_rows": 12,
    "path_vs_frozen_write": {
      "log_probability:aggregate_visual_write": {
        "count": 1,
        "mae": 0.0019642519764602184,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:highest_attention": {
        "count": 1,
        "mae": 0.00045350193977355957,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:highest_jffn": {
        "count": 1,
        "mae": 0.0004431493580341339,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:highest_local_riesz": {
        "count": 1,
        "mae": 0.0004431493580341339,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:highest_path": {
        "count": 1,
        "mae": 0.0004431493580341339,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:highest_write": {
        "count": 1,
        "mae": 0.00045350193977355957,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:most_negative_path": {
        "count": 1,
        "mae": 0.00045350193977355957,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:random_token": {
        "count": 1,
        "mae": 0.00032839924097061157,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:top16_positive_path": {
        "count": 1,
        "mae": 0.0015942901372909546,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "log_probability:top32_positive_path": {
        "count": 1,
        "mae": 0.00012037158012390137,
        "pearson": null,
        "sign_accuracy": 1.0,
        "spearman": null
      },
      "log_probability:top4_positive_path": {
        "count": 1,
        "mae": 0.0006212815642356873,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:aggregate_visual_write": {
        "count": 1,
        "mae": 0.1130845844745636,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_attention": {
        "count": 1,
        "mae": 0.00213623046875,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_jffn": {
        "count": 1,
        "mae": 0.00762939453125,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_local_riesz": {
        "count": 1,
        "mae": 0.00762939453125,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_path": {
        "count": 1,
        "mae": 0.00762939453125,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:highest_write": {
        "count": 1,
        "mae": 0.00213623046875,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:most_negative_path": {
        "count": 1,
        "mae": 0.00213623046875,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:random_token": {
        "count": 1,
        "mae": 0.00022125244140625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:top16_positive_path": {
        "count": 1,
        "mae": 0.034610748291015625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:top32_positive_path": {
        "count": 1,
        "mae": 0.04506683349609375,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "logit:top4_positive_path": {
        "count": 1,
        "mae": 0.0213165283203125,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:aggregate_visual_write": {
        "count": 1,
        "mae": 0.06594216823577881,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:highest_attention": {
        "count": 1,
        "mae": 0.00136566162109375,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:highest_jffn": {
        "count": 1,
        "mae": 0.003387451171875,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:highest_local_riesz": {
        "count": 1,
        "mae": 0.003387451171875,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:highest_path": {
        "count": 1,
        "mae": 0.003387451171875,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:highest_write": {
        "count": 1,
        "mae": 0.00136566162109375,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:most_negative_path": {
        "count": 1,
        "mae": 0.00136566162109375,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:random_token": {
        "count": 1,
        "mae": 0.00014209747314453125,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:top16_positive_path": {
        "count": 1,
        "mae": 0.0189361572265625,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:top32_positive_path": {
        "count": 1,
        "mae": 0.027797698974609375,
        "pearson": null,
        "sign_accuracy": 0.0,
        "spearman": null
      },
      "margin:top4_positive_path": {
        "count": 1,
        "mae": 0.0085906982421875,
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
          "sha256": "e6a36725a78f4a6d4a254fdb4c41b92a4dd115f4205442131ea3e6ef49f5472a"
        }
      ],
      "unique_cases": 1
    },
    "spatial_metric_rows": 0,
    "token_score_sample_rows": 5376
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
    "model": "internvl_2_5_8b",
    "resume_command": "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model internvl_2_5_8b --device cuda:1 --devices cuda:1 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
    "rows": 33
  },
  "fp32": {
    "blocked_layer_rows": 0,
    "elapsed_seconds": 10.039759651001077,
    "failures": [],
    "measured_rows": 336,
    "model": "internvl_2_5_8b",
    "protocol": "tc_fvpa_true_fp32_final_block_v1"
  },
  "shapley": {},
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
  "created_unix": 1787959544.2380319,
  "failed_cases": [],
  "failed_shards": [],
  "resume_commands": [
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_path_attribution.py --model internvl_2_5_8b --device cuda:1 --devices cuda:1 --num-shards 1 --shard-id 0 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --num-images 1 --max-targets-per-image 1 --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4 --quadrature trapezoid,gauss_legendre --audit-vector-cases 1",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_fp32_causal.py --model internvl_2_5_8b --device cuda:1 --devices cuda:1 --num-shards 1 --shard-id 0 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --path-result-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --num-images 1 --max-targets-per-image 1 --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4 --logit-batch-size 64",
    "/opt/conda/private/envs/vicr/bin/python scripts/analyze_tc_fvpa_comprehensive.py --model internvl_2_5_8b --device cuda:1 --devices cuda:1 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model internvl_2_5_8b --device cuda:1 --devices cuda:1 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
    "/opt/conda/private/envs/vicr/bin/python scripts/build_tc_fvpa_reports.py --model internvl_2_5_8b --device cuda:1 --devices cuda:1 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
    "/opt/conda/private/envs/vicr/bin/python scripts/build_tc_fvpa_reports.py --model internvl_2_5_8b --device cuda:0 --devices cuda:0 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4"
  ],
  "schema_version": "tc-fvpa-comprehensive-v1",
  "stages": {
    "analysis:internvl_2_5_8b": {
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
            "rows": 5376,
            "status": "BLOCKED"
          }
        },
        "required_case_rows": 1
      },
      "started_unix": 1787960192.2067668,
      "status": "PASS",
      "updated_unix": 1787960192.2067668
    },
    "counterfactuals:internvl_2_5_8b": {
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
        "model": "internvl_2_5_8b",
        "resume_command": "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model internvl_2_5_8b --device cuda:1 --devices cuda:1 --output-dir outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --smoke --layers 32 --target-scalars log_probability,margin,logit --integration-points 1,4",
        "rows": 33
      },
      "started_unix": 1787960194.9156775,
      "status": "BLOCKED",
      "updated_unix": 1787960194.9156775
    },
    "fp32_causal:internvl_2_5_8b:shard0": {
      "details": {
        "blocked_layers": [],
        "failures": 0,
        "measured_rows": 336,
        "route": "FP32 final-block FFN + final norm + LM head"
      },
      "started_unix": 1787960153.9959462,
      "status": "PASS",
      "updated_unix": 1787960165.3341722
    },
    "path_attribution:internvl_2_5_8b:shard0": {
      "details": {
        "elapsed_seconds": 248.02784807799617,
        "failed_cases": 0,
        "images": 1,
        "measured_cases": 1
      },
      "started_unix": 1787959544.3214247,
      "status": "PASS",
      "updated_unix": 1787959793.478947
    },
    "reports:internvl_2_5_8b": {
      "details": {
        "reports": 22
      },
      "started_unix": 1787960197.8349283,
      "status": "PASS",
      "updated_unix": 1787961357.9072514
    }
  },
  "updated_unix": 1787961357.9072514
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
- Loader: repository `scripts/load_tc_fvpa_results.py`; run `python scripts/load_tc_fvpa_results.py /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_smoke_20260829 --model internvl_2_5_8b --method WRITE` after output checksums exist.
- Complete status and resume commands: `manifests/run_status.json`.

No missing measurement is replaced by zero or inference.
