# HANDOFF TO CHATGPT

Repository commit: `dd3c1ecec02c37bac23687959a0508b0b53a452f`

Modified/untracked files at final report build:

```text
## main...origin/main
 M README.md
 M docs/CURRENT_TASK.md
 M scripts/run_jffn_second_round_logit_causal.py
?? docs/TC_FVPA_RUNBOOK.md
?? features/ffn_visual_interactions.py
?? features/ffn_visual_path_attribution.py
?? features/qwen_fp32_suffix.py
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
?? tests/test_qwen_fp32_suffix.py
?? tests/test_representation_geometry_controls.py
?? tests/test_swiglu_visual_mechanism.py
?? tests/test_tc_fvpa_analysis.py
?? tests/test_tc_fvpa_artifacts.py
?? tests/test_tc_fvpa_path_runner.py
?? tests/test_tc_fvpa_qwen_adapter.py
```

Model: `qwen2_5_vl_7b`

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
  "fp32_intervention_rows": 216384,
  "frozen_write_rows": 43428,
  "images": 500,
  "labels": {
    "HALL": 1148,
    "REAL": 2348
  },
  "layers": [
    7,
    14,
    21,
    28
  ],
  "model": "qwen2_5_vl_7b",
  "path_convergence_rows": 39480,
  "shapley_rows": 300,
  "spatial_metric_rows": 53200,
  "target_layer_cases": 3496
}
```

## Persisted numerical summaries

```json
{
  "comprehensive": {
    "case_rows": 3496,
    "cohort_case_counts": {
      "local": 3496,
      "path": 1316
    },
    "cohort_image_counts": {
      "local": 500,
      "path": 200
    },
    "counterfactual_rows": 43428,
    "images": 500,
    "labels": {
      "HALL": 1148,
      "REAL": 2348
    },
    "layers": [
      7,
      14,
      21,
      28
    ],
    "model": "qwen2_5_vl_7b",
    "parquet": {
      "case_layer": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 3496,
        "status": "BLOCKED"
      },
      "path_convergence": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 39480,
        "status": "BLOCKED"
      },
      "spatial_metrics": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 53200,
        "status": "BLOCKED"
      },
      "token_scores_sample": {
        "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
        "rows": 30888,
        "status": "BLOCKED"
      }
    },
    "path_convergence_rows": 39480,
    "path_vs_frozen_write": {
      "log_probability:aggregate_visual_write": {
        "count": 1316,
        "mae": 0.03304824554354089,
        "pearson": 0.8745418813850748,
        "sign_accuracy": 0.6747720364741642,
        "spearman": 0.5654525206070877
      },
      "log_probability:highest_attention": {
        "count": 1316,
        "mae": 0.029702839631175273,
        "pearson": 0.555524683722176,
        "sign_accuracy": 0.5562310030395137,
        "spearman": 0.2219092606563377
      },
      "log_probability:highest_jffn": {
        "count": 1316,
        "mae": 0.0307054532748724,
        "pearson": 0.6195452533597706,
        "sign_accuracy": 0.5661094224924013,
        "spearman": 0.2539379744608402
      },
      "log_probability:highest_local_riesz": {
        "count": 1316,
        "mae": 0.028589824556838637,
        "pearson": 0.3900965288372638,
        "sign_accuracy": 0.5539513677811551,
        "spearman": 0.1804102807732233
      },
      "log_probability:highest_path": {
        "count": 1316,
        "mae": 0.028557623807332993,
        "pearson": 0.3907293098284998,
        "sign_accuracy": 0.5448328267477204,
        "spearman": 0.20432187638219396
      },
      "log_probability:highest_write": {
        "count": 1316,
        "mae": 0.03055539990129676,
        "pearson": 0.596628965259538,
        "sign_accuracy": 0.5668693009118541,
        "spearman": 0.2549345901826949
      },
      "log_probability:most_negative_path": {
        "count": 1316,
        "mae": 0.030044984021606345,
        "pearson": 0.6440854581011801,
        "sign_accuracy": 0.5600303951367781,
        "spearman": 0.28905692675805855
      },
      "log_probability:random_token": {
        "count": 1316,
        "mae": 0.026524657880830818,
        "pearson": 0.09834196252390504,
        "sign_accuracy": 0.4726443768996961,
        "spearman": 0.019615065838666814
      },
      "log_probability:top16_positive_path": {
        "count": 1316,
        "mae": 0.033396732519291775,
        "pearson": 0.7227137956462657,
        "sign_accuracy": 0.648176291793313,
        "spearman": 0.4610374635718141
      },
      "log_probability:top32_positive_path": {
        "count": 1316,
        "mae": 0.03278741006459287,
        "pearson": 0.7924052364204723,
        "sign_accuracy": 0.6740121580547113,
        "spearman": 0.5171753702012457
      },
      "log_probability:top4_positive_path": {
        "count": 1316,
        "mae": 0.03008490563488791,
        "pearson": 0.5842934476930848,
        "sign_accuracy": 0.6208206686930091,
        "spearman": 0.30585018915053913
      },
      "logit:aggregate_visual_write": {
        "count": 1316,
        "mae": 0.21781810509893454,
        "pearson": 0.7317704990569588,
        "sign_accuracy": 0.5668693009118541,
        "spearman": 0.4724840703731251
      },
      "logit:highest_attention": {
        "count": 1316,
        "mae": 0.1910448826673953,
        "pearson": 0.375208319317826,
        "sign_accuracy": 0.40501519756838905,
        "spearman": 0.17081255503930684
      },
      "logit:highest_jffn": {
        "count": 1316,
        "mae": 0.18864800856876157,
        "pearson": 0.43197298538818574,
        "sign_accuracy": 0.40805471124620063,
        "spearman": 0.1893761998210205
      },
      "logit:highest_local_riesz": {
        "count": 1316,
        "mae": 0.1850069633495391,
        "pearson": 0.10767933401583221,
        "sign_accuracy": 0.3594224924012158,
        "spearman": 0.09453240042286824
      },
      "logit:highest_path": {
        "count": 1316,
        "mae": 0.18546630385314825,
        "pearson": 0.10494197170900674,
        "sign_accuracy": 0.3617021276595745,
        "spearman": 0.08964972367112133
      },
      "logit:highest_write": {
        "count": 1316,
        "mae": 0.18720815782534314,
        "pearson": 0.4119648971667345,
        "sign_accuracy": 0.4072948328267477,
        "spearman": 0.18901002710176207
      },
      "logit:most_negative_path": {
        "count": 1316,
        "mae": 0.18039052794746896,
        "pearson": 0.4457138387685007,
        "sign_accuracy": 0.42857142857142855,
        "spearman": 0.20594787153880076
      },
      "logit:random_token": {
        "count": 1316,
        "mae": 0.1667999138901775,
        "pearson": 0.045466302029371644,
        "sign_accuracy": 0.30851063829787234,
        "spearman": 0.019654428379275557
      },
      "logit:top16_positive_path": {
        "count": 1316,
        "mae": 0.18655256051302138,
        "pearson": 0.3566662524747739,
        "sign_accuracy": 0.5091185410334347,
        "spearman": 0.32247323856645477
      },
      "logit:top32_positive_path": {
        "count": 1316,
        "mae": 0.2006066598576769,
        "pearson": 0.4145851166423319,
        "sign_accuracy": 0.5349544072948328,
        "spearman": 0.33758688513438323
      },
      "logit:top4_positive_path": {
        "count": 1316,
        "mae": 0.19112496460008782,
        "pearson": 0.17821846475146108,
        "sign_accuracy": 0.42325227963525835,
        "spearman": 0.1677196839529173
      },
      "margin:aggregate_visual_write": {
        "count": 1316,
        "mae": 0.11250247486628519,
        "pearson": 0.9057721463927535,
        "sign_accuracy": 0.5562310030395137,
        "spearman": 0.6218594941055305
      },
      "margin:highest_attention": {
        "count": 1316,
        "mae": 0.09669420632638467,
        "pearson": 0.6157465973108489,
        "sign_accuracy": 0.34422492401215804,
        "spearman": 0.23006022874132565
      },
      "margin:highest_jffn": {
        "count": 1316,
        "mae": 0.10075342765015671,
        "pearson": 0.6712224285253726,
        "sign_accuracy": 0.3655015197568389,
        "spearman": 0.265610984318087
      },
      "margin:highest_local_riesz": {
        "count": 1316,
        "mae": 0.09328362939024407,
        "pearson": 0.32447159339888937,
        "sign_accuracy": 0.3183890577507599,
        "spearman": 0.14675954702856123
      },
      "margin:highest_path": {
        "count": 1316,
        "mae": 0.09274310631094371,
        "pearson": 0.3315920752512427,
        "sign_accuracy": 0.32370820668693007,
        "spearman": 0.17845854427099833
      },
      "margin:highest_write": {
        "count": 1316,
        "mae": 0.1008080204886506,
        "pearson": 0.6426616437400161,
        "sign_accuracy": 0.3617021276595745,
        "spearman": 0.25058356045815866
      },
      "margin:most_negative_path": {
        "count": 1316,
        "mae": 0.10219004261378188,
        "pearson": 0.6686237523450429,
        "sign_accuracy": 0.3708206686930091,
        "spearman": 0.28458121739713726
      },
      "margin:random_token": {
        "count": 1316,
        "mae": 0.08233952121704298,
        "pearson": 0.07797094924912794,
        "sign_accuracy": 0.23252279635258358,
        "spearman": -0.023735580395948933
      },
      "margin:top16_positive_path": {
        "count": 1316,
        "mae": 0.10656194847309254,
        "pearson": 0.61089487467292,
        "sign_accuracy": 0.4566869300911854,
        "spearman": 0.4087997444699276
      },
      "margin:top32_positive_path": {
        "count": 1316,
        "mae": 0.10687975415853447,
        "pearson": 0.6624874284010732,
        "sign_accuracy": 0.5007598784194529,
        "spearman": 0.4715967967969127
      },
      "margin:top4_positive_path": {
        "count": 1316,
        "mae": 0.10264515239281133,
        "pearson": 0.44282829718205446,
        "sign_accuracy": 0.3958966565349544,
        "spearman": 0.242023940122394
      }
    },
    "shard_audit": {
      "local": {
        "cohort": "local",
        "failures": [],
        "schema_version": "tc-fvpa-comprehensive-v1",
        "shards": [
          {
            "failures": 0,
            "path": "shards/local_case_rank00_shard_00000.pt",
            "rows": 1620,
            "sha256": "3978c5753759285015a8b8e206a56bb3c11f3d8fc852e95c062f8a2887818410"
          },
          {
            "failures": 0,
            "path": "shards/local_case_rank01_shard_00000.pt",
            "rows": 1876,
            "sha256": "cd48afe7ef64fcb9880973e4dbdd18261bdff0f370bff3f89e2eb8c811044d49"
          }
        ],
        "unique_cases": 3496
      },
      "path": {
        "cohort": "path",
        "failures": [],
        "schema_version": "tc-fvpa-comprehensive-v1",
        "shards": [
          {
            "failures": 0,
            "path": "shards/case_rank00_shard_00000.pt",
            "rows": 640,
            "sha256": "19863d455a166dfdbf135a010edc4d97793797fb5b44f36398f8e5b2d05191d7"
          },
          {
            "failures": 0,
            "path": "shards/case_rank01_shard_00000.pt",
            "rows": 676,
            "sha256": "a8c7db0dfe7ce03f0a93147960c06edbed704219f19fa702ca49217b41391556"
          }
        ],
        "unique_cases": 1316
      }
    },
    "spatial_metric_rows": 53200,
    "token_score_sample_rows": 30888
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
    "model": "qwen2_5_vl_7b",
    "resume_command": "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --resume --formal",
    "rows": 43428
  },
  "fp32": {
    "aggregate_gpu_seconds": 533.8664438429987,
    "all_ranks_present": true,
    "blocked_layer_rows": 0,
    "completed_ranks": [
      0,
      1
    ],
    "expected_ranks": 2,
    "failures": [],
    "measured_rows": 216384,
    "model": "qwen2_5_vl_7b",
    "peak_gpu_memory_bytes": 22023510016,
    "protocol": "tc_fvpa_true_fp32_causal_suffix_v3_microbatched",
    "query_microbatch_size": 32
  },
  "shapley": {
    "all_ranks_present": true,
    "blocked_layers": [],
    "completed_ranks": [
      0,
      1
    ],
    "expected_ranks": 2,
    "failures": [],
    "measured_rows": 300,
    "model": "qwen2_5_vl_7b",
    "not_in_scope_layers": [
      7,
      14,
      21,
      7,
      14,
      21
    ],
    "peak_gpu_memory_bytes": 21764888064,
    "permutations": 128,
    "protocol": "tc_fvpa_shapley_final_block_fp32_v1"
  },
  "validation": {}
}
```

## Experiment status

```json
{
  "completed_shards": [],
  "created_unix": 1787999472.3995109,
  "failed_cases": [],
  "failed_shards": [],
  "resume_commands": [
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_path_attribution.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --no-resume --formal --attribution-mode local --audit-vector-cases 0 --num-images 500",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_path_attribution.py --model qwen2_5_vl_7b --device cuda:1 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 1 --no-resume --formal --attribution-mode local --audit-vector-cases 0 --num-images 500",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_path_attribution.py --model qwen2_5_vl_7b --device cuda:1 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 1 --resume --formal --attribution-mode path --num-images 200",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_path_attribution.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --resume --formal --attribution-mode path --num-images 200",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_fp32_causal.py --model qwen2_5_vl_7b --device cuda:1 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 1 --resume --formal --num-images 100",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_fp32_causal.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --resume --formal --num-images 100",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_shapley.py --model qwen2_5_vl_7b --device cuda:1 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 1 --resume --formal --num-images 50",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_shapley.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --resume --formal --num-images 50",
    "/opt/conda/private/envs/vicr/bin/python scripts/analyze_tc_fvpa_comprehensive.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --resume --formal",
    "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --resume --formal"
  ],
  "schema_version": "tc-fvpa-comprehensive-v1",
  "stages": {
    "analysis:qwen2_5_vl_7b": {
      "details": {
        "case_rows": 3496,
        "cohort_case_counts": {
          "local": 3496,
          "path": 1316
        },
        "cohort_image_counts": {
          "local": 500,
          "path": 200
        },
        "formal_requirements": {
          "local_images": 500,
          "path_case_rows": 800
        },
        "parquet": {
          "case_layer": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 3496,
            "status": "BLOCKED"
          },
          "path_convergence": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 39480,
            "status": "BLOCKED"
          },
          "spatial_metrics": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 53200,
            "status": "BLOCKED"
          },
          "token_scores_sample": {
            "error": "ModuleNotFoundError(\"No module named 'pandas'\")",
            "rows": 30888,
            "status": "BLOCKED"
          }
        }
      },
      "started_unix": 1788079558.393346,
      "status": "PASS",
      "updated_unix": 1788079558.393346
    },
    "counterfactuals:qwen2_5_vl_7b": {
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
        "model": "qwen2_5_vl_7b",
        "resume_command": "/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_counterfactuals.py --model qwen2_5_vl_7b --device cuda:0 --devices cuda:0,cuda:1 --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --seed 20260829 --layers 7,14,21,28 --target-scalars log_probability,margin,logit --integration-points 1,4,8,16,32 --config /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/configs/model_configs_inslen_official_target.yaml --num-shards 2 --shard-id 0 --resume --formal",
        "rows": 43428
      },
      "started_unix": 1788079575.537069,
      "status": "BLOCKED",
      "updated_unix": 1788079575.537069
    },
    "fp32_causal:qwen2_5_vl_7b:shard0": {
      "details": {
        "blocked_layers": [],
        "failures": 0,
        "measured_rows": 109536,
        "path": "/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829/shards/fp32_causal_rank00_shard_00000.pt",
        "query_microbatch_size": 32,
        "resumed_existing": true
      },
      "started_unix": 1788021916.341132,
      "status": "PASS",
      "updated_unix": 1788079226.5520308
    },
    "fp32_causal:qwen2_5_vl_7b:shard1": {
      "details": {
        "blocked_layers": [],
        "failures": 0,
        "measured_rows": 106848,
        "path": "/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829/shards/fp32_causal_rank01_shard_00000.pt",
        "query_microbatch_size": 32,
        "resumed_existing": true
      },
      "started_unix": 1788021885.9912238,
      "status": "PASS",
      "updated_unix": 1788079196.3471518
    },
    "local_riesz:qwen2_5_vl_7b:shard0": {
      "details": {
        "elapsed_seconds": 889.3753788640024,
        "failed_cases": 0,
        "images": 250,
        "measured_cases": 1620,
        "peak_gpu_memory_bytes": 21428404736
      },
      "started_unix": 1787999472.5888038,
      "status": "PASS",
      "updated_unix": 1788000370.073862
    },
    "local_riesz:qwen2_5_vl_7b:shard1": {
      "details": {
        "elapsed_seconds": 1037.2082894070772,
        "failed_cases": 0,
        "images": 250,
        "measured_cases": 1876,
        "peak_gpu_memory_bytes": 21860895232
      },
      "started_unix": 1787999493.8594036,
      "status": "PASS",
      "updated_unix": 1788000538.9167278
    },
    "path_attribution:qwen2_5_vl_7b:shard0": {
      "details": {
        "elapsed_seconds": 16558.25944970909,
        "failed_cases": 0,
        "images": 100,
        "measured_cases": 640,
        "peak_gpu_memory_bytes": 21615140352
      },
      "started_unix": 1788000820.4064553,
      "status": "PASS",
      "updated_unix": 1788020339.4577403
    },
    "path_attribution:qwen2_5_vl_7b:shard1": {
      "details": {
        "elapsed_seconds": 17917.030190572026,
        "failed_cases": 0,
        "images": 100,
        "measured_cases": 676,
        "peak_gpu_memory_bytes": 22256279552
      },
      "started_unix": 1788000790.3096943,
      "status": "PASS",
      "updated_unix": 1788021668.112282
    },
    "shapley:qwen2_5_vl_7b:shard0": {
      "details": {
        "failures": 0,
        "measured_rows": 150,
        "not_in_scope_layers": [
          7,
          14,
          21
        ],
        "scope": "preregistered final-decoder-layer subset"
      },
      "started_unix": 1788079457.452942,
      "status": "PASS",
      "updated_unix": 1788079500.6395776
    },
    "shapley:qwen2_5_vl_7b:shard1": {
      "details": {
        "failures": 0,
        "measured_rows": 150,
        "not_in_scope_layers": [
          7,
          14,
          21
        ],
        "scope": "preregistered final-decoder-layer subset"
      },
      "started_unix": 1788079427.1358624,
      "status": "PASS",
      "updated_unix": 1788079463.1351602
    }
  },
  "updated_unix": 1788079575.537069
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
- Loader: repository `scripts/load_tc_fvpa_results.py`; run `python scripts/load_tc_fvpa_results.py /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829 --model qwen2_5_vl_7b --method WRITE` after output checksums exist.
- Complete status and resume commands: `manifests/run_status.json`.

No missing measurement is replaced by zero or inference.
