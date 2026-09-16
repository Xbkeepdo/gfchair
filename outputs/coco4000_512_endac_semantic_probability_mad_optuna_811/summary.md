# Four-model ENDAC-811 target-probability MAD (Optuna)

Per layer: `median_v |p_v(y) - median_u p_u(y)|` over all visual tokens. The probability is the original vocabulary softmax, not spatially normalized. Standalone and +`log1p(S_E)` each use validation-selected parameters and checkpoints.

Test AUROC and HALL-AUPR are three-seed mean ± population SD, in percent. Test-based feature ranking is exploratory.

| Model | Feature | Val AUROC | Test AUROC | Test HALL-AUPR |
|---|---|---:|---:|---:|
| qwen2_5_vl_7b | raw_mad | 65.16 | 61.51 ± 0.96 | 18.87 ± 0.40 |
| qwen2_5_vl_7b | norm_mad | 77.83 | 81.54 ± 0.47 | 47.40 ± 1.37 |
| qwen2_5_vl_7b | raw_mad_logs | 87.16 | 84.57 ± 0.37 | 44.93 ± 1.71 |
| qwen2_5_vl_7b | norm_mad_logs | 86.77 | 85.91 ± 0.53 | 49.32 ± 2.13 |
| llava_1_5_7b | raw_mad | 77.92 | 78.23 ± 0.18 | 49.55 ± 0.33 |
| llava_1_5_7b | norm_mad | 80.97 | 80.18 ± 0.12 | 51.21 ± 0.52 |
| llava_1_5_7b | raw_mad_logs | 90.00 | 88.81 ± 0.08 | 70.30 ± 0.40 |
| llava_1_5_7b | norm_mad_logs | 89.87 | 89.11 ± 0.22 | 70.53 ± 0.28 |
| qwen3_vl_8b | raw_mad | 78.87 | 81.10 ± 0.22 | 57.57 ± 0.65 |
| qwen3_vl_8b | norm_mad | 78.30 | 80.08 ± 0.13 | 56.40 ± 0.30 |
| qwen3_vl_8b | raw_mad_logs | 89.35 | 89.75 ± 0.10 | 74.04 ± 0.35 |
| qwen3_vl_8b | norm_mad_logs | 89.16 | 89.90 ± 0.05 | 72.75 ± 0.16 |
| internvl_2_5_8b | raw_mad | 73.10 | 75.23 ± 0.11 | 41.81 ± 0.53 |
| internvl_2_5_8b | norm_mad | 76.61 | 75.93 ± 0.08 | 44.08 ± 0.32 |
| internvl_2_5_8b | raw_mad_logs | 86.31 | 85.93 ± 0.11 | 62.51 ± 0.39 |
| internvl_2_5_8b | norm_mad_logs | 87.08 | 85.91 ± 0.25 | 60.03 ± 0.97 |

## Numerical notes

A zero MAD can be genuine flatness or information lost when the saved FP32 probability underflowed. The existing trainer sets the z-score scale to 1 for training columns with SD below `1e-12`; these columns are effectively not standardized.

| Model | raw zero layer-rows | norm zero layer-rows | raw affected mentions | total layer-rows | raw SD<1e-12 columns | norm SD<1e-12 columns |
|---|---:|---:|---:|---:|---:|---:|
| qwen2_5_vl_7b | 19372 | 0 | 7923 | 221872 | 11 | 0 |
| llava_1_5_7b | 0 | 0 | 0 | 405952 | 0 | 0 |
| qwen3_vl_8b | 23 | 0 | 21 | 423036 | 4 | 0 |
| internvl_2_5_8b | 0 | 0 | 0 | 344992 | 2 | 0 |

Full per-seed metrics and predictions are in each model directory. Selected hyperparameters are in `selected_configs.csv`; all trials are in `studies/*.db`. For comparison, see `../coco4000_512_endac_semantic_js_full_optuna_811/summary.md` and `../coco4000_512_endac_four_ae_semantic_optuna_811/summary.md`.

Checkpoint reload: 48/48 passed; maximum CPU/GPU probability difference 5.1e-06 (qwen2_5_vl_7b/raw_mad_logs/seed43), tolerance 1e-05; maximum metric difference 3.3e-16.
