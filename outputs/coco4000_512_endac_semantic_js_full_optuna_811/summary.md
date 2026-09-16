# Four-model ENDAC-811 JS and full-visual product (Optuna)

Each variant uses its own validation-selected parameters/checkpoint; the table reports three-seed test mean ± population SD. All inputs are [new per-layer feature, log1p(S_E)].

JS is base-2 after separately renormalizing attention and object-probability spatial distributions within the selected region. `attention32` uses attention Top32; `union32` uses the union of attention Top32 and semantic Top32. `full_product` is the original unnormalized sum over all visual tokens.

| Model | Feature | Val AUROC (%) | Test AUROC (%) | Test HALL-AUPR (%) |
|---|---|---:|---:|---:|
| qwen2_5_vl_7b | raw_js_attention32 | 84.19 | 83.84 ± 0.23 | 42.79 ± 0.35 |
| qwen2_5_vl_7b | norm_js_attention32 | 85.03 | 82.71 ± 0.38 | 42.42 ± 2.58 |
| qwen2_5_vl_7b | raw_js_union32 | 84.81 | 84.13 ± 0.37 | 44.81 ± 1.45 |
| qwen2_5_vl_7b | norm_js_union32 (test AUROC best) (test AP best) | 85.94 | 85.74 ± 0.18 | 49.73 ± 0.26 |
| qwen2_5_vl_7b | raw_full_product | 85.79 | 82.50 ± 0.72 | 40.85 ± 1.45 |
| qwen2_5_vl_7b | norm_full_product | 86.45 | 84.05 ± 0.27 | 42.84 ± 0.70 |
| llava_1_5_7b | raw_js_attention32 | 90.41 | 89.45 ± 0.05 | 71.18 ± 0.33 |
| llava_1_5_7b | norm_js_attention32 | 90.47 | 89.71 ± 0.02 | 70.30 ± 0.09 |
| llava_1_5_7b | raw_js_union32 | 90.74 | 91.00 ± 0.05 | 74.70 ± 0.07 |
| llava_1_5_7b | norm_js_union32 | 91.49 | 91.61 ± 0.03 | 75.95 ± 0.17 |
| llava_1_5_7b | raw_full_product | 92.16 | 92.00 ± 0.06 | 76.37 ± 0.31 |
| llava_1_5_7b | norm_full_product (test AUROC best) (test AP best) | 92.27 | 92.59 ± 0.05 | 77.42 ± 0.25 |
| qwen3_vl_8b | raw_js_attention32 | 88.95 | 90.02 ± 0.09 | 70.95 ± 0.20 |
| qwen3_vl_8b | norm_js_attention32 (test AUROC best) (test AP best) | 87.70 | 90.47 ± 0.25 | 72.46 ± 0.54 |
| qwen3_vl_8b | raw_js_union32 | 88.99 | 90.19 ± 0.06 | 72.08 ± 0.81 |
| qwen3_vl_8b | norm_js_union32 | 88.58 | 89.60 ± 0.15 | 71.76 ± 0.18 |
| qwen3_vl_8b | raw_full_product | 89.59 | 89.72 ± 0.04 | 72.06 ± 0.55 |
| qwen3_vl_8b | norm_full_product | 89.80 | 90.00 ± 0.05 | 71.98 ± 0.36 |
| internvl_2_5_8b | raw_js_attention32 | 85.76 | 86.90 ± 0.03 | 61.45 ± 0.34 |
| internvl_2_5_8b | norm_js_attention32 | 87.07 | 85.95 ± 0.26 | 58.43 ± 0.47 |
| internvl_2_5_8b | raw_js_union32 | 87.15 | 87.32 ± 0.07 | 63.02 ± 0.43 |
| internvl_2_5_8b | norm_js_union32 | 86.56 | 87.01 ± 0.15 | 61.00 ± 0.34 |
| internvl_2_5_8b | raw_full_product (test AUROC best) (test AP best) | 87.47 | 88.98 ± 0.10 | 64.23 ± 1.10 |
| internvl_2_5_8b | norm_full_product | 87.94 | 88.36 ± 0.13 | 62.92 ± 0.72 |

## Validation-selected hyperparameters

| Model | Feature | Trial | Width | Dropout | Activation | LR | WD | Batch | Monitor | Best epochs 43/44/45 |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---|
| qwen2_5_vl_7b | raw_js_attention32 | 22 | 64 | 0.5 | gelu | 0.00211 | 0.0001 | 128 | val_auroc | 30/29/29 |
| qwen2_5_vl_7b | norm_js_attention32 | 21 | 128 | 0.5 | relu | 0.00891 | 0.001 | 256 | val_auroc | 65/59/39 |
| qwen2_5_vl_7b | raw_js_union32 | 21 | 128 | 0.5 | gelu | 0.00874 | 0 | 128 | val_auroc | 15/28/21 |
| qwen2_5_vl_7b | norm_js_union32 | 21 | 64 | 0.5 | relu | 0.00709 | 0 | 256 | val_auroc | 41/38/30 |
| qwen2_5_vl_7b | raw_full_product | 19 | 128 | 0.5 | gelu | 0.00585 | 0.0001 | 64 | val_auroc | 26/55/37 |
| qwen2_5_vl_7b | norm_full_product | 16 | 512 | 0.1 | gelu | 0.00635 | 0.001 | 64 | val_auroc | 26/24/36 |
| llava_1_5_7b | raw_js_attention32 | 14 | 128 | 0.1 | gelu | 0.000252 | 0 | 64 | val_auroc | 45/51/41 |
| llava_1_5_7b | norm_js_attention32 | 8 | 256 | 0.5 | relu | 0.00468 | 1e-06 | 512 | val_auroc | 25/28/26 |
| llava_1_5_7b | raw_js_union32 | 22 | 248 | 0.5 | relu | 0.00658 | 0.001 | 512 | val_loss | 45/52/51 |
| llava_1_5_7b | norm_js_union32 | 6 | 256 | 0.5 | relu | 0.0027 | 1e-05 | 512 | val_auroc | 24/20/26 |
| llava_1_5_7b | raw_full_product | 13 | 1024 | 0.0 | relu | 0.000411 | 0.001 | 256 | val_loss | 44/44/71 |
| llava_1_5_7b | norm_full_product | 10 | 1024 | 0.1 | relu | 0.000949 | 0.0001 | 256 | val_loss | 19/23/23 |
| qwen3_vl_8b | raw_js_attention32 | 21 | 512 | 0.5 | relu | 0.000491 | 1e-06 | 64 | val_auroc | 59/40/34 |
| qwen3_vl_8b | norm_js_attention32 | 17 | 128 | 0.5 | relu | 0.00257 | 0.001 | 64 | val_auroc | 29/42/27 |
| qwen3_vl_8b | raw_js_union32 | 18 | 256 | 0.5 | relu | 0.00076 | 1e-05 | 64 | val_auroc | 50/58/42 |
| qwen3_vl_8b | norm_js_union32 | 6 | 256 | 0.5 | relu | 0.0027 | 1e-05 | 512 | val_auroc | 45/48/39 |
| qwen3_vl_8b | raw_full_product | 6 | 256 | 0.5 | relu | 0.0027 | 1e-05 | 512 | val_auroc | 49/54/45 |
| qwen3_vl_8b | norm_full_product | 23 | 256 | 0.3 | relu | 0.000156 | 0 | 64 | val_auroc | 132/132/122 |
| internvl_2_5_8b | raw_js_attention32 | 6 | 256 | 0.5 | relu | 0.0027 | 1e-05 | 512 | val_auroc | 23/28/34 |
| internvl_2_5_8b | norm_js_attention32 | 19 | 128 | 0.3 | gelu | 0.000948 | 1e-05 | 64 | val_auroc | 61/44/53 |
| internvl_2_5_8b | raw_js_union32 | 8 | 256 | 0.5 | relu | 0.00468 | 1e-06 | 512 | val_auroc | 65/56/59 |
| internvl_2_5_8b | norm_js_union32 | 17 | 256 | 0.3 | gelu | 0.000963 | 0 | 512 | val_auroc | 83/71/60 |
| internvl_2_5_8b | raw_full_product | 10 | 1024 | 0.5 | relu | 0.00119 | 1e-05 | 256 | val_auroc | 70/47/47 |
| internvl_2_5_8b | norm_full_product | 17 | 128 | 0.5 | relu | 0.00608 | 0.0001 | 256 | val_loss | 46/37/45 |

## Numerical note

A spatial distribution with zero mass inside the selected region cannot be renormalized. For such mention-by-layer rows, this experiment uses a uniform distribution over that region and records the count below. Qwen2.5 raw attention32 has 7,543/221,872 affected rows (3.40%); its attention mass itself is never zero.

| Model | raw attention32 | norm attention32 | raw union32 | norm union32 |
|---|---:|---:|---:|---:|
| qwen2_5_vl_7b | 7543 | 0 | 37 | 0 |
| llava_1_5_7b | 0 | 0 | 0 | 0 |
| qwen3_vl_8b | 0 | 0 | 0 | 0 |
| internvl_2_5_8b | 0 | 0 | 0 | 0 |

The full 24-trial Optuna histories are in each model's `studies/*.db`. Existing AE and Top-K baselines are in `../coco4000_512_endac_four_ae_semantic_optuna_811/summary.md`. Test-based feature ranking is exploratory, not an independent test estimate.
