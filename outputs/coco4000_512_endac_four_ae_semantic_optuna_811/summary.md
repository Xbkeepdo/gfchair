# Four-model AE vs semantic×attention + log1p(S_E): Optuna single MLP

Train-only standardization, no BN; TPE 24 trials per feature;
seed43 shortlist top3, seeds44/45 validation selection, then test.
3200/400/400 image split; AUROC/HALL-AUPR are three-seed test mean ± population std.

| Model | Feature | Trial | Val AUROC (%) | Test AUROC (%) | Test HALL-AUPR (%) |
|---|---|---:|---:|---:|---:|
| qwen2_5_vl_7b | ae_logs (selected) | 11 | 88.02 | 87.18 ± 0.05 | 54.77 ± 0.44 |
| qwen2_5_vl_7b | raw_top16 | 20 | 85.30 | 81.60 ± 0.37 | 39.92 ± 1.65 |
| qwen2_5_vl_7b | raw_top32 | 21 | 85.17 | 82.74 ± 0.08 | 41.02 ± 0.22 |
| qwen2_5_vl_7b | norm_top16 | 23 | 86.06 | 82.95 ± 0.31 | 42.68 ± 0.62 |
| qwen2_5_vl_7b | norm_top32 | 20 | 86.21 | 82.38 ± 0.15 | 43.58 ± 0.79 |
| llava_1_5_7b | ae_logs | 18 | 90.26 | 90.39 ± 0.15 | 72.63 ± 0.09 |
| llava_1_5_7b | raw_top16 | 22 | 91.76 | 91.88 ± 0.09 | 76.23 ± 0.11 |
| llava_1_5_7b | raw_top32 | 6 | 91.74 | 91.68 ± 0.06 | 75.76 ± 0.32 |
| llava_1_5_7b | norm_top16 | 12 | 92.09 | 91.95 ± 0.04 | 76.24 ± 0.22 |
| llava_1_5_7b | norm_top32 (selected) | 22 | 92.24 | 92.41 ± 0.22 | 76.99 ± 0.65 |
| qwen3_vl_8b | ae_logs (selected) | 22 | 90.16 | 89.85 ± 0.18 | 71.75 ± 0.29 |
| qwen3_vl_8b | raw_top16 | 6 | 89.63 | 89.51 ± 0.06 | 71.95 ± 0.13 |
| qwen3_vl_8b | raw_top32 | 6 | 89.62 | 89.51 ± 0.07 | 71.99 ± 0.08 |
| qwen3_vl_8b | norm_top16 | 18 | 89.80 | 89.77 ± 0.12 | 71.34 ± 0.61 |
| qwen3_vl_8b | norm_top32 | 13 | 89.76 | 89.82 ± 0.03 | 71.21 ± 0.17 |
| internvl_2_5_8b | ae_logs | 16 | 87.37 | 87.81 ± 0.11 | 62.36 ± 0.35 |
| internvl_2_5_8b | raw_top16 | 14 | 87.69 | 89.40 ± 0.10 | 64.81 ± 0.45 |
| internvl_2_5_8b | raw_top32 | 10 | 87.72 | 89.61 ± 0.16 | 66.88 ± 0.64 |
| internvl_2_5_8b | norm_top16 | 17 | 87.71 | 88.22 ± 0.43 | 62.86 ± 1.19 |
| internvl_2_5_8b | norm_top32 (selected) | 21 | 87.95 | 88.23 ± 0.11 | 62.41 ± 0.19 |

## Validation-selected hyperparameters

All rows use train-only standardization and no BatchNorm.

| Model | Feature | Width | Dropout | Activation | LR | WD | Batch | Monitor | Best epochs 43/44/45 |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| qwen2_5_vl_7b | ae_logs (selected) | 256 | 0.5 | relu | 0.000547 | 1e-06 | 128 | val_auroc | 39/36/40 |
| qwen2_5_vl_7b | raw_top16 | 512 | 0.1 | relu | 0.00503 | 0.0001 | 64 | val_auroc | 24/13/15 |
| qwen2_5_vl_7b | raw_top32 | 128 | 0.5 | gelu | 0.00653 | 1e-06 | 256 | val_auroc | 38/25/31 |
| qwen2_5_vl_7b | norm_top16 | 512 | 0.3 | relu | 0.00249 | 0.001 | 128 | val_auroc | 75/58/37 |
| qwen2_5_vl_7b | norm_top32 | 512 | 0.1 | relu | 0.0027 | 0.0001 | 64 | val_auroc | 26/23/29 |
| llava_1_5_7b | ae_logs | 128 | 0.1 | relu | 0.00247 | 0 | 512 | val_loss | 23/30/41 |
| llava_1_5_7b | raw_top16 | 512 | 0.1 | relu | 0.00149 | 0.001 | 256 | val_auroc | 26/45/57 |
| llava_1_5_7b | raw_top32 | 256 | 0.5 | relu | 0.0027 | 1e-05 | 512 | val_auroc | 32/40/56 |
| llava_1_5_7b | norm_top16 | 256 | 0.1 | relu | 0.0096 | 1e-06 | 512 | val_auroc | 31/21/22 |
| llava_1_5_7b | norm_top32 (selected) | 512 | 0.5 | relu | 0.00112 | 1e-06 | 64 | val_auroc | 40/23/43 |
| qwen3_vl_8b | ae_logs (selected) | 256 | 0.5 | relu | 0.00128 | 0.0001 | 128 | val_auroc | 34/33/29 |
| qwen3_vl_8b | raw_top16 | 256 | 0.5 | relu | 0.0027 | 1e-05 | 512 | val_auroc | 62/50/45 |
| qwen3_vl_8b | raw_top32 | 256 | 0.5 | relu | 0.0027 | 1e-05 | 512 | val_auroc | 49/54/45 |
| qwen3_vl_8b | norm_top16 | 256 | 0.0 | relu | 0.00139 | 0 | 512 | val_auroc | 40/38/56 |
| qwen3_vl_8b | norm_top32 | 512 | 0.0 | relu | 0.0005 | 1e-05 | 128 | val_auroc | 37/34/33 |
| internvl_2_5_8b | ae_logs | 1024 | 0.1 | relu | 0.000678 | 0 | 512 | val_auroc | 51/42/67 |
| internvl_2_5_8b | raw_top16 | 1024 | 0.5 | relu | 0.000696 | 1e-06 | 256 | val_auroc | 66/60/80 |
| internvl_2_5_8b | raw_top32 | 1024 | 0.5 | relu | 0.00119 | 1e-05 | 256 | val_auroc | 66/57/59 |
| internvl_2_5_8b | norm_top16 | 128 | 0.5 | relu | 0.00608 | 0.0001 | 256 | val_loss | 46/21/36 |
| internvl_2_5_8b | norm_top32 (selected) | 128 | 0.5 | relu | 0.00415 | 0 | 128 | val_auroc | 49/33/39 |
