# Four-model ENDAC-811 results

Exact first-canonical mentions; 3200/400/400 image split; seeds 43/44/45.

| model | family | head | AUROC (%) | HALL-AUPR (%) |
|---|---|---|---:|---:|
| qwen2_5_vl_7b | native | svar_native | 86.19 ± 0.09 | 43.67 ± 0.91 |
| qwen2_5_vl_7b | native | metatoken_lr | 85.97 ± 0.00 | 50.80 ± 0.00 |
| qwen2_5_vl_7b | native | metatoken_gb | 86.22 ± 0.00 | 50.86 ± 0.58 |
| qwen2_5_vl_7b | shared_mlp_fixed | svar | 86.21 ± 0.26 | 43.63 ± 1.56 |
| qwen2_5_vl_7b | shared_mlp_fixed | metatoken | 75.78 ± 1.35 | 25.80 ± 1.65 |
| qwen2_5_vl_7b | shared_mlp_fixed | visual_only | 84.93 ± 0.53 | 45.82 ± 0.40 |
| qwen2_5_vl_7b | shared_mlp_fixed | all_attention | 85.16 ± 0.72 | 46.11 ± 0.99 |
| qwen2_5_vl_7b | shared_mlp_single | svar | 86.54 ± 0.32 | 49.73 ± 1.13 |
| qwen2_5_vl_7b | shared_mlp_single | metatoken | 87.47 ± 0.33 | 53.31 ± 0.68 |
| qwen2_5_vl_7b | shared_mlp_single | visual_only | 86.71 ± 0.31 | 53.47 ± 0.56 |
| qwen2_5_vl_7b | shared_mlp_single | all_attention | 87.11 ± 0.39 | 53.90 ± 0.63 |
| llava_1_5_7b | native | svar_native | 90.68 ± 0.02 | 72.37 ± 0.02 |
| llava_1_5_7b | native | metatoken_lr | 90.21 ± 0.00 | 72.32 ± 0.00 |
| llava_1_5_7b | native | metatoken_gb | 89.90 ± 0.00 | 71.37 ± 0.01 |
| llava_1_5_7b | shared_mlp_fixed | svar | 90.89 ± 0.05 | 72.70 ± 0.29 |
| llava_1_5_7b | shared_mlp_fixed | metatoken | 88.15 ± 0.10 | 67.89 ± 0.30 |
| llava_1_5_7b | shared_mlp_fixed | visual_only | 89.84 ± 0.08 | 71.69 ± 0.28 |
| llava_1_5_7b | shared_mlp_fixed | all_attention | 90.62 ± 0.11 | 75.06 ± 0.20 |
| llava_1_5_7b | shared_mlp_single | svar | 91.05 ± 0.01 | 73.14 ± 0.29 |
| llava_1_5_7b | shared_mlp_single | metatoken | 90.67 ± 0.19 | 72.56 ± 0.51 |
| llava_1_5_7b | shared_mlp_single | visual_only | 90.01 ± 0.08 | 71.68 ± 0.06 |
| llava_1_5_7b | shared_mlp_single | all_attention | 90.52 ± 0.10 | 74.97 ± 0.32 |
| qwen3_vl_8b | native | svar_native | 88.95 ± 0.13 | 68.25 ± 0.35 |
| qwen3_vl_8b | native | metatoken_lr | 87.68 ± 0.00 | 65.80 ± 0.00 |
| qwen3_vl_8b | native | metatoken_gb | 88.29 ± 0.02 | 68.88 ± 0.28 |
| qwen3_vl_8b | shared_mlp_fixed | svar | 89.40 ± 0.59 | 69.29 ± 1.29 |
| qwen3_vl_8b | shared_mlp_fixed | metatoken | 82.23 ± 0.82 | 49.93 ± 1.82 |
| qwen3_vl_8b | shared_mlp_fixed | visual_only | 90.29 ± 0.24 | 71.59 ± 0.48 |
| qwen3_vl_8b | shared_mlp_fixed | all_attention | 92.78 ± 0.14 | 76.46 ± 0.06 |
| qwen3_vl_8b | shared_mlp_single | svar | 88.98 ± 0.10 | 69.23 ± 0.62 |
| qwen3_vl_8b | shared_mlp_single | metatoken | 89.39 ± 0.14 | 69.56 ± 0.85 |
| qwen3_vl_8b | shared_mlp_single | visual_only | 89.79 ± 0.26 | 72.40 ± 0.53 |
| qwen3_vl_8b | shared_mlp_single | all_attention | 92.69 ± 0.06 | 77.20 ± 0.37 |
| internvl_2_5_8b | native | svar_native | 88.47 ± 0.31 | 66.46 ± 1.11 |
| internvl_2_5_8b | native | metatoken_lr | 86.39 ± 0.00 | 60.88 ± 0.00 |
| internvl_2_5_8b | native | metatoken_gb | 87.15 ± 0.02 | 63.30 ± 0.06 |
| internvl_2_5_8b | shared_mlp_fixed | svar | 88.66 ± 0.25 | 66.78 ± 1.01 |
| internvl_2_5_8b | shared_mlp_fixed | metatoken | 84.08 ± 0.44 | 54.63 ± 0.69 |
| internvl_2_5_8b | shared_mlp_fixed | visual_only | 87.42 ± 0.68 | 61.42 ± 1.66 |
| internvl_2_5_8b | shared_mlp_fixed | all_attention | 88.09 ± 0.35 | 62.25 ± 0.65 |
| internvl_2_5_8b | shared_mlp_single | svar | 88.80 ± 0.10 | 67.17 ± 0.22 |
| internvl_2_5_8b | shared_mlp_single | metatoken | 87.67 ± 0.25 | 63.08 ± 0.75 |
| internvl_2_5_8b | shared_mlp_single | visual_only | 88.12 ± 0.29 | 63.02 ± 0.82 |
| internvl_2_5_8b | shared_mlp_single | all_attention | 88.58 ± 0.20 | 64.00 ± 0.39 |
