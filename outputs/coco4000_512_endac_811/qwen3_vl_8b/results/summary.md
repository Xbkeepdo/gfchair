# qwen3_vl_8b ENDAC-811 results

3200/400/400 image split; seeds 43/44/45. Scores are mean ± population std.

| family | head | AUROC (%) | HALL-AUPR (%) |
|---|---|---:|---:|
| native | svar_native | 88.95 ± 0.13 | 68.25 ± 0.35 |
| native | metatoken_lr | 87.68 ± 0.00 | 65.80 ± 0.00 |
| native | metatoken_gb | 88.29 ± 0.02 | 68.88 ± 0.28 |
| shared_mlp_fixed | svar | 89.40 ± 0.59 | 69.29 ± 1.29 |
| shared_mlp_fixed | metatoken | 82.23 ± 0.82 | 49.93 ± 1.82 |
| shared_mlp_fixed | visual_only | 90.29 ± 0.24 | 71.59 ± 0.48 |
| shared_mlp_fixed | all_attention | 92.78 ± 0.14 | 76.46 ± 0.06 |
| shared_mlp_single | svar | 88.98 ± 0.10 | 69.23 ± 0.62 |
| shared_mlp_single | metatoken | 89.39 ± 0.14 | 69.56 ± 0.85 |
| shared_mlp_single | visual_only | 89.79 ± 0.26 | 72.40 ± 0.53 |
| shared_mlp_single | all_attention | 92.69 ± 0.06 | 77.20 ± 0.37 |
