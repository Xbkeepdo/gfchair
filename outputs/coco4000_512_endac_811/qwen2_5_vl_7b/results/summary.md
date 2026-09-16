# qwen2_5_vl_7b ENDAC-811 results

3200/400/400 image split; seeds 43/44/45. Scores are mean ± population std.

| family | head | AUROC (%) | HALL-AUPR (%) |
|---|---|---:|---:|
| native | svar_native | 86.19 ± 0.09 | 43.67 ± 0.91 |
| native | metatoken_lr | 85.97 ± 0.00 | 50.80 ± 0.00 |
| native | metatoken_gb | 86.22 ± 0.00 | 50.86 ± 0.58 |
| shared_mlp_fixed | svar | 86.21 ± 0.26 | 43.63 ± 1.56 |
| shared_mlp_fixed | metatoken | 75.78 ± 1.35 | 25.80 ± 1.65 |
| shared_mlp_fixed | visual_only | 84.93 ± 0.53 | 45.82 ± 0.40 |
| shared_mlp_fixed | all_attention | 85.16 ± 0.72 | 46.11 ± 0.99 |
| shared_mlp_single | svar | 86.54 ± 0.32 | 49.73 ± 1.13 |
| shared_mlp_single | metatoken | 87.47 ± 0.33 | 53.31 ± 0.68 |
| shared_mlp_single | visual_only | 86.71 ± 0.31 | 53.47 ± 0.56 |
| shared_mlp_single | all_attention | 87.11 ± 0.39 | 53.90 ± 0.63 |
