# llava_1_5_7b ENDAC-811 results

3200/400/400 image split; seeds 43/44/45. Scores are mean ± population std.

| family | head | AUROC (%) | HALL-AUPR (%) |
|---|---|---:|---:|
| native | svar_native | 90.68 ± 0.02 | 72.37 ± 0.02 |
| native | metatoken_lr | 90.21 ± 0.00 | 72.32 ± 0.00 |
| native | metatoken_gb | 89.90 ± 0.00 | 71.37 ± 0.01 |
| shared_mlp_fixed | svar | 90.89 ± 0.05 | 72.70 ± 0.29 |
| shared_mlp_fixed | metatoken | 88.15 ± 0.10 | 67.89 ± 0.30 |
| shared_mlp_fixed | visual_only | 89.84 ± 0.08 | 71.69 ± 0.28 |
| shared_mlp_fixed | all_attention | 90.62 ± 0.11 | 75.06 ± 0.20 |
| shared_mlp_single | svar | 91.05 ± 0.01 | 73.14 ± 0.29 |
| shared_mlp_single | metatoken | 90.67 ± 0.19 | 72.56 ± 0.51 |
| shared_mlp_single | visual_only | 90.01 ± 0.08 | 71.68 ± 0.06 |
| shared_mlp_single | all_attention | 90.52 ± 0.10 | 74.97 ± 0.32 |
