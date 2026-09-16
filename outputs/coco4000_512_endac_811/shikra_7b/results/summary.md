# shikra_7b ENDAC-811 results

3200/400/400 image split; seeds 43/44/45. Scores are mean ± population std.

| family | head | AUROC (%) | HALL-AUPR (%) |
|---|---|---:|---:|
| native | svar_native | 88.23 ± 0.03 | 68.96 ± 0.32 |
| native | metatoken_lr | 87.91 ± 0.00 | 68.07 ± 0.00 |
| native | metatoken_gb | 87.59 ± 0.00 | 66.39 ± 0.02 |
| shared_mlp_fixed | svar | 87.56 ± 0.03 | 67.76 ± 0.11 |
| shared_mlp_fixed | metatoken | 87.47 ± 0.06 | 66.72 ± 0.17 |
| shared_mlp_fixed | visual_only | 86.11 ± 0.11 | 66.26 ± 0.15 |
| shared_mlp_fixed | all_attention | 87.32 ± 0.29 | 68.81 ± 0.42 |
| shared_mlp_single | svar | 88.08 ± 0.02 | 69.65 ± 0.13 |
| shared_mlp_single | metatoken | 87.48 ± 0.08 | 66.60 ± 0.80 |
| shared_mlp_single | visual_only | 86.29 ± 0.03 | 65.31 ± 0.27 |
| shared_mlp_single | all_attention | 87.60 ± 0.19 | 67.65 ± 0.83 |
