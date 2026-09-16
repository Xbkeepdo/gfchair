# minigpt4_7b ENDAC-811 results

3200/400/400 image split; seeds 43/44/45. Scores are mean ± population std.

| family | head | AUROC (%) | HALL-AUPR (%) |
|---|---|---:|---:|
| native | svar_native | 93.81 ± 0.35 | 79.39 ± 0.48 |
| native | metatoken_lr | 91.94 ± 0.00 | 70.00 ± 0.00 |
| native | metatoken_gb | 92.24 ± 0.00 | 73.35 ± 0.05 |
| shared_mlp_fixed | svar | 94.17 ± 0.21 | 80.85 ± 0.68 |
| shared_mlp_fixed | metatoken | 89.78 ± 0.59 | 63.02 ± 2.22 |
| shared_mlp_fixed | visual_only | 92.42 ± 0.30 | 76.41 ± 1.03 |
| shared_mlp_fixed | all_attention | 93.56 ± 0.13 | 79.08 ± 0.85 |
| shared_mlp_single | svar | 94.26 ± 0.10 | 80.20 ± 0.45 |
| shared_mlp_single | metatoken | 92.08 ± 0.12 | 74.08 ± 0.46 |
| shared_mlp_single | visual_only | 93.10 ± 0.32 | 77.20 ± 0.43 |
| shared_mlp_single | all_attention | 93.50 ± 0.08 | 78.45 ± 0.13 |
