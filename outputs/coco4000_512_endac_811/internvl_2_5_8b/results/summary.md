# internvl_2_5_8b ENDAC-811 results

3200/400/400 image split; seeds 43/44/45. Scores are mean ± population std.

| family | head | AUROC (%) | HALL-AUPR (%) |
|---|---|---:|---:|
| native | svar_native | 88.47 ± 0.31 | 66.46 ± 1.11 |
| native | metatoken_lr | 86.39 ± 0.00 | 60.88 ± 0.00 |
| native | metatoken_gb | 87.15 ± 0.02 | 63.30 ± 0.06 |
| shared_mlp_fixed | svar | 88.66 ± 0.25 | 66.78 ± 1.01 |
| shared_mlp_fixed | metatoken | 84.08 ± 0.44 | 54.63 ± 0.69 |
| shared_mlp_fixed | visual_only | 87.42 ± 0.68 | 61.42 ± 1.66 |
| shared_mlp_fixed | all_attention | 88.09 ± 0.35 | 62.25 ± 0.65 |
| shared_mlp_single | svar | 88.80 ± 0.10 | 67.17 ± 0.22 |
| shared_mlp_single | metatoken | 87.67 ± 0.25 | 63.08 ± 0.75 |
| shared_mlp_single | visual_only | 88.12 ± 0.29 | 63.02 ± 0.82 |
| shared_mlp_single | all_attention | 88.58 ± 0.20 | 64.00 ± 0.39 |
