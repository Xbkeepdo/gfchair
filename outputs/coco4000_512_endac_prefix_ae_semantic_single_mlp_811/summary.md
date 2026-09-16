# Prefix AE vs semantic×attention + log1p(S_E), Torch single MLP

Train-only standardization, no BN; 24 frozen candidates per feature;
seed43 shortlist top3, seeds44/45 validation selection, then test.
3200/400/400 image split; AUROC/HALL-AUPR are three-seed test mean ± population std.

| Model | Feature | Val AUROC (%) | Test AUROC (%) | Test HALL-AUPR (%) |
|---|---|---:|---:|---:|
| minigpt4_7b | ae_logs | 92.27 | 93.53 ± 0.01 | 77.95 ± 0.20 |
| minigpt4_7b | raw_top16 | 93.43 | 93.38 ± 0.08 | 76.86 ± 0.31 |
| minigpt4_7b | raw_top32 | 93.34 | 93.20 ± 0.12 | 76.07 ± 0.49 |
| minigpt4_7b | norm_top16 | 93.89 | 93.46 ± 0.05 | 76.83 ± 0.25 |
| minigpt4_7b | norm_top32 (selected) | 93.91 | 93.47 ± 0.06 | 76.84 ± 0.29 |
| shikra_7b | ae_logs | 86.18 | 86.22 ± 0.18 | 64.43 ± 0.84 |
| shikra_7b | raw_top16 (selected) | 88.27 | 87.82 ± 0.12 | 68.51 ± 0.25 |
| shikra_7b | raw_top32 | 88.17 | 87.90 ± 0.19 | 69.00 ± 0.53 |
| shikra_7b | norm_top16 | 88.20 | 87.48 ± 0.23 | 68.24 ± 0.75 |
| shikra_7b | norm_top32 | 88.20 | 87.34 ± 0.30 | 67.53 ± 0.50 |
