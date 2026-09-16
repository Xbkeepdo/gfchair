# Raw S_E vs log1p(S_E), ENDAC-811

Same full-visual features, image split, three seeds, and fixed 128/no-BN MLP.
Raw S_E = expm1(saved float32 log1p(S_E)); validation selection was frozen before test.
All entries are test AUROC / HALL-AUPR (%), three-seed means.

## qwen2_5_vl_7b

| feature | raw + S | raw + logS | norm + S | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 84.12 / 47.13 | 83.94 / 45.86 | 82.87 / 45.07 | 82.90 / 44.28 |
| attention_only | 83.39 / 42.05 | 83.37 / 42.39 | 83.39 / 42.05 | 83.37 / 42.39 |
| cosine_only | 85.59 / 48.16 | 85.29 / 46.49 | 85.59 / 48.16 | 85.29 / 46.49 |
| semantic_attention | 82.14 / 38.24 | 81.86 / 37.91 | 82.76 / 41.13 | 82.84 / 41.44 |
| cosine_attention | 83.49 / 42.29 | 83.65 / 43.09 | 83.49 / 42.29 | 83.65 / 43.09 |
| semantic_attention_positive_cosine | 81.96 / 36.68 | 81.65 / 37.15 | 82.76 / 41.09 | 82.60 / 41.94 |
| all_six | 84.72 / 45.50 | 84.57 / 46.52 | 85.84 / 52.01 | 85.99 / 52.51 |

| reference | AUROC / HALL-AUPR |
|---|---:|
| S_only | 82.38 / 40.52 |
| logS_only | 81.93 / 38.69 |
| AE_only | 78.07 / 37.37 |
| AE_plus_S | 86.57 / 52.38 |
| AE_plus_logS | 86.98 / 53.06 |

## llava_1_5_7b

| feature | raw + S | raw + logS | norm + S | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 92.19 / 77.41 | 92.42 / 78.30 | 92.27 / 77.71 | 92.40 / 78.23 |
| attention_only | 89.01 / 69.56 | 89.38 / 69.71 | 89.01 / 69.56 | 89.38 / 69.71 |
| cosine_only | 90.25 / 71.60 | 90.20 / 71.35 | 90.25 / 71.60 | 90.20 / 71.35 |
| semantic_attention | 91.58 / 75.60 | 91.73 / 76.09 | 92.34 / 76.83 | 92.56 / 77.71 |
| cosine_attention | 88.73 / 69.86 | 89.05 / 70.46 | 88.73 / 69.86 | 89.05 / 70.46 |
| semantic_attention_positive_cosine | 91.66 / 75.02 | 91.96 / 76.15 | 92.25 / 76.63 | 92.44 / 77.24 |
| all_six | 92.56 / 78.55 | 92.86 / 79.14 | 93.29 / 79.88 | 93.24 / 79.47 |

| reference | AUROC / HALL-AUPR |
|---|---:|
| S_only | 88.64 / 68.81 |
| logS_only | 89.00 / 69.89 |
| AE_only | 85.62 / 61.35 |
| AE_plus_S | 90.21 / 72.33 |
| AE_plus_logS | 90.32 / 72.37 |

## qwen3_vl_8b

| feature | raw + S | raw + logS | norm + S | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 91.06 / 74.46 | 91.38 / 74.51 | 90.28 / 72.22 | 90.67 / 72.27 |
| attention_only | 89.43 / 71.52 | 89.66 / 71.31 | 89.43 / 71.52 | 89.66 / 71.31 |
| cosine_only | 90.41 / 74.33 | 90.49 / 74.03 | 90.41 / 74.33 | 90.49 / 74.03 |
| semantic_attention | 89.16 / 71.60 | 89.43 / 71.38 | 89.78 / 72.40 | 90.10 / 72.58 |
| cosine_attention | 89.86 / 72.38 | 90.12 / 72.83 | 89.86 / 72.38 | 90.12 / 72.83 |
| semantic_attention_positive_cosine | 89.03 / 71.67 | 89.22 / 71.17 | 89.91 / 72.68 | 90.19 / 72.66 |
| all_six | 92.01 / 76.99 | 92.09 / 76.67 | 92.19 / 76.92 | 92.53 / 76.87 |

| reference | AUROC / HALL-AUPR |
|---|---:|
| S_only | 88.19 / 69.03 |
| logS_only | 88.45 / 68.36 |
| AE_only | 83.17 / 57.71 |
| AE_plus_S | 89.76 / 71.87 |
| AE_plus_logS | 89.83 / 71.73 |

## internvl_2_5_8b

| feature | raw + S | raw + logS | norm + S | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 88.41 / 64.96 | 88.57 / 65.32 | 89.03 / 63.21 | 88.98 / 63.87 |
| attention_only | 86.81 / 61.48 | 86.89 / 61.19 | 86.81 / 61.48 | 86.89 / 61.19 |
| cosine_only | 88.11 / 64.23 | 88.21 / 64.93 | 88.11 / 64.23 | 88.21 / 64.93 |
| semantic_attention | 88.07 / 62.57 | 88.27 / 63.75 | 88.13 / 61.69 | 88.21 / 62.61 |
| cosine_attention | 87.40 / 63.75 | 87.11 / 62.59 | 87.40 / 63.75 | 87.11 / 62.59 |
| semantic_attention_positive_cosine | 88.18 / 64.02 | 88.41 / 64.69 | 88.12 / 61.76 | 88.23 / 62.57 |
| all_six | 89.27 / 67.62 | 88.78 / 66.83 | 89.86 / 66.66 | 89.96 / 66.55 |

| reference | AUROC / HALL-AUPR |
|---|---:|
| S_only | 85.97 / 57.68 |
| logS_only | 86.45 / 60.22 |
| AE_only | 82.75 / 53.21 |
| AE_plus_S | 87.10 / 60.42 |
| AE_plus_logS | 86.98 / 60.23 |
