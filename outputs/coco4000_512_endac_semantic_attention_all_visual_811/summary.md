# All-visual-token semantic-attention ENDAC-811

T_k is replaced by every visual token. Original mean and sum reductions are retained.
Fixed 3200/400/400 image split; seeds 43/44/45; fixed single-hidden 128 MLP.
All validation choices were frozen before test evaluation.

## qwen2_5_vl_7b

| feature | raw | norm | raw + logS | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 63.12 / 16.90 | 76.24 / 34.30 | 83.94 / 45.86 | 82.90 / 44.28 |
| attention_only | 81.59 / 38.11 | 81.59 / 38.11 | 83.37 / 42.39 | 83.37 / 42.39 |
| cosine_only | 79.61 / 34.67 | 79.61 / 34.67 | 85.29 / 46.49 | 85.29 / 46.49 |
| semantic_attention | 73.56 / 26.18 | 76.31 / 29.95 | 81.86 / 37.91 | 82.84 / 41.44 |
| cosine_attention | 82.52 / 39.32 | 82.52 / 39.32 | 83.65 / 43.09 | 83.65 / 43.09 |
| semantic_attention_positive_cosine | 73.49 / 25.82 | 75.28 / 28.99 | 81.65 / 37.15 | 82.60 / 41.94 |
| all_six | 84.02 / 48.23 | 85.52 / 53.25 | 84.57 / 46.52 | 85.99 / 52.51 |

Validation champion: plus_logS/norm/all_six.

## llava_1_5_7b

| feature | raw | norm | raw + logS | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 89.30 / 73.24 | 85.59 / 65.53 | 92.42 / 78.30 | 92.40 / 78.23 |
| attention_only | 87.73 / 67.76 | 87.73 / 67.76 | 89.38 / 69.71 | 89.38 / 69.71 |
| cosine_only | 87.88 / 66.11 | 87.88 / 66.11 | 90.20 / 71.35 | 90.20 / 71.35 |
| semantic_attention | 90.14 / 74.15 | 86.13 / 63.45 | 91.73 / 76.09 | 92.56 / 77.71 |
| cosine_attention | 87.71 / 67.17 | 87.71 / 67.17 | 89.05 / 70.46 | 89.05 / 70.46 |
| semantic_attention_positive_cosine | 89.78 / 73.80 | 86.48 / 63.31 | 91.96 / 76.15 | 92.44 / 77.24 |
| all_six | 92.27 / 77.75 | 92.76 / 78.25 | 92.86 / 79.14 | 93.24 / 79.47 |

Validation champion: plus_logS/norm/all_six.

## qwen3_vl_8b

| feature | raw | norm | raw + logS | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 85.08 / 62.06 | 80.64 / 51.91 | 91.38 / 74.51 | 90.67 / 72.27 |
| attention_only | 87.75 / 68.11 | 87.75 / 68.11 | 89.66 / 71.31 | 89.66 / 71.31 |
| cosine_only | 84.33 / 62.61 | 84.33 / 62.61 | 90.49 / 74.03 | 90.49 / 74.03 |
| semantic_attention | 82.81 / 56.27 | 79.84 / 47.52 | 89.43 / 71.38 | 90.10 / 72.58 |
| cosine_attention | 87.90 / 68.93 | 87.90 / 68.93 | 90.12 / 72.83 | 90.12 / 72.83 |
| semantic_attention_positive_cosine | 81.92 / 53.55 | 79.96 / 47.40 | 89.22 / 71.17 | 90.19 / 72.66 |
| all_six | 91.31 / 76.39 | 91.70 / 76.02 | 92.09 / 76.67 | 92.53 / 76.87 |

Validation champion: plus_logS/raw/all_six.

## internvl_2_5_8b

| feature | raw | norm | raw + logS | norm + logS |
|---|---:|---:|---:|---:|
| semantic_only | 85.93 / 62.61 | 84.89 / 55.22 | 88.57 / 65.32 | 88.98 / 63.87 |
| attention_only | 85.16 / 59.33 | 85.16 / 59.33 | 86.89 / 61.19 | 86.89 / 61.19 |
| cosine_only | 84.10 / 57.83 | 84.10 / 57.83 | 88.21 / 64.93 | 88.21 / 64.93 |
| semantic_attention | 86.26 / 59.88 | 84.41 / 55.36 | 88.27 / 63.75 | 88.21 / 62.61 |
| cosine_attention | 84.31 / 58.62 | 84.31 / 58.62 | 87.11 / 62.59 | 87.11 / 62.59 |
| semantic_attention_positive_cosine | 85.89 / 61.33 | 84.68 / 56.29 | 88.41 / 64.69 | 88.23 / 62.57 |
| all_six | 89.37 / 67.13 | 89.99 / 66.72 | 88.78 / 66.83 | 89.96 / 66.55 |

Validation champion: plus_logS/norm/all_six.
