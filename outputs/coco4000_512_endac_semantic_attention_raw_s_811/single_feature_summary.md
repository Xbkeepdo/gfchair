# Each all-visual semantic-attention feature: raw S_E vs log1p(S_E)

For each feature, raw/norm is fixed by the earlier logS three-seed validation AUROC, then HALL-AUPR; the same variant is used for S and logS.
Test scores are three-seed means (%); delta is S minus logS in percentage points.

## qwen2_5_vl_7b

| feature | fixed variant | +S AUROC/AP | +logS AUROC/AP | delta pp |
|---|---|---:|---:|---:|
| semantic_only | raw | 84.12 / 47.13 | 83.94 / 45.86 | +0.18 / +1.27 |
| attention_only | raw | 83.39 / 42.05 | 83.37 / 42.39 | +0.02 / -0.34 |
| cosine_only | raw | 85.59 / 48.16 | 85.29 / 46.49 | +0.30 / +1.67 |
| semantic_attention | norm | 82.76 / 41.13 | 82.84 / 41.44 | -0.08 / -0.31 |
| cosine_attention | raw | 83.49 / 42.29 | 83.65 / 43.09 | -0.16 / -0.79 |
| semantic_attention_positive_cosine | norm | 82.76 / 41.09 | 82.60 / 41.94 | +0.16 / -0.85 |

## llava_1_5_7b

| feature | fixed variant | +S AUROC/AP | +logS AUROC/AP | delta pp |
|---|---|---:|---:|---:|
| semantic_only | norm | 92.27 / 77.71 | 92.40 / 78.23 | -0.13 / -0.53 |
| attention_only | raw | 89.01 / 69.56 | 89.38 / 69.71 | -0.37 / -0.15 |
| cosine_only | raw | 90.25 / 71.60 | 90.20 / 71.35 | +0.05 / +0.25 |
| semantic_attention | norm | 92.34 / 76.83 | 92.56 / 77.71 | -0.22 / -0.88 |
| cosine_attention | raw | 88.73 / 69.86 | 89.05 / 70.46 | -0.32 / -0.60 |
| semantic_attention_positive_cosine | norm | 92.25 / 76.63 | 92.44 / 77.24 | -0.18 / -0.61 |

## qwen3_vl_8b

| feature | fixed variant | +S AUROC/AP | +logS AUROC/AP | delta pp |
|---|---|---:|---:|---:|
| semantic_only | raw | 91.06 / 74.46 | 91.38 / 74.51 | -0.32 / -0.05 |
| attention_only | raw | 89.43 / 71.52 | 89.66 / 71.31 | -0.23 / +0.21 |
| cosine_only | raw | 90.41 / 74.33 | 90.49 / 74.03 | -0.07 / +0.30 |
| semantic_attention | norm | 89.78 / 72.40 | 90.10 / 72.58 | -0.31 / -0.18 |
| cosine_attention | raw | 89.86 / 72.38 | 90.12 / 72.83 | -0.26 / -0.45 |
| semantic_attention_positive_cosine | norm | 89.91 / 72.68 | 90.19 / 72.66 | -0.29 / +0.01 |

## internvl_2_5_8b

| feature | fixed variant | +S AUROC/AP | +logS AUROC/AP | delta pp |
|---|---|---:|---:|---:|
| semantic_only | raw | 88.41 / 64.96 | 88.57 / 65.32 | -0.16 / -0.36 |
| attention_only | raw | 86.81 / 61.48 | 86.89 / 61.19 | -0.08 / +0.29 |
| cosine_only | raw | 88.11 / 64.23 | 88.21 / 64.93 | -0.10 / -0.70 |
| semantic_attention | norm | 88.13 / 61.69 | 88.21 / 62.61 | -0.08 / -0.92 |
| cosine_attention | raw | 87.40 / 63.75 | 87.11 / 62.59 | +0.30 / +1.15 |
| semantic_attention_positive_cosine | norm | 88.12 / 61.76 | 88.23 / 62.57 | -0.11 / -0.81 |
