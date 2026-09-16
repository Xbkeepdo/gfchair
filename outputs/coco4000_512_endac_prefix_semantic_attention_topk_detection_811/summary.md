# Semantic-attention Top-K ENDAC-811 detection

固定 3200/400/400 图片划分；seeds 43/44/45；train-only z-score；
单隐藏 128/ReLU/dropout .3、无 BN；最低 validation BCE loss checkpoint。
所有 84 个预注册特征组在访问 test 前完成 validation 选择。

## Validation-selected champions

| model | variant/group | test AUROC (%) | test HALL-AUPR (%) |
|---|---|---:|---:|
| minigpt4_7b | norm_top16/all_six+S | 93.67 ± 0.04 | 78.16 ± 0.08 |
| shikra_7b | raw_top32/all_six+S | 87.79 ± 0.24 | 69.91 ± 0.37 |

## 2-model test macro average (supplementary)

| variant/group | AUROC (%) | HALL-AUPR (%) |
|---|---:|---:|
| raw_top32/all_six+logS | 90.94 | 74.43 |
| norm_top32/all_six+S **(global validation champion)** | 90.91 | 74.69 |
| raw_top32/semantic_only+S | 90.89 | 72.96 |
| norm_top32/all_six+logS | 90.88 | 74.52 |
| norm_top16/semantic_only+S | 90.88 | 73.02 |
| norm_top32/semantic_only+S | 90.87 | 73.10 |
| raw_top32/semantic_only+logS | 90.85 | 72.91 |
| raw_top32/all_six+S | 90.83 | 74.15 |
| raw_top16/semantic_only+logS | 90.82 | 72.73 |
| norm_top32/semantic_only+logS | 90.81 | 72.95 |
| norm_top16/semantic_only+logS | 90.80 | 72.99 |
| raw_top16/semantic_only+S | 90.79 | 72.64 |
| norm_top16/all_six+logS | 90.78 | 74.07 |
| norm_top16/all_six+S | 90.78 | 74.28 |
| raw_top16/all_six+logS | 90.69 | 73.15 |
| raw_top16/all_six+S | 90.67 | 73.22 |
| raw_top16/semantic_attention+logS | 90.57 | 72.36 |
| norm_top16/semantic_attention+S | 90.57 | 72.53 |
| raw_top32/semantic_attention+S | 90.55 | 72.14 |
| raw_top32/semantic_attention+logS | 90.55 | 72.25 |
| norm_top32/semantic_attention+logS | 90.53 | 72.59 |
| raw_top16/semantic_attention+S | 90.52 | 72.07 |
| norm_top16/semantic_attention+logS | 90.52 | 72.51 |
| norm_top32/semantic_attention+S | 90.49 | 72.39 |
| raw_top16/semantic_attention_positive_cosine+logS | 90.49 | 72.18 |
| raw_top32/semantic_attention_positive_cosine+logS | 90.48 | 72.22 |
| raw_top16/semantic_attention_positive_cosine+S | 90.46 | 72.01 |
| raw_top32/semantic_attention_positive_cosine+S | 90.43 | 71.90 |
| norm_top16/semantic_attention_positive_cosine+S | 90.42 | 72.07 |
| norm_top32/semantic_attention_positive_cosine+S | 90.42 | 72.17 |
| norm_top32/semantic_attention_positive_cosine+logS | 90.42 | 72.37 |
| norm_top16/semantic_attention_positive_cosine+logS | 90.41 | 72.21 |
| norm_top16/cosine_attention+S | 90.03 | 72.55 |
| raw_top16/cosine_only+S | 89.99 | 71.80 |
| raw_top16/cosine_only+logS | 89.91 | 71.38 |
| raw_top32/attention_only+logS | 89.89 | 71.70 |
| norm_top32/cosine_attention+logS | 89.88 | 72.07 |
| raw_top32/cosine_only+logS | 89.88 | 72.22 |
| raw_top32/cosine_attention+logS | 89.87 | 71.90 |
| norm_top16/cosine_attention+logS | 89.87 | 72.25 |
| raw_top16/cosine_attention+logS | 89.86 | 72.91 |
| raw_top32/cosine_attention+S | 89.85 | 71.97 |
| raw_top16/cosine_attention+S | 89.84 | 72.69 |
| raw_top16/attention_only+logS | 89.84 | 72.66 |
| norm_top32/cosine_attention+S | 89.84 | 71.94 |
| raw_top16/attention_only+S | 89.82 | 72.73 |
| raw_top32/attention_only+S | 89.81 | 71.47 |
| raw_top32/cosine_only+S | 89.79 | 72.33 |
| norm_top16/attention_only+S | 89.79 | 71.99 |
| norm_top16/cosine_only+S | 89.78 | 71.85 |
| norm_top32/attention_only+logS | 89.77 | 72.01 |
| norm_top32/attention_only+S | 89.71 | 71.56 |
| norm_top16/cosine_only+logS | 89.71 | 71.58 |
| norm_top32/cosine_only+S | 89.70 | 72.39 |
| norm_top32/cosine_only+logS | 89.70 | 71.93 |
| norm_top16/attention_only+logS | 89.68 | 71.89 |
| raw_top32/all_six | 89.47 | 70.67 |
| norm_top32/all_six | 89.41 | 70.68 |
| norm_top16/all_six | 89.40 | 70.68 |
| raw_top16/all_six | 88.84 | 69.09 |
| raw_top32/attention_only | 86.59 | 64.01 |
| raw_top32/cosine_only | 86.22 | 66.32 |
| raw_top32/cosine_attention | 86.19 | 64.72 |
| norm_top32/attention_only | 86.05 | 63.45 |
| raw_top16/cosine_only | 85.96 | 65.29 |
| norm_top32/cosine_only | 85.93 | 66.07 |
| raw_top16/attention_only | 85.84 | 64.60 |
| norm_top32/cosine_attention | 85.71 | 63.85 |
| norm_top16/cosine_only | 85.32 | 63.55 |
| raw_top16/cosine_attention | 85.17 | 62.70 |
| norm_top16/cosine_attention | 84.98 | 62.97 |
| norm_top16/attention_only | 84.90 | 62.98 |
| norm_top32/semantic_attention_positive_cosine | 81.46 | 51.20 |
| norm_top16/semantic_attention_positive_cosine | 81.26 | 50.80 |
| norm_top32/semantic_attention | 80.71 | 50.55 |
| norm_top16/semantic_attention | 80.50 | 49.97 |
| raw_top32/semantic_attention_positive_cosine | 80.08 | 48.30 |
| raw_top16/semantic_attention_positive_cosine | 79.74 | 47.63 |
| norm_top16/semantic_only | 79.47 | 51.13 |
| norm_top32/semantic_only | 79.35 | 50.60 |
| raw_top32/semantic_attention | 79.08 | 46.85 |
| raw_top16/semantic_attention | 78.94 | 46.67 |
| raw_top32/semantic_only | 77.53 | 45.38 |
| raw_top16/semantic_only | 77.50 | 45.29 |
