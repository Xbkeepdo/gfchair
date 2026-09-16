# Semantic-attention Top-K ENDAC-811 detection

固定 3200/400/400 图片划分；seeds 43/44/45；train-only z-score；
单隐藏 128/ReLU/dropout .3、无 BN；最低 validation BCE loss checkpoint。
所有 28 个预注册特征组在访问 test 前完成 validation 选择。

## Validation-selected champions

| model | variant/group | test AUROC (%) | test HALL-AUPR (%) |
|---|---|---:|---:|
| qwen2_5_vl_7b | norm_top32/all_six | 82.54 ± 0.58 | 45.92 ± 2.94 |
| llava_1_5_7b | norm_top32/all_six | 91.79 ± 0.13 | 74.85 ± 0.07 |
| qwen3_vl_8b | raw_top32/all_six | 88.60 ± 0.04 | 72.05 ± 0.38 |
| internvl_2_5_8b | norm_top32/all_six | 89.79 ± 0.10 | 64.84 ± 0.22 |

## Four-model test macro average

| variant/group | AUROC (%) | HALL-AUPR (%) |
|---|---:|---:|
| norm_top32/all_six **(global validation champion)** | 88.18 | 63.48 |
| norm_top16/all_six | 87.20 | 60.97 |
| raw_top32/all_six | 86.61 | 61.03 |
| raw_top16/all_six | 85.79 | 59.84 |
| norm_top32/attention_only | 83.96 | 54.99 |
| norm_top32/cosine_attention | 83.74 | 54.72 |
| norm_top16/attention_only | 83.27 | 53.05 |
| norm_top16/cosine_attention | 83.06 | 53.28 |
| raw_top32/attention_only | 82.97 | 53.09 |
| raw_top32/cosine_attention | 82.84 | 52.44 |
| norm_top32/cosine_only | 82.68 | 54.26 |
| raw_top16/cosine_attention | 82.47 | 53.66 |
| raw_top16/attention_only | 82.23 | 54.06 |
| raw_top32/semantic_attention | 81.95 | 52.01 |
| raw_top32/cosine_only | 81.77 | 50.68 |
| norm_top16/cosine_only | 81.62 | 52.35 |
| norm_top32/semantic_attention_positive_cosine | 81.57 | 49.72 |
| raw_top32/semantic_attention_positive_cosine | 81.40 | 50.91 |
| raw_top16/semantic_attention | 81.40 | 51.10 |
| norm_top32/semantic_attention | 81.31 | 48.70 |
| norm_top16/semantic_attention_positive_cosine | 81.23 | 49.25 |
| raw_top16/semantic_attention_positive_cosine | 81.00 | 49.50 |
| norm_top16/semantic_attention | 80.99 | 48.24 |
| norm_top16/semantic_only | 80.82 | 49.38 |
| norm_top32/semantic_only | 80.79 | 49.41 |
| raw_top16/cosine_only | 80.41 | 49.73 |
| raw_top32/semantic_only | 78.70 | 50.19 |
| raw_top16/semantic_only | 78.22 | 49.47 |
