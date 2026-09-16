# Single semantic-attention feature + log1p(S_E)

每个模型×单特征仅用三 seed validation AUROC/HALL-AUPR 在 raw/norm×top16/32 中选择变体，然后报告对应 test。standalone 增量始终与该选中变体的未拼接结果比较。

## Four-model macro average

| single feature + logS | AUROC / AP (%) | vs standalone (pp) | vs logS-only (pp) |
|---|---:|---:|---:|
| semantic_only + logS | 88.90 / 65.82 | +11.36 / +17.79 | +2.43 / +6.52 |
| attention_only + logS | 87.75 / 63.90 | +3.59 / +8.28 | +1.28 / +4.61 |
| cosine_only + logS | 88.44 / 64.42 | +6.04 / +11.21 | +1.97 / +5.13 |
| semantic_attention + logS | 88.39 / 63.61 | +7.08 / +14.90 | +1.93 / +4.31 |
| cosine_attention + logS | 87.83 / 63.80 | +4.04 / +8.81 | +1.37 / +4.50 |
| semantic_attention_positive_cosine + logS | 88.21 / 63.41 | +6.83 / +13.90 | +1.74 / +4.11 |

## Per-model results

| model | feature + logS | validation-selected variant | AUROC / AP (%) | vs standalone (pp) | vs logS-only (pp) |
|---|---|---|---:|---:|---:|
| qwen2_5_vl_7b | semantic_only + logS | raw_top32 | 82.90 / 42.07 | +24.94 / +27.77 | +0.96 / +3.39 |
| qwen2_5_vl_7b | attention_only + logS | norm_top16 | 83.18 / 48.21 | +5.60 / +13.57 | +1.24 / +9.52 |
| qwen2_5_vl_7b | cosine_only + logS | norm_top32 | 84.12 / 44.87 | +7.87 / +11.40 | +2.19 / +6.19 |
| qwen2_5_vl_7b | semantic_attention + logS | norm_top32 | 82.55 / 41.66 | +7.82 / +13.43 | +0.62 / +2.97 |
| qwen2_5_vl_7b | cosine_attention + logS | norm_top16 | 83.06 / 48.07 | +5.79 / +13.51 | +1.12 / +9.39 |
| qwen2_5_vl_7b | semantic_attention_positive_cosine + logS | norm_top16 | 81.99 / 40.89 | +8.32 / +13.37 | +0.05 / +2.21 |
| llava_1_5_7b | semantic_only + logS | norm_top32 | 92.46 / 78.25 | +8.19 / +15.23 | +3.49 / +8.41 |
| llava_1_5_7b | attention_only + logS | norm_top32 | 90.36 / 73.24 | +3.93 / +8.97 | +1.39 / +3.40 |
| llava_1_5_7b | cosine_only + logS | norm_top16 | 90.71 / 74.00 | +4.40 / +10.38 | +1.74 / +4.16 |
| llava_1_5_7b | semantic_attention + logS | norm_top32 | 92.47 / 77.40 | +6.18 / +13.59 | +3.50 / +7.55 |
| llava_1_5_7b | cosine_attention + logS | norm_top16 | 90.28 / 72.27 | +5.27 / +11.75 | +1.30 / +2.43 |
| llava_1_5_7b | semantic_attention_positive_cosine + logS | norm_top32 | 92.45 / 77.51 | +5.24 / +10.82 | +3.48 / +7.67 |
| qwen3_vl_8b | semantic_only + logS | raw_top32 | 91.28 / 74.17 | +8.40 / +19.84 | +2.82 / +5.71 |
| qwen3_vl_8b | attention_only + logS | raw_top32 | 89.42 / 70.86 | +3.50 / +9.44 | +0.96 / +2.39 |
| qwen3_vl_8b | cosine_only + logS | raw_top32 | 90.42 / 72.71 | +6.56 / +12.95 | +1.97 / +4.25 |
| qwen3_vl_8b | semantic_attention + logS | norm_top16 | 90.20 / 72.64 | +10.32 / +24.83 | +1.75 / +4.18 |
| qwen3_vl_8b | cosine_attention + logS | raw_top32 | 89.54 / 70.82 | +3.31 / +8.62 | +1.08 / +2.36 |
| qwen3_vl_8b | semantic_attention_positive_cosine + logS | norm_top32 | 90.09 / 72.40 | +10.11 / +24.74 | +1.63 / +3.94 |
| internvl_2_5_8b | semantic_only + logS | raw_top16 | 88.94 / 68.78 | +3.91 / +8.34 | +2.45 / +8.59 |
| internvl_2_5_8b | attention_only + logS | norm_top32 | 88.04 / 63.31 | +1.33 / +1.13 | +1.55 / +3.11 |
| internvl_2_5_8b | cosine_only + logS | norm_top16 | 88.48 / 66.11 | +5.31 / +10.11 | +1.99 / +5.92 |
| internvl_2_5_8b | semantic_attention + logS | norm_top32 | 88.35 / 62.73 | +3.99 / +7.74 | +1.85 / +2.54 |
| internvl_2_5_8b | cosine_attention + logS | norm_top32 | 88.45 / 64.04 | +1.79 / +1.35 | +1.95 / +3.84 |
| internvl_2_5_8b | semantic_attention_positive_cosine + logS | norm_top32 | 88.30 / 62.81 | +3.65 / +6.66 | +1.81 / +2.62 |
