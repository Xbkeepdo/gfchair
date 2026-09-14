# 区域 AE + All-attention gross：三层MLP / 单层MLP / XGB

原4000图3200/800、全部mentions、三seed均值±总体std；每域独立MAD gate与attention归一化，强度为真实RMS All-attention K32 gross。
单层MLP 12候选、XGB18候选；训练内2560/640选参，原800图最终评估。AE-only为相同分类器对照；仅S取log1p，无标准化。

| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---|---:|---:|
| qwen2_5_vl_7b | visual | three_hidden | 87.537 ± 0.185 | 44.198 ± 0.377 |
| qwen2_5_vl_7b | visual | one_hidden | 82.408 ± 0.577 | 32.529 ± 1.170 |
| qwen2_5_vl_7b | visual | xgb | 85.380 ± 0.000 | 41.831 ± 0.000 |
| qwen2_5_vl_7b | visual_prompt_sum | three_hidden | 85.289 ± 0.947 | 43.955 ± 1.427 |
| qwen2_5_vl_7b | visual_prompt_sum | one_hidden | 82.006 ± 0.134 | 33.285 ± 0.710 |
| qwen2_5_vl_7b | visual_prompt_sum | xgb | 85.465 ± 0.000 | 44.710 ± 0.000 |
| qwen2_5_vl_7b | generation | three_hidden | 80.473 ± 1.371 | 35.227 ± 1.761 |
| qwen2_5_vl_7b | generation | one_hidden | 78.519 ± 0.193 | 27.212 ± 0.412 |
| qwen2_5_vl_7b | generation | xgb | 81.440 ± 0.000 | 31.668 ± 0.000 |
| qwen2_5_vl_7b | vp_generation | three_hidden | 85.699 ± 0.765 | 45.648 ± 1.645 |
| qwen2_5_vl_7b | vp_generation | one_hidden | 82.221 ± 0.716 | 33.618 ± 1.091 |
| qwen2_5_vl_7b | vp_generation | xgb | 86.864 ± 0.000 | 48.038 ± 0.000 |
| qwen2_5_vl_7b | ae_only_visual | three_hidden | 77.887 ± 0.595 | 32.755 ± 0.179 |
| qwen2_5_vl_7b | ae_only_visual | one_hidden | 69.897 ± 0.687 | 26.622 ± 0.668 |
| qwen2_5_vl_7b | ae_only_visual | xgb | 73.676 ± 0.000 | 27.665 ± 0.000 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | three_hidden | 82.957 ± 0.309 | 44.214 ± 0.524 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | one_hidden | 76.889 ± 0.173 | 34.002 ± 0.150 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | xgb | 80.626 ± 0.000 | 36.504 ± 0.000 |
| qwen2_5_vl_7b | ae_only_generation | three_hidden | 72.749 ± 0.390 | 21.699 ± 0.498 |
| qwen2_5_vl_7b | ae_only_generation | one_hidden | 68.349 ± 0.305 | 17.854 ± 0.231 |
| qwen2_5_vl_7b | ae_only_generation | xgb | 71.055 ± 0.000 | 23.306 ± 0.000 |
| qwen2_5_vl_7b | ae_only_vp_generation | three_hidden | 82.260 ± 0.107 | 43.104 ± 1.121 |
| qwen2_5_vl_7b | ae_only_vp_generation | one_hidden | 77.577 ± 0.265 | 36.316 ± 0.682 |
| qwen2_5_vl_7b | ae_only_vp_generation | xgb | 81.516 ± 0.000 | 38.134 ± 0.000 |
