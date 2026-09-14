# 区域 AE + All-attention gross：三层MLP / 单层MLP / XGB

原4000图3200/800、全部mentions、三seed均值±总体std；每域独立MAD gate与attention归一化，强度为真实RMS All-attention K32 gross。
单层MLP 12候选、XGB18候选；训练内2560/640选参，原800图最终评估。AE-only为相同分类器对照；仅S取log1p，无标准化。

| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---|---:|---:|
| qwen3_vl_8b | visual | three_hidden | 88.321 ± 0.327 | 61.803 ± 0.837 |
| qwen3_vl_8b | visual | one_hidden | 85.975 ± 0.804 | 56.766 ± 2.121 |
| qwen3_vl_8b | visual | xgb | 89.054 ± 0.000 | 63.137 ± 0.000 |
| qwen3_vl_8b | visual_prompt_sum | three_hidden | 88.256 ± 0.252 | 63.260 ± 0.417 |
| qwen3_vl_8b | visual_prompt_sum | one_hidden | 85.187 ± 0.529 | 58.057 ± 0.400 |
| qwen3_vl_8b | visual_prompt_sum | xgb | 90.204 ± 0.000 | 68.059 ± 0.000 |
| qwen3_vl_8b | generation | three_hidden | 85.538 ± 1.238 | 57.842 ± 2.952 |
| qwen3_vl_8b | generation | one_hidden | 82.731 ± 0.085 | 50.496 ± 0.609 |
| qwen3_vl_8b | generation | xgb | 86.864 ± 0.000 | 59.790 ± 0.000 |
| qwen3_vl_8b | vp_generation | three_hidden | 90.118 ± 0.917 | 66.915 ± 2.882 |
| qwen3_vl_8b | vp_generation | one_hidden | 87.871 ± 0.366 | 63.331 ± 0.320 |
| qwen3_vl_8b | vp_generation | xgb | 91.219 ± 0.000 | 70.190 ± 0.000 |
| qwen3_vl_8b | ae_only_visual | three_hidden | 83.732 ± 0.395 | 54.752 ± 0.648 |
| qwen3_vl_8b | ae_only_visual | one_hidden | 80.382 ± 0.599 | 49.219 ± 1.078 |
| qwen3_vl_8b | ae_only_visual | xgb | 81.214 ± 0.000 | 50.122 ± 0.000 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | three_hidden | 86.402 ± 0.440 | 58.853 ± 0.732 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | one_hidden | 82.764 ± 0.292 | 56.103 ± 0.470 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | xgb | 87.774 ± 0.000 | 63.581 ± 0.000 |
| qwen3_vl_8b | ae_only_generation | three_hidden | 76.896 ± 0.996 | 46.140 ± 1.587 |
| qwen3_vl_8b | ae_only_generation | one_hidden | 70.351 ± 0.323 | 34.677 ± 0.452 |
| qwen3_vl_8b | ae_only_generation | xgb | 76.637 ± 0.000 | 44.650 ± 0.000 |
| qwen3_vl_8b | ae_only_vp_generation | three_hidden | 86.194 ± 0.583 | 61.880 ± 0.665 |
| qwen3_vl_8b | ae_only_vp_generation | one_hidden | 79.099 ± 0.427 | 51.652 ± 0.592 |
| qwen3_vl_8b | ae_only_vp_generation | xgb | 89.152 ± 0.000 | 65.202 ± 0.000 |
