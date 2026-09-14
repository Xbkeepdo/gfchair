# 区域 AE + All-attention gross：三层MLP / 单层MLP / XGB

原4000图3200/800、全部mentions、三seed均值±总体std；每域独立MAD gate与attention归一化，强度为真实RMS All-attention K32 gross。
单层MLP 12候选、XGB18候选；训练内2560/640选参，原800图最终评估。AE-only为相同分类器对照；仅S取log1p，无标准化。

| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---|---:|---:|
| internvl_2_5_8b | visual | three_hidden | 85.587 ± 0.470 | 53.852 ± 1.381 |
| internvl_2_5_8b | visual | one_hidden | 84.523 ± 0.229 | 53.920 ± 0.299 |
| internvl_2_5_8b | visual | xgb | 85.300 ± 0.000 | 50.695 ± 0.000 |
| internvl_2_5_8b | visual_prompt_sum | three_hidden | 85.844 ± 0.553 | 52.085 ± 0.801 |
| internvl_2_5_8b | visual_prompt_sum | one_hidden | 81.442 ± 0.689 | 46.679 ± 1.401 |
| internvl_2_5_8b | visual_prompt_sum | xgb | 85.489 ± 0.000 | 51.579 ± 0.000 |
| internvl_2_5_8b | generation | three_hidden | 84.417 ± 0.600 | 49.604 ± 1.989 |
| internvl_2_5_8b | generation | one_hidden | 81.155 ± 0.064 | 41.856 ± 0.305 |
| internvl_2_5_8b | generation | xgb | 83.074 ± 0.000 | 47.161 ± 0.000 |
| internvl_2_5_8b | vp_generation | three_hidden | 85.375 ± 1.015 | 54.005 ± 3.143 |
| internvl_2_5_8b | vp_generation | one_hidden | 83.698 ± 0.238 | 49.912 ± 0.814 |
| internvl_2_5_8b | vp_generation | xgb | 87.073 ± 0.000 | 58.386 ± 0.000 |
| internvl_2_5_8b | ae_only_visual | three_hidden | 83.119 ± 0.151 | 46.330 ± 0.825 |
| internvl_2_5_8b | ae_only_visual | one_hidden | 79.083 ± 0.264 | 41.733 ± 0.380 |
| internvl_2_5_8b | ae_only_visual | xgb | 80.565 ± 0.000 | 43.315 ± 0.000 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | three_hidden | 80.744 ± 0.491 | 43.481 ± 0.631 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | one_hidden | 75.712 ± 0.095 | 34.843 ± 0.118 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | xgb | 80.216 ± 0.000 | 43.217 ± 0.000 |
| internvl_2_5_8b | ae_only_generation | three_hidden | 76.668 ± 0.831 | 36.726 ± 2.292 |
| internvl_2_5_8b | ae_only_generation | one_hidden | 72.074 ± 0.152 | 32.339 ± 0.085 |
| internvl_2_5_8b | ae_only_generation | xgb | 74.480 ± 0.000 | 34.506 ± 0.000 |
| internvl_2_5_8b | ae_only_vp_generation | three_hidden | 80.892 ± 0.784 | 45.533 ± 1.712 |
| internvl_2_5_8b | ae_only_vp_generation | one_hidden | 78.127 ± 0.212 | 41.813 ± 0.219 |
| internvl_2_5_8b | ae_only_vp_generation | xgb | 82.744 ± 0.000 | 46.527 ± 0.000 |
