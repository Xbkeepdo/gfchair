# 区域 AE + All-attention gross：三层MLP / 单层MLP / XGB

原4000图3200/800、全部mentions、三seed均值±总体std；每域独立MAD gate与attention归一化，强度为真实RMS All-attention K32 gross。
单层MLP 12候选、XGB18候选；训练内2560/640选参，原800图最终评估。AE-only为相同分类器对照；仅S取log1p，无标准化。

| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---|---:|---:|
| llava_1_5_7b | visual | three_hidden | 90.044 ± 0.195 | 70.688 ± 0.614 |
| llava_1_5_7b | visual | one_hidden | 89.024 ± 0.124 | 68.704 ± 0.137 |
| llava_1_5_7b | visual | xgb | 89.711 ± 0.000 | 69.878 ± 0.000 |
| llava_1_5_7b | visual_prompt_sum | three_hidden | 89.588 ± 0.240 | 70.133 ± 0.122 |
| llava_1_5_7b | visual_prompt_sum | one_hidden | 87.330 ± 0.298 | 66.244 ± 0.642 |
| llava_1_5_7b | visual_prompt_sum | xgb | 89.037 ± 0.000 | 69.566 ± 0.000 |
| llava_1_5_7b | generation | three_hidden | 87.553 ± 0.246 | 64.632 ± 0.971 |
| llava_1_5_7b | generation | one_hidden | 86.699 ± 0.155 | 64.105 ± 0.261 |
| llava_1_5_7b | generation | xgb | 87.899 ± 0.000 | 65.211 ± 0.000 |
| llava_1_5_7b | vp_generation | three_hidden | 89.850 ± 0.360 | 70.477 ± 1.230 |
| llava_1_5_7b | vp_generation | one_hidden | 89.450 ± 0.200 | 70.891 ± 0.244 |
| llava_1_5_7b | vp_generation | xgb | 90.346 ± 0.000 | 71.655 ± 0.000 |
| llava_1_5_7b | ae_only_visual | three_hidden | 86.831 ± 0.497 | 64.016 ± 0.686 |
| llava_1_5_7b | ae_only_visual | one_hidden | 84.267 ± 0.131 | 59.215 ± 0.257 |
| llava_1_5_7b | ae_only_visual | xgb | 86.170 ± 0.000 | 64.295 ± 0.000 |
| llava_1_5_7b | ae_only_visual_prompt_sum | three_hidden | 85.824 ± 0.240 | 61.073 ± 0.994 |
| llava_1_5_7b | ae_only_visual_prompt_sum | one_hidden | 84.676 ± 0.299 | 60.470 ± 0.139 |
| llava_1_5_7b | ae_only_visual_prompt_sum | xgb | 86.579 ± 0.000 | 63.304 ± 0.000 |
| llava_1_5_7b | ae_only_generation | three_hidden | 82.076 ± 0.373 | 55.669 ± 0.565 |
| llava_1_5_7b | ae_only_generation | one_hidden | 79.492 ± 0.363 | 50.198 ± 0.807 |
| llava_1_5_7b | ae_only_generation | xgb | 80.220 ± 0.000 | 53.283 ± 0.000 |
| llava_1_5_7b | ae_only_vp_generation | three_hidden | 86.710 ± 0.135 | 63.620 ± 0.541 |
| llava_1_5_7b | ae_only_vp_generation | one_hidden | 85.335 ± 0.336 | 62.243 ± 0.345 |
| llava_1_5_7b | ae_only_vp_generation | xgb | 87.092 ± 0.000 | 65.715 ± 0.000 |
