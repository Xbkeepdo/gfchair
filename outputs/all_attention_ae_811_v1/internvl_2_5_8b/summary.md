# 区域AE 8:1:1检测

3200训练/400验证/400测试；全部mentions；三seed均值±std。旧800图已被查看，此轮为探索性分析。
三层checkpoint/调度/早停依据验证loss；单层/XGB依据验证AUROC选参。均只拟合3200训练图，无标准化。

| 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---:|---:|
| visual | three_hidden | 87.16 ± 0.36 | 54.93 ± 1.55 |
| visual | one_hidden | 85.77 ± 0.15 | 56.40 ± 1.30 |
| visual | xgb | 85.56 ± 0.00 | 50.37 ± 0.00 |
| visual_prompt_sum | three_hidden | 86.74 ± 0.55 | 51.92 ± 1.44 |
| visual_prompt_sum | one_hidden | 82.75 ± 0.90 | 48.60 ± 1.85 |
| visual_prompt_sum | xgb | 86.32 ± 0.00 | 51.58 ± 0.00 |
| generation | three_hidden | 84.83 ± 0.50 | 49.70 ± 0.57 |
| generation | one_hidden | 82.15 ± 0.29 | 44.81 ± 1.73 |
| generation | xgb | 82.52 ± 0.00 | 45.66 ± 0.00 |
| vp_generation | three_hidden | 87.21 ± 0.26 | 53.30 ± 0.83 |
| vp_generation | one_hidden | 84.40 ± 0.32 | 50.52 ± 0.47 |
| vp_generation | xgb | 88.39 ± 0.00 | 55.92 ± 0.00 |
| ae_only_visual | three_hidden | 83.88 ± 0.12 | 46.10 ± 1.00 |
| ae_only_visual | one_hidden | 80.54 ± 0.23 | 42.38 ± 0.12 |
| ae_only_visual | xgb | 81.66 ± 0.00 | 43.21 ± 0.00 |
| ae_only_visual_prompt_sum | three_hidden | 80.13 ± 0.91 | 39.49 ± 2.65 |
| ae_only_visual_prompt_sum | one_hidden | 75.09 ± 0.42 | 36.65 ± 0.09 |
| ae_only_visual_prompt_sum | xgb | 81.54 ± 0.00 | 45.50 ± 0.00 |
| ae_only_generation | three_hidden | 77.34 ± 0.42 | 39.21 ± 2.13 |
| ae_only_generation | one_hidden | 74.17 ± 0.34 | 32.75 ± 0.71 |
| ae_only_generation | xgb | 72.49 ± 0.00 | 31.16 ± 0.00 |
| ae_only_vp_generation | three_hidden | 82.71 ± 0.15 | 45.63 ± 0.81 |
| ae_only_vp_generation | one_hidden | 79.07 ± 0.32 | 43.26 ± 0.47 |
| ae_only_vp_generation | xgb | 82.66 ± 0.00 | 46.06 ± 0.00 |
| legacy_visual | three_hidden | 86.73 ± 0.51 | 54.51 ± 1.07 |
| legacy_visual | one_hidden | 85.94 ± 0.26 | 56.28 ± 0.08 |
| legacy_visual | xgb | 85.62 ± 0.00 | 51.42 ± 0.00 |
