# 区域AE 8:1:1检测

3200训练/400验证/400测试；全部mentions；三seed均值±std。旧800图已被查看，此轮为探索性分析。
三层checkpoint/调度/早停依据验证loss；单层/XGB依据验证AUROC选参。均只拟合3200训练图，无标准化。

| 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---:|---:|
| visual | three_hidden | 90.12 ± 0.12 | 66.51 ± 0.51 |
| visual | one_hidden | 88.02 ± 0.53 | 61.80 ± 2.10 |
| visual | xgb | 90.31 ± 0.00 | 66.15 ± 0.00 |
| visual_prompt_sum | three_hidden | 89.88 ± 0.19 | 66.38 ± 0.33 |
| visual_prompt_sum | one_hidden | 86.36 ± 0.57 | 60.87 ± 0.84 |
| visual_prompt_sum | xgb | 91.45 ± 0.00 | 70.44 ± 0.00 |
| generation | three_hidden | 88.12 ± 0.29 | 60.57 ± 0.46 |
| generation | one_hidden | 82.92 ± 0.61 | 50.58 ± 0.63 |
| generation | xgb | 88.14 ± 0.00 | 59.48 ± 0.00 |
| vp_generation | three_hidden | 92.29 ± 0.26 | 70.04 ± 0.10 |
| vp_generation | one_hidden | 88.90 ± 0.41 | 64.01 ± 0.23 |
| vp_generation | xgb | 91.93 ± 0.00 | 70.77 ± 0.00 |
| ae_only_visual | three_hidden | 83.92 ± 0.63 | 52.81 ± 1.59 |
| ae_only_visual | one_hidden | 81.15 ± 0.56 | 49.63 ± 0.96 |
| ae_only_visual | xgb | 81.63 ± 0.00 | 49.27 ± 0.00 |
| ae_only_visual_prompt_sum | three_hidden | 86.49 ± 0.57 | 59.01 ± 0.49 |
| ae_only_visual_prompt_sum | one_hidden | 81.78 ± 0.85 | 53.66 ± 1.30 |
| ae_only_visual_prompt_sum | xgb | 88.68 ± 0.00 | 64.36 ± 0.00 |
| ae_only_generation | three_hidden | 76.79 ± 0.77 | 46.09 ± 1.42 |
| ae_only_generation | one_hidden | 73.83 ± 0.59 | 39.38 ± 1.35 |
| ae_only_generation | xgb | 76.02 ± 0.00 | 44.35 ± 0.00 |
| ae_only_vp_generation | three_hidden | 87.26 ± 0.67 | 63.14 ± 1.36 |
| ae_only_vp_generation | one_hidden | 84.29 ± 0.73 | 57.99 ± 1.14 |
| ae_only_vp_generation | xgb | 89.88 ± 0.00 | 66.41 ± 0.00 |
| legacy_visual | three_hidden | 90.12 ± 0.14 | 66.95 ± 0.64 |
| legacy_visual | one_hidden | 87.91 ± 0.39 | 62.01 ± 1.85 |
| legacy_visual | xgb | 89.36 ± 0.00 | 63.89 ± 0.00 |
