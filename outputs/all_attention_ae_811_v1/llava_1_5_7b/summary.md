# 区域AE 8:1:1检测

3200训练/400验证/400测试；全部mentions；三seed均值±std。旧800图已被查看，此轮为探索性分析。
三层checkpoint/调度/早停依据验证loss；单层/XGB依据验证AUROC选参。均只拟合3200训练图，无标准化。

| 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---:|---:|
| visual | three_hidden | 90.03 ± 0.12 | 69.38 ± 0.23 |
| visual | one_hidden | 88.91 ± 0.35 | 66.72 ± 0.78 |
| visual | xgb | 90.21 ± 0.00 | 69.80 ± 0.00 |
| visual_prompt_sum | three_hidden | 89.02 ± 0.41 | 68.10 ± 0.49 |
| visual_prompt_sum | one_hidden | 87.03 ± 0.18 | 65.09 ± 0.29 |
| visual_prompt_sum | xgb | 89.40 ± 0.00 | 67.83 ± 0.00 |
| generation | three_hidden | 87.31 ± 0.30 | 63.35 ± 0.40 |
| generation | one_hidden | 86.38 ± 0.18 | 63.66 ± 0.12 |
| generation | xgb | 88.68 ± 0.00 | 65.87 ± 0.00 |
| vp_generation | three_hidden | 90.05 ± 0.13 | 72.06 ± 0.08 |
| vp_generation | one_hidden | 89.11 ± 0.28 | 70.11 ± 0.27 |
| vp_generation | xgb | 90.24 ± 0.00 | 70.46 ± 0.00 |
| ae_only_visual | three_hidden | 87.66 ± 0.15 | 63.52 ± 0.29 |
| ae_only_visual | one_hidden | 84.46 ± 0.53 | 58.92 ± 0.57 |
| ae_only_visual | xgb | 86.72 ± 0.00 | 64.92 ± 0.00 |
| ae_only_visual_prompt_sum | three_hidden | 85.76 ± 0.22 | 58.40 ± 0.89 |
| ae_only_visual_prompt_sum | one_hidden | 84.46 ± 0.38 | 58.40 ± 0.65 |
| ae_only_visual_prompt_sum | xgb | 85.73 ± 0.00 | 61.27 ± 0.00 |
| ae_only_generation | three_hidden | 81.66 ± 0.33 | 53.02 ± 0.83 |
| ae_only_generation | one_hidden | 79.22 ± 0.34 | 49.78 ± 0.29 |
| ae_only_generation | xgb | 81.27 ± 0.00 | 52.38 ± 0.00 |
| ae_only_vp_generation | three_hidden | 86.70 ± 0.40 | 63.21 ± 0.38 |
| ae_only_vp_generation | one_hidden | 85.67 ± 0.41 | 63.18 ± 0.54 |
| ae_only_vp_generation | xgb | 87.20 ± 0.00 | 64.59 ± 0.00 |
| legacy_visual | three_hidden | 90.09 ± 0.14 | 69.52 ± 0.12 |
| legacy_visual | one_hidden | 88.76 ± 0.46 | 66.22 ± 0.84 |
| legacy_visual | xgb | 90.24 ± 0.00 | 69.13 ± 0.00 |
