# 区域AE 8:1:1检测

3200训练/400验证/400测试；全部mentions；三seed均值±std。旧800图已被查看，此轮为探索性分析。
三层checkpoint/调度/早停依据验证loss；单层/XGB依据验证AUROC选参。均只拟合3200训练图，无标准化。

| 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---:|---:|
| visual | three_hidden | 86.21 ± 0.50 | 39.25 ± 0.53 |
| visual | one_hidden | 82.05 ± 0.52 | 32.24 ± 1.45 |
| visual | xgb | 86.55 ± 0.00 | 40.79 ± 0.00 |
| visual_prompt_sum | three_hidden | 86.21 ± 0.48 | 40.03 ± 0.68 |
| visual_prompt_sum | one_hidden | 81.79 ± 0.05 | 33.02 ± 0.16 |
| visual_prompt_sum | xgb | 86.37 ± 0.00 | 45.64 ± 0.00 |
| generation | three_hidden | 82.02 ± 0.70 | 35.22 ± 1.83 |
| generation | one_hidden | 80.64 ± 0.13 | 29.09 ± 0.34 |
| generation | xgb | 82.45 ± 0.00 | 34.94 ± 0.00 |
| vp_generation | three_hidden | 87.36 ± 0.25 | 46.83 ± 1.35 |
| vp_generation | one_hidden | 83.75 ± 0.62 | 36.98 ± 0.98 |
| vp_generation | xgb | 87.54 ± 0.00 | 48.40 ± 0.00 |
| ae_only_visual | three_hidden | 79.68 ± 0.10 | 33.30 ± 1.03 |
| ae_only_visual | one_hidden | 71.38 ± 0.56 | 29.15 ± 0.52 |
| ae_only_visual | xgb | 75.82 ± 0.00 | 27.65 ± 0.00 |
| ae_only_visual_prompt_sum | three_hidden | 83.67 ± 0.34 | 42.46 ± 1.21 |
| ae_only_visual_prompt_sum | one_hidden | 77.94 ± 0.16 | 31.65 ± 0.13 |
| ae_only_visual_prompt_sum | xgb | 81.83 ± 0.00 | 36.63 ± 0.00 |
| ae_only_generation | three_hidden | 75.92 ± 0.69 | 26.53 ± 1.87 |
| ae_only_generation | one_hidden | 70.89 ± 0.80 | 19.39 ± 0.86 |
| ae_only_generation | xgb | 72.12 ± 0.00 | 28.16 ± 0.00 |
| ae_only_vp_generation | three_hidden | 83.02 ± 0.20 | 42.16 ± 0.85 |
| ae_only_vp_generation | one_hidden | 78.97 ± 0.57 | 36.13 ± 1.03 |
| ae_only_vp_generation | xgb | 81.54 ± 0.00 | 37.70 ± 0.00 |
| legacy_visual | three_hidden | 85.69 ± 0.77 | 37.21 ± 1.07 |
| legacy_visual | one_hidden | 82.69 ± 0.52 | 33.01 ± 1.53 |
| legacy_visual | xgb | 85.72 ± 0.00 | 38.84 ± 0.00 |
