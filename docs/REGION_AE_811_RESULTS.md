# 区域AE 8:1:1检测结果

四模型各4000图，按图片分3200训练/400验证/400测试，保留全部mentions。seed20260912冻结划分，检测器seeds43/44/45；下面是均值±总体标准差。
默认保留原3200训练图，将原800图分成400验证/400测试。原800图已有历史研究使用，因此本轮属于探索性评估。
三层MLP保持[128,64,32]/BN/dropout.3/Adam等配置，改为验证loss选择checkpoint、学习率调度及早停。单层sklearn12候选、XGB18候选，仅根据400验证图的seed43 AUROC/AP选参；最终三seed只拟合3200训练图。
V、VP、G、VP+G四组为区域原值AE＋对应All-attention gross的log1p，VP先加SV+SP再log；VP区域AE独立重算。ae_only为去掉gross的对应控制；legacy_visual为旧[AE_V,log1p(S_E)]，同样重训。
新模型和旧82模型均在相同400测试图片、相同mentions评估；vs82_same400.csv保存配对点差。不能直接用新400图分数减旧800图总分。无新增bootstrap。

| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR |
|---|---|---|---:|---:|
| qwen2_5_vl_7b | visual | three_hidden | 86.21 ± 0.50 | 39.25 ± 0.53 |
| qwen2_5_vl_7b | visual | one_hidden | 82.05 ± 0.52 | 32.24 ± 1.45 |
| qwen2_5_vl_7b | visual | xgb | 86.55 ± 0.00 | 40.79 ± 0.00 |
| qwen2_5_vl_7b | visual_prompt_sum | three_hidden | 86.21 ± 0.48 | 40.03 ± 0.68 |
| qwen2_5_vl_7b | visual_prompt_sum | one_hidden | 81.79 ± 0.05 | 33.02 ± 0.16 |
| qwen2_5_vl_7b | visual_prompt_sum | xgb | 86.37 ± 0.00 | 45.64 ± 0.00 |
| qwen2_5_vl_7b | generation | three_hidden | 82.02 ± 0.70 | 35.22 ± 1.83 |
| qwen2_5_vl_7b | generation | one_hidden | 80.64 ± 0.13 | 29.09 ± 0.34 |
| qwen2_5_vl_7b | generation | xgb | 82.45 ± 0.00 | 34.94 ± 0.00 |
| qwen2_5_vl_7b | vp_generation | three_hidden | 87.36 ± 0.25 | 46.83 ± 1.35 |
| qwen2_5_vl_7b | vp_generation | one_hidden | 83.75 ± 0.62 | 36.98 ± 0.98 |
| qwen2_5_vl_7b | vp_generation | xgb | 87.54 ± 0.00 | 48.40 ± 0.00 |
| qwen2_5_vl_7b | ae_only_visual | three_hidden | 79.68 ± 0.10 | 33.30 ± 1.03 |
| qwen2_5_vl_7b | ae_only_visual | one_hidden | 71.38 ± 0.56 | 29.15 ± 0.52 |
| qwen2_5_vl_7b | ae_only_visual | xgb | 75.82 ± 0.00 | 27.65 ± 0.00 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | three_hidden | 83.67 ± 0.34 | 42.46 ± 1.21 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | one_hidden | 77.94 ± 0.16 | 31.65 ± 0.13 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | xgb | 81.83 ± 0.00 | 36.63 ± 0.00 |
| qwen2_5_vl_7b | ae_only_generation | three_hidden | 75.92 ± 0.69 | 26.53 ± 1.87 |
| qwen2_5_vl_7b | ae_only_generation | one_hidden | 70.89 ± 0.80 | 19.39 ± 0.86 |
| qwen2_5_vl_7b | ae_only_generation | xgb | 72.12 ± 0.00 | 28.16 ± 0.00 |
| qwen2_5_vl_7b | ae_only_vp_generation | three_hidden | 83.02 ± 0.20 | 42.16 ± 0.85 |
| qwen2_5_vl_7b | ae_only_vp_generation | one_hidden | 78.97 ± 0.57 | 36.13 ± 1.03 |
| qwen2_5_vl_7b | ae_only_vp_generation | xgb | 81.54 ± 0.00 | 37.70 ± 0.00 |
| qwen2_5_vl_7b | legacy_visual | three_hidden | 85.69 ± 0.77 | 37.21 ± 1.07 |
| qwen2_5_vl_7b | legacy_visual | one_hidden | 82.69 ± 0.52 | 33.01 ± 1.53 |
| qwen2_5_vl_7b | legacy_visual | xgb | 85.72 ± 0.00 | 38.84 ± 0.00 |
| llava_1_5_7b | visual | three_hidden | 90.03 ± 0.12 | 69.38 ± 0.23 |
| llava_1_5_7b | visual | one_hidden | 88.91 ± 0.35 | 66.72 ± 0.78 |
| llava_1_5_7b | visual | xgb | 90.21 ± 0.00 | 69.80 ± 0.00 |
| llava_1_5_7b | visual_prompt_sum | three_hidden | 89.02 ± 0.41 | 68.10 ± 0.49 |
| llava_1_5_7b | visual_prompt_sum | one_hidden | 87.03 ± 0.18 | 65.09 ± 0.29 |
| llava_1_5_7b | visual_prompt_sum | xgb | 89.40 ± 0.00 | 67.83 ± 0.00 |
| llava_1_5_7b | generation | three_hidden | 87.31 ± 0.30 | 63.35 ± 0.40 |
| llava_1_5_7b | generation | one_hidden | 86.38 ± 0.18 | 63.66 ± 0.12 |
| llava_1_5_7b | generation | xgb | 88.68 ± 0.00 | 65.87 ± 0.00 |
| llava_1_5_7b | vp_generation | three_hidden | 90.05 ± 0.13 | 72.06 ± 0.08 |
| llava_1_5_7b | vp_generation | one_hidden | 89.11 ± 0.28 | 70.11 ± 0.27 |
| llava_1_5_7b | vp_generation | xgb | 90.24 ± 0.00 | 70.46 ± 0.00 |
| llava_1_5_7b | ae_only_visual | three_hidden | 87.66 ± 0.15 | 63.52 ± 0.29 |
| llava_1_5_7b | ae_only_visual | one_hidden | 84.46 ± 0.53 | 58.92 ± 0.57 |
| llava_1_5_7b | ae_only_visual | xgb | 86.72 ± 0.00 | 64.92 ± 0.00 |
| llava_1_5_7b | ae_only_visual_prompt_sum | three_hidden | 85.76 ± 0.22 | 58.40 ± 0.89 |
| llava_1_5_7b | ae_only_visual_prompt_sum | one_hidden | 84.46 ± 0.38 | 58.40 ± 0.65 |
| llava_1_5_7b | ae_only_visual_prompt_sum | xgb | 85.73 ± 0.00 | 61.27 ± 0.00 |
| llava_1_5_7b | ae_only_generation | three_hidden | 81.66 ± 0.33 | 53.02 ± 0.83 |
| llava_1_5_7b | ae_only_generation | one_hidden | 79.22 ± 0.34 | 49.78 ± 0.29 |
| llava_1_5_7b | ae_only_generation | xgb | 81.27 ± 0.00 | 52.38 ± 0.00 |
| llava_1_5_7b | ae_only_vp_generation | three_hidden | 86.70 ± 0.40 | 63.21 ± 0.38 |
| llava_1_5_7b | ae_only_vp_generation | one_hidden | 85.67 ± 0.41 | 63.18 ± 0.54 |
| llava_1_5_7b | ae_only_vp_generation | xgb | 87.20 ± 0.00 | 64.59 ± 0.00 |
| llava_1_5_7b | legacy_visual | three_hidden | 90.09 ± 0.14 | 69.52 ± 0.12 |
| llava_1_5_7b | legacy_visual | one_hidden | 88.76 ± 0.46 | 66.22 ± 0.84 |
| llava_1_5_7b | legacy_visual | xgb | 90.24 ± 0.00 | 69.13 ± 0.00 |
| qwen3_vl_8b | visual | three_hidden | 90.12 ± 0.12 | 66.51 ± 0.51 |
| qwen3_vl_8b | visual | one_hidden | 88.02 ± 0.53 | 61.80 ± 2.10 |
| qwen3_vl_8b | visual | xgb | 90.31 ± 0.00 | 66.15 ± 0.00 |
| qwen3_vl_8b | visual_prompt_sum | three_hidden | 89.88 ± 0.19 | 66.38 ± 0.33 |
| qwen3_vl_8b | visual_prompt_sum | one_hidden | 86.36 ± 0.57 | 60.87 ± 0.84 |
| qwen3_vl_8b | visual_prompt_sum | xgb | 91.45 ± 0.00 | 70.44 ± 0.00 |
| qwen3_vl_8b | generation | three_hidden | 88.12 ± 0.29 | 60.57 ± 0.46 |
| qwen3_vl_8b | generation | one_hidden | 82.92 ± 0.61 | 50.58 ± 0.63 |
| qwen3_vl_8b | generation | xgb | 88.14 ± 0.00 | 59.48 ± 0.00 |
| qwen3_vl_8b | vp_generation | three_hidden | 92.29 ± 0.26 | 70.04 ± 0.10 |
| qwen3_vl_8b | vp_generation | one_hidden | 88.90 ± 0.41 | 64.01 ± 0.23 |
| qwen3_vl_8b | vp_generation | xgb | 91.93 ± 0.00 | 70.77 ± 0.00 |
| qwen3_vl_8b | ae_only_visual | three_hidden | 83.92 ± 0.63 | 52.81 ± 1.59 |
| qwen3_vl_8b | ae_only_visual | one_hidden | 81.15 ± 0.56 | 49.63 ± 0.96 |
| qwen3_vl_8b | ae_only_visual | xgb | 81.63 ± 0.00 | 49.27 ± 0.00 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | three_hidden | 86.49 ± 0.57 | 59.01 ± 0.49 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | one_hidden | 81.78 ± 0.85 | 53.66 ± 1.30 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | xgb | 88.68 ± 0.00 | 64.36 ± 0.00 |
| qwen3_vl_8b | ae_only_generation | three_hidden | 76.79 ± 0.77 | 46.09 ± 1.42 |
| qwen3_vl_8b | ae_only_generation | one_hidden | 73.83 ± 0.59 | 39.38 ± 1.35 |
| qwen3_vl_8b | ae_only_generation | xgb | 76.02 ± 0.00 | 44.35 ± 0.00 |
| qwen3_vl_8b | ae_only_vp_generation | three_hidden | 87.26 ± 0.67 | 63.14 ± 1.36 |
| qwen3_vl_8b | ae_only_vp_generation | one_hidden | 84.29 ± 0.73 | 57.99 ± 1.14 |
| qwen3_vl_8b | ae_only_vp_generation | xgb | 89.88 ± 0.00 | 66.41 ± 0.00 |
| qwen3_vl_8b | legacy_visual | three_hidden | 90.12 ± 0.14 | 66.95 ± 0.64 |
| qwen3_vl_8b | legacy_visual | one_hidden | 87.91 ± 0.39 | 62.01 ± 1.85 |
| qwen3_vl_8b | legacy_visual | xgb | 89.36 ± 0.00 | 63.89 ± 0.00 |
| internvl_2_5_8b | visual | three_hidden | 87.16 ± 0.36 | 54.93 ± 1.55 |
| internvl_2_5_8b | visual | one_hidden | 85.77 ± 0.15 | 56.40 ± 1.30 |
| internvl_2_5_8b | visual | xgb | 85.56 ± 0.00 | 50.37 ± 0.00 |
| internvl_2_5_8b | visual_prompt_sum | three_hidden | 86.74 ± 0.55 | 51.92 ± 1.44 |
| internvl_2_5_8b | visual_prompt_sum | one_hidden | 82.75 ± 0.90 | 48.60 ± 1.85 |
| internvl_2_5_8b | visual_prompt_sum | xgb | 86.32 ± 0.00 | 51.58 ± 0.00 |
| internvl_2_5_8b | generation | three_hidden | 84.83 ± 0.50 | 49.70 ± 0.57 |
| internvl_2_5_8b | generation | one_hidden | 82.15 ± 0.29 | 44.81 ± 1.73 |
| internvl_2_5_8b | generation | xgb | 82.52 ± 0.00 | 45.66 ± 0.00 |
| internvl_2_5_8b | vp_generation | three_hidden | 87.21 ± 0.26 | 53.30 ± 0.83 |
| internvl_2_5_8b | vp_generation | one_hidden | 84.40 ± 0.32 | 50.52 ± 0.47 |
| internvl_2_5_8b | vp_generation | xgb | 88.39 ± 0.00 | 55.92 ± 0.00 |
| internvl_2_5_8b | ae_only_visual | three_hidden | 83.88 ± 0.12 | 46.10 ± 1.00 |
| internvl_2_5_8b | ae_only_visual | one_hidden | 80.54 ± 0.23 | 42.38 ± 0.12 |
| internvl_2_5_8b | ae_only_visual | xgb | 81.66 ± 0.00 | 43.21 ± 0.00 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | three_hidden | 80.13 ± 0.91 | 39.49 ± 2.65 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | one_hidden | 75.09 ± 0.42 | 36.65 ± 0.09 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | xgb | 81.54 ± 0.00 | 45.50 ± 0.00 |
| internvl_2_5_8b | ae_only_generation | three_hidden | 77.34 ± 0.42 | 39.21 ± 2.13 |
| internvl_2_5_8b | ae_only_generation | one_hidden | 74.17 ± 0.34 | 32.75 ± 0.71 |
| internvl_2_5_8b | ae_only_generation | xgb | 72.49 ± 0.00 | 31.16 ± 0.00 |
| internvl_2_5_8b | ae_only_vp_generation | three_hidden | 82.71 ± 0.15 | 45.63 ± 0.81 |
| internvl_2_5_8b | ae_only_vp_generation | one_hidden | 79.07 ± 0.32 | 43.26 ± 0.47 |
| internvl_2_5_8b | ae_only_vp_generation | xgb | 82.66 ± 0.00 | 46.06 ± 0.00 |
| internvl_2_5_8b | legacy_visual | three_hidden | 86.73 ± 0.51 | 54.51 ± 1.07 |
| internvl_2_5_8b | legacy_visual | one_hidden | 85.94 ± 0.26 | 56.28 ± 0.08 |
| internvl_2_5_8b | legacy_visual | xgb | 85.62 ± 0.00 | 51.42 ± 0.00 |

## 与旧82训练在相同400测试图上比较

下面的变化包含训练协议变化：三层使用验证loss而非训练loss；单层/XGB改用3200/400候选拟合/验证而非旧2560/640。测试图片固定。

| 模型 | 特征 | 分类器 | ΔAUROC百分点 | ΔHALL-AUPR百分点 |
|---|---|---|---:|---:|
| qwen2_5_vl_7b | visual | three_hidden | -1.81 | -3.43 |
| qwen2_5_vl_7b | visual | one_hidden | +0.16 | +1.39 |
| qwen2_5_vl_7b | visual | xgb | +0.07 | -0.72 |
| qwen2_5_vl_7b | visual_prompt_sum | three_hidden | +2.17 | +1.69 |
| qwen2_5_vl_7b | visual_prompt_sum | one_hidden | +0.19 | -0.54 |
| qwen2_5_vl_7b | visual_prompt_sum | xgb | +0.54 | +0.97 |
| qwen2_5_vl_7b | generation | three_hidden | +0.09 | -3.10 |
| qwen2_5_vl_7b | generation | one_hidden | +0.16 | +0.79 |
| qwen2_5_vl_7b | generation | xgb | -0.38 | +0.33 |
| qwen2_5_vl_7b | vp_generation | three_hidden | +0.14 | -1.06 |
| qwen2_5_vl_7b | vp_generation | one_hidden | +0.70 | +2.50 |
| qwen2_5_vl_7b | vp_generation | xgb | -0.44 | -2.36 |
| qwen2_5_vl_7b | ae_only_visual | three_hidden | +0.52 | +0.98 |
| qwen2_5_vl_7b | ae_only_visual | one_hidden | -1.16 | +0.43 |
| qwen2_5_vl_7b | ae_only_visual | xgb | +1.61 | +2.29 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | three_hidden | +0.39 | +0.94 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | one_hidden | +0.00 | +0.00 |
| qwen2_5_vl_7b | ae_only_visual_prompt_sum | xgb | +0.20 | +0.69 |
| qwen2_5_vl_7b | ae_only_generation | three_hidden | +0.64 | +2.65 |
| qwen2_5_vl_7b | ae_only_generation | one_hidden | +0.26 | +0.06 |
| qwen2_5_vl_7b | ae_only_generation | xgb | +0.03 | +2.20 |
| qwen2_5_vl_7b | ae_only_vp_generation | three_hidden | +0.50 | +0.71 |
| qwen2_5_vl_7b | ae_only_vp_generation | one_hidden | -0.26 | -0.90 |
| qwen2_5_vl_7b | ae_only_vp_generation | xgb | -0.11 | +0.68 |
| qwen2_5_vl_7b | legacy_visual | three_hidden | -2.09 | -4.96 |
| llava_1_5_7b | visual | three_hidden | -0.46 | -0.45 |
| llava_1_5_7b | visual | one_hidden | -0.12 | -0.04 |
| llava_1_5_7b | visual | xgb | +0.00 | +0.00 |
| llava_1_5_7b | visual_prompt_sum | three_hidden | -0.37 | -0.52 |
| llava_1_5_7b | visual_prompt_sum | one_hidden | +0.01 | +0.14 |
| llava_1_5_7b | visual_prompt_sum | xgb | +0.37 | +0.72 |
| llava_1_5_7b | generation | three_hidden | -0.64 | -2.46 |
| llava_1_5_7b | generation | one_hidden | +0.11 | +0.42 |
| llava_1_5_7b | generation | xgb | +0.39 | +0.81 |
| llava_1_5_7b | vp_generation | three_hidden | +0.54 | +2.34 |
| llava_1_5_7b | vp_generation | one_hidden | +0.00 | +0.00 |
| llava_1_5_7b | vp_generation | xgb | +0.00 | +0.00 |
| llava_1_5_7b | ae_only_visual | three_hidden | +0.44 | +0.18 |
| llava_1_5_7b | ae_only_visual | one_hidden | -0.47 | -0.24 |
| llava_1_5_7b | ae_only_visual | xgb | -0.29 | +0.12 |
| llava_1_5_7b | ae_only_visual_prompt_sum | three_hidden | -0.42 | -1.55 |
| llava_1_5_7b | ae_only_visual_prompt_sum | one_hidden | -0.53 | -1.41 |
| llava_1_5_7b | ae_only_visual_prompt_sum | xgb | -0.53 | -0.62 |
| llava_1_5_7b | ae_only_generation | three_hidden | -0.70 | -2.12 |
| llava_1_5_7b | ae_only_generation | one_hidden | +0.25 | +0.62 |
| llava_1_5_7b | ae_only_generation | xgb | +0.95 | +1.19 |
| llava_1_5_7b | ae_only_vp_generation | three_hidden | -0.65 | -0.66 |
| llava_1_5_7b | ae_only_vp_generation | one_hidden | +0.30 | +1.02 |
| llava_1_5_7b | ae_only_vp_generation | xgb | +0.00 | +0.00 |
| llava_1_5_7b | legacy_visual | three_hidden | -0.20 | -0.14 |
| qwen3_vl_8b | visual | three_hidden | +0.67 | +1.82 |
| qwen3_vl_8b | visual | one_hidden | +0.49 | +1.42 |
| qwen3_vl_8b | visual | xgb | +0.00 | +0.00 |
| qwen3_vl_8b | visual_prompt_sum | three_hidden | +0.18 | +0.06 |
| qwen3_vl_8b | visual_prompt_sum | one_hidden | +0.00 | +0.00 |
| qwen3_vl_8b | visual_prompt_sum | xgb | +0.00 | +0.00 |
| qwen3_vl_8b | generation | three_hidden | +1.54 | +2.13 |
| qwen3_vl_8b | generation | one_hidden | -0.43 | -0.61 |
| qwen3_vl_8b | generation | xgb | +0.83 | +0.37 |
| qwen3_vl_8b | vp_generation | three_hidden | +0.67 | +0.58 |
| qwen3_vl_8b | vp_generation | one_hidden | -0.16 | -1.24 |
| qwen3_vl_8b | vp_generation | xgb | -0.40 | -0.75 |
| qwen3_vl_8b | ae_only_visual | three_hidden | -0.51 | -1.68 |
| qwen3_vl_8b | ae_only_visual | one_hidden | +0.00 | +0.00 |
| qwen3_vl_8b | ae_only_visual | xgb | +0.57 | -0.49 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | three_hidden | +0.13 | +0.38 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | one_hidden | -0.14 | -0.33 |
| qwen3_vl_8b | ae_only_visual_prompt_sum | xgb | -0.08 | -0.05 |
| qwen3_vl_8b | ae_only_generation | three_hidden | -0.91 | -1.28 |
| qwen3_vl_8b | ae_only_generation | one_hidden | +1.87 | +3.69 |
| qwen3_vl_8b | ae_only_generation | xgb | -1.11 | -1.10 |
| qwen3_vl_8b | ae_only_vp_generation | three_hidden | +0.80 | +0.56 |
| qwen3_vl_8b | ae_only_vp_generation | one_hidden | +4.99 | +6.07 |
| qwen3_vl_8b | ae_only_vp_generation | xgb | +0.08 | +0.80 |
| qwen3_vl_8b | legacy_visual | three_hidden | +0.41 | +1.32 |
| internvl_2_5_8b | visual | three_hidden | +0.14 | -1.65 |
| internvl_2_5_8b | visual | one_hidden | -0.15 | -0.53 |
| internvl_2_5_8b | visual | xgb | -0.42 | +0.03 |
| internvl_2_5_8b | visual_prompt_sum | three_hidden | -0.43 | -0.86 |
| internvl_2_5_8b | visual_prompt_sum | one_hidden | +0.00 | +0.00 |
| internvl_2_5_8b | visual_prompt_sum | xgb | +0.00 | +0.00 |
| internvl_2_5_8b | generation | three_hidden | -0.01 | -0.31 |
| internvl_2_5_8b | generation | one_hidden | +0.68 | +3.09 |
| internvl_2_5_8b | generation | xgb | -0.09 | +0.11 |
| internvl_2_5_8b | vp_generation | three_hidden | +0.35 | -1.53 |
| internvl_2_5_8b | vp_generation | one_hidden | +0.16 | +1.09 |
| internvl_2_5_8b | vp_generation | xgb | +0.66 | -2.35 |
| internvl_2_5_8b | ae_only_visual | three_hidden | -0.81 | -0.90 |
| internvl_2_5_8b | ae_only_visual | one_hidden | -0.08 | -0.14 |
| internvl_2_5_8b | ae_only_visual | xgb | +0.00 | +0.00 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | three_hidden | -2.34 | -6.38 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | one_hidden | -0.59 | -0.19 |
| internvl_2_5_8b | ae_only_visual_prompt_sum | xgb | +0.98 | +1.12 |
| internvl_2_5_8b | ae_only_generation | three_hidden | +0.21 | +0.71 |
| internvl_2_5_8b | ae_only_generation | one_hidden | +0.75 | -0.07 |
| internvl_2_5_8b | ae_only_generation | xgb | -1.08 | -2.49 |
| internvl_2_5_8b | ae_only_vp_generation | three_hidden | +0.50 | -1.01 |
| internvl_2_5_8b | ae_only_vp_generation | one_hidden | +0.00 | +0.00 |
| internvl_2_5_8b | ae_only_vp_generation | xgb | -0.29 | -0.95 |
| internvl_2_5_8b | legacy_visual | three_hidden | -0.80 | -1.43 |
