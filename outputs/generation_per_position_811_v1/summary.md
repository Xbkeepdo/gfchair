# 生成来源 S_G / position：固定811检测结果

真RMS All-attention K32缓存，position=response_index=N_G；S_G为逐生成token响应范数和。无新VLM提取。
四模型原4000图全部mentions，固定3200/400/400，seeds43/44/45。先求比，再log1p和train-only逐列Z-score；位置单独组为log1p(position)一个输入维度。统一单隐藏128/ReLU/dropout.3，无BN，Adam lr.001/wd1e-5/batch128，最多150epoch/早停20/最低val BCE checkpoint，无HPO。
position=0时raw ratio=NaN、检测输入置0，保留样本。已有测试集曾被查看，以下为探索性比较。

| 模型 | S_G-only AUROC/AP | S_G/position-only AUROC/AP | position-only AUROC/AP |
|---|---:|---:|---:|
| qwen2_5_vl_7b | 80.73±0.34 / 33.69±0.54 | 80.44±0.34 / 30.05±1.19 | 73.58±0.00 / 20.36±0.00 |
| llava_1_5_7b | 87.16±0.08 / 61.52±0.03 | 86.61±0.10 / 61.63±0.20 | 81.43±0.00 / 50.77±0.00 |
| qwen3_vl_8b | 85.19±0.23 / 54.17±0.59 | 83.99±0.29 / 50.43±0.66 | 65.91±0.04 / 22.46±0.09 |
| internvl_2_5_8b | 84.24±0.09 / 49.55±0.37 | 81.84±0.28 / 43.25±0.75 | 70.49±0.00 / 27.16±0.00 |

数值为百分比、三seed指标均值±总体std，不是ensemble。AP以HALL为正类。

| 模型 | 比较 | 指标 | 差值pp [名义95% CI] |
|---|---|---|---:|
| qwen2_5_vl_7b | S_G_per_position minus S_G | AUROC | -0.29 [-2.09,+1.56] |
| qwen2_5_vl_7b | S_G_per_position minus S_G | HALL_AUPR | -3.65 [-8.48,+0.86] |
| qwen2_5_vl_7b | S_G_per_position minus position | AUROC | +6.86 [+3.31,+11.08] |
| qwen2_5_vl_7b | S_G_per_position minus position | HALL_AUPR | +9.69 [+3.27,+17.22] |
| qwen2_5_vl_7b | S_G minus position | AUROC | +7.15 [+3.08,+11.43] |
| qwen2_5_vl_7b | S_G minus position | HALL_AUPR | +13.34 [+5.01,+23.21] |
| llava_1_5_7b | S_G_per_position minus S_G | AUROC | -0.56 [-1.37,+0.28] |
| llava_1_5_7b | S_G_per_position minus S_G | HALL_AUPR | +0.11 [-3.04,+3.32] |
| llava_1_5_7b | S_G_per_position minus position | AUROC | +5.17 [+3.73,+6.83] |
| llava_1_5_7b | S_G_per_position minus position | HALL_AUPR | +10.86 [+6.55,+15.48] |
| llava_1_5_7b | S_G minus position | AUROC | +5.73 [+3.88,+7.65] |
| llava_1_5_7b | S_G minus position | HALL_AUPR | +10.75 [+5.43,+16.65] |
| qwen3_vl_8b | S_G_per_position minus S_G | AUROC | -1.20 [-2.48,+0.08] |
| qwen3_vl_8b | S_G_per_position minus S_G | HALL_AUPR | -3.74 [-7.22,-0.56] |
| qwen3_vl_8b | S_G_per_position minus position | AUROC | +18.08 [+14.65,+21.87] |
| qwen3_vl_8b | S_G_per_position minus position | HALL_AUPR | +27.97 [+22.14,+34.16] |
| qwen3_vl_8b | S_G minus position | AUROC | +19.28 [+15.77,+23.15] |
| qwen3_vl_8b | S_G minus position | HALL_AUPR | +31.71 [+25.54,+37.81] |
| internvl_2_5_8b | S_G_per_position minus S_G | AUROC | -2.40 [-4.05,-0.75] |
| internvl_2_5_8b | S_G_per_position minus S_G | HALL_AUPR | -6.29 [-10.78,-1.56] |
| internvl_2_5_8b | S_G_per_position minus position | AUROC | +11.35 [+7.82,+15.37] |
| internvl_2_5_8b | S_G_per_position minus position | HALL_AUPR | +16.09 [+8.92,+23.36] |
| internvl_2_5_8b | S_G minus position | AUROC | +13.75 [+10.11,+17.72] |
| internvl_2_5_8b | S_G minus position | HALL_AUPR | +22.39 [+14.91,+29.57] |

每模型2000次测试图片簇配对bootstrap，包括无mention图片的400图抽样框架。每次分别算3seed指标差再平均，区间未多重校正。
36头CPU重载、train-only scaler、最佳val-loss epoch、train/validation/test概率和指标复算均通过。比例归一化不等同于完全消除位置相关性或控制所有长度混杂。

[检测图](detection.png) · [REAL/HALL比值曲线](ratio_curves.png) · [核验](validation.json)
