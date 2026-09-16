# 五个来源/残差几何标量：固定811幻觉检测结果

复用四模型4000图的真实RMS All-attention K32与数值通过的B1缓存；没有重跑VLM。B2 K4未使用。
固定3200训练/400验证/400测试图片及全部mentions；seeds43/44/45。原几何量不取log，position取log1p；逐列StandardScaler只拟合训练集。统一单隐藏128/ReLU/dropout.3、无BN、Adam lr.001/wd1e-5/batch128、最多150epoch、早停20，最低validation BCE loss checkpoint；无HPO。

## 检测结果（%，三seed均值±总体std）

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| prompt kappa | 83.44±0.09 / 36.13±0.40 | 87.35±0.03 / 59.90±0.04 | 87.57±0.04 / 57.82±0.16 | 85.37±0.20 / 48.70±0.36 |
| generation kappa | 81.23±0.06 / 30.10±0.65 | 86.79±0.13 / 64.56±0.57 | 83.91±0.08 / 53.26±0.17 | 79.34±0.25 / 42.09±0.82 |
| B1 residual G-V balance | 81.25±0.24 / 37.23±1.01 | 88.17±0.12 / 63.92±0.45 | 84.66±0.20 / 57.90±0.41 | 82.38±0.31 / 47.42±0.57 |
| B1 delta cos(R,G) | 84.50±0.25 / 33.00±1.12 | 88.65±0.14 / 66.86±0.12 | 84.65±0.04 / 54.88±0.46 | 82.86±0.35 / 48.74±1.43 |
| visual context relative change | 78.82±0.25 / 31.05±0.44 | 87.00±0.23 / 62.37±0.44 | 83.64±0.20 / 50.88±1.90 | 81.22±0.23 / 46.13±0.75 |
| 五量拼接 | 85.54±0.31 / 41.01±0.64 | 90.08±0.11 / 68.39±0.50 | 90.77±0.16 / 66.95±0.56 | 87.44±0.27 / 56.12±0.66 |
| generation kappa + position | 82.02±0.31 / 31.71±1.11 | 86.78±0.09 / 64.20±0.24 | 83.98±0.18 / 52.99±0.10 | 80.46±0.09 / 43.73±0.77 |
| position-only（复用） | 73.58±0.00 / 20.36±0.00 | 81.43±0.00 / 50.77±0.00 | 65.91±0.04 / 22.46±0.09 | 70.49±0.00 / 27.16±0.00 |

每格为AUROC / HALL-AUPR；HALL为AP正类。position-only直接复用完全相同配置与811划分的既有三头，其余共84个新头。

## 配对图片bootstrap

| 模型 | 比较 | 指标 | 差值pp [名义95% CI] |
|---|---|---|---:|
| qwen2_5_vl_7b | prompt kappa − position-only（复用） | AUROC | +9.86 [+5.89,+13.83] |
| qwen2_5_vl_7b | prompt kappa − position-only（复用） | HALL_AUPR | +15.77 [+7.51,+23.89] |
| qwen2_5_vl_7b | generation kappa − position-only（复用） | AUROC | +7.65 [+3.04,+12.65] |
| qwen2_5_vl_7b | generation kappa − position-only（复用） | HALL_AUPR | +9.75 [+1.47,+19.11] |
| qwen2_5_vl_7b | B1 residual G-V balance − position-only（复用） | AUROC | +7.68 [+2.79,+12.94] |
| qwen2_5_vl_7b | B1 residual G-V balance − position-only（复用） | HALL_AUPR | +16.87 [+6.21,+27.56] |
| qwen2_5_vl_7b | B1 delta cos(R,G) − position-only（复用） | AUROC | +10.92 [+6.65,+15.76] |
| qwen2_5_vl_7b | B1 delta cos(R,G) − position-only（复用） | HALL_AUPR | +12.64 [+4.63,+20.68] |
| qwen2_5_vl_7b | visual context relative change − position-only（复用） | AUROC | +5.24 [+0.28,+10.05] |
| qwen2_5_vl_7b | visual context relative change − position-only（复用） | HALL_AUPR | +10.70 [+3.03,+18.11] |
| qwen2_5_vl_7b | generation kappa + position − generation kappa | AUROC | +0.79 [-0.70,+2.32] |
| qwen2_5_vl_7b | generation kappa + position − generation kappa | HALL_AUPR | +1.61 [-2.17,+4.75] |
| qwen2_5_vl_7b | generation kappa + position − position-only（复用） | AUROC | +8.45 [+4.37,+12.83] |
| qwen2_5_vl_7b | generation kappa + position − position-only（复用） | HALL_AUPR | +11.36 [+3.46,+19.99] |
| qwen2_5_vl_7b | 五量拼接 − prompt kappa | AUROC | +2.10 [-0.29,+4.56] |
| qwen2_5_vl_7b | 五量拼接 − prompt kappa | HALL_AUPR | +4.89 [-2.26,+12.97] |
| qwen2_5_vl_7b | 五量拼接 − generation kappa | AUROC | +4.31 [+0.58,+7.86] |
| qwen2_5_vl_7b | 五量拼接 − generation kappa | HALL_AUPR | +10.91 [+2.61,+18.21] |
| qwen2_5_vl_7b | 五量拼接 − B1 residual G-V balance | AUROC | +4.29 [+1.10,+7.49] |
| qwen2_5_vl_7b | 五量拼接 − B1 residual G-V balance | HALL_AUPR | +3.79 [-4.58,+12.50] |
| qwen2_5_vl_7b | 五量拼接 − B1 delta cos(R,G) | AUROC | +1.04 [-1.95,+3.77] |
| qwen2_5_vl_7b | 五量拼接 − B1 delta cos(R,G) | HALL_AUPR | +8.02 [+1.20,+14.78] |
| qwen2_5_vl_7b | 五量拼接 − visual context relative change | AUROC | +6.72 [+3.35,+10.36] |
| qwen2_5_vl_7b | 五量拼接 − visual context relative change | HALL_AUPR | +9.96 [+2.25,+18.18] |
| qwen2_5_vl_7b | 五量拼接 − position-only（复用） | AUROC | +11.96 [+7.43,+16.61] |
| qwen2_5_vl_7b | 五量拼接 − position-only（复用） | HALL_AUPR | +20.66 [+11.41,+29.79] |
| llava_1_5_7b | prompt kappa − position-only（复用） | AUROC | +5.92 [+4.06,+7.78] |
| llava_1_5_7b | prompt kappa − position-only（复用） | HALL_AUPR | +9.13 [+3.16,+15.09] |
| llava_1_5_7b | generation kappa − position-only（复用） | AUROC | +5.36 [+3.68,+6.88] |
| llava_1_5_7b | generation kappa − position-only（复用） | HALL_AUPR | +13.79 [+8.83,+18.59] |
| llava_1_5_7b | B1 residual G-V balance − position-only（复用） | AUROC | +6.74 [+4.89,+8.49] |
| llava_1_5_7b | B1 residual G-V balance − position-only（复用） | HALL_AUPR | +13.15 [+7.36,+18.73] |
| llava_1_5_7b | B1 delta cos(R,G) − position-only（复用） | AUROC | +7.22 [+5.49,+8.88] |
| llava_1_5_7b | B1 delta cos(R,G) − position-only（复用） | HALL_AUPR | +16.09 [+10.83,+21.06] |
| llava_1_5_7b | visual context relative change − position-only（复用） | AUROC | +5.57 [+3.81,+7.26] |
| llava_1_5_7b | visual context relative change − position-only（复用） | HALL_AUPR | +11.61 [+6.39,+16.54] |
| llava_1_5_7b | generation kappa + position − generation kappa | AUROC | -0.02 [-0.37,+0.33] |
| llava_1_5_7b | generation kappa + position − generation kappa | HALL_AUPR | -0.36 [-1.61,+0.92] |
| llava_1_5_7b | generation kappa + position − position-only（复用） | AUROC | +5.34 [+3.83,+6.74] |
| llava_1_5_7b | generation kappa + position − position-only（复用） | HALL_AUPR | +13.43 [+8.95,+17.90] |
| llava_1_5_7b | 五量拼接 − prompt kappa | AUROC | +2.73 [+1.61,+3.88] |
| llava_1_5_7b | 五量拼接 − prompt kappa | HALL_AUPR | +8.49 [+4.43,+12.31] |
| llava_1_5_7b | 五量拼接 − generation kappa | AUROC | +3.28 [+1.92,+4.79] |
| llava_1_5_7b | 五量拼接 − generation kappa | HALL_AUPR | +3.83 [-0.27,+8.36] |
| llava_1_5_7b | 五量拼接 − B1 residual G-V balance | AUROC | +1.91 [+0.82,+3.16] |
| llava_1_5_7b | 五量拼接 − B1 residual G-V balance | HALL_AUPR | +4.47 [+1.28,+7.91] |
| llava_1_5_7b | 五量拼接 − B1 delta cos(R,G) | AUROC | +1.42 [+0.40,+2.51] |
| llava_1_5_7b | 五量拼接 − B1 delta cos(R,G) | HALL_AUPR | +1.53 [-1.93,+5.01] |
| llava_1_5_7b | 五量拼接 − visual context relative change | AUROC | +3.08 [+1.79,+4.44] |
| llava_1_5_7b | 五量拼接 − visual context relative change | HALL_AUPR | +6.02 [+1.97,+9.98] |
| llava_1_5_7b | 五量拼接 − position-only（复用） | AUROC | +8.64 [+6.58,+10.69] |
| llava_1_5_7b | 五量拼接 − position-only（复用） | HALL_AUPR | +17.62 [+11.34,+23.65] |
| qwen3_vl_8b | prompt kappa − position-only（复用） | AUROC | +21.66 [+18.57,+24.90] |
| qwen3_vl_8b | prompt kappa − position-only（复用） | HALL_AUPR | +35.36 [+29.12,+41.65] |
| qwen3_vl_8b | generation kappa − position-only（复用） | AUROC | +18.00 [+14.38,+21.68] |
| qwen3_vl_8b | generation kappa − position-only（复用） | HALL_AUPR | +30.80 [+24.76,+36.08] |
| qwen3_vl_8b | B1 residual G-V balance − position-only（复用） | AUROC | +18.75 [+15.57,+22.30] |
| qwen3_vl_8b | B1 residual G-V balance − position-only（复用） | HALL_AUPR | +35.44 [+30.01,+40.70] |
| qwen3_vl_8b | B1 delta cos(R,G) − position-only（复用） | AUROC | +18.73 [+15.18,+22.56] |
| qwen3_vl_8b | B1 delta cos(R,G) − position-only（复用） | HALL_AUPR | +32.43 [+26.27,+38.03] |
| qwen3_vl_8b | visual context relative change − position-only（复用） | AUROC | +17.73 [+14.31,+21.29] |
| qwen3_vl_8b | visual context relative change − position-only（复用） | HALL_AUPR | +28.42 [+22.46,+34.56] |
| qwen3_vl_8b | generation kappa + position − generation kappa | AUROC | +0.07 [-0.57,+0.70] |
| qwen3_vl_8b | generation kappa + position − generation kappa | HALL_AUPR | -0.27 [-2.26,+1.66] |
| qwen3_vl_8b | generation kappa + position − position-only（复用） | AUROC | +18.07 [+14.71,+21.62] |
| qwen3_vl_8b | generation kappa + position − position-only（复用） | HALL_AUPR | +30.54 [+24.29,+35.78] |
| qwen3_vl_8b | 五量拼接 − prompt kappa | AUROC | +3.20 [+1.57,+4.83] |
| qwen3_vl_8b | 五量拼接 − prompt kappa | HALL_AUPR | +9.13 [+3.77,+14.30] |
| qwen3_vl_8b | 五量拼接 − generation kappa | AUROC | +6.86 [+4.81,+9.03] |
| qwen3_vl_8b | 五量拼接 − generation kappa | HALL_AUPR | +13.69 [+8.46,+19.26] |
| qwen3_vl_8b | 五量拼接 − B1 residual G-V balance | AUROC | +6.11 [+4.07,+8.05] |
| qwen3_vl_8b | 五量拼接 − B1 residual G-V balance | HALL_AUPR | +9.05 [+3.57,+14.57] |
| qwen3_vl_8b | 五量拼接 − B1 delta cos(R,G) | AUROC | +6.12 [+4.08,+8.15] |
| qwen3_vl_8b | 五量拼接 − B1 delta cos(R,G) | HALL_AUPR | +12.07 [+6.79,+17.63] |
| qwen3_vl_8b | 五量拼接 − visual context relative change | AUROC | +7.13 [+5.48,+8.99] |
| qwen3_vl_8b | 五量拼接 − visual context relative change | HALL_AUPR | +16.07 [+10.95,+21.18] |
| qwen3_vl_8b | 五量拼接 − position-only（复用） | AUROC | +24.86 [+21.64,+28.28] |
| qwen3_vl_8b | 五量拼接 − position-only（复用） | HALL_AUPR | +44.49 [+38.59,+49.96] |
| internvl_2_5_8b | prompt kappa − position-only（复用） | AUROC | +14.88 [+11.59,+18.37] |
| internvl_2_5_8b | prompt kappa − position-only（复用） | HALL_AUPR | +21.54 [+14.01,+28.40] |
| internvl_2_5_8b | generation kappa − position-only（复用） | AUROC | +8.85 [+5.03,+12.53] |
| internvl_2_5_8b | generation kappa − position-only（复用） | HALL_AUPR | +14.93 [+7.78,+21.90] |
| internvl_2_5_8b | B1 residual G-V balance − position-only（复用） | AUROC | +11.89 [+7.94,+15.99] |
| internvl_2_5_8b | B1 residual G-V balance − position-only（复用） | HALL_AUPR | +20.26 [+12.17,+27.89] |
| internvl_2_5_8b | B1 delta cos(R,G) − position-only（复用） | AUROC | +12.38 [+8.45,+16.59] |
| internvl_2_5_8b | B1 delta cos(R,G) − position-only（复用） | HALL_AUPR | +21.59 [+14.16,+29.24] |
| internvl_2_5_8b | visual context relative change − position-only（复用） | AUROC | +10.73 [+6.69,+15.19] |
| internvl_2_5_8b | visual context relative change − position-only（复用） | HALL_AUPR | +18.97 [+10.98,+26.79] |
| internvl_2_5_8b | generation kappa + position − generation kappa | AUROC | +1.12 [-0.10,+2.40] |
| internvl_2_5_8b | generation kappa + position − generation kappa | HALL_AUPR | +1.64 [-1.07,+4.35] |
| internvl_2_5_8b | generation kappa + position − position-only（复用） | AUROC | +9.97 [+6.53,+13.34] |
| internvl_2_5_8b | generation kappa + position − position-only（复用） | HALL_AUPR | +16.57 [+9.42,+23.42] |
| internvl_2_5_8b | 五量拼接 − prompt kappa | AUROC | +2.07 [-0.17,+4.39] |
| internvl_2_5_8b | 五量拼接 − prompt kappa | HALL_AUPR | +7.42 [+1.43,+13.19] |
| internvl_2_5_8b | 五量拼接 − generation kappa | AUROC | +8.10 [+5.32,+11.12] |
| internvl_2_5_8b | 五量拼接 − generation kappa | HALL_AUPR | +14.03 [+7.43,+20.29] |
| internvl_2_5_8b | 五量拼接 − B1 residual G-V balance | AUROC | +5.06 [+2.77,+7.34] |
| internvl_2_5_8b | 五量拼接 − B1 residual G-V balance | HALL_AUPR | +8.70 [+2.56,+15.22] |
| internvl_2_5_8b | 五量拼接 − B1 delta cos(R,G) | AUROC | +4.58 [+2.52,+6.68] |
| internvl_2_5_8b | 五量拼接 − B1 delta cos(R,G) | HALL_AUPR | +7.37 [+1.57,+12.55] |
| internvl_2_5_8b | 五量拼接 − visual context relative change | AUROC | +6.22 [+4.00,+8.60] |
| internvl_2_5_8b | 五量拼接 − visual context relative change | HALL_AUPR | +9.99 [+3.96,+15.52] |
| internvl_2_5_8b | 五量拼接 − position-only（复用） | AUROC | +16.95 [+13.33,+21.06] |
| internvl_2_5_8b | 五量拼接 − position-only（复用） | HALL_AUPR | +28.96 [+21.30,+36.29] |

每模型2000次测试图片簇配对bootstrap，抽样框架包含无mention测试图片；区间未作多重比较校正。测试集已在此前实验中查看，本结果属于探索性复核。

## 核验

84个新头均已CPU重载；train-only scaler、最低val-loss checkpoint、train/validation/test概率和指标全部复算。最大概率误差：qwen2_5_vl_7b=1.79e-07；llava_1_5_7b=1.79e-07；qwen3_vl_8b=4.17e-07；internvl_2_5_8b=2.38e-07。

[检测图](../outputs/selected_geometry_scalars_811_v1/detection.png) · [逐seed](../outputs/selected_geometry_scalars_811_v1/seed_metrics.csv) · [bootstrap](../outputs/selected_geometry_scalars_811_v1/bootstrap.csv) · [完整JSON](../outputs/selected_geometry_scalars_811_v1/summary.json)
