# AE + log1p(S) 分别与五个几何量拼接：固定811检测结果

这里的F严格采用项目既有`legacy_visual=[AE_V, log1p(S_E)]`：Visual-only条件路径，真RMS沿`z-A_V -> z`求差。五个几何块为此前选定的全层prompt/generation kappa、B1 residual G-V balance、B1 delta cos(R,G)与visual context relative change；B2 K4未使用。

固定3200训练/400验证/400测试图片及全部mentions，seeds43/44/45。先拼接原始块，再逐列用训练集拟合StandardScaler；统一单隐藏128/ReLU/dropout.3、无BN、Adam lr.001/wd1e-5/batch128、最多150epoch、早停20，最低validation BCE loss checkpoint；无HPO。

## 检测结果（%，三seed均值±总体std）

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| F = AE_V + log1p(S_E) | 88.29±0.24 / 42.39±0.82 | 89.83±0.06 / 68.55±0.37 | 89.85±0.08 / 67.84±0.33 | 86.24±0.37 / 53.86±0.35 |
| F + prompt kappa | 88.96±0.26 / 45.54±1.32 | 90.59±0.19 / 68.74±0.71 | 91.93±0.21 / 71.92±0.66 | 88.17±0.10 / 56.85±0.22 |
| F + generation kappa | 88.00±0.46 / 44.62±1.13 | 90.37±0.11 / 70.08±0.49 | 90.92±0.22 / 69.81±0.54 | 86.99±0.10 / 54.77±0.72 |
| F + B1 residual G-V balance | 88.92±0.15 / 50.00±0.60 | 90.29±0.05 / 68.09±0.39 | 91.28±0.26 / 68.50±0.45 | 86.31±0.25 / 54.11±0.72 |
| F + B1 delta cos(R,G) | 89.85±0.26 / 47.49±1.89 | 90.81±0.12 / 69.42±0.31 | 91.25±0.12 / 69.25±0.27 | 87.53±0.18 / 56.94±0.26 |
| F + visual context change | 88.96±0.19 / 45.82±0.69 | 90.73±0.14 / 70.31±0.72 | 90.88±0.23 / 68.10±0.83 | 86.15±0.39 / 54.70±1.37 |

每格为AUROC / HALL-AUPR；HALL为AP正类。主结果包含F与五种分别拼接的72个新头。用户澄清到达前队列已额外启动五量合并的12个头；它们只保存在`supplementary_*`文件，不进入本表、图或结论。

## 结论

- prompt kappa跨模型宏平均增量为AUROC +1.36pp、HALL-AUPR +2.60pp；AUROC在LLaVA/Qwen3/InternVL的名义95%区间全正，Qwen3的两项区间均全正。
- B1 delta cos(R,G)跨模型宏平均增量为AUROC +1.31pp、HALL-AUPR +2.62pp；AUROC在Qwen2.5/LLaVA/Qwen3区间全正，Qwen2.5的AP也全正。它与prompt kappa是最稳定的两个增量块。
- generation kappa、B1 residual balance和visual context change都有局部收益，但跨模型一致性较弱：分别仅Qwen3 AUROC、Qwen2.5 AP与Qwen3 AUROC、LLaVA AUROC得到全正区间。
- 按点估计，Qwen2.5最优AUROC/AP分别来自B1 delta cos与B1 residual balance；LLaVA来自B1 delta cos与visual context change；Qwen3两项均来自prompt kappa；InternVL来自prompt kappa与B1 delta cos。

## 主要配对图片bootstrap

| 模型 | 比较 | 指标 | 差值pp [名义95% CI] |
|---|---|---|---:|
| qwen2_5_vl_7b | F + prompt kappa − F = AE_V + log1p(S_E) | AUROC | +0.68 [-0.91,+2.34] |
| qwen2_5_vl_7b | F + prompt kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +3.15 [-2.34,+9.17] |
| qwen2_5_vl_7b | F + generation kappa − F = AE_V + log1p(S_E) | AUROC | -0.29 [-1.72,+1.10] |
| qwen2_5_vl_7b | F + generation kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +2.24 [-2.44,+6.83] |
| qwen2_5_vl_7b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | AUROC | +0.63 [-1.11,+2.32] |
| qwen2_5_vl_7b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | HALL_AUPR | +7.61 [+2.12,+13.92] |
| qwen2_5_vl_7b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | AUROC | +1.57 [+0.27,+2.85] |
| qwen2_5_vl_7b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | HALL_AUPR | +5.11 [+1.12,+9.82] |
| qwen2_5_vl_7b | F + visual context change − F = AE_V + log1p(S_E) | AUROC | +0.67 [-0.97,+2.22] |
| qwen2_5_vl_7b | F + visual context change − F = AE_V + log1p(S_E) | HALL_AUPR | +3.43 [-1.91,+8.47] |
| llava_1_5_7b | F + prompt kappa − F = AE_V + log1p(S_E) | AUROC | +0.76 [+0.06,+1.51] |
| llava_1_5_7b | F + prompt kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +0.19 [-1.94,+2.35] |
| llava_1_5_7b | F + generation kappa − F = AE_V + log1p(S_E) | AUROC | +0.54 [-0.02,+1.12] |
| llava_1_5_7b | F + generation kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +1.54 [-0.10,+3.44] |
| llava_1_5_7b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | AUROC | +0.46 [-0.28,+1.16] |
| llava_1_5_7b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | HALL_AUPR | -0.46 [-2.79,+1.83] |
| llava_1_5_7b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | AUROC | +0.98 [+0.28,+1.68] |
| llava_1_5_7b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | HALL_AUPR | +0.88 [-1.57,+3.52] |
| llava_1_5_7b | F + visual context change − F = AE_V + log1p(S_E) | AUROC | +0.90 [+0.17,+1.66] |
| llava_1_5_7b | F + visual context change − F = AE_V + log1p(S_E) | HALL_AUPR | +1.77 [-1.06,+4.95] |
| qwen3_vl_8b | F + prompt kappa − F = AE_V + log1p(S_E) | AUROC | +2.07 [+1.02,+3.12] |
| qwen3_vl_8b | F + prompt kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +4.09 [+1.34,+7.12] |
| qwen3_vl_8b | F + generation kappa − F = AE_V + log1p(S_E) | AUROC | +1.07 [+0.13,+2.02] |
| qwen3_vl_8b | F + generation kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +1.97 [-0.88,+4.97] |
| qwen3_vl_8b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | AUROC | +1.43 [+0.50,+2.34] |
| qwen3_vl_8b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | HALL_AUPR | +0.66 [-1.89,+3.38] |
| qwen3_vl_8b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | AUROC | +1.39 [+0.14,+2.63] |
| qwen3_vl_8b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | HALL_AUPR | +1.41 [-1.62,+4.56] |
| qwen3_vl_8b | F + visual context change − F = AE_V + log1p(S_E) | AUROC | +1.03 [-0.20,+2.28] |
| qwen3_vl_8b | F + visual context change − F = AE_V + log1p(S_E) | HALL_AUPR | +0.26 [-3.65,+3.79] |
| internvl_2_5_8b | F + prompt kappa − F = AE_V + log1p(S_E) | AUROC | +1.93 [+0.85,+3.07] |
| internvl_2_5_8b | F + prompt kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +2.99 [-0.04,+5.91] |
| internvl_2_5_8b | F + generation kappa − F = AE_V + log1p(S_E) | AUROC | +0.75 [-0.33,+1.80] |
| internvl_2_5_8b | F + generation kappa − F = AE_V + log1p(S_E) | HALL_AUPR | +0.92 [-2.05,+3.82] |
| internvl_2_5_8b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | AUROC | +0.07 [-1.05,+1.13] |
| internvl_2_5_8b | F + B1 residual G-V balance − F = AE_V + log1p(S_E) | HALL_AUPR | +0.25 [-3.01,+3.48] |
| internvl_2_5_8b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | AUROC | +1.29 [-0.17,+2.76] |
| internvl_2_5_8b | F + B1 delta cos(R,G) − F = AE_V + log1p(S_E) | HALL_AUPR | +3.08 [-0.28,+6.27] |
| internvl_2_5_8b | F + visual context change − F = AE_V + log1p(S_E) | AUROC | -0.09 [-1.39,+1.29] |
| internvl_2_5_8b | F + visual context change − F = AE_V + log1p(S_E) | HALL_AUPR | +0.84 [-3.08,+4.63] |

每模型2000次测试图片簇配对bootstrap，抽样框架包含无mention测试图片；五个逐项比较见主CSV，额外队列结果见supplementary CSV。区间未作多重比较校正。测试集已在此前实验中查看，本结果属于探索性复核。

## 核验

84个新头均已CPU重载；train-only scaler、最低val-loss checkpoint、train/validation/test概率和指标全部复算。最大概率误差：qwen2_5_vl_7b=2.38e-07；llava_1_5_7b=2.98e-07；qwen3_vl_8b=3.58e-07；internvl_2_5_8b=2.38e-07。

[检测图](../outputs/legacy_visual_geometry_fusion_811_v1/detection.png) · [逐seed](../outputs/legacy_visual_geometry_fusion_811_v1/seed_metrics.csv) · [bootstrap](../outputs/legacy_visual_geometry_fusion_811_v1/bootstrap.csv) · [完整JSON](../outputs/legacy_visual_geometry_fusion_811_v1/summary.json)
