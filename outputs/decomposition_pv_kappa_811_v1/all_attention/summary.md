# 同一 All-attention 分解的 S_PV、κ_P 与 κ_V∪P：固定811检测

本表只使用真RMS All-attention K32路径 `z−A_all→z` 的逐token FFN响应 `e_m`：`S_PV=Σ_{m∈P∪V}||e_m||`；两个κ的分母均为各自token集合的范数和，κ_V∪P不是两个组向量的κ。AE_V是所有实验共用的视觉语义注意力块，不混入旧Visual-only路径的S。

固定4000图及3200/400/400图片级811划分，全部mentions、seeds43/44/45、训练集逐列z-score、128/ReLU/dropout0.3/无BN单隐藏层、最低validation BCE-loss checkpoint，无HPO及新VLM前向。κ_P-only复用同设置旧头，其余四组共48个新头。测试集此前已查看，属于探索性比较。

## 测试集 AUROC / HALL-AUPR（%，三seed均值±总体std）

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| AE_V+log1p(S_PV) | 87.19±0.22 / 40.73±0.91 | 90.09±0.05 / 69.16±0.44 | 90.57±0.31 / 69.34±0.69 | 86.62±0.23 / 52.03±0.29 |
| AE_V+log1p(S_PV)+κ_P | 89.00±0.20 / 45.94±0.39 | 90.53±0.12 / 68.75±0.41 | 92.50±0.14 / 72.28±0.13 | 88.61±0.23 / 56.59±0.57 |
| AE_V+log1p(S_PV)+κ_V∪P | 88.01±0.42 / 46.19±0.08 | 90.30±0.02 / 68.56±0.60 | 91.74±0.10 / 68.52±0.94 | 87.52±0.16 / 55.06±0.97 |
| κ_V∪P-only | 78.36±0.08 / 29.43±0.51 | 87.82±0.28 / 63.19±0.47 | 86.49±0.43 / 58.51±1.33 | 82.68±0.28 / 46.66±0.89 |
| κ_P-only | 83.44±0.09 / 36.13±0.40 | 87.35±0.03 / 59.90±0.04 | 87.57±0.04 / 57.82±0.16 | 85.37±0.20 / 48.70±0.36 |

## 测试图片内 REAL/HALL 的κ差异

先排除同一target标签冲突，再对每张同时有两类mention的图片算HALL−REAL均值，最后跨图片等权平均；下表为正差层数，不是独立图片数。逐层的2000次配对图片bootstrap区间见CSV。

| 模型 | 同图图片数 | κ_P正差层数 | κ_V∪P正差层数 | 层均κ_P差 | 层均κ_V∪P差 |
|---|---:|---:|---:|---:|---:|
| qwen2_5_vl_7b | 60 | 28/28 | 27/28 | +0.0581 | +0.0225 |
| llava_1_5_7b | 187 | 32/32 | 21/32 | +0.0700 | +0.0291 |
| qwen3_vl_8b | 170 | 34/36 | 33/36 | +0.0494 | +0.0336 |
| internvl_2_5_8b | 123 | 31/32 | 31/32 | +0.0802 | +0.0569 |

## 配对测试图片bootstrap差值（pp，名义95%区间）

| 模型 | 比较 | 指标 | 差值 [95% CI] |
|---|---|---|---:|
| qwen2_5_vl_7b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | AUROC | +1.81 [+0.06, +3.59] |
| qwen2_5_vl_7b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | HALL_AUPR | +5.21 [-0.59, +10.98] |
| qwen2_5_vl_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | AUROC | +0.82 [-1.01, +2.69] |
| qwen2_5_vl_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | HALL_AUPR | +5.46 [-0.60, +10.41] |
| qwen2_5_vl_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | AUROC | -0.99 [-2.81, +0.70] |
| qwen2_5_vl_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | HALL_AUPR | +0.25 [-6.37, +5.44] |
| qwen2_5_vl_7b | κ_V∪P-only − κ_P-only | AUROC | -5.08 [-9.91, -0.76] |
| qwen2_5_vl_7b | κ_V∪P-only − κ_P-only | HALL_AUPR | -6.70 [-15.40, +3.28] |
| llava_1_5_7b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | AUROC | +0.44 [-0.27, +1.16] |
| llava_1_5_7b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | HALL_AUPR | -0.41 [-2.51, +1.87] |
| llava_1_5_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | AUROC | +0.21 [-0.57, +1.00] |
| llava_1_5_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | HALL_AUPR | -0.60 [-3.11, +2.00] |
| llava_1_5_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | AUROC | -0.23 [-1.00, +0.50] |
| llava_1_5_7b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | HALL_AUPR | -0.19 [-2.98, +2.38] |
| llava_1_5_7b | κ_V∪P-only − κ_P-only | AUROC | +0.47 [-0.89, +1.83] |
| llava_1_5_7b | κ_V∪P-only − κ_P-only | HALL_AUPR | +3.29 [-1.82, +8.09] |
| qwen3_vl_8b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | AUROC | +1.93 [+0.84, +3.05] |
| qwen3_vl_8b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | HALL_AUPR | +2.93 [-0.11, +5.96] |
| qwen3_vl_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | AUROC | +1.17 [+0.18, +2.16] |
| qwen3_vl_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | HALL_AUPR | -0.82 [-4.17, +2.55] |
| qwen3_vl_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | AUROC | -0.76 [-1.81, +0.21] |
| qwen3_vl_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | HALL_AUPR | -3.75 [-7.70, +0.21] |
| qwen3_vl_8b | κ_V∪P-only − κ_P-only | AUROC | -1.08 [-3.45, +1.06] |
| qwen3_vl_8b | κ_V∪P-only − κ_P-only | HALL_AUPR | +0.69 [-5.61, +6.75] |
| internvl_2_5_8b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | AUROC | +2.00 [+0.83, +3.26] |
| internvl_2_5_8b | AE_V+log1p(S_PV)+κ_P − AE_V+log1p(S_PV) | HALL_AUPR | +4.55 [+1.49, +7.32] |
| internvl_2_5_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | AUROC | +0.91 [-0.32, +2.18] |
| internvl_2_5_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV) | HALL_AUPR | +3.02 [-0.25, +6.09] |
| internvl_2_5_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | AUROC | -1.09 [-2.14, -0.06] |
| internvl_2_5_8b | AE_V+log1p(S_PV)+κ_V∪P − AE_V+log1p(S_PV)+κ_P | HALL_AUPR | -1.53 [-4.89, +1.98] |
| internvl_2_5_8b | κ_V∪P-only − κ_P-only | AUROC | -2.68 [-5.39, -0.21] |
| internvl_2_5_8b | κ_V∪P-only − κ_P-only | HALL_AUPR | -2.04 [-8.79, +4.24] |

每模型2000次以全部400张测试图片为抽样框架的配对bootstrap，含无mention图片；未作多重比较校正。48个新头均CPU重载复算，最大三划分概率误差 internvl_2_5_8b=1.79e-07；llava_1_5_7b=2.38e-07；qwen2_5_vl_7b=2.09e-07；qwen3_vl_8b=2.98e-07。

[逐层曲线](kappa_test.png) · [同图差值](kappa_paired_images.csv) · [检测CSV](detection_summary.csv) · [bootstrap CSV](bootstrap.csv) · [完整JSON](summary.json)
