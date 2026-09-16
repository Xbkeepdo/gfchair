# Prompt WRITE κ_P^a 的幻觉检测及与 F 拼接

κ_P^a = ||Σ_{m∈P} a_m|| / Σ_{m∈P}||a_m||，与此前 FFN 响应 κ_P^e = ||Σ e_m|| / Σ||e_m|| 区分。缓存中用 `prompt_net_norm / prompt_group_gain` 还原分子 `||Σa_m||`，分母取逐token WRITE 范数和（prompt I）；这与原提取代码的 gain 定义严格对应，精度受已存F1 float32影响。四模型全量值有限且位于(0,1]。

固定811：3200/400/400图、全部mentions、seeds43/44/45、train-only逐列z-score、单隐藏128/ReLU/dropout.3、无BN、val-loss checkpoint、无HPO。κ_P^e、F、F+κ_P^e直接复用完全同配置的既有三seed头；仅κ_P^a与F+κ_P^a新训练24头。无新VLM前向、无B2。

## 测试集检测（AUROC / HALL-AUPR，%，三seed均值±总体std）

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| κ_P^a only | 83.96±0.39 / 35.90±0.62 | 87.23±0.07 / 60.21±0.67 | 87.18±0.22 / 58.14±0.36 | 85.44±0.15 / 51.48±0.65 |
| κ_P^e only（复用） | 83.44±0.09 / 36.13±0.40 | 87.35±0.03 / 59.90±0.04 | 87.57±0.04 / 57.82±0.16 | 85.37±0.20 / 48.70±0.36 |
| F=[AE_V,log1p(S_E)]（复用） | 88.29±0.24 / 42.39±0.82 | 89.83±0.06 / 68.55±0.37 | 89.85±0.08 / 67.84±0.33 | 86.24±0.37 / 53.86±0.35 |
| F+κ_P^a | 89.17±0.22 / 47.02±0.17 | 90.47±0.13 / 69.12±0.72 | 91.49±0.14 / 69.80±0.54 | 87.97±0.20 / 57.32±0.17 |
| F+κ_P^e（复用） | 88.96±0.26 / 45.54±1.32 | 90.59±0.19 / 68.74±0.71 | 91.93±0.21 / 71.92±0.66 | 88.17±0.10 / 56.85±0.22 |

## 结论

- κ_P^a 单独与 κ_P^e 单独在四模型的AUROC与AP差值名义区间均跨零，没有明确赢家。
- F+κ_P^a 相对F的四模型宏平均AUROC/AP提高+1.22/+2.66pp；AUROC在LLaVA、Qwen3、InternVL名义区间全正，AP仅InternVL全正。Qwen2.5两项区间均跨零。
- F+κ_P^a 对F+κ_P^e的差值，仅Qwen3 AP名义区间全负（−2.12pp [−3.99,−0.16]）；其余模型两项区间均跨零。不能笼统地说WRITE κ比FFN响应κ更好。
- 同cohort原始逐层曲线显示HALL的两种κ普遍高于REAL，但数值分离不等同于多特征分类器中的独立增益。

## 配对图片bootstrap差值

每模型2000次400测试图片簇配对抽样，含无mention图片；差值为左−右，单位pp，名义95% CI未作多重比较校正。

| 模型 | 对比 | 指标 | 差值pp [95% CI] |
|---|---|---|---:|
| qwen2_5_vl_7b | κ_P^a only − κ_P^e only（复用） | AUROC | +0.52 [-0.71,+1.73] |
| qwen2_5_vl_7b | κ_P^a only − κ_P^e only（复用） | HALL_AUPR | -0.22 [-4.90,+4.90] |
| qwen2_5_vl_7b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | AUROC | +0.88 [-0.85,+2.75] |
| qwen2_5_vl_7b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +4.63 [-1.19,+10.95] |
| qwen2_5_vl_7b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | AUROC | +0.68 [-0.91,+2.34] |
| qwen2_5_vl_7b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +3.15 [-2.34,+9.17] |
| qwen2_5_vl_7b | F+κ_P^a − F+κ_P^e（复用） | AUROC | +0.20 [-0.55,+0.95] |
| qwen2_5_vl_7b | F+κ_P^a − F+κ_P^e（复用） | HALL_AUPR | +1.48 [-1.33,+4.50] |
| llava_1_5_7b | κ_P^a only − κ_P^e only（复用） | AUROC | -0.12 [-0.74,+0.56] |
| llava_1_5_7b | κ_P^a only − κ_P^e only（复用） | HALL_AUPR | +0.32 [-2.35,+3.07] |
| llava_1_5_7b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | AUROC | +0.64 [+0.05,+1.27] |
| llava_1_5_7b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +0.57 [-1.47,+2.70] |
| llava_1_5_7b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | AUROC | +0.76 [+0.06,+1.51] |
| llava_1_5_7b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +0.19 [-1.94,+2.35] |
| llava_1_5_7b | F+κ_P^a − F+κ_P^e（复用） | AUROC | -0.12 [-0.54,+0.27] |
| llava_1_5_7b | F+κ_P^a − F+κ_P^e（复用） | HALL_AUPR | +0.38 [-1.00,+1.67] |
| qwen3_vl_8b | κ_P^a only − κ_P^e only（复用） | AUROC | -0.39 [-1.49,+0.72] |
| qwen3_vl_8b | κ_P^a only − κ_P^e only（复用） | HALL_AUPR | +0.32 [-3.39,+4.05] |
| qwen3_vl_8b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | AUROC | +1.63 [+0.62,+2.71] |
| qwen3_vl_8b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +1.97 [-0.96,+5.11] |
| qwen3_vl_8b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | AUROC | +2.07 [+1.02,+3.12] |
| qwen3_vl_8b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +4.09 [+1.34,+7.12] |
| qwen3_vl_8b | F+κ_P^a − F+κ_P^e（复用） | AUROC | -0.44 [-1.05,+0.16] |
| qwen3_vl_8b | F+κ_P^a − F+κ_P^e（复用） | HALL_AUPR | -2.12 [-3.99,-0.16] |
| internvl_2_5_8b | κ_P^a only − κ_P^e only（复用） | AUROC | +0.07 [-1.12,+1.30] |
| internvl_2_5_8b | κ_P^a only − κ_P^e only（复用） | HALL_AUPR | +2.78 [-1.19,+7.28] |
| internvl_2_5_8b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | AUROC | +1.73 [+0.54,+2.93] |
| internvl_2_5_8b | F+κ_P^a − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +3.46 [+0.14,+6.49] |
| internvl_2_5_8b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | AUROC | +1.93 [+0.85,+3.07] |
| internvl_2_5_8b | F+κ_P^e（复用） − F=[AE_V,log1p(S_E)]（复用） | HALL_AUPR | +2.99 [-0.04,+5.91] |
| internvl_2_5_8b | F+κ_P^a − F+κ_P^e（复用） | AUROC | -0.20 [-0.84,+0.43] |
| internvl_2_5_8b | F+κ_P^a − F+κ_P^e（复用） | HALL_AUPR | +0.48 [-1.57,+2.25] |

测试集此前已查看，本结果是探索性比较，不能作为独立泛化或因果结论。

## 核验

新头全部CPU重载并复算训练/验证/测试概率、train-only scaler、最低validation-loss checkpoint；最大概率误差：qwen2_5_vl_7b=1.79e-07；llava_1_5_7b=2.38e-07；qwen3_vl_8b=2.68e-07；internvl_2_5_8b=2.09e-07。

[检测图](../outputs/prompt_write_kappa_811_v1/detection.png) · [a_m/e_m全量逐层图](../outputs/prompt_write_kappa_811_v1/kappa_curves_all.png) · [811测试逐层图](../outputs/prompt_write_kappa_811_v1/kappa_curves_test.png) · [逐seed](../outputs/prompt_write_kappa_811_v1/seed_metrics.csv) · [bootstrap](../outputs/prompt_write_kappa_811_v1/bootstrap.csv) · [完整JSON](../outputs/prompt_write_kappa_811_v1/summary.json)
