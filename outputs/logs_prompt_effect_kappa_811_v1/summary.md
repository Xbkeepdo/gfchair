# log1p(S_E) + κ_P^e：固定811幻觉检测

S_E为Visual-only真实RMS条件路径的逐视觉token FFN响应范数和；κ_P^e=||Σ_P e_m||/Σ_P||e_m||，来自All-attention K32。两块各取全层向量，直接拼接后仅用训练集拟合逐列z-score；不包含AE_V。

图片级3200/400/400，保留全部mentions；seeds43/44/45，单隐藏128/ReLU/dropout0.3、无BN、Adam，最低validation BCE-loss checkpoint，无HPO、无新VLM前向。κ-only与AE+logS+κ直接复用同协议已有头。测试集已多次查看，以下为探索性结果。

## 测试集 AUROC / HALL-AUPR（%，三seed均值±总体std）

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| log1p(S_E) | 82.63±0.27 / 35.18±0.18 | 87.93±0.06 / 65.37±0.25 | 88.14±0.08 / 62.57±0.46 | 85.09±0.30 / 53.62±0.25 |
| κ_P^e | 83.44±0.09 / 36.13±0.40 | 87.35±0.03 / 59.90±0.04 | 87.57±0.04 / 57.82±0.16 | 85.37±0.20 / 48.70±0.36 |
| log1p(S_E)+κ_P^e | 85.61±0.47 / 40.29±0.45 | 89.37±0.15 / 64.79±0.61 | 91.00±0.27 / 68.09±0.33 | 87.42±0.01 / 54.34±0.76 |
| AE_V+log1p(S_E)+κ_P^e | 88.96±0.26 / 45.54±1.32 | 90.59±0.19 / 68.74±0.71 | 91.93±0.21 / 71.92±0.66 | 88.17±0.10 / 56.85±0.22 |

## 配对测试图片bootstrap差值（pp，名义95%区间）

| 模型 | 比较 | 指标 | 差值 [95% CI] |
|---|---|---|---:|
| qwen2_5_vl_7b | log1p(S_E)+κ_P^e − log1p(S_E) | AUROC | +2.98 [+0.53, +5.59] |
| qwen2_5_vl_7b | log1p(S_E)+κ_P^e − log1p(S_E) | HALL_AUPR | +5.11 [-1.81, +10.82] |
| qwen2_5_vl_7b | log1p(S_E)+κ_P^e − κ_P^e | AUROC | +2.17 [+0.00, +4.37] |
| qwen2_5_vl_7b | log1p(S_E)+κ_P^e − κ_P^e | HALL_AUPR | +4.17 [-3.06, +11.65] |
| qwen2_5_vl_7b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | AUROC | +3.35 [+1.33, +5.44] |
| qwen2_5_vl_7b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | HALL_AUPR | +5.25 [-0.16, +11.17] |
| llava_1_5_7b | log1p(S_E)+κ_P^e − log1p(S_E) | AUROC | +1.45 [+0.49, +2.39] |
| llava_1_5_7b | log1p(S_E)+κ_P^e − log1p(S_E) | HALL_AUPR | -0.59 [-3.31, +2.73] |
| llava_1_5_7b | log1p(S_E)+κ_P^e − κ_P^e | AUROC | +2.02 [+1.09, +2.97] |
| llava_1_5_7b | log1p(S_E)+κ_P^e − κ_P^e | HALL_AUPR | +4.89 [+1.85, +8.14] |
| llava_1_5_7b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | AUROC | +1.22 [+0.53, +1.96] |
| llava_1_5_7b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | HALL_AUPR | +3.95 [+0.77, +6.81] |
| qwen3_vl_8b | log1p(S_E)+κ_P^e − log1p(S_E) | AUROC | +2.87 [+1.75, +4.07] |
| qwen3_vl_8b | log1p(S_E)+κ_P^e − log1p(S_E) | HALL_AUPR | +5.52 [+2.21, +8.99] |
| qwen3_vl_8b | log1p(S_E)+κ_P^e − κ_P^e | AUROC | +3.43 [+2.01, +4.92] |
| qwen3_vl_8b | log1p(S_E)+κ_P^e − κ_P^e | HALL_AUPR | +10.27 [+6.00, +14.54] |
| qwen3_vl_8b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | AUROC | +0.92 [-0.00, +1.90] |
| qwen3_vl_8b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | HALL_AUPR | +3.83 [+0.70, +7.31] |
| internvl_2_5_8b | log1p(S_E)+κ_P^e − log1p(S_E) | AUROC | +2.33 [+0.65, +4.15] |
| internvl_2_5_8b | log1p(S_E)+κ_P^e − log1p(S_E) | HALL_AUPR | +0.72 [-3.31, +4.86] |
| internvl_2_5_8b | log1p(S_E)+κ_P^e − κ_P^e | AUROC | +2.05 [+0.40, +3.78] |
| internvl_2_5_8b | log1p(S_E)+κ_P^e − κ_P^e | HALL_AUPR | +5.64 [+0.35, +11.02] |
| internvl_2_5_8b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | AUROC | +0.75 [-0.53, +1.99] |
| internvl_2_5_8b | AE_V+log1p(S_E)+κ_P^e − log1p(S_E)+κ_P^e | HALL_AUPR | +2.51 [-2.44, +7.48] |

每模型2000次以400张测试图片为簇的配对bootstrap（包括无mention图片）；区间未作多重比较校正。24个新头均CPU重载、train-only scaler、最低val-loss checkpoint和三划分概率复算通过；最大概率误差 qwen2_5_vl_7b=1.79e-07；llava_1_5_7b=1.79e-07；qwen3_vl_8b=3.58e-07；internvl_2_5_8b=2.38e-07。

[逐seed](seed_metrics.csv) · [bootstrap](bootstrap.csv) · [完整JSON](summary.json)
