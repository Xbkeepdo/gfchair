# SVAR/MetaToken：相同811划分与分类器对照

逐模型严格复用区域AE的3200训练/400验证/400测试图片、全部mentions、原标签及seeds43/44/45。SVAR为controlled样本、零基索引[5,19)层各head视觉attention mass；MetaToken为原10+H维特征。
三层[128,64,32]/BN/dropout.3，验证loss选择checkpoint/调度/早停；sklearn单层12候选、XGB18候选按验证AUROC/AP选参。无额外标准化。这是统一分类器的特征对照，不是SVAR原生248隐藏单层或MetaToken原生Scaler+LR/GB。
MetaToken保留完整回答长度和对象span统计，信息范围比严格pre-target信号更宽。原800图已有研究使用，本轮为探索性评估；无新bootstrap。

| 模型 | 特征 | 分类器 | AUROC均值±std | HALL-AUPR均值±std |
|---|---|---|---:|---:|
| qwen2_5_vl_7b | svar | three_hidden | 86.48 ± 0.37 | 43.34 ± 1.52 |
| qwen2_5_vl_7b | svar | one_hidden | 86.81 ± 0.34 | 45.90 ± 0.54 |
| qwen2_5_vl_7b | svar | xgb | 87.49 ± 0.00 | 44.39 ± 0.00 |
| qwen2_5_vl_7b | metatoken | three_hidden | 76.85 ± 0.53 | 22.20 ± 0.84 |
| qwen2_5_vl_7b | metatoken | one_hidden | 77.92 ± 0.24 | 27.12 ± 0.91 |
| qwen2_5_vl_7b | metatoken | xgb | 83.74 ± 0.00 | 38.66 ± 0.00 |
| llava_1_5_7b | svar | three_hidden | 90.62 ± 0.19 | 70.79 ± 0.36 |
| llava_1_5_7b | svar | one_hidden | 89.87 ± 0.13 | 68.39 ± 0.12 |
| llava_1_5_7b | svar | xgb | 90.08 ± 0.00 | 69.03 ± 0.00 |
| llava_1_5_7b | metatoken | three_hidden | 86.22 ± 0.35 | 63.77 ± 0.63 |
| llava_1_5_7b | metatoken | one_hidden | 86.33 ± 0.09 | 62.05 ± 0.37 |
| llava_1_5_7b | metatoken | xgb | 89.14 ± 0.00 | 66.33 ± 0.00 |
| qwen3_vl_8b | svar | three_hidden | 89.20 ± 0.23 | 61.92 ± 0.78 |
| qwen3_vl_8b | svar | one_hidden | 89.05 ± 0.11 | 60.48 ± 0.35 |
| qwen3_vl_8b | svar | xgb | 88.53 ± 0.00 | 60.98 ± 0.00 |
| qwen3_vl_8b | metatoken | three_hidden | 77.20 ± 0.35 | 36.93 ± 0.20 |
| qwen3_vl_8b | metatoken | one_hidden | 77.00 ± 0.14 | 37.24 ± 0.36 |
| qwen3_vl_8b | metatoken | xgb | 87.67 ± 0.00 | 59.19 ± 0.00 |
| internvl_2_5_8b | svar | three_hidden | 87.93 ± 0.26 | 54.28 ± 1.00 |
| internvl_2_5_8b | svar | one_hidden | 86.35 ± 0.08 | 52.40 ± 0.49 |
| internvl_2_5_8b | svar | xgb | 88.31 ± 0.00 | 57.34 ± 0.00 |
| internvl_2_5_8b | metatoken | three_hidden | 78.52 ± 0.26 | 38.85 ± 0.79 |
| internvl_2_5_8b | metatoken | one_hidden | 78.72 ± 0.55 | 37.86 ± 2.20 |
| internvl_2_5_8b | metatoken | xgb | 85.69 ± 0.00 | 50.35 ± 0.00 |

逐seed及概率ensemble见CSV/保存的result.pt；ae_vs_baselines.csv中的差值为AE组减baseline，始终固定相同分类器及400测试图。正值表示AE组更高，不据此宣称显著性。

## 同分类器下与AE拼接直接比较

每格AUROC / HALL-AUPR（%，三seed均值）；V、VP、G均指区域AE＋对应All-attention gross的log1p。

| 模型 | 分类器 | SVAR | MetaToken | V | VP | G | VP+G |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen2_5_vl_7b | three_hidden | 86.48 / 43.34 | 76.85 / 22.20 | 86.21 / 39.25 | 86.21 / 40.03 | 82.02 / 35.22 | 87.36 / 46.83 |
| qwen2_5_vl_7b | one_hidden | 86.81 / 45.90 | 77.92 / 27.12 | 82.05 / 32.24 | 81.79 / 33.02 | 80.64 / 29.09 | 83.75 / 36.98 |
| qwen2_5_vl_7b | xgb | 87.49 / 44.39 | 83.74 / 38.66 | 86.55 / 40.79 | 86.37 / 45.64 | 82.45 / 34.94 | 87.54 / 48.40 |
| llava_1_5_7b | three_hidden | 90.62 / 70.79 | 86.22 / 63.77 | 90.03 / 69.38 | 89.02 / 68.10 | 87.31 / 63.35 | 90.05 / 72.06 |
| llava_1_5_7b | one_hidden | 89.87 / 68.39 | 86.33 / 62.05 | 88.91 / 66.72 | 87.03 / 65.09 | 86.38 / 63.66 | 89.11 / 70.11 |
| llava_1_5_7b | xgb | 90.08 / 69.03 | 89.14 / 66.33 | 90.21 / 69.80 | 89.40 / 67.83 | 88.68 / 65.87 | 90.24 / 70.46 |
| qwen3_vl_8b | three_hidden | 89.20 / 61.92 | 77.20 / 36.93 | 90.12 / 66.51 | 89.88 / 66.38 | 88.12 / 60.57 | 92.29 / 70.04 |
| qwen3_vl_8b | one_hidden | 89.05 / 60.48 | 77.00 / 37.24 | 88.02 / 61.80 | 86.36 / 60.87 | 82.92 / 50.58 | 88.90 / 64.01 |
| qwen3_vl_8b | xgb | 88.53 / 60.98 | 87.67 / 59.19 | 90.31 / 66.15 | 91.45 / 70.44 | 88.14 / 59.48 | 91.93 / 70.77 |
| internvl_2_5_8b | three_hidden | 87.93 / 54.28 | 78.52 / 38.85 | 87.16 / 54.93 | 86.74 / 51.92 | 84.83 / 49.70 | 87.21 / 53.30 |
| internvl_2_5_8b | one_hidden | 86.35 / 52.40 | 78.72 / 37.86 | 85.77 / 56.40 | 82.75 / 48.60 | 82.15 / 44.81 | 84.40 / 50.52 |
| internvl_2_5_8b | xgb | 88.31 / 57.34 | 85.69 / 50.35 | 85.56 / 50.37 | 86.32 / 51.58 | 82.52 / 45.66 | 88.39 / 55.92 |