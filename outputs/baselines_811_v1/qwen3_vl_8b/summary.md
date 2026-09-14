# SVAR/MetaToken：相同811划分与分类器对照

逐模型严格复用区域AE的3200训练/400验证/400测试图片、全部mentions、原标签及seeds43/44/45。SVAR为controlled样本、零基索引[5,19)层各head视觉attention mass；MetaToken为原10+H维特征。
三层[128,64,32]/BN/dropout.3，验证loss选择checkpoint/调度/早停；sklearn单层12候选、XGB18候选按验证AUROC/AP选参。无额外标准化。这是统一分类器的特征对照，不是SVAR原生248隐藏单层或MetaToken原生Scaler+LR/GB。
MetaToken保留完整回答长度和对象span统计，信息范围比严格pre-target信号更宽。原800图已有研究使用，本轮为探索性评估；无新bootstrap。

| 模型 | 特征 | 分类器 | AUROC均值±std | HALL-AUPR均值±std |
|---|---|---|---:|---:|
| qwen3_vl_8b | svar | three_hidden | 89.20 ± 0.23 | 61.92 ± 0.78 |
| qwen3_vl_8b | svar | one_hidden | 89.05 ± 0.11 | 60.48 ± 0.35 |
| qwen3_vl_8b | svar | xgb | 88.53 ± 0.00 | 60.98 ± 0.00 |
| qwen3_vl_8b | metatoken | three_hidden | 77.20 ± 0.35 | 36.93 ± 0.20 |
| qwen3_vl_8b | metatoken | one_hidden | 77.00 ± 0.14 | 37.24 ± 0.36 |
| qwen3_vl_8b | metatoken | xgb | 87.67 ± 0.00 | 59.19 ± 0.00 |

逐seed及概率ensemble见CSV/保存的result.pt；ae_vs_baselines.csv中的差值为AE组减baseline，始终固定相同分类器及400测试图。正值表示AE组更高，不据此宣称显著性。

## 同分类器下与AE拼接直接比较

每格AUROC / HALL-AUPR（%，三seed均值）；V、VP、G均指区域AE＋对应All-attention gross的log1p。

| 模型 | 分类器 | SVAR | MetaToken | V | VP | G | VP+G |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_vl_8b | three_hidden | 89.20 / 61.92 | 77.20 / 36.93 | 90.12 / 66.51 | 89.88 / 66.38 | 88.12 / 60.57 | 92.29 / 70.04 |
| qwen3_vl_8b | one_hidden | 89.05 / 60.48 | 77.00 / 37.24 | 88.02 / 61.80 | 86.36 / 60.87 | 82.92 / 50.58 | 88.90 / 64.01 |
| qwen3_vl_8b | xgb | 88.53 / 60.98 | 87.67 / 59.19 | 90.31 / 66.15 | 91.45 / 70.44 | 88.14 / 59.48 | 91.93 / 70.77 |