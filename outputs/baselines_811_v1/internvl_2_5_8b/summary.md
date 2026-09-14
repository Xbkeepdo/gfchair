# SVAR/MetaToken：相同811划分与分类器对照

逐模型严格复用区域AE的3200训练/400验证/400测试图片、全部mentions、原标签及seeds43/44/45。SVAR为controlled样本、零基索引[5,19)层各head视觉attention mass；MetaToken为原10+H维特征。
三层[128,64,32]/BN/dropout.3，验证loss选择checkpoint/调度/早停；sklearn单层12候选、XGB18候选按验证AUROC/AP选参。无额外标准化。这是统一分类器的特征对照，不是SVAR原生248隐藏单层或MetaToken原生Scaler+LR/GB。
MetaToken保留完整回答长度和对象span统计，信息范围比严格pre-target信号更宽。原800图已有研究使用，本轮为探索性评估；无新bootstrap。

| 模型 | 特征 | 分类器 | AUROC均值±std | HALL-AUPR均值±std |
|---|---|---|---:|---:|
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
| internvl_2_5_8b | three_hidden | 87.93 / 54.28 | 78.52 / 38.85 | 87.16 / 54.93 | 86.74 / 51.92 | 84.83 / 49.70 | 87.21 / 53.30 |
| internvl_2_5_8b | one_hidden | 86.35 / 52.40 | 78.72 / 37.86 | 85.77 / 56.40 | 82.75 / 48.60 | 82.15 / 44.81 | 84.40 / 50.52 |
| internvl_2_5_8b | xgb | 88.31 / 57.34 | 85.69 / 50.35 | 85.56 / 50.37 | 86.32 / 51.58 | 82.52 / 45.66 | 88.39 / 55.92 |