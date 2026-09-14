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

逐seed及概率ensemble见CSV/保存的result.pt；ae_vs_baselines.csv中的差值为AE组减baseline，始终固定相同分类器及400测试图。正值表示AE组更高，不据此宣称显著性。
