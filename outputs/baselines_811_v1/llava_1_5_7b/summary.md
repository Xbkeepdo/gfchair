# SVAR/MetaToken：相同811划分与分类器对照

逐模型严格复用区域AE的3200训练/400验证/400测试图片、全部mentions、原标签及seeds43/44/45。SVAR为controlled样本、零基索引[5,19)层各head视觉attention mass；MetaToken为原10+H维特征。
三层[128,64,32]/BN/dropout.3，验证loss选择checkpoint/调度/早停；sklearn单层12候选、XGB18候选按验证AUROC/AP选参。无额外标准化。这是统一分类器的特征对照，不是SVAR原生248隐藏单层或MetaToken原生Scaler+LR/GB。
MetaToken保留完整回答长度和对象span统计，信息范围比严格pre-target信号更宽。原800图已有研究使用，本轮为探索性评估；无新bootstrap。

| 模型 | 特征 | 分类器 | AUROC均值±std | HALL-AUPR均值±std |
|---|---|---|---:|---:|
| llava_1_5_7b | svar | three_hidden | 90.62 ± 0.19 | 70.79 ± 0.36 |
| llava_1_5_7b | svar | one_hidden | 89.87 ± 0.13 | 68.39 ± 0.12 |
| llava_1_5_7b | svar | xgb | 90.08 ± 0.00 | 69.03 ± 0.00 |
| llava_1_5_7b | metatoken | three_hidden | 86.22 ± 0.35 | 63.77 ± 0.63 |
| llava_1_5_7b | metatoken | one_hidden | 86.33 ± 0.09 | 62.05 ± 0.37 |
| llava_1_5_7b | metatoken | xgb | 89.14 ± 0.00 | 66.33 ± 0.00 |

逐seed及概率ensemble见CSV/保存的result.pt；ae_vs_baselines.csv中的差值为AE组减baseline，始终固定相同分类器及400测试图。正值表示AE组更高，不据此宣称显著性。
