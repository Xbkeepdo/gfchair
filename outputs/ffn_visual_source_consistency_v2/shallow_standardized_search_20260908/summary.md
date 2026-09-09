# AE + log1p(S)：StandardScaler 与浅层 probe 两阶段搜索

原4000图按图片分为2560训练/640验证/800测试。只在2560图拟合StandardScaler。
AdamW、最多150epochs、patience15，validation AUROC最佳checkpoint；平局HALL-AUPR，再取最早epoch。
六结构固定参数对照，验证最好的两个结构做48组网格（Linear为12个不重复配置），再比较batch64/128/256。
全部最终组seeds43/44/45；冻结后才评价800图；不合并训练与验证集重训。
BN-on与scaler-off是同参数单因素控制，不用于重新选择主分类器。
旧800图已经用于研究探索，不是独立确认；保留数值FAIL例外，不做bootstrap。
旧三隐藏层reference使用3200图训练、BatchNorm、Adam及训练损失checkpoint，不能把差异仅归因于深度或标准化。

## 结构对照：ensemble AUROC / HALL-AUPR（%）

| 组 | llava_1_5_7b | internvl_2_5_8b | qwen2_5_vl_7b | qwen3_vl_8b |
|---|---:|---:|---:|---:|
| fixed_arch0 | 87.675 / 66.698 | 82.576 / 48.267 | 83.252 / 36.003 | 84.550 / 54.810 |
| fixed_arch1 | 89.713 / 70.393 | 85.338 / 51.076 | 86.650 / 43.033 | 88.951 / 65.524 |
| fixed_arch2 | 89.493 / 69.953 | 85.185 / 50.650 | 87.187 / 45.142 | 89.145 / 65.882 |
| fixed_arch3 | 90.181 / 71.992 | 85.611 / 50.621 | 86.497 / 43.715 | 89.214 / 65.458 |
| fixed_arch4 | 90.074 / 70.935 | 85.500 / 51.804 | 87.507 / 46.010 | 89.081 / 65.670 |
| fixed_arch5 | 90.158 / 70.972 | 85.685 / 50.669 | 87.182 / 43.070 | 89.434 / 66.034 |
| best_tuned | 89.534 / 70.109 | 85.712 / 52.493 | 87.644 / 47.109 | 89.227 / 65.798 |
| best_tuned_mlp | 89.534 / 70.109 | 85.712 / 52.493 | 87.644 / 47.109 | 89.227 / 65.798 |
| matched_BN_on | 89.639 / 70.316 | 86.038 / 53.373 | 88.018 / 47.282 | 89.217 / 65.866 |
| matched_scaler_off | 88.604 / 66.877 | 83.691 / 50.489 | 84.037 / 37.675 | 85.996 / 56.866 |
| old_three_hidden_direct_reference | 90.496 / 71.928 | 86.381 / 54.615 | 88.214 / 45.774 | 89.651 / 64.934 |

## 验证集选定参数

- llava_1_5_7b，64维：`{"batch_norm": false, "batch_size": 128, "dropout": 0.3, "hidden_sizes": [256], "learning_rate": 0.001, "standardize": true, "weight_decay": 0.0}`。
- internvl_2_5_8b，64维：`{"batch_norm": false, "batch_size": 128, "dropout": 0.2, "hidden_sizes": [128, 64], "learning_rate": 0.003, "standardize": true, "weight_decay": 0.0}`。
- qwen2_5_vl_7b，56维：`{"batch_norm": false, "batch_size": 128, "dropout": 0.3, "hidden_sizes": [256, 128], "learning_rate": 0.001, "standardize": true, "weight_decay": 0.0001}`。
- qwen3_vl_8b，72维：`{"batch_norm": false, "batch_size": 128, "dropout": 0.3, "hidden_sizes": [256], "learning_rate": 0.0003, "standardize": true, "weight_decay": 0.0}`。

完整逐seed、mean±std、REAL/HALL AUPR、P/R/F1和阈值见summary.json；各模型stage1/stage2文件公开所有验证候选。
