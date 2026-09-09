# AE + log1p(S)：冻结配置后3200图重训

四模型上一轮10组各3seed，共120个头；StandardScaler只在完整3200训练图mentions重新拟合。
各组轮数固定为上一轮三seed最佳验证epoch的中位数，保存末轮；不再搜索、不以800图挑epoch或结构。
以下均为3seed概率ensemble，AUROC / HALL-AUPR（%）。旧800图已被探索，非独立确认；数值FAIL保留。

| 组 | llava_1_5_7b | internvl_2_5_8b | qwen2_5_vl_7b | qwen3_vl_8b |
|---|---:|---:|---:|---:|
| fixed_arch0 | 87.806 / 66.916 | 82.597 / 48.195 | 83.395 / 36.358 | 84.726 / 54.943 |
| fixed_arch1 | 89.911 / 70.724 | 85.632 / 53.793 | 87.052 / 44.338 | 89.293 / 65.282 |
| fixed_arch2 | 89.562 / 70.384 | 85.338 / 51.855 | 87.677 / 45.712 | 89.564 / 65.462 |
| fixed_arch3 | 90.366 / 72.044 | 85.778 / 51.825 | 87.058 / 43.704 | 89.585 / 65.536 |
| fixed_arch4 | 89.958 / 71.388 | 85.832 / 52.760 | 87.517 / 45.697 | 89.785 / 67.092 |
| fixed_arch5 | 90.405 / 71.210 | 85.625 / 51.868 | 87.652 / 44.453 | 89.872 / 67.117 |
| primary | 89.744 / 70.173 | 85.928 / 52.737 | 87.647 / 46.945 | 89.719 / 65.916 |
| matched_BN_on | 89.784 / 70.553 | 85.688 / 52.100 | 88.158 / 47.536 | 89.631 / 65.810 |
| matched_scaler_off | 88.751 / 67.805 | 84.218 / 51.202 | 84.387 / 38.010 | 86.516 / 58.123 |
| previous_primary | 89.534 / 70.109 | 85.712 / 52.493 | 87.644 / 47.109 | 89.227 / 65.798 |
| old_three_hidden_direct_reference | 90.496 / 71.928 | 86.381 / 54.615 | 88.214 / 45.774 | 89.651 / 64.934 |

## 冻结配置与训练轮数

- llava_1_5_7b：主组`tuned_arch2`，`{"batch_norm": false, "batch_size": 128, "dropout": 0.3, "hidden_sizes": [256], "learning_rate": 0.001, "standardize": true, "weight_decay": 0.0}`，54epochs；三层固定组20epochs。
- internvl_2_5_8b：主组`tuned_arch3`，`{"batch_norm": false, "batch_size": 128, "dropout": 0.2, "hidden_sizes": [128, 64], "learning_rate": 0.003, "standardize": true, "weight_decay": 0.0}`，15epochs；三层固定组22epochs。
- qwen2_5_vl_7b：主组`tuned_arch4`，`{"batch_norm": false, "batch_size": 128, "dropout": 0.3, "hidden_sizes": [256, 128], "learning_rate": 0.001, "standardize": true, "weight_decay": 0.0001}`，31epochs；三层固定组13epochs。
- qwen3_vl_8b：主组`tuned_arch2`，`{"batch_norm": false, "batch_size": 128, "dropout": 0.3, "hidden_sizes": [256], "learning_rate": 0.0003, "standardize": true, "weight_decay": 0.0}`，139epochs；三层固定组23epochs。

完整10组、逐seed、均值/标准差、REAL/HALL AUPR与双阈值P/R/F1见summary.json和groups.csv。
三层只参加了上一轮第一阶段固定参数比较，没有进入前二精搜；本轮没有为三层追加精搜。
旧三层参考虽然也用3200图，但BN/Adam/train-loss checkpoint等不同；与本轮不是仅改变深度的对照。
