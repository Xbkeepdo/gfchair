# llava_1_5_7b Baseline 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 12317/0/3146。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed43/llava_1_5_7b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8506 | 0.8990 | 0.9093 | 0.9041 | 0.8986 | 0.9679 | 0.6755 | 0.6488 | 0.6619 | 0.8986 | 0.7040 |
| MetaToken-LR | 0.8408 | 0.8583 | 0.9516 | 0.9025 | 0.8892 | 0.9643 | 0.7342 | 0.4598 | 0.5655 | 0.8892 | 0.7011 |
| MetaToken-GB | 0.8490 | 0.8841 | 0.9265 | 0.9048 | 0.8928 | 0.9660 | 0.6976 | 0.5825 | 0.6349 | 0.8928 | 0.7050 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8347 | 0.9239 | 0.8572 | 0.8893 | 0.8986 | 0.9679 | 0.6068 | 0.7574 | 0.6738 | 0.8986 | 0.7040 |
| MetaToken-LR | 0.8382 | 0.8760 | 0.9216 | 0.8982 | 0.8892 | 0.9643 | 0.6718 | 0.5515 | 0.6057 | 0.8892 | 0.7011 |
| MetaToken-GB | 0.8506 | 0.8822 | 0.9315 | 0.9062 | 0.8928 | 0.9660 | 0.7086 | 0.5726 | 0.6334 | 0.8928 | 0.7050 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8887 | 0.9197 | 0.9381 | 0.9288 | 0.9453 | 0.9840 | 0.7719 | 0.7189 | 0.7445 | 0.9453 | 0.8250 |
| MetaToken-LR | 0.8366 | 0.8540 | 0.9518 | 0.9002 | 0.8875 | 0.9642 | 0.7272 | 0.4413 | 0.5493 | 0.8875 | 0.6819 |
| MetaToken-GB | 0.8695 | 0.8970 | 0.9394 | 0.9177 | 0.9218 | 0.9754 | 0.7516 | 0.6296 | 0.6852 | 0.9218 | 0.7772 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-GB**：0.9048。
