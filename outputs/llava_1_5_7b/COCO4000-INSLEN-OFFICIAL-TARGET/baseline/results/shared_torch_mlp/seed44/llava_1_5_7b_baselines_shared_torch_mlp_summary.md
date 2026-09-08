# llava_1_5_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 12317/0/3146。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/llava_1_5_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8312 | 0.8818 | 0.9032 | 0.8924 | 0.8795 | 0.9616 | 0.6369 | 0.5839 | 0.6093 | 0.8795 | 0.6310 |
| MetaToken-MLP | 0.8331 | 0.8461 | 0.9590 | 0.8990 | 0.8703 | 0.9540 | 0.7396 | 0.4006 | 0.5197 | 0.8703 | 0.6769 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8315 | 0.8791 | 0.9073 | 0.8930 | 0.8795 | 0.9616 | 0.6418 | 0.5712 | 0.6045 | 0.8795 | 0.6310 |
| MetaToken-MLP | 0.8427 | 0.8698 | 0.9372 | 0.9022 | 0.8703 | 0.9540 | 0.7058 | 0.5176 | 0.5972 | 0.8703 | 0.6769 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9613 | 0.9715 | 0.9787 | 0.9751 | 0.9931 | 0.9981 | 0.9250 | 0.9014 | 0.9130 | 0.9931 | 0.9763 |
| MetaToken-MLP | 0.8306 | 0.8437 | 0.9588 | 0.8976 | 0.8702 | 0.9563 | 0.7339 | 0.3902 | 0.5095 | 0.8702 | 0.6579 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.8990。
