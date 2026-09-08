# llava_1_5_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 12317/0/3146。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43_ep200/llava_1_5_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8385 | 0.8916 | 0.9011 | 0.8963 | 0.8799 | 0.9585 | 0.6471 | 0.6234 | 0.6351 | 0.8799 | 0.6635 |
| MetaToken-MLP | 0.8446 | 0.8634 | 0.9495 | 0.9044 | 0.8705 | 0.9540 | 0.7361 | 0.4838 | 0.5838 | 0.8705 | 0.6788 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8328 | 0.8973 | 0.8855 | 0.8914 | 0.8799 | 0.9585 | 0.6235 | 0.6516 | 0.6372 | 0.8799 | 0.6635 |
| MetaToken-MLP | 0.8423 | 0.8692 | 0.9376 | 0.9021 | 0.8705 | 0.9540 | 0.7060 | 0.5148 | 0.5954 | 0.8705 | 0.6788 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9963 | 0.9983 | 0.9969 | 0.9976 | 0.9999 | 1.0000 | 0.9893 | 0.9942 | 0.9917 | 0.9999 | 0.9998 |
| MetaToken-MLP | 0.8326 | 0.8543 | 0.9450 | 0.8974 | 0.8685 | 0.9558 | 0.7027 | 0.4467 | 0.5462 | 0.8685 | 0.6556 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9044。
