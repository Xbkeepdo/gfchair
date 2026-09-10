# shikra_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 13361/0/3343。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/shikra_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed44/shikra_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8127 | 0.8712 | 0.8900 | 0.8805 | 0.8489 | 0.9498 | 0.5899 | 0.5459 | 0.5671 | 0.8489 | 0.5783 |
| ProjectAway | 0.7757 | 0.7769 | 0.9969 | 0.8733 | 0.6937 | 0.8848 | 0.5294 | 0.0120 | 0.0234 | 0.6937 | 0.3683 |
| MetaToken-MLP | 0.8148 | 0.8212 | 0.9730 | 0.8907 | 0.8543 | 0.9498 | 0.7426 | 0.2690 | 0.3949 | 0.8543 | 0.6243 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.7795 | 0.9135 | 0.7905 | 0.8476 | 0.8489 | 0.9498 | 0.5064 | 0.7417 | 0.6018 | 0.8489 | 0.5783 |
| ProjectAway | 0.7751 | 0.7753 | 0.9996 | 0.8733 | 0.6937 | 0.8848 | 0.0000 | 0.0000 | 0.0000 | 0.6937 | 0.3683 |
| MetaToken-MLP | 0.8199 | 0.8501 | 0.9321 | 0.8892 | 0.8543 | 0.9498 | 0.6487 | 0.4328 | 0.5192 | 0.8543 | 0.6243 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9445 | 0.9625 | 0.9650 | 0.9638 | 0.9853 | 0.9955 | 0.8849 | 0.8775 | 0.8812 | 0.9853 | 0.9563 |
| ProjectAway | 0.7666 | 0.7672 | 0.9979 | 0.8675 | 0.7390 | 0.9079 | 0.6441 | 0.0121 | 0.0238 | 0.7390 | 0.4195 |
| MetaToken-MLP | 0.8050 | 0.8109 | 0.9718 | 0.8841 | 0.8509 | 0.9484 | 0.7394 | 0.2606 | 0.3854 | 0.8509 | 0.6190 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.8907。
