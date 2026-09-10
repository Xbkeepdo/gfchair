# minigpt4_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 8728/0/2229。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/minigpt4_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed44/minigpt4_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8883 | 0.9180 | 0.9511 | 0.9343 | 0.9029 | 0.9758 | 0.6977 | 0.5707 | 0.6278 | 0.9029 | 0.6552 |
| ProjectAway | 0.8354 | 0.8377 | 0.9957 | 0.9099 | 0.7801 | 0.9444 | 0.5294 | 0.0245 | 0.0468 | 0.7801 | 0.4036 |
| MetaToken-MLP | 0.8421 | 0.8564 | 0.9742 | 0.9115 | 0.8668 | 0.9687 | 0.5714 | 0.1739 | 0.2667 | 0.8668 | 0.5028 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8883 | 0.9180 | 0.9511 | 0.9343 | 0.9029 | 0.9758 | 0.6977 | 0.5707 | 0.6278 | 0.9029 | 0.6552 |
| ProjectAway | 0.8349 | 0.8349 | 1.0000 | 0.9100 | 0.7801 | 0.9444 | 0.0000 | 0.0000 | 0.0000 | 0.7801 | 0.4036 |
| MetaToken-MLP | 0.8461 | 0.8642 | 0.9678 | 0.9131 | 0.8668 | 0.9687 | 0.5862 | 0.2310 | 0.3314 | 0.8668 | 0.5028 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9738 | 0.9774 | 0.9920 | 0.9847 | 0.9947 | 0.9991 | 0.9507 | 0.8700 | 0.9086 | 0.9947 | 0.9736 |
| ProjectAway | 0.8535 | 0.8553 | 0.9961 | 0.9204 | 0.8042 | 0.9572 | 0.6667 | 0.0443 | 0.0832 | 0.8042 | 0.4214 |
| MetaToken-MLP | 0.8594 | 0.8694 | 0.9822 | 0.9224 | 0.8580 | 0.9708 | 0.6174 | 0.1628 | 0.2577 | 0.8580 | 0.4855 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9343。
