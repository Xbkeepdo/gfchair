# minigpt4_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 8728/0/2229。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/minigpt4_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed43/minigpt4_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8834 | 0.9146 | 0.9490 | 0.9314 | 0.9074 | 0.9764 | 0.6812 | 0.5516 | 0.6096 | 0.9074 | 0.6668 |
| ProjectAway | 0.8367 | 0.8512 | 0.9747 | 0.9088 | 0.7788 | 0.9443 | 0.5204 | 0.1386 | 0.2189 | 0.7788 | 0.3996 |
| MetaToken-MLP | 0.8479 | 0.8659 | 0.9678 | 0.9140 | 0.8729 | 0.9712 | 0.5973 | 0.2418 | 0.3443 | 0.8729 | 0.5170 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8834 | 0.9185 | 0.9441 | 0.9311 | 0.9074 | 0.9764 | 0.6709 | 0.5761 | 0.6199 | 0.9074 | 0.6668 |
| ProjectAway | 0.8349 | 0.8349 | 1.0000 | 0.9100 | 0.7788 | 0.9443 | 0.0000 | 0.0000 | 0.0000 | 0.7788 | 0.3996 |
| MetaToken-MLP | 0.8470 | 0.8636 | 0.9699 | 0.9137 | 0.8729 | 0.9712 | 0.5971 | 0.2255 | 0.3274 | 0.8729 | 0.5170 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9734 | 0.9809 | 0.9880 | 0.9844 | 0.9946 | 0.9991 | 0.9290 | 0.8907 | 0.9094 | 0.9946 | 0.9709 |
| ProjectAway | 0.8563 | 0.8671 | 0.9814 | 0.9207 | 0.8060 | 0.9578 | 0.5818 | 0.1468 | 0.2344 | 0.8060 | 0.4266 |
| MetaToken-MLP | 0.8613 | 0.8746 | 0.9768 | 0.9229 | 0.8641 | 0.9723 | 0.6100 | 0.2057 | 0.3076 | 0.8641 | 0.4970 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9314。
