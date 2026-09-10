# minigpt4_7b Baseline 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 8728/0/2229。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/minigpt4_7b/COCO4000-JACOBIAN-PATH/baseline/results/native_paper/seed45/minigpt4_7b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8847 | 0.9168 | 0.9479 | 0.9321 | 0.9183 | 0.9822 | 0.6820 | 0.5652 | 0.6181 | 0.9183 | 0.6838 |
| ProjectAway | 0.8358 | 0.8372 | 0.9973 | 0.9103 | 0.7397 | 0.9280 | 0.5833 | 0.0190 | 0.0368 | 0.7397 | 0.3416 |
| MetaToken-LR | 0.8636 | 0.8766 | 0.9737 | 0.9226 | 0.8882 | 0.9727 | 0.6975 | 0.3071 | 0.4264 | 0.8882 | 0.5988 |
| MetaToken-GB | 0.8591 | 0.8881 | 0.9511 | 0.9185 | 0.8909 | 0.9760 | 0.6144 | 0.3940 | 0.4801 | 0.8909 | 0.5691 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8847 | 0.9092 | 0.9575 | 0.9327 | 0.9183 | 0.9822 | 0.7063 | 0.5163 | 0.5965 | 0.9183 | 0.6838 |
| ProjectAway | 0.5935 | 0.9393 | 0.5486 | 0.6927 | 0.7397 | 0.9280 | 0.2644 | 0.8207 | 0.4000 | 0.7397 | 0.3416 |
| MetaToken-LR | 0.8668 | 0.8875 | 0.9624 | 0.9234 | 0.8882 | 0.9727 | 0.6682 | 0.3832 | 0.4870 | 0.8882 | 0.5988 |
| MetaToken-GB | 0.8596 | 0.8847 | 0.9565 | 0.9192 | 0.8909 | 0.9760 | 0.6267 | 0.3696 | 0.4650 | 0.8909 | 0.5691 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9215 | 0.9383 | 0.9716 | 0.9546 | 0.9528 | 0.9911 | 0.7981 | 0.6376 | 0.7089 | 0.9528 | 0.8043 |
| ProjectAway | 0.8512 | 0.8523 | 0.9978 | 0.9194 | 0.7406 | 0.9387 | 0.6098 | 0.0191 | 0.0371 | 0.7406 | 0.3165 |
| MetaToken-LR | 0.8697 | 0.8858 | 0.9721 | 0.9269 | 0.8770 | 0.9746 | 0.6462 | 0.2890 | 0.3994 | 0.8770 | 0.5427 |
| MetaToken-GB | 0.9044 | 0.9168 | 0.9761 | 0.9456 | 0.9230 | 0.9844 | 0.7862 | 0.4977 | 0.6096 | 0.9230 | 0.7373 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9321。
