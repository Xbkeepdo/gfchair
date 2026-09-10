# shikra_7b Baseline 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 13361/0/3343。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/shikra_7b/COCO4000-JACOBIAN-PATH/baseline/results/native_paper/seed45/shikra_7b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8349 | 0.8709 | 0.9240 | 0.8967 | 0.8806 | 0.9618 | 0.6678 | 0.5273 | 0.5893 | 0.8806 | 0.6528 |
| ProjectAway | 0.7754 | 0.7757 | 0.9992 | 0.8734 | 0.6518 | 0.8692 | 0.5000 | 0.0027 | 0.0053 | 0.6518 | 0.3272 |
| MetaToken-LR | 0.8316 | 0.8690 | 0.9217 | 0.8946 | 0.8635 | 0.9534 | 0.6582 | 0.5206 | 0.5814 | 0.8635 | 0.6482 |
| MetaToken-GB | 0.8277 | 0.8585 | 0.9313 | 0.8934 | 0.8669 | 0.9553 | 0.6648 | 0.4700 | 0.5507 | 0.8669 | 0.6410 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8292 | 0.9060 | 0.8700 | 0.8876 | 0.8806 | 0.9618 | 0.6054 | 0.6884 | 0.6442 | 0.8806 | 0.6528 |
| ProjectAway | 0.5025 | 0.8967 | 0.4051 | 0.5581 | 0.6518 | 0.8692 | 0.2901 | 0.8389 | 0.4311 | 0.6518 | 0.3272 |
| MetaToken-LR | 0.8304 | 0.8667 | 0.9232 | 0.8941 | 0.8635 | 0.9534 | 0.6581 | 0.5100 | 0.5746 | 0.8635 | 0.6482 |
| MetaToken-GB | 0.8304 | 0.8638 | 0.9275 | 0.8945 | 0.8669 | 0.9553 | 0.6643 | 0.4953 | 0.5675 | 0.8669 | 0.6410 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8483 | 0.8778 | 0.9314 | 0.9038 | 0.8987 | 0.9655 | 0.7207 | 0.5770 | 0.6409 | 0.8987 | 0.7244 |
| ProjectAway | 0.7663 | 0.7665 | 0.9991 | 0.8675 | 0.6784 | 0.8745 | 0.7097 | 0.0070 | 0.0139 | 0.6784 | 0.3535 |
| MetaToken-LR | 0.8245 | 0.8585 | 0.9228 | 0.8895 | 0.8583 | 0.9504 | 0.6668 | 0.5037 | 0.5739 | 0.8583 | 0.6376 |
| MetaToken-GB | 0.8474 | 0.8676 | 0.9447 | 0.9045 | 0.8920 | 0.9628 | 0.7462 | 0.5298 | 0.6197 | 0.8920 | 0.7266 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.8967。
