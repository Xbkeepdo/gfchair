# shikra_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 13361/0/3343。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/shikra_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed45/shikra_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8250 | 0.8771 | 0.9005 | 0.8886 | 0.8588 | 0.9507 | 0.6217 | 0.5646 | 0.5918 | 0.8588 | 0.5964 |
| ProjectAway | 0.7751 | 0.7753 | 0.9996 | 0.8733 | 0.6847 | 0.8853 | 0.0000 | 0.0000 | 0.0000 | 0.6847 | 0.3505 |
| MetaToken-MLP | 0.8196 | 0.8347 | 0.9568 | 0.8916 | 0.8534 | 0.9497 | 0.6989 | 0.3462 | 0.4630 | 0.8534 | 0.6212 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8238 | 0.8772 | 0.8985 | 0.8877 | 0.8588 | 0.9507 | 0.6177 | 0.5659 | 0.5907 | 0.8588 | 0.5964 |
| ProjectAway | 0.7754 | 0.7754 | 1.0000 | 0.8735 | 0.6847 | 0.8853 | 0.0000 | 0.0000 | 0.0000 | 0.6847 | 0.3505 |
| MetaToken-MLP | 0.8181 | 0.8437 | 0.9394 | 0.8890 | 0.8534 | 0.9497 | 0.6565 | 0.3995 | 0.4967 | 0.8534 | 0.6212 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9213 | 0.9385 | 0.9600 | 0.9491 | 0.9709 | 0.9910 | 0.8590 | 0.7949 | 0.8257 | 0.9709 | 0.9159 |
| ProjectAway | 0.7656 | 0.7658 | 0.9994 | 0.8671 | 0.7138 | 0.8935 | 0.6000 | 0.0029 | 0.0057 | 0.7138 | 0.3958 |
| MetaToken-MLP | 0.8080 | 0.8226 | 0.9552 | 0.8839 | 0.8493 | 0.9478 | 0.6918 | 0.3279 | 0.4449 | 0.8493 | 0.6166 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.8916。
