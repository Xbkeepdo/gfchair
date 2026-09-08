# qwen2_5_vl_7b Baseline 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 6985/0/1732。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed44/qwen2_5_vl_7b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8915 | 0.9304 | 0.9496 | 0.9399 | 0.8589 | 0.9797 | 0.4834 | 0.3989 | 0.4371 | 0.8589 | 0.4240 |
| MetaToken-LR | 0.9001 | 0.9052 | 0.9923 | 0.9467 | 0.8116 | 0.9698 | 0.6471 | 0.1202 | 0.2028 | 0.8116 | 0.3921 |
| MetaToken-GB | 0.8926 | 0.9128 | 0.9729 | 0.9419 | 0.8163 | 0.9731 | 0.4815 | 0.2131 | 0.2955 | 0.8163 | 0.3842 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9007 | 0.9170 | 0.9774 | 0.9463 | 0.8589 | 0.9797 | 0.5679 | 0.2514 | 0.3485 | 0.8589 | 0.4240 |
| MetaToken-LR | 0.8995 | 0.9046 | 0.9923 | 0.9464 | 0.8116 | 0.9698 | 0.6364 | 0.1148 | 0.1944 | 0.8116 | 0.3921 |
| MetaToken-GB | 0.9024 | 0.9059 | 0.9942 | 0.9480 | 0.8163 | 0.9731 | 0.7188 | 0.1257 | 0.2140 | 0.8163 | 0.3842 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9563 | 0.9691 | 0.9826 | 0.9758 | 0.9744 | 0.9968 | 0.8313 | 0.7326 | 0.7788 | 0.9744 | 0.8325 |
| MetaToken-LR | 0.9024 | 0.9093 | 0.9896 | 0.9478 | 0.8400 | 0.9758 | 0.6409 | 0.1583 | 0.2538 | 0.8400 | 0.4257 |
| MetaToken-GB | 0.9386 | 0.9438 | 0.9904 | 0.9665 | 0.9244 | 0.9890 | 0.8585 | 0.4966 | 0.6292 | 0.9244 | 0.7321 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-LR**：0.9467。
