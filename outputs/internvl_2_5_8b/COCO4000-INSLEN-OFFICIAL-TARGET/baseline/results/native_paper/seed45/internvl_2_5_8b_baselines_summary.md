# internvl_2_5_8b Baseline 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 9378/0/2381。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed45/internvl_2_5_8b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8627 | 0.9035 | 0.9373 | 0.9201 | 0.8760 | 0.9738 | 0.5772 | 0.4611 | 0.5127 | 0.8760 | 0.5485 |
| MetaToken-LR | 0.8555 | 0.8668 | 0.9791 | 0.9196 | 0.8349 | 0.9617 | 0.6283 | 0.1903 | 0.2922 | 0.8349 | 0.4867 |
| MetaToken-GB | 0.8627 | 0.8836 | 0.9641 | 0.9221 | 0.8533 | 0.9656 | 0.6211 | 0.3164 | 0.4192 | 0.8533 | 0.5287 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8564 | 0.9136 | 0.9163 | 0.9150 | 0.8760 | 0.9738 | 0.5422 | 0.5335 | 0.5378 | 0.8760 | 0.5485 |
| MetaToken-LR | 0.8564 | 0.8689 | 0.9771 | 0.9198 | 0.8349 | 0.9617 | 0.6260 | 0.2064 | 0.3105 | 0.8349 | 0.4867 |
| MetaToken-GB | 0.8610 | 0.8772 | 0.9711 | 0.9218 | 0.8533 | 0.9656 | 0.6329 | 0.2681 | 0.3766 | 0.8533 | 0.5287 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9379 | 0.9508 | 0.9777 | 0.9641 | 0.9668 | 0.9938 | 0.8475 | 0.7100 | 0.7727 | 0.9668 | 0.8665 |
| MetaToken-LR | 0.8635 | 0.8755 | 0.9790 | 0.9243 | 0.8442 | 0.9658 | 0.6258 | 0.2017 | 0.3051 | 0.8442 | 0.4798 |
| MetaToken-GB | 0.8965 | 0.9105 | 0.9742 | 0.9413 | 0.9110 | 0.9823 | 0.7530 | 0.4508 | 0.5640 | 0.9110 | 0.6920 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-GB**：0.9221。
