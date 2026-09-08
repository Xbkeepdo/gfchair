# internvl_2_5_8b Baseline 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 9378/0/2381。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed43/internvl_2_5_8b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8589 | 0.8985 | 0.9387 | 0.9182 | 0.8702 | 0.9727 | 0.5654 | 0.4290 | 0.4878 | 0.8702 | 0.5377 |
| MetaToken-LR | 0.8555 | 0.8668 | 0.9791 | 0.9196 | 0.8349 | 0.9617 | 0.6283 | 0.1903 | 0.2922 | 0.8349 | 0.4867 |
| MetaToken-GB | 0.8631 | 0.8837 | 0.9646 | 0.9224 | 0.8534 | 0.9656 | 0.6243 | 0.3164 | 0.4199 | 0.8534 | 0.5304 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8606 | 0.9025 | 0.9358 | 0.9188 | 0.8702 | 0.9727 | 0.5686 | 0.4558 | 0.5060 | 0.8702 | 0.5377 |
| MetaToken-LR | 0.8564 | 0.8689 | 0.9771 | 0.9198 | 0.8349 | 0.9617 | 0.6260 | 0.2064 | 0.3105 | 0.8349 | 0.4867 |
| MetaToken-GB | 0.8614 | 0.8772 | 0.9716 | 0.9220 | 0.8534 | 0.9656 | 0.6369 | 0.2681 | 0.3774 | 0.8534 | 0.5304 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9364 | 0.9523 | 0.9742 | 0.9631 | 0.9681 | 0.9940 | 0.8296 | 0.7200 | 0.7709 | 0.9681 | 0.8703 |
| MetaToken-LR | 0.8635 | 0.8755 | 0.9790 | 0.9243 | 0.8442 | 0.9658 | 0.6258 | 0.2017 | 0.3051 | 0.8442 | 0.4798 |
| MetaToken-GB | 0.8965 | 0.9105 | 0.9742 | 0.9413 | 0.9110 | 0.9823 | 0.7530 | 0.4508 | 0.5640 | 0.9110 | 0.6920 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-GB**：0.9224。
