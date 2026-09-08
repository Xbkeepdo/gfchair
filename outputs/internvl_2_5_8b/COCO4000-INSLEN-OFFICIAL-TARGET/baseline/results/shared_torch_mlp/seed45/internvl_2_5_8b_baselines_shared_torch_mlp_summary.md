# internvl_2_5_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 9378/0/2381。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/internvl_2_5_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8601 | 0.8990 | 0.9397 | 0.9189 | 0.8543 | 0.9665 | 0.5709 | 0.4316 | 0.4916 | 0.8543 | 0.5108 |
| MetaToken-MLP | 0.8412 | 0.8430 | 0.9975 | 0.9138 | 0.7859 | 0.9494 | 0.0000 | 0.0000 | 0.0000 | 0.7859 | 0.3757 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8580 | 0.8928 | 0.9452 | 0.9182 | 0.8543 | 0.9665 | 0.5686 | 0.3887 | 0.4618 | 0.8543 | 0.5108 |
| MetaToken-MLP | 0.8433 | 0.8433 | 1.0000 | 0.9150 | 0.7859 | 0.9494 | 0.0000 | 0.0000 | 0.0000 | 0.7859 | 0.3757 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9896 | 0.9949 | 0.9929 | 0.9939 | 0.9993 | 0.9999 | 0.9595 | 0.9706 | 0.9650 | 0.9993 | 0.9959 |
| MetaToken-MLP | 0.8530 | 0.8537 | 0.9984 | 0.9204 | 0.8096 | 0.9600 | 0.6750 | 0.0194 | 0.0377 | 0.8096 | 0.3773 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9189。
