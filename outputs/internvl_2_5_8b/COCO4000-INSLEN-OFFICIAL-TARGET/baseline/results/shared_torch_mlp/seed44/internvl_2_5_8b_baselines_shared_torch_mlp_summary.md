# internvl_2_5_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 9378/0/2381。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/internvl_2_5_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8559 | 0.8959 | 0.9382 | 0.9166 | 0.8569 | 0.9670 | 0.5540 | 0.4129 | 0.4731 | 0.8569 | 0.5102 |
| MetaToken-MLP | 0.8446 | 0.8444 | 1.0000 | 0.9156 | 0.7853 | 0.9490 | 1.0000 | 0.0080 | 0.0160 | 0.7853 | 0.3856 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8551 | 0.8950 | 0.9382 | 0.9161 | 0.8569 | 0.9670 | 0.5507 | 0.4075 | 0.4684 | 0.8569 | 0.5102 |
| MetaToken-MLP | 0.8433 | 0.8433 | 1.0000 | 0.9150 | 0.7853 | 0.9490 | 0.0000 | 0.0000 | 0.0000 | 0.7853 | 0.3856 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9801 | 0.9840 | 0.9927 | 0.9883 | 0.9971 | 0.9995 | 0.9561 | 0.9074 | 0.9311 | 0.9971 | 0.9848 |
| MetaToken-MLP | 0.8523 | 0.8526 | 0.9994 | 0.9201 | 0.8081 | 0.9597 | 0.7222 | 0.0093 | 0.0184 | 0.8081 | 0.3767 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9166。
