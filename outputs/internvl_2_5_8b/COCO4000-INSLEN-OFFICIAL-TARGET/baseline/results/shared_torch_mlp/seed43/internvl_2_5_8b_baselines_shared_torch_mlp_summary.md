# internvl_2_5_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 9378/0/2381。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43/internvl_2_5_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8551 | 0.9019 | 0.9293 | 0.9154 | 0.8472 | 0.9640 | 0.5449 | 0.4558 | 0.4964 | 0.8472 | 0.4980 |
| MetaToken-MLP | 0.8446 | 0.8444 | 1.0000 | 0.9156 | 0.7831 | 0.9490 | 1.0000 | 0.0080 | 0.0160 | 0.7831 | 0.3693 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8551 | 0.9027 | 0.9283 | 0.9153 | 0.8472 | 0.9640 | 0.5443 | 0.4611 | 0.4993 | 0.8472 | 0.4980 |
| MetaToken-MLP | 0.8433 | 0.8433 | 1.0000 | 0.9150 | 0.7831 | 0.9490 | 0.0000 | 0.0000 | 0.0000 | 0.7831 | 0.3693 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9816 | 0.9882 | 0.9901 | 0.9892 | 0.9974 | 0.9995 | 0.9427 | 0.9325 | 0.9376 | 0.9974 | 0.9863 |
| MetaToken-MLP | 0.8521 | 0.8521 | 0.9999 | 0.9201 | 0.8071 | 0.9596 | 0.8750 | 0.0050 | 0.0100 | 0.8071 | 0.3750 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9156。
