# internvl_2_5_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 9378/0/2381。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43/internvl_2_5_8b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/internvl_2_5_8b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/internvl_2_5_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8571 ± 0.0022 | 0.8989 ± 0.0025 | 0.9358 ± 0.0046 | 0.9170 ± 0.0015 | 0.8528 ± 0.0041 | 0.9658 ± 0.0013 | 0.5566 ± 0.0108 | 0.4334 ± 0.0176 | 0.4870 ± 0.0100 | 0.8528 ± 0.0041 | 0.5063 ± 0.0059 |
| MetaToken-MLP | 0.8435 ± 0.0016 | 0.8439 ± 0.0007 | 0.9992 ± 0.0012 | 0.9150 ± 0.0009 | 0.7848 ± 0.0012 | 0.9491 ± 0.0002 | 0.6667 ± 0.4714 | 0.0054 ± 0.0038 | 0.0106 ± 0.0075 | 0.7848 ± 0.0012 | 0.3768 ± 0.0067 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8561 ± 0.0014 | 0.8968 ± 0.0042 | 0.9373 ± 0.0069 | 0.9166 ± 0.0012 | 0.8528 ± 0.0041 | 0.9658 ± 0.0013 | 0.5546 ± 0.0103 | 0.4191 ± 0.0307 | 0.4765 ± 0.0163 | 0.8528 ± 0.0041 | 0.5063 ± 0.0059 |
| MetaToken-MLP | 0.8433 ± 0.0000 | 0.8433 ± 0.0000 | 1.0000 ± 0.0000 | 0.9150 ± 0.0000 | 0.7848 ± 0.0012 | 0.9491 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7848 ± 0.0012 | 0.3768 ± 0.0067 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9837 ± 0.0042 | 0.9890 ± 0.0045 | 0.9919 ± 0.0013 | 0.9905 ± 0.0024 | 0.9979 ± 0.0010 | 0.9996 ± 0.0002 | 0.9528 ± 0.0073 | 0.9368 ± 0.0260 | 0.9446 ± 0.0147 | 0.9979 ± 0.0010 | 0.9890 ± 0.0049 |
| MetaToken-MLP | 0.8525 ± 0.0004 | 0.8528 ± 0.0007 | 0.9992 ± 0.0006 | 0.9202 ± 0.0001 | 0.8083 ± 0.0010 | 0.9597 ± 0.0002 | 0.7574 ± 0.0854 | 0.0112 ± 0.0060 | 0.0220 ± 0.0116 | 0.8083 ± 0.0010 | 0.3763 ± 0.0010 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.9154 | 0.9166 | 0.9189 |
| MetaToken-MLP | 0.9156 | 0.9156 | 0.9138 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9170 ± 0.0015。
