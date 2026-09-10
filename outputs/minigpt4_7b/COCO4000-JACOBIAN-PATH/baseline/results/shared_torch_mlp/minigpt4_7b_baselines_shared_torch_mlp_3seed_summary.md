# minigpt4_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 8728/0/2229。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/minigpt4_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed43/minigpt4_7b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/minigpt4_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed44/minigpt4_7b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/minigpt4_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed45/minigpt4_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8843 ± 0.0030 | 0.9151 ± 0.0022 | 0.9495 ± 0.0012 | 0.9320 ± 0.0017 | 0.9046 ± 0.0020 | 0.9764 ± 0.0004 | 0.6845 ± 0.0097 | 0.5543 ± 0.0124 | 0.6126 ± 0.0114 | 0.9046 ± 0.0020 | 0.6568 ± 0.0076 |
| ProjectAway | 0.8376 ± 0.0023 | 0.8464 ± 0.0062 | 0.9842 ± 0.0087 | 0.9101 ± 0.0011 | 0.7787 ± 0.0013 | 0.9439 ± 0.0006 | 0.5440 ± 0.0273 | 0.0960 ± 0.0509 | 0.1572 ± 0.0782 | 0.7787 ± 0.0013 | 0.4011 ± 0.0018 |
| MetaToken-MLP | 0.8446 ± 0.0024 | 0.8618 ± 0.0040 | 0.9694 ± 0.0035 | 0.9124 ± 0.0011 | 0.8662 ± 0.0058 | 0.9689 ± 0.0018 | 0.5791 ± 0.0130 | 0.2138 ± 0.0290 | 0.3113 ± 0.0327 | 0.8662 ± 0.0058 | 0.5032 ± 0.0112 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8834 ± 0.0040 | 0.9190 ± 0.0011 | 0.9434 ± 0.0066 | 0.9310 ± 0.0027 | 0.9046 ± 0.0020 | 0.9764 ± 0.0004 | 0.6705 ± 0.0223 | 0.5797 ± 0.0092 | 0.6215 ± 0.0047 | 0.9046 ± 0.0020 | 0.6568 ± 0.0076 |
| ProjectAway | 0.8349 ± 0.0000 | 0.8349 ± 0.0000 | 1.0000 ± 0.0000 | 0.9100 ± 0.0000 | 0.7787 ± 0.0013 | 0.9439 ± 0.0006 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7787 ± 0.0013 | 0.4011 ± 0.0018 |
| MetaToken-MLP | 0.8446 ± 0.0028 | 0.8609 ± 0.0043 | 0.9708 ± 0.0029 | 0.9125 ± 0.0012 | 0.8662 ± 0.0058 | 0.9689 ± 0.0018 | 0.5814 ± 0.0152 | 0.2065 ± 0.0308 | 0.3038 ± 0.0362 | 0.8662 ± 0.0058 | 0.5032 ± 0.0112 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9801 ± 0.0092 | 0.9849 ± 0.0082 | 0.9918 ± 0.0030 | 0.9883 ± 0.0053 | 0.9963 ± 0.0023 | 0.9993 ± 0.0004 | 0.9513 ± 0.0184 | 0.9134 ± 0.0474 | 0.9316 ± 0.0319 | 0.9963 ± 0.0023 | 0.9805 ± 0.0118 |
| ProjectAway | 0.8557 ± 0.0016 | 0.8629 ± 0.0054 | 0.9871 ± 0.0064 | 0.9208 ± 0.0004 | 0.8040 ± 0.0017 | 0.9572 ± 0.0005 | 0.6170 ± 0.0361 | 0.1101 ± 0.0466 | 0.1812 ± 0.0694 | 0.8040 ± 0.0017 | 0.4247 ± 0.0024 |
| MetaToken-MLP | 0.8600 ± 0.0009 | 0.8739 ± 0.0034 | 0.9762 ± 0.0052 | 0.9222 ± 0.0007 | 0.8569 ± 0.0063 | 0.9705 ± 0.0017 | 0.6006 ± 0.0188 | 0.2006 ± 0.0289 | 0.2990 ± 0.0308 | 0.8569 ± 0.0063 | 0.4824 ± 0.0134 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.9314 | 0.9343 | 0.9302 |
| ProjectAway | 0.9088 | 0.9099 | 0.9115 |
| MetaToken-MLP | 0.9140 | 0.9115 | 0.9118 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9320 ± 0.0017。
