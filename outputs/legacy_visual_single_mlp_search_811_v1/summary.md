# Visual-only 条件路径：Torch 单层 MLP 搜参（811）

固定3200/400/400图片划分及全部mentions。特征为旧Visual-only条件路径 `[AE_V, log1p(S_E)]`；复用缓存，不重新提取。24候选先以seed43验证集排序，前三名补seeds44/45，再以三seed验证AUROC/HALL-AUPR冻结配置；四模型全部冻结后才读取测试集。

表中均为三个训练seed的测试均值 ± 总体标准差，单位%。XGB为同一811实验中已有的18候选验证搜参结果。

| 模型 | Torch单层搜参 AUROC/AP | 早期三层MLP | 早期sklearn单层 | 搜参XGB | 原生SVAR |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 87.95±0.09 / 45.73±0.80 | 85.69±0.77 / 37.21±1.07 | 82.69±0.52 / 33.01±1.53 | 85.72±0.00 / 38.84±0.00 | 86.53±0.39 / 43.77±0.99 |
| LLaVA-1.5-7B | 89.74±0.02 / 68.37±0.26 | 90.09±0.14 / 69.52±0.12 | 88.76±0.46 / 66.22±0.84 | 90.24±0.00 / 69.13±0.00 | 90.40±0.01 / 71.15±0.36 |
| Qwen3-VL-8B | 89.81±0.08 / 68.45±0.54 | 90.12±0.14 / 66.95±0.64 | 87.91±0.39 / 62.01±1.85 | 89.36±0.00 / 63.89±0.00 | 89.31±0.08 / 63.66±0.82 |
| InternVL2.5-8B | 86.69±0.25 / 53.70±0.67 | 86.73±0.51 / 54.51±1.07 | 85.94±0.26 / 56.28±0.08 | 85.62±0.00 / 51.42±0.00 | 87.65±0.26 / 54.94±0.81 |

## 验证集选出的参数

- Qwen2.5-VL-7B：candidate 4，`{"activation": "relu", "batch_norm": true, "batch_size": 256, "dropout": 0.1, "learning_rate": 0.001, "lr_patience": 6, "max_epochs": 150, "monitor": "val_loss", "patience": 20, "standardize": false, "weight_decay": 1e-05, "width": 512}`
- LLaVA-1.5-7B：candidate 4，`{"activation": "relu", "batch_norm": true, "batch_size": 256, "dropout": 0.1, "learning_rate": 0.001, "lr_patience": 6, "max_epochs": 150, "monitor": "val_loss", "patience": 20, "standardize": false, "weight_decay": 1e-05, "width": 512}`
- Qwen3-VL-8B：candidate 9，`{"activation": "relu", "batch_norm": true, "batch_size": 256, "dropout": 0.3, "learning_rate": 0.001, "lr_patience": 6, "max_epochs": 150, "monitor": "val_loss", "patience": 20, "standardize": true, "weight_decay": 1e-05, "width": 128}`
- InternVL2.5-8B：candidate 19，`{"activation": "relu", "batch_norm": false, "batch_size": 256, "dropout": 0.3, "learning_rate": 0.003, "lr_patience": 6, "max_epochs": 150, "monitor": "val_auroc", "patience": 20, "standardize": true, "weight_decay": 0.0001, "width": 128}`

本实验沿用已经多次访问过的固定测试集，属于探索性比较。不同分类器的搜参预算也不同，因此只能比较当前完整管线，不能把差值完全归因于特征或分类器。

完整逐seed指标见 `outputs/legacy_visual_single_mlp_search_811_v1/*/seed_metrics.csv`。
