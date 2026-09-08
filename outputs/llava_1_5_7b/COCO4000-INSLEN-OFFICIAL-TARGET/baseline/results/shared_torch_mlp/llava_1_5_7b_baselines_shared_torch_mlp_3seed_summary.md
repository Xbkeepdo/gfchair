# llava_1_5_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 12317/0/3146。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43/llava_1_5_7b_baselines_shared_torch_mlp.json`、`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/llava_1_5_7b_baselines_shared_torch_mlp.json`、`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/llava_1_5_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8375 ± 0.0052 | 0.8851 ± 0.0090 | 0.9082 ± 0.0083 | 0.8965 ± 0.0030 | 0.8824 ± 0.0024 | 0.9617 ± 0.0001 | 0.6536 ± 0.0127 | 0.5943 ± 0.0387 | 0.6217 ± 0.0208 | 0.8824 ± 0.0024 | 0.6530 ± 0.0157 |
| MetaToken-MLP | 0.8400 ± 0.0050 | 0.8555 ± 0.0071 | 0.9549 ± 0.0040 | 0.9024 ± 0.0024 | 0.8700 ± 0.0005 | 0.9539 ± 0.0002 | 0.7417 ± 0.0057 | 0.4452 ± 0.0342 | 0.5556 ± 0.0268 | 0.8700 ± 0.0005 | 0.6763 ± 0.0023 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8362 ± 0.0035 | 0.8874 ± 0.0139 | 0.9037 ± 0.0150 | 0.8953 ± 0.0019 | 0.8824 ± 0.0024 | 0.9617 ± 0.0001 | 0.6475 ± 0.0148 | 0.6041 ± 0.0600 | 0.6230 ± 0.0258 | 0.8824 ± 0.0024 | 0.6530 ± 0.0157 |
| MetaToken-MLP | 0.8423 ± 0.0003 | 0.8682 ± 0.0018 | 0.9390 ± 0.0022 | 0.9022 ± 0.0001 | 0.8700 ± 0.0005 | 0.9539 ± 0.0002 | 0.7088 ± 0.0041 | 0.5101 ± 0.0087 | 0.5932 ± 0.0045 | 0.8700 ± 0.0005 | 0.6763 ± 0.0023 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9618 ± 0.0060 | 0.9724 ± 0.0066 | 0.9785 ± 0.0010 | 0.9755 ± 0.0038 | 0.9926 ± 0.0022 | 0.9979 ± 0.0006 | 0.9245 ± 0.0051 | 0.9046 ± 0.0232 | 0.9143 ± 0.0143 | 0.9926 ± 0.0022 | 0.9745 ± 0.0073 |
| MetaToken-MLP | 0.8313 ± 0.0009 | 0.8489 ± 0.0043 | 0.9515 ± 0.0057 | 0.8973 ± 0.0003 | 0.8689 ± 0.0009 | 0.9559 ± 0.0003 | 0.7164 ± 0.0130 | 0.4184 ± 0.0231 | 0.5276 ± 0.0150 | 0.8689 ± 0.0009 | 0.6560 ± 0.0014 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.8995 | 0.8924 | 0.8975 |
| MetaToken-MLP | 0.9044 | 0.8990 | 0.9038 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9024 ± 0.0024。
