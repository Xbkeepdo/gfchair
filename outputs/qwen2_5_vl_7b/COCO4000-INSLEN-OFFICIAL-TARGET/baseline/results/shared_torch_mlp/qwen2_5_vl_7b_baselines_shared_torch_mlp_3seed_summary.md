# qwen2_5_vl_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 6985/0/1732。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43/qwen2_5_vl_7b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/qwen2_5_vl_7b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/qwen2_5_vl_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8926 ± 0.0033 | 0.9255 ± 0.0046 | 0.9570 ± 0.0019 | 0.9410 ± 0.0016 | 0.8390 ± 0.0085 | 0.9730 ± 0.0019 | 0.4865 ± 0.0245 | 0.3479 ± 0.0451 | 0.4050 ± 0.0396 | 0.8390 ± 0.0085 | 0.3960 ± 0.0095 |
| MetaToken-MLP | 0.8936 ± 0.0007 | 0.8943 ± 0.0001 | 0.9991 ± 0.0008 | 0.9438 ± 0.0004 | 0.7539 ± 0.0028 | 0.9601 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7539 ± 0.0028 | 0.2326 ± 0.0070 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8909 ± 0.0064 | 0.9284 ± 0.0073 | 0.9516 ± 0.0139 | 0.9397 ± 0.0040 | 0.8390 ± 0.0085 | 0.9730 ± 0.0019 | 0.4865 ± 0.0422 | 0.3770 ± 0.0762 | 0.4180 ± 0.0433 | 0.8390 ± 0.0085 | 0.3960 ± 0.0095 |
| MetaToken-MLP | 0.8943 ± 0.0000 | 0.8943 ± 0.0000 | 1.0000 ± 0.0000 | 0.9442 ± 0.0000 | 0.7539 ± 0.0028 | 0.9601 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7539 ± 0.0028 | 0.2326 ± 0.0070 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9953 ± 0.0018 | 0.9975 ± 0.0009 | 0.9972 ± 0.0017 | 0.9974 ± 0.0010 | 0.9996 ± 0.0003 | 1.0000 ± 0.0000 | 0.9766 ± 0.0138 | 0.9786 ± 0.0078 | 0.9775 ± 0.0083 | 0.9996 ± 0.0003 | 0.9966 ± 0.0022 |
| MetaToken-MLP | 0.8952 ± 0.0001 | 0.8952 ± 0.0002 | 0.9999 ± 0.0001 | 0.9447 ± 0.0001 | 0.7858 ± 0.0016 | 0.9681 ± 0.0003 | 0.5833 ± 0.4249 | 0.0018 ± 0.0017 | 0.0036 ± 0.0034 | 0.7858 ± 0.0016 | 0.2650 ± 0.0028 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.9417 | 0.9424 | 0.9387 |
| MetaToken-MLP | 0.9439 | 0.9442 | 0.9433 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9438 ± 0.0004。
