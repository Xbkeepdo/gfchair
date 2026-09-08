# qwen3_vl_8b Baseline 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 11846/0/3027。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed43/qwen3_vl_8b_baselines.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed44/qwen3_vl_8b_baselines.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed45/qwen3_vl_8b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8654 ± 0.0007 | 0.9064 ± 0.0011 | 0.9343 ± 0.0011 | 0.9201 ± 0.0004 | 0.8841 ± 0.0035 | 0.9725 ± 0.0009 | 0.6239 ± 0.0025 | 0.5304 ± 0.0064 | 0.5733 ± 0.0037 | 0.8841 ± 0.0035 | 0.6101 ± 0.0076 |
| MetaToken-LR | 0.8427 ± 0.0000 | 0.8615 ± 0.0000 | 0.9658 ± 0.0000 | 0.9106 ± 0.0000 | 0.8227 ± 0.0000 | 0.9548 ± 0.0000 | 0.5943 ± 0.0000 | 0.2442 ± 0.0000 | 0.3462 ± 0.0000 | 0.8227 ± 0.0000 | 0.4708 ± 0.0000 |
| MetaToken-GB | 0.8468 ± 0.0002 | 0.8726 ± 0.0000 | 0.9547 ± 0.0002 | 0.9118 ± 0.0001 | 0.8322 ± 0.0001 | 0.9581 ± 0.0000 | 0.5936 ± 0.0010 | 0.3217 ± 0.0000 | 0.4173 ± 0.0002 | 0.8322 ± 0.0001 | 0.5165 ± 0.0008 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8652 ± 0.0010 | 0.8899 ± 0.0126 | 0.9563 ± 0.0165 | 0.9217 ± 0.0011 | 0.8841 ± 0.0035 | 0.9725 ± 0.0009 | 0.6749 ± 0.0386 | 0.4218 ± 0.0830 | 0.5113 ± 0.0491 | 0.8841 ± 0.0035 | 0.6101 ± 0.0076 |
| MetaToken-LR | 0.8421 ± 0.0000 | 0.8578 ± 0.0000 | 0.9705 ± 0.0000 | 0.9107 ± 0.0000 | 0.8227 ± 0.0000 | 0.9548 ± 0.0000 | 0.6022 ± 0.0000 | 0.2171 ± 0.0000 | 0.3191 ± 0.0000 | 0.8227 ± 0.0000 | 0.4708 ± 0.0000 |
| MetaToken-GB | 0.8466 ± 0.0002 | 0.8662 ± 0.0000 | 0.9640 ± 0.0002 | 0.9125 ± 0.0001 | 0.8322 ± 0.0001 | 0.9581 ± 0.0000 | 0.6112 ± 0.0012 | 0.2752 ± 0.0000 | 0.3795 ± 0.0002 | 0.8322 ± 0.0001 | 0.5165 ± 0.0008 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9164 ± 0.0022 | 0.9366 ± 0.0026 | 0.9639 ± 0.0015 | 0.9501 ± 0.0012 | 0.9577 ± 0.0015 | 0.9907 ± 0.0003 | 0.8034 ± 0.0060 | 0.6930 ± 0.0135 | 0.7440 ± 0.0082 | 0.9577 ± 0.0015 | 0.8380 ± 0.0052 |
| MetaToken-LR | 0.8431 ± 0.0000 | 0.8614 ± 0.0000 | 0.9650 ± 0.0000 | 0.9102 ± 0.0000 | 0.8415 ± 0.0000 | 0.9600 ± 0.0000 | 0.6208 ± 0.0000 | 0.2696 ± 0.0000 | 0.3760 ± 0.0000 | 0.8415 ± 0.0000 | 0.5137 ± 0.0000 |
| MetaToken-GB | 0.8749 ± 0.0000 | 0.8888 ± 0.0000 | 0.9696 ± 0.0000 | 0.9274 ± 0.0000 | 0.8974 ± 0.0000 | 0.9748 ± 0.0000 | 0.7502 ± 0.0000 | 0.4295 ± 0.0000 | 0.5462 ± 0.0000 | 0.8974 ± 0.0000 | 0.6816 ± 0.0000 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.9203 | 0.9196 | 0.9205 |
| MetaToken-LR | 0.9106 | 0.9106 | 0.9106 |
| MetaToken-GB | 0.9118 | 0.9120 | 0.9118 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9201 ± 0.0004。
