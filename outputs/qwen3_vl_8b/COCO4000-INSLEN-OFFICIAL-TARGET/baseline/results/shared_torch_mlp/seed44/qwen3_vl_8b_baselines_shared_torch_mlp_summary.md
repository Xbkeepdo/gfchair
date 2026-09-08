# qwen3_vl_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 11846/0/3027。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/qwen3_vl_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8639 | 0.9041 | 0.9351 | 0.9193 | 0.8763 | 0.9704 | 0.6209 | 0.5174 | 0.5645 | 0.8763 | 0.5773 |
| MetaToken-MLP | 0.8295 | 0.8313 | 0.9968 | 0.9066 | 0.7654 | 0.9383 | 0.5000 | 0.0155 | 0.0301 | 0.7654 | 0.3669 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8652 | 0.9021 | 0.9395 | 0.9204 | 0.8763 | 0.9704 | 0.6311 | 0.5039 | 0.5603 | 0.8763 | 0.5773 |
| MetaToken-MLP | 0.8282 | 0.8328 | 0.9920 | 0.9055 | 0.7654 | 0.9383 | 0.4444 | 0.0310 | 0.0580 | 0.7654 | 0.3669 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9827 | 0.9863 | 0.9928 | 0.9895 | 0.9982 | 0.9996 | 0.9652 | 0.9350 | 0.9499 | 0.9982 | 0.9920 |
| MetaToken-MLP | 0.8271 | 0.8278 | 0.9980 | 0.9049 | 0.7905 | 0.9463 | 0.7101 | 0.0236 | 0.0457 | 0.7905 | 0.4094 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9193。
