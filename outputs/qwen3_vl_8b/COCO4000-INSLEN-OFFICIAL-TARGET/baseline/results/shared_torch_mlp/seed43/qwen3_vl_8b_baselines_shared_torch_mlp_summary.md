# qwen3_vl_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 11846/0/3027。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43/qwen3_vl_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8599 | 0.9068 | 0.9263 | 0.9165 | 0.8666 | 0.9655 | 0.5996 | 0.5368 | 0.5665 | 0.8666 | 0.5651 |
| MetaToken-MLP | 0.8315 | 0.8356 | 0.9920 | 0.9071 | 0.7710 | 0.9395 | 0.5652 | 0.0504 | 0.0925 | 0.7710 | 0.3945 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8576 | 0.9075 | 0.9223 | 0.9149 | 0.8666 | 0.9655 | 0.5895 | 0.5426 | 0.5651 | 0.8666 | 0.5651 |
| MetaToken-MLP | 0.8322 | 0.8364 | 0.9916 | 0.9074 | 0.7710 | 0.9395 | 0.5800 | 0.0562 | 0.1025 | 0.7710 | 0.3945 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9818 | 0.9890 | 0.9888 | 0.9889 | 0.9979 | 0.9996 | 0.9476 | 0.9485 | 0.9480 | 0.9979 | 0.9908 |
| MetaToken-MLP | 0.8296 | 0.8326 | 0.9929 | 0.9057 | 0.7987 | 0.9482 | 0.6480 | 0.0611 | 0.1117 | 0.7987 | 0.4305 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9165。
