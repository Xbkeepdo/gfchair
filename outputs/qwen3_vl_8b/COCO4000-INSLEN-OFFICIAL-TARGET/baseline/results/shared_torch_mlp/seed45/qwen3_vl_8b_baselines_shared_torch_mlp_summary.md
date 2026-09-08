# qwen3_vl_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 11846/0/3027。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/qwen3_vl_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8722 | 0.9085 | 0.9407 | 0.9243 | 0.8741 | 0.9668 | 0.6511 | 0.5388 | 0.5896 | 0.8741 | 0.5810 |
| MetaToken-MLP | 0.8342 | 0.8356 | 0.9960 | 0.9088 | 0.7725 | 0.9396 | 0.7059 | 0.0465 | 0.0873 | 0.7725 | 0.4059 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8715 | 0.9087 | 0.9395 | 0.9238 | 0.8741 | 0.9668 | 0.6473 | 0.5407 | 0.5892 | 0.8741 | 0.5810 |
| MetaToken-MLP | 0.8325 | 0.8327 | 0.9988 | 0.9082 | 0.7725 | 0.9396 | 0.8000 | 0.0233 | 0.0452 | 0.7725 | 0.4059 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9813 | 0.9879 | 0.9895 | 0.9887 | 0.9974 | 0.9995 | 0.9500 | 0.9432 | 0.9466 | 0.9974 | 0.9883 |
| MetaToken-MLP | 0.8285 | 0.8308 | 0.9947 | 0.9054 | 0.7938 | 0.9471 | 0.6533 | 0.0472 | 0.0880 | 0.7938 | 0.4168 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9243。
