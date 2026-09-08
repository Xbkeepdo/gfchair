# qwen2_5_vl_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 6985/0/1732。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/qwen2_5_vl_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8880 | 0.9190 | 0.9593 | 0.9387 | 0.8329 | 0.9721 | 0.4522 | 0.2842 | 0.3490 | 0.8329 | 0.3885 |
| MetaToken-MLP | 0.8926 | 0.8942 | 0.9981 | 0.9433 | 0.7511 | 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.7511 | 0.2259 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8834 | 0.9265 | 0.9445 | 0.9354 | 0.8329 | 0.9721 | 0.4379 | 0.3661 | 0.3988 | 0.8329 | 0.3885 |
| MetaToken-MLP | 0.8943 | 0.8943 | 1.0000 | 0.9442 | 0.7511 | 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.7511 | 0.2259 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9976 | 0.9986 | 0.9987 | 0.9986 | 0.9999 | 1.0000 | 0.9891 | 0.9877 | 0.9884 | 0.9999 | 0.9990 |
| MetaToken-MLP | 0.8953 | 0.8954 | 0.9998 | 0.9448 | 0.7836 | 0.9677 | 0.7500 | 0.0041 | 0.0081 | 0.7836 | 0.2610 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9433。
