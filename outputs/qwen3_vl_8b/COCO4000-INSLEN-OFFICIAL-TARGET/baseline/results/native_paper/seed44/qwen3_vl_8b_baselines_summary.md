# qwen3_vl_8b Baseline 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 11846/0/3027。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed44/qwen3_vl_8b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8646 | 0.9061 | 0.9335 | 0.9196 | 0.8825 | 0.9719 | 0.6205 | 0.5291 | 0.5711 | 0.8825 | 0.6108 |
| MetaToken-LR | 0.8427 | 0.8615 | 0.9658 | 0.9106 | 0.8227 | 0.9548 | 0.5943 | 0.2442 | 0.3462 | 0.8227 | 0.4708 |
| MetaToken-GB | 0.8470 | 0.8726 | 0.9550 | 0.9120 | 0.8323 | 0.9581 | 0.5950 | 0.3217 | 0.4176 | 0.8323 | 0.5170 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8662 | 0.8863 | 0.9622 | 0.9227 | 0.8825 | 0.9719 | 0.6844 | 0.3992 | 0.5043 | 0.8825 | 0.6108 |
| MetaToken-LR | 0.8421 | 0.8578 | 0.9705 | 0.9107 | 0.8227 | 0.9548 | 0.6022 | 0.2171 | 0.3191 | 0.8227 | 0.4708 |
| MetaToken-GB | 0.8467 | 0.8662 | 0.9642 | 0.9126 | 0.8323 | 0.9581 | 0.6121 | 0.2752 | 0.3797 | 0.8323 | 0.5170 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9194 | 0.9401 | 0.9637 | 0.9517 | 0.9598 | 0.9911 | 0.8062 | 0.7111 | 0.7557 | 0.9598 | 0.8451 |
| MetaToken-LR | 0.8431 | 0.8614 | 0.9650 | 0.9102 | 0.8415 | 0.9600 | 0.6208 | 0.2696 | 0.3760 | 0.8415 | 0.5137 |
| MetaToken-GB | 0.8749 | 0.8888 | 0.9696 | 0.9274 | 0.8974 | 0.9748 | 0.7502 | 0.4295 | 0.5462 | 0.8974 | 0.6816 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9196。
