# qwen2_5_vl_7b Baseline 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 6985/0/1732。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed43/qwen2_5_vl_7b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8990 | 0.9288 | 0.9606 | 0.9445 | 0.8620 | 0.9803 | 0.5308 | 0.3770 | 0.4409 | 0.8620 | 0.4347 |
| MetaToken-LR | 0.9001 | 0.9052 | 0.9923 | 0.9467 | 0.8116 | 0.9698 | 0.6471 | 0.1202 | 0.2028 | 0.8116 | 0.3921 |
| MetaToken-GB | 0.8909 | 0.9141 | 0.9690 | 0.9408 | 0.8171 | 0.9729 | 0.4667 | 0.2295 | 0.3077 | 0.8171 | 0.3648 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8978 | 0.9353 | 0.9516 | 0.9434 | 0.8620 | 0.9803 | 0.5192 | 0.4426 | 0.4779 | 0.8620 | 0.4347 |
| MetaToken-LR | 0.8995 | 0.9046 | 0.9923 | 0.9464 | 0.8116 | 0.9698 | 0.6364 | 0.1148 | 0.1944 | 0.8116 | 0.3921 |
| MetaToken-GB | 0.8984 | 0.9045 | 0.9910 | 0.9458 | 0.8171 | 0.9729 | 0.6000 | 0.1148 | 0.1927 | 0.8171 | 0.3648 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9535 | 0.9617 | 0.9874 | 0.9744 | 0.9735 | 0.9967 | 0.8604 | 0.6644 | 0.7498 | 0.9735 | 0.8507 |
| MetaToken-LR | 0.9024 | 0.9093 | 0.9896 | 0.9478 | 0.8400 | 0.9758 | 0.6409 | 0.1583 | 0.2538 | 0.8400 | 0.4257 |
| MetaToken-GB | 0.9386 | 0.9465 | 0.9872 | 0.9664 | 0.9249 | 0.9891 | 0.8276 | 0.5239 | 0.6416 | 0.9249 | 0.7377 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-LR**：0.9467。
