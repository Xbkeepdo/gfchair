# qwen2_5_vl_7b Baseline 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 6985/0/1732。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed43/qwen2_5_vl_7b_baselines.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed44/qwen2_5_vl_7b_baselines.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed45/qwen2_5_vl_7b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8961 ± 0.0033 | 0.9306 ± 0.0015 | 0.9550 ± 0.0045 | 0.9426 ± 0.0020 | 0.8598 ± 0.0015 | 0.9799 ± 0.0003 | 0.5116 ± 0.0203 | 0.3971 ± 0.0157 | 0.4467 ± 0.0109 | 0.8598 ± 0.0015 | 0.4293 ± 0.0044 |
| MetaToken-LR | 0.9001 ± 0.0000 | 0.9052 ± 0.0000 | 0.9923 ± 0.0000 | 0.9467 ± 0.0000 | 0.8116 ± 0.0000 | 0.9698 ± 0.0000 | 0.6471 ± 0.0000 | 0.1202 ± 0.0000 | 0.2028 ± 0.0000 | 0.8116 ± 0.0000 | 0.3921 ± 0.0000 |
| MetaToken-GB | 0.8920 ± 0.0008 | 0.9132 ± 0.0006 | 0.9716 ± 0.0018 | 0.9415 ± 0.0005 | 0.8167 ± 0.0003 | 0.9730 ± 0.0001 | 0.4765 ± 0.0070 | 0.2186 ± 0.0077 | 0.2995 ± 0.0058 | 0.8167 ± 0.0003 | 0.3788 ± 0.0100 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8997 ± 0.0014 | 0.9255 ± 0.0075 | 0.9658 ± 0.0107 | 0.9451 ± 0.0013 | 0.8598 ± 0.0015 | 0.9799 ± 0.0003 | 0.5459 ± 0.0201 | 0.3406 ± 0.0786 | 0.4124 ± 0.0528 | 0.8598 ± 0.0015 | 0.4293 ± 0.0044 |
| MetaToken-LR | 0.8995 ± 0.0000 | 0.9046 ± 0.0000 | 0.9923 ± 0.0000 | 0.9464 ± 0.0000 | 0.8116 ± 0.0000 | 0.9698 ± 0.0000 | 0.6364 ± 0.0000 | 0.1148 ± 0.0000 | 0.1944 ± 0.0000 | 0.8116 ± 0.0000 | 0.3921 ± 0.0000 |
| MetaToken-GB | 0.9013 ± 0.0021 | 0.9055 ± 0.0006 | 0.9933 ± 0.0017 | 0.9474 ± 0.0011 | 0.8167 ± 0.0003 | 0.9730 ± 0.0001 | 0.6869 ± 0.0622 | 0.1220 ± 0.0052 | 0.2072 ± 0.0103 | 0.8167 ± 0.0003 | 0.3788 ± 0.0100 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9539 ± 0.0018 | 0.9647 ± 0.0032 | 0.9846 ± 0.0020 | 0.9745 ± 0.0010 | 0.9724 ± 0.0022 | 0.9966 ± 0.0003 | 0.8410 ± 0.0137 | 0.6921 ± 0.0293 | 0.7588 ± 0.0142 | 0.9724 ± 0.0022 | 0.8359 ± 0.0110 |
| MetaToken-LR | 0.9024 ± 0.0000 | 0.9093 ± 0.0000 | 0.9896 ± 0.0000 | 0.9478 ± 0.0000 | 0.8400 ± 0.0000 | 0.9758 ± 0.0000 | 0.6409 ± 0.0000 | 0.1583 ± 0.0000 | 0.2538 ± 0.0000 | 0.8400 ± 0.0000 | 0.4257 ± 0.0000 |
| MetaToken-GB | 0.9386 ± 0.0000 | 0.9447 ± 0.0013 | 0.9893 ± 0.0015 | 0.9665 ± 0.0000 | 0.9246 ± 0.0002 | 0.9890 ± 0.0000 | 0.8482 ± 0.0146 | 0.5057 ± 0.0129 | 0.6333 ± 0.0058 | 0.9246 ± 0.0002 | 0.7340 ± 0.0027 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.9445 | 0.9399 | 0.9435 |
| MetaToken-LR | 0.9467 | 0.9467 | 0.9467 |
| MetaToken-GB | 0.9408 | 0.9419 | 0.9419 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-LR**：0.9467 ± 0.0000。
