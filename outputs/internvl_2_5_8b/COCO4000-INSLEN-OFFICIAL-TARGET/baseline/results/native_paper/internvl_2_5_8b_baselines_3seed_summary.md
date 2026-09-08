# internvl_2_5_8b Baseline 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 9378/0/2381。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed43/internvl_2_5_8b_baselines.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed44/internvl_2_5_8b_baselines.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/native_paper/seed45/internvl_2_5_8b_baselines.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8606 ± 0.0016 | 0.9041 ± 0.0049 | 0.9338 ± 0.0060 | 0.9187 ± 0.0010 | 0.8758 ± 0.0044 | 0.9739 ± 0.0010 | 0.5671 ± 0.0076 | 0.4665 ± 0.0331 | 0.5111 ± 0.0185 | 0.8758 ± 0.0044 | 0.5522 ± 0.0136 |
| MetaToken-LR | 0.8555 ± 0.0000 | 0.8668 ± 0.0000 | 0.9791 ± 0.0000 | 0.9196 ± 0.0000 | 0.8349 ± 0.0000 | 0.9617 ± 0.0000 | 0.6283 ± 0.0000 | 0.1903 ± 0.0000 | 0.2922 ± 0.0000 | 0.8349 ± 0.0000 | 0.4867 ± 0.0000 |
| MetaToken-GB | 0.8628 ± 0.0002 | 0.8836 ± 0.0000 | 0.9643 ± 0.0002 | 0.9222 ± 0.0001 | 0.8534 ± 0.0001 | 0.9656 ± 0.0000 | 0.6221 ± 0.0015 | 0.3164 ± 0.0000 | 0.4194 ± 0.0004 | 0.8534 ± 0.0001 | 0.5298 ± 0.0007 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8562 ± 0.0036 | 0.9140 ± 0.0096 | 0.9160 ± 0.0163 | 0.9148 ± 0.0033 | 0.8758 ± 0.0044 | 0.9739 ± 0.0010 | 0.5445 ± 0.0187 | 0.5344 ± 0.0646 | 0.5362 ± 0.0240 | 0.8758 ± 0.0044 | 0.5522 ± 0.0136 |
| MetaToken-LR | 0.8564 ± 0.0000 | 0.8689 ± 0.0000 | 0.9771 ± 0.0000 | 0.9198 ± 0.0000 | 0.8349 ± 0.0000 | 0.9617 ± 0.0000 | 0.6260 ± 0.0000 | 0.2064 ± 0.0000 | 0.3105 ± 0.0000 | 0.8349 ± 0.0000 | 0.4867 ± 0.0000 |
| MetaToken-GB | 0.8614 ± 0.0003 | 0.8774 ± 0.0002 | 0.9714 ± 0.0002 | 0.9220 ± 0.0002 | 0.8534 ± 0.0001 | 0.9656 ± 0.0000 | 0.6364 ± 0.0026 | 0.2690 ± 0.0013 | 0.3781 ± 0.0016 | 0.8534 ± 0.0001 | 0.5298 ± 0.0007 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9403 ± 0.0045 | 0.9549 ± 0.0047 | 0.9761 ± 0.0014 | 0.9653 ± 0.0025 | 0.9701 ± 0.0038 | 0.9944 ± 0.0007 | 0.8428 ± 0.0094 | 0.7353 ± 0.0291 | 0.7852 ± 0.0189 | 0.9701 ± 0.0038 | 0.8794 ± 0.0156 |
| MetaToken-LR | 0.8635 ± 0.0000 | 0.8755 ± 0.0000 | 0.9790 ± 0.0000 | 0.9243 ± 0.0000 | 0.8442 ± 0.0000 | 0.9658 ± 0.0000 | 0.6258 ± 0.0000 | 0.2017 ± 0.0000 | 0.3051 ± 0.0000 | 0.8442 ± 0.0000 | 0.4798 ± 0.0000 |
| MetaToken-GB | 0.8965 ± 0.0000 | 0.9105 ± 0.0000 | 0.9742 ± 0.0000 | 0.9413 ± 0.0000 | 0.9110 ± 0.0000 | 0.9823 ± 0.0000 | 0.7530 ± 0.0000 | 0.4508 ± 0.0000 | 0.5640 ± 0.0000 | 0.9110 ± 0.0000 | 0.6920 ± 0.0000 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.9182 | 0.9178 | 0.9201 |
| MetaToken-LR | 0.9196 | 0.9196 | 0.9196 |
| MetaToken-GB | 0.9224 | 0.9221 | 0.9221 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-GB**：0.9222 ± 0.0001。
