# shikra_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 13361/0/3343。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/shikra_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed43/shikra_7b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/shikra_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed44/shikra_7b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/shikra_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed45/shikra_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8218 ± 0.0065 | 0.8787 ± 0.0068 | 0.8936 ± 0.0048 | 0.8861 ± 0.0039 | 0.8560 ± 0.0051 | 0.9499 ± 0.0007 | 0.6098 ± 0.0141 | 0.5739 ± 0.0274 | 0.5911 ± 0.0193 | 0.8560 ± 0.0051 | 0.5957 ± 0.0140 |
| ProjectAway | 0.7753 ± 0.0003 | 0.7758 ± 0.0008 | 0.9987 ± 0.0013 | 0.8733 ± 0.0000 | 0.6903 ± 0.0040 | 0.8858 ± 0.0011 | 0.1765 ± 0.2496 | 0.0040 ± 0.0056 | 0.0078 ± 0.0110 | 0.6903 ± 0.0040 | 0.3588 ± 0.0073 |
| MetaToken-MLP | 0.8175 ± 0.0020 | 0.8281 ± 0.0055 | 0.9651 ± 0.0066 | 0.8913 ± 0.0005 | 0.8537 ± 0.0004 | 0.9496 ± 0.0001 | 0.7214 ± 0.0179 | 0.3080 ± 0.0315 | 0.4303 ± 0.0279 | 0.8537 ± 0.0004 | 0.6227 ± 0.0012 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8116 ± 0.0229 | 0.8875 ± 0.0185 | 0.8690 ± 0.0560 | 0.8765 ± 0.0206 | 0.8560 ± 0.0051 | 0.9499 ± 0.0007 | 0.5924 ± 0.0625 | 0.6138 ± 0.0913 | 0.5934 ± 0.0061 | 0.8560 ± 0.0051 | 0.5957 ± 0.0140 |
| ProjectAway | 0.7753 ± 0.0001 | 0.7753 ± 0.0000 | 0.9999 ± 0.0002 | 0.8734 ± 0.0001 | 0.6903 ± 0.0040 | 0.8858 ± 0.0011 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6903 ± 0.0040 | 0.3588 ± 0.0073 |
| MetaToken-MLP | 0.8198 ± 0.0013 | 0.8480 ± 0.0030 | 0.9353 ± 0.0031 | 0.8895 ± 0.0006 | 0.8537 ± 0.0004 | 0.9496 ± 0.0001 | 0.6537 ± 0.0035 | 0.4212 ± 0.0154 | 0.5121 ± 0.0109 | 0.8537 ± 0.0004 | 0.6227 ± 0.0012 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9515 ± 0.0280 | 0.9654 ± 0.0231 | 0.9717 ± 0.0132 | 0.9685 ± 0.0181 | 0.9851 ± 0.0116 | 0.9954 ± 0.0036 | 0.9042 ± 0.0468 | 0.8854 ± 0.0773 | 0.8943 ± 0.0621 | 0.9851 ± 0.0116 | 0.9566 ± 0.0334 |
| ProjectAway | 0.7660 ± 0.0005 | 0.7662 ± 0.0007 | 0.9991 ± 0.0008 | 0.8673 ± 0.0001 | 0.7252 ± 0.0104 | 0.9000 ± 0.0060 | 0.6739 ± 0.0756 | 0.0057 ± 0.0045 | 0.0113 ± 0.0088 | 0.7252 ± 0.0104 | 0.4045 ± 0.0107 |
| MetaToken-MLP | 0.8067 ± 0.0013 | 0.8169 ± 0.0048 | 0.9635 ± 0.0068 | 0.8841 ± 0.0002 | 0.8498 ± 0.0008 | 0.9480 ± 0.0003 | 0.7149 ± 0.0194 | 0.2953 ± 0.0275 | 0.4167 ± 0.0244 | 0.8498 ± 0.0008 | 0.6173 ± 0.0013 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.8891 | 0.8805 | 0.8886 |
| ProjectAway | 0.8733 | 0.8733 | 0.8733 |
| MetaToken-MLP | 0.8917 | 0.8907 | 0.8916 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.8913 ± 0.0005。
