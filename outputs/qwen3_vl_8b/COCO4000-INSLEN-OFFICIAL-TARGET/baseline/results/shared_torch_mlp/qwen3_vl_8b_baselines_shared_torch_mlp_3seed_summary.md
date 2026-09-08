# qwen3_vl_8b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43, 44, 45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 11846/0/3027。
- 表中数值为 3 个随机种子的总体均值 ± 总体标准差。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43/qwen3_vl_8b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/qwen3_vl_8b_baselines_shared_torch_mlp.json`、`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/qwen3_vl_8b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8653 ± 0.0051 | 0.9065 ± 0.0018 | 0.9340 ± 0.0059 | 0.9200 ± 0.0032 | 0.8723 ± 0.0041 | 0.9675 ± 0.0021 | 0.6239 ± 0.0211 | 0.5310 ± 0.0096 | 0.5735 ± 0.0114 | 0.8723 ± 0.0041 | 0.5744 ± 0.0068 |
| MetaToken-MLP | 0.8317 ± 0.0019 | 0.8342 ± 0.0020 | 0.9950 ± 0.0021 | 0.9075 ± 0.0009 | 0.7696 ± 0.0031 | 0.9391 ± 0.0006 | 0.5904 ± 0.0859 | 0.0375 ± 0.0156 | 0.0700 ± 0.0283 | 0.7696 ± 0.0031 | 0.3891 ± 0.0164 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8648 ± 0.0057 | 0.9061 ± 0.0029 | 0.9338 ± 0.0081 | 0.9197 ± 0.0037 | 0.8723 ± 0.0041 | 0.9675 ± 0.0021 | 0.6226 ± 0.0244 | 0.5291 ± 0.0178 | 0.5716 ± 0.0126 | 0.8723 ± 0.0041 | 0.5744 ± 0.0068 |
| MetaToken-MLP | 0.8310 ± 0.0020 | 0.8340 ± 0.0017 | 0.9942 ± 0.0033 | 0.9070 ± 0.0011 | 0.7696 ± 0.0031 | 0.9391 ± 0.0006 | 0.6081 ± 0.1465 | 0.0368 ± 0.0141 | 0.0685 ± 0.0245 | 0.7696 ± 0.0031 | 0.3891 ± 0.0164 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9819 ± 0.0006 | 0.9878 ± 0.0011 | 0.9904 ± 0.0018 | 0.9891 ± 0.0004 | 0.9979 ± 0.0003 | 0.9995 ± 0.0001 | 0.9543 ± 0.0078 | 0.9422 ± 0.0055 | 0.9482 ± 0.0013 | 0.9979 ± 0.0003 | 0.9903 ± 0.0015 |
| MetaToken-MLP | 0.8284 ± 0.0010 | 0.8304 ± 0.0020 | 0.9952 ± 0.0021 | 0.9054 ± 0.0003 | 0.7943 ± 0.0034 | 0.9472 ± 0.0008 | 0.6705 ± 0.0281 | 0.0440 ± 0.0155 | 0.0818 ± 0.0273 | 0.7943 ± 0.0034 | 0.4189 ± 0.0087 |

## 各随机种子的 Test Real F1

| 方法 | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| SVAR | 0.9165 | 0.9193 | 0.9243 |
| MetaToken-MLP | 0.9071 | 0.9066 | 0.9088 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9200 ± 0.0032。
