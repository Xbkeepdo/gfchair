# minigpt4_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 8728/0/2229。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/minigpt4_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed45/minigpt4_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8811 | 0.9126 | 0.9484 | 0.9302 | 0.9035 | 0.9769 | 0.6746 | 0.5408 | 0.6003 | 0.9035 | 0.6484 |
| ProjectAway | 0.8407 | 0.8502 | 0.9823 | 0.9115 | 0.7771 | 0.9430 | 0.5823 | 0.1250 | 0.2058 | 0.7771 | 0.4002 |
| MetaToken-MLP | 0.8439 | 0.8632 | 0.9661 | 0.9118 | 0.8588 | 0.9668 | 0.5685 | 0.2255 | 0.3230 | 0.8588 | 0.4897 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8784 | 0.9206 | 0.9350 | 0.9278 | 0.9035 | 0.9769 | 0.6431 | 0.5924 | 0.6167 | 0.9035 | 0.6484 |
| ProjectAway | 0.8349 | 0.8349 | 1.0000 | 0.9100 | 0.7771 | 0.9430 | 0.0000 | 0.0000 | 0.0000 | 0.7771 | 0.4002 |
| MetaToken-MLP | 0.8407 | 0.8549 | 0.9747 | 0.9109 | 0.8588 | 0.9668 | 0.5607 | 0.1630 | 0.2526 | 0.8588 | 0.4897 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9930 | 0.9964 | 0.9954 | 0.9959 | 0.9995 | 0.9999 | 0.9741 | 0.9794 | 0.9767 | 0.9995 | 0.9972 |
| ProjectAway | 0.8572 | 0.8664 | 0.9838 | 0.9214 | 0.8019 | 0.9566 | 0.6026 | 0.1391 | 0.2261 | 0.8019 | 0.4261 |
| MetaToken-MLP | 0.8592 | 0.8776 | 0.9695 | 0.9213 | 0.8488 | 0.9683 | 0.5744 | 0.2332 | 0.3317 | 0.8488 | 0.4647 |

## 简要结论

- Test Real F1 最高的方法是 **SVAR**：0.9302。
