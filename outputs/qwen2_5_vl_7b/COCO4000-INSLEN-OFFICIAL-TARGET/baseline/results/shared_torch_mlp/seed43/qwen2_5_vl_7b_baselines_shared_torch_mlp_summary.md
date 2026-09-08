# qwen2_5_vl_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 6985/0/1732。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed43/qwen2_5_vl_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8943 | 0.9290 | 0.9548 | 0.9417 | 0.8510 | 0.9756 | 0.5000 | 0.3825 | 0.4334 | 0.8510 | 0.4095 |
| MetaToken-MLP | 0.8938 | 0.8943 | 0.9994 | 0.9439 | 0.7531 | 0.9601 | 0.0000 | 0.0000 | 0.0000 | 0.7531 | 0.2298 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8903 | 0.9381 | 0.9393 | 0.9387 | 0.8510 | 0.9756 | 0.4807 | 0.4754 | 0.4780 | 0.8510 | 0.4095 |
| MetaToken-MLP | 0.8943 | 0.8943 | 1.0000 | 0.9442 | 0.7531 | 0.9601 | 0.0000 | 0.0000 | 0.0000 | 0.7531 | 0.2298 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9950 | 0.9963 | 0.9981 | 0.9972 | 0.9996 | 1.0000 | 0.9834 | 0.9686 | 0.9759 | 0.9996 | 0.9971 |
| MetaToken-MLP | 0.8952 | 0.8952 | 1.0000 | 0.9447 | 0.7867 | 0.9683 | 1.0000 | 0.0014 | 0.0027 | 0.7867 | 0.2671 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9439。
