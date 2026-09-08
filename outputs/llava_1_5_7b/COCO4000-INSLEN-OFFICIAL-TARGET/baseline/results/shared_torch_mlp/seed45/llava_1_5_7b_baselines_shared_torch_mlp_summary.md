# llava_1_5_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`45`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 12317/0/3146。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed45/llava_1_5_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8373 | 0.8761 | 0.9200 | 0.8975 | 0.8824 | 0.9618 | 0.6678 | 0.5529 | 0.6049 | 0.8824 | 0.6625 |
| MetaToken-MLP | 0.8423 | 0.8569 | 0.9561 | 0.9038 | 0.8694 | 0.9537 | 0.7494 | 0.4513 | 0.5634 | 0.8694 | 0.6732 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8373 | 0.8761 | 0.9200 | 0.8975 | 0.8824 | 0.9618 | 0.6678 | 0.5529 | 0.6049 | 0.8824 | 0.6625 |
| MetaToken-MLP | 0.8420 | 0.8658 | 0.9421 | 0.9023 | 0.8694 | 0.9537 | 0.7146 | 0.4979 | 0.5869 | 0.8694 | 0.6732 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9548 | 0.9649 | 0.9771 | 0.9710 | 0.9896 | 0.9970 | 0.9180 | 0.8780 | 0.8975 | 0.9896 | 0.9648 |
| MetaToken-MLP | 0.8307 | 0.8488 | 0.9508 | 0.8969 | 0.8679 | 0.9555 | 0.7124 | 0.4183 | 0.5271 | 0.8679 | 0.6546 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9038。
