# shikra_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`43`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 13361/0/3343。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/shikra_7b/COCO4000-JACOBIAN-PATH/baseline/results/shared_torch_mlp/seed43/shikra_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8277 | 0.8877 | 0.8904 | 0.8891 | 0.8602 | 0.9490 | 0.6178 | 0.6112 | 0.6145 | 0.8602 | 0.6125 |
| ProjectAway | 0.7751 | 0.7753 | 0.9996 | 0.8733 | 0.6926 | 0.8873 | 0.0000 | 0.0000 | 0.0000 | 0.6926 | 0.3577 |
| MetaToken-MLP | 0.8181 | 0.8283 | 0.9657 | 0.8917 | 0.8535 | 0.9494 | 0.7227 | 0.3089 | 0.4328 | 0.8535 | 0.6226 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8316 | 0.8717 | 0.9178 | 0.8942 | 0.8602 | 0.9490 | 0.6531 | 0.5340 | 0.5875 | 0.8602 | 0.6125 |
| ProjectAway | 0.7754 | 0.7754 | 1.0000 | 0.8735 | 0.6926 | 0.8873 | 0.0000 | 0.0000 | 0.0000 | 0.6926 | 0.3577 |
| MetaToken-MLP | 0.8214 | 0.8501 | 0.9344 | 0.8903 | 0.8535 | 0.9494 | 0.6559 | 0.4314 | 0.5205 | 0.8535 | 0.6226 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9887 | 0.9950 | 0.9902 | 0.9926 | 0.9993 | 0.9998 | 0.9686 | 0.9837 | 0.9761 | 0.9993 | 0.9977 |
| ProjectAway | 0.7657 | 0.7657 | 0.9998 | 0.8672 | 0.7229 | 0.8986 | 0.7778 | 0.0022 | 0.0045 | 0.7229 | 0.3980 |
| MetaToken-MLP | 0.8071 | 0.8173 | 0.9634 | 0.8843 | 0.8492 | 0.9477 | 0.7136 | 0.2973 | 0.4197 | 0.8492 | 0.6162 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.8917。
