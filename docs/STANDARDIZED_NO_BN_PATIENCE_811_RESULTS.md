# StandardScaler、无 BatchNorm：early-stop patience 5/10/20

## 实验设置

- 固定811图片划分：3200 train / 400 validation / 400 test，split seed `20260912`，全部mentions。
- 分类器训练seeds `43/44/45`，不改变图片划分；表中测试指标为三seed独立指标的均值 ± 总体标准差，单位%。
- 每列独立Z-score，均值/总体标准差只拟合训练mentions；验证和测试复用训练统计量。
- `BatchNorm=False`，BCE（REAL=1），Adam，最多150 epochs，checkpoint/调度/早停均监控验证BCE loss。
- `ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6)`固定，只改变early-stop patience为5/10/20。
- 每模型沿用原验证搜索选出的champion特征和其他参数：Qwen2.5为True-RMS V，其余三个模型为True-RMS VP+G。
- 本轮在测试集已访问后追加，完整报告三种patience，不按测试结果事后隐藏或挑选。

## 与原生 SVAR 比较

原生SVAR使用相同固定811划分与seeds；分类器为`Linear(D,248)-ReLU-Linear(248,2)`，Adam lr `.001`、batch32、最多50 epochs、最低验证loss checkpoint、early-stop patience5，无StandardScaler/BN/dropout。

| 模型 | patience | 最佳epoch 43/44/45 | 验证 AUROC / HALL-AUPR | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD | 相对SVAR ΔAUROC / ΔAP (pp) |
|---|---:|---|---:|---:|---:|---:|
| Qwen2.5 | 5 | 13/20/21 | 87.77 / 48.62 | 87.41 ± 0.77 | 41.05 ± 1.47 | +0.07 / -6.01 |
| Qwen2.5 | 10 | 30/31/21 | 88.46 / 50.90 | **87.96 ± 0.16** | **42.23 ± 0.81** | +0.62 / -4.83 |
| Qwen2.5 | 20 | 30/31/21 | 88.46 / 50.90 | **87.96 ± 0.16** | **42.23 ± 0.81** | +0.62 / -4.83 |
| Qwen2.5 SVAR | 5 | — | — | 87.34 ± 0.22 | 47.06 ± 0.66 | — |
| LLaVA | 5 | 22/18/26 | 90.26 / 73.47 | 89.67 ± 0.08 | 71.40 ± 0.44 | -0.78 / +0.44 |
| LLaVA | 10 | 72/50/26 | 90.70 / **73.89** | 89.91 ± 0.11 | 71.56 ± 0.49 | -0.53 / +0.60 |
| LLaVA | 20 | 72/50/38 | **90.73** / 73.71 | **89.94 ± 0.07** | **71.76 ± 0.21** | -0.51 / +0.80 |
| LLaVA SVAR | 5 | — | — | 90.44 ± 0.04 | 70.96 ± 0.53 | — |
| Qwen3 | 5 | 36/36/38 | 89.76 / 66.91 | 92.49 ± 0.04 | 70.79 ± 0.05 | +3.54 / +8.13 |
| Qwen3 | 10 | 61/69/67 | 90.21 / 67.51 | 92.97 ± 0.06 | 72.09 ± 0.16 | +4.02 / +9.43 |
| Qwen3 | 20 | 74/69/67 | **90.22 / 67.57** | **92.97 ± 0.06** | **72.10 ± 0.14** | +4.02 / +9.44 |
| Qwen3 SVAR | 5 | — | — | 88.95 ± 0.23 | 62.66 ± 0.39 | — |
| InternVL | 5 | 56/35/42 | 85.02 / 57.50 | 87.29 ± 0.43 | 55.70 ± 0.97 | +0.31 / +1.97 |
| InternVL | 10 | 56/93/42 | 85.44 / 58.19 | 87.89 ± 0.45 | 57.07 ± 1.04 | +0.91 / +3.34 |
| InternVL | 20 | 83/93/59 | **85.59 / 58.50** | **88.14 ± 0.27** | **57.63 ± 0.63** | +1.16 / +3.90 |
| InternVL SVAR | 5 | — | — | 86.98 ± 0.61 | 53.73 ± 1.14 | — |

## 结论

Patience 5通常过早停止。由于学习率调度器自身patience为6，early-stop patience5会在连续5轮不改善时退出，来不及利用首次plateau降学习率。Qwen2.5的patience10和20选择了完全相同的三个checkpoint；Qwen3也几乎饱和。LLaVA仍有很小增益，InternVL从10延长到20的测试AUROC/HALL-AUPR提高`+0.25/+0.56 pp`。

按验证AUROC优先，Qwen2.5的10和20并列，LLaVA/Qwen3/InternVL均为20最高。与SVAR相比，Qwen3和InternVL在三种patience下均同时提高AUROC和HALL-AUPR；Qwen2.5只有AUROC提高、AP下降，LLaVA只有AP提高、AUROC下降。

本方法沿用先前24候选搜索得到的champion参数，原生SVAR使用固定原生分类器，没有相同HPO预算。因此差值比较的是两套完整方法，不能单独归因于特征。

## 不使用 early stopping

补充实验将四模型全部训练满150 epochs，学习率调度保持不变，并继续从完整150轮中选择最低验证BCE loss checkpoint。它不是直接使用第150轮权重。

| 模型 | 训练轮数 43/44/45 | 最低loss epoch 43/44/45 | 无早停测试 AUROC / HALL-AUPR | 与patience20差值 (pp) |
|---|---|---|---:|---:|
| Qwen2.5 | 150/150/150 | 30/31/21 | 87.96 ± 0.16 / 42.23 ± 0.81 | 0.00 / 0.00 |
| LLaVA | 150/150/150 | 72/50/38 | 89.94 ± 0.07 / 71.76 ± 0.21 | 0.00 / 0.00 |
| Qwen3 | 150/150/150 | 74/69/67 | 92.97 ± 0.06 / 72.10 ± 0.14 | 0.00 / 0.00 |
| InternVL | 150/150/150 | 83/93/59 | 88.14 ± 0.27 / 57.63 ± 0.63 | 0.00 / 0.00 |

十二个无早停训练均没有在patience20原停止点之后发现更低的验证loss，因此保存的checkpoint及测试概率与patience20逐项相同。继续到150轮只增加训练计算。这里能说明patience20对当前四组训练轨迹已经足够，不能推导为所有超参数或其他特征组都适用。

## 复核文件

- 汇总：`outputs/standardized_no_bn_patience_811_v1/summary.json`
- 36个训练头：`outputs/standardized_no_bn_patience_811_v1/seed_metrics.csv`
- 每模型协议、checkpoint、训练历史及概率：`outputs/standardized_no_bn_patience_811_v1/<model>/`
- 无早停150轮汇总：`outputs/standardized_no_bn_no_early_stop_811_v1/summary.json`
- 复现实验：`scripts/evaluate_standardized_no_bn_patience_811.py`
