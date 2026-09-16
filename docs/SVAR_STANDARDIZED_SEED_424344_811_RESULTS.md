# 比例层SVAR训练集标准化结果（固定811，seeds42/43/44）

## 问题与单因素设置

本实验回答：保持比例层SVAR的输入、数据划分、PyTorch MLP及优化参数不变，只增加逐特征train-only z-score，检测结果如何。

- 固定图片级3200 train / 400 validation / 400 test及全部mentions，split seed 20260912。
- LLaVA零基层5–18（`[5,19)`）；其他模型按decoder深度同比例映射。
- seeds42/43/44；`Linear(D,248)-ReLU-Linear(248,2)`，Adam lr0.001、batch32、最多50 epochs、validation loss checkpoint、patience5。
- 每列均值和总体标准差仅由train mentions拟合；相同mean/scale用于train、validation、test。无BN、dropout、weighted sampler或weight decay。
- 对照为同层、同划分、同种子、同训练设置的未标准化SVAR。

完整预设见`docs/SVAR_STANDARDIZED_SEED_424344_811_PROTOCOL.md`。

## 结果

以下均为三次独立训练的测试AUROC和HALL-AUPR均值±总体标准差，单位为%。

| 模型 | 标准化 AUROC / AP | 未标准化 AUROC / AP | 标准化变化（AUROC / AP pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **85.19±0.14 / 39.46±1.13** | 86.61±0.37 / 43.90±0.95 | **−1.42 / −4.45** |
| LLaVA-1.5-7B | **90.21±0.10 / 70.39±0.55** | 90.42±0.02 / 71.24±0.46 | **−0.21 / −0.85** |
| Qwen3-VL-8B | **89.26±0.38 / 63.29±1.08** | 89.23±0.09 / 64.28±1.27 | **+0.03 / −0.99** |
| InternVL2.5-8B | **87.11±1.00 / 53.21±2.50** | 87.85±0.45 / 55.28±1.00 | **−0.74 / −2.08** |
| 四模型宏平均 | **87.94 / 56.59** | 88.53 / 58.68 | **−0.58 / −2.09** |

标准化没有产生跨模型收益：四模型AP全部下降；AUROC在三模型下降，Qwen3仅提高0.03个百分点，可视为点估计基本不变。损失最大的是Qwen2.5，AUROC/AP分别下降1.42/4.45个百分点。

## 训练行为与解释边界

标准化头的最佳epoch均为1–4；未标准化头为8–35。因为本实验刻意固定lr0.001和patience5，这一结果测量的是“在原生SVAR训练协议中加入标准化”的整体效果，也包含标准化与学习率、早停的交互。它不能证明经过重新调学习率后的标准化SVAR仍必然更差。

固定test此前已经访问，本实验是探索性消融；未计算独立cohort推广或显著性检验。

## 产物与核验

- 入口：`scripts/evaluate_svar_standardized_seed_424344_811.py`
- 输出：`outputs/svar_proportional_standardized_seed424344_811_v1/`
- 三seed汇总：`summary.md`、`summary.json`、`comparison.csv`、`seed_metrics.csv`
- 每模型保存协议、mean/scale、训练历史、权重、概率和指标。
- 12个checkpoint的CPU重载与指标复算全部通过；mean/scale相对train-only重算最大误差均为0，最大概率绝对误差`1.7881393e-7`，`validation.json`状态为`PASS`。
