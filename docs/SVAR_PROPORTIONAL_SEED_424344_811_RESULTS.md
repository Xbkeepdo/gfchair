# 比例层原生SVAR：训练种子42/43/44（固定811）

## 设置

- 数据严格复用当前路径实验的固定图片级3200 train / 400 validation / 400 test划分及全部mentions。
- 原生SVAR分类器为`Linear(D,248)-ReLU-Linear(248,2)`，Adam lr0.001、batch32、最多50 epochs、validation loss checkpoint、patience5；无标准化、BN、dropout或调度。
- LLaVA使用零基层5–18，即`[5,19)`；其他模型按decoder深度映射为Qwen2.5层4–16、Qwen3层6–20、InternVL层5–18。
- 本轮完整重训seeds42/43/44共12个头，并与原seeds43/44/45结果对照。
- token-detector中的Qwen3 image-level 82结果使用2800/400/800及不同test cohort，未混入本表。

## SVAR结果

| 模型 | seeds42/43/44 AUROC / HALL-AUPR | seeds43/44/45 | 变化（AUROC/AP pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **86.61±0.37 / 43.90±0.95** | 86.53±0.39 / 43.77±0.99 | +0.08 / +0.13 |
| LLaVA-1.5-7B | **90.42±0.02 / 71.24±0.46** | 90.40±0.01 / 71.15±0.36 | +0.02 / +0.09 |
| Qwen3-VL-8B | **89.23±0.09 / 64.28±1.27** | 89.31±0.08 / 63.66±0.82 | −0.08 / +0.62 |
| InternVL2.5-8B | **87.85±0.45 / 55.28±1.00** | 87.65±0.26 / 54.94±0.81 | +0.20 / +0.35 |
| 四模型宏平均 | **88.53 / 58.68** | 88.47 / 58.38 | +0.05 / +0.29 |

两组三seed共享43/44，所以变化只反映用seed42替换seed45，不是两组三次独立重复。

## 与同样seeds42/43/44的Visual-only单层比较

| 模型 | Visual-only AUROC/AP | SVAR AUROC/AP | Visual-only − SVAR（pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **87.90 / 45.75** | 86.61 / 43.90 | **+1.29 / +1.85** |
| LLaVA-1.5-7B | 89.66 / 67.57 | **90.42 / 71.24** | −0.76 / −3.67 |
| Qwen3-VL-8B | **89.68 / 68.26** | 89.23 / 64.28 | **+0.45 / +3.99** |
| InternVL2.5-8B | 86.66 / 53.59 | **87.85 / 55.28** | −1.19 / −1.69 |
| 四模型宏平均 | 88.47 / **58.80** | **88.53** / 58.68 | −0.05 / +0.12 |

定性结论与43/44/45一致：Visual-only在Qwen2.5、Qwen3同时超过SVAR，在LLaVA、InternVL低于SVAR。宏平均表现为极小取舍：SVAR的AUROC高0.05个百分点，Visual-only的HALL-AUPR高0.12个百分点。

固定test此前已访问，本轮仍为探索性对照；三seed不足以完整刻画训练随机性。

## 产物与核验

- 汇总：`outputs/svar_proportional_seed424344_811_v1/summary.md`、`summary.json`、`comparison.csv`
- 每模型协议、三seed权重、概率及指标：`outputs/svar_proportional_seed424344_811_v1/<model>/`
- 12个头CPU重载复算全部通过，最大概率绝对误差`4.7683716e-7`；`validation.json`为`PASS`
- 入口：`scripts/evaluate_svar_proportional_seed_424344_811.py`
