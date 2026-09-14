# Qwen3 SVAR 5–20层对照

## 实验设置

- 固定811图片划分：3200 train / 400 validation / 400 test，split seed `20260912`，全部mentions。
- 原生SVAR分类器：`Linear(D,248)-ReLU-Linear(248,2)`；Adam lr `.001`、batch32、最多50 epochs、最低验证交叉熵checkpoint、early-stop patience5；无标准化、BN、dropout、学习率调度和类别加权。
- seeds `43/44/45`只控制分类器训练，不改变图片划分；指标为三seed测试均值 ± 总体标准差，单位%。
- 同时报告代码`[5,20)`（零基索引5–19）和中文闭区间“5–20层”对应的`[5,21)`（零基索引5–20）。原生参考为`[5,19)`，即索引5–18。

## 结果

Qwen3每层32个attention heads；各层、各头的视觉attention mass展开为分类器输入。

| 层范围（零基） | 层数 | 输入维度 | 最佳epoch 43/44/45 | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD | 相对原`[5,19)` ΔAUROC / ΔAP (pp) |
|---|---:|---:|---|---:|---:|---:|
| 原生`[5,19)`，索引5–18 | 14 | 448 | — | 88.95 ± 0.23 | 62.66 ± 0.39 | — |
| `[5,20)`，索引5–19 | 15 | 480 | 20/30/17 | 89.21 ± 0.04 | 62.38 ± 0.67 | +0.25 / -0.28 |
| `[5,21)`，索引5–20 | 16 | 512 | 20/24/22 | **89.27 ± 0.01** | **63.34 ± 0.63** | **+0.32 / +0.69** |

若“5–20层”包含索引20，权威结果为`[5,21)`：`89.27/63.34`。加入索引19单独改善AUROC但略损失AP；再加入索引20后，相对`[5,20)`提高约`+0.07 AUROC/+0.96 HALL-AUPR pp`，最终两项都超过原14层SVAR。

该层范围SVAR仍低于当前相同811测试上的Qwen3 True-RMS VP+G标准化无BN头`92.97/72.10`，差约`-3.70/-8.75 pp`。两套方法分类器和特征不同，这不是纯层范围归因。

这是测试集已访问后的探索性端点消融，没有根据测试结果继续搜索其他层范围。

## 复核文件

- 协议：`outputs/qwen3_svar_layer_ranges_811_v1/protocol.json`
- 汇总：`outputs/qwen3_svar_layer_ranges_811_v1/summary.json`
- 六个训练头：`outputs/qwen3_svar_layer_ranges_811_v1/seed_metrics.csv`
- 复现实验：`scripts/evaluate_qwen3_svar_layer_ranges_811.py`
