# Qwen2.5 SVAR 5–16层对照

## 实验设置

- 固定811图片划分：3200 train / 400 validation / 400 test，split seed `20260912`，全部mentions。
- 原生SVAR分类器：`Linear(D,248)-ReLU-Linear(248,2)`；Adam lr `.001`、batch32、最多50 epochs、最低验证交叉熵checkpoint、early-stop patience5；无标准化、BN、dropout、学习率调度和类别加权。
- seeds `43/44/45`只控制分类器初始化和minibatch顺序，不改变图片划分；指标为三seed测试均值 ± 总体标准差，单位%。
- 同时报告两种端点语义：代码配置`[5,16)`为索引5–15；中文闭区间“5–16层”为代码`[5,17)`，即索引5–16。
- 当前原生参考为`[5,19)`，即索引5–18，共14层。

## 结果

Qwen2.5每层28个attention heads；每层每头的视觉attention mass展开为分类器输入。

| 层范围（零基） | 层数 | 输入维度 | 最佳epoch 43/44/45 | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD | 相对原`[5,19)` ΔAUROC / ΔAP (pp) |
|---|---:|---:|---|---:|---:|---:|
| `[5,16)`，索引5–15 | 11 | 308 | 8/8/8 | 86.12 ± 0.19 | 40.88 ± 0.59 | -1.22 / -6.18 |
| `[5,17)`，索引5–16 | 12 | 336 | 8/11/8 | 86.42 ± 0.32 | 43.54 ± 1.01 | -0.92 / -3.52 |
| 原生`[5,19)`，索引5–18 | 14 | 392 | — | **87.34 ± 0.22** | **47.06 ± 0.66** | — |

若用户所说“5–16层”是包含16层的闭区间，权威结果是`[5,17)`这一行：`86.42/43.54`。加入索引16相对只到15层提高约`+0.30 AUROC/+2.66 HALL-AUPR pp`；再保留原配置中的索引17和18后提高约`+0.92/+3.52 pp`。因此在当前固定811划分上，缩短到5–16没有改善，较后的三层尤其对HALL-AUPR有贡献。

这是测试集已经访问后的探索性层范围消融，没有据此继续搜索其他端点。

## 复核文件

- 协议：`outputs/qwen25_svar_layer_ranges_811_v1/protocol.json`
- 汇总：`outputs/qwen25_svar_layer_ranges_811_v1/summary.json`
- 六个训练头：`outputs/qwen25_svar_layer_ranges_811_v1/seed_metrics.csv`
- 复现实验：`scripts/evaluate_qwen25_svar_layer_ranges_811.py`
