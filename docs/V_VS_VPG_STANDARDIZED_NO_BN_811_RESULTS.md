# 四模型 True-RMS V 与 VP+G：统一头对照

## 实验设置

- 固定811图片划分：3200 train / 400 validation / 400 test，split seed `20260912`，全部mentions。
- 分类器seeds `43/44/45`只影响训练随机性，不改变划分；测试指标为三seed独立结果的均值 ± 总体标准差，单位%。
- True-RMS All-attention K32。V=`[AE_V,log1p(S_V)]`；VP+G=`[AE_VP,log1p(S_V+S_P),AE_G,log1p(S_G)]`。
- 每列独立StandardScaler，只拟合训练mentions；无BatchNorm；训练满150 epochs；从完整轨迹选择最低验证BCE loss checkpoint。
- 每个模型固定一套分类器参数同时训练V和VP+G，只改变输入特征。参数沿用该模型原验证champion：Qwen2.5为width256/GELU/dropout.5/lr.003/wd1e-6/batch256；LLaVA和Qwen3为width128/ReLU/dropout.3/lr.001/wd.001/batch512；InternVL为width248/GELU/dropout.1/lr.0003/wd.001/batch512。
- 本轮在测试集已访问后追加，完整报告两组，不按测试结果事后选择。

## V 与 VP+G

| 模型 | 特征 | 维度 | 最低loss epoch 43/44/45 | 验证 AUROC / HALL-AUPR | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD |
|---|---|---:|---|---:|---:|---:|
| Qwen2.5 | V | 56 | 30/31/21 | **88.46 / 50.90** | **87.96 ± 0.16** | 42.23 ± 0.81 |
| Qwen2.5 | VP+G | 112 | 9/16/8 | 84.88 / 45.03 | 86.68 ± 0.33 | **46.20 ± 0.60** |
| LLaVA | V | 64 | 46/60/67 | 89.77 / 72.13 | 89.71 ± 0.13 | 68.43 ± 0.25 |
| LLaVA | VP+G | 128 | 72/50/38 | **90.73 / 73.71** | **89.94 ± 0.07** | **71.76 ± 0.21** |
| Qwen3 | V | 72 | 88/105/81 | 88.75 / 64.23 | 90.00 ± 0.12 | 67.78 ± 0.07 |
| Qwen3 | VP+G | 144 | 74/69/67 | **90.22 / 67.57** | **92.97 ± 0.06** | **72.10 ± 0.14** |
| InternVL | V | 64 | 76/84/65 | 83.93 / 51.89 | 85.45 ± 0.14 | 53.05 ± 0.22 |
| InternVL | VP+G | 128 | 83/93/59 | **85.59 / 58.50** | **88.14 ± 0.27** | **57.63 ± 0.63** |

| 模型 | VP+G − V ΔAUROC (pp) | VP+G − V ΔHALL-AUPR (pp) |
|---|---:|---:|
| Qwen2.5 | -1.28 | +3.98 |
| LLaVA | +0.22 | +3.33 |
| Qwen3 | +2.97 | +4.32 |
| InternVL | +2.70 | +4.58 |

按预定验证AUROC优先规则，Qwen2.5应保留V；LLaVA、Qwen3、InternVL应选VP+G。Qwen2.5加入prompt和generation后提高测试HALL-AUPR，但验证AUROC/AP均更低，不能按测试AP反向将VP+G设为正式赢家。

## 与原生 SVAR 比较

SVAR使用相同固定811划分和seeds；原生分类器为248单隐藏ReLU、Adam lr `.001`、batch32、最多50 epochs、最低验证loss/patience5，无StandardScaler/BN/dropout。

| 模型 | SVAR AUROC / AP | V相对SVAR ΔAUROC / ΔAP (pp) | VP+G相对SVAR ΔAUROC / ΔAP (pp) |
|---|---:|---:|---:|
| Qwen2.5 | 87.34 / 47.06 | +0.62 / -4.83 | -0.66 / -0.85 |
| LLaVA | 90.44 / 70.96 | -0.73 / -2.53 | -0.51 / +0.80 |
| Qwen3 | 88.95 / 62.66 | +1.05 / +5.12 | +4.02 / +9.44 |
| InternVL | 86.98 / 53.73 | -1.53 / -0.68 | +1.16 / +3.90 |

Qwen3的V和VP+G都同时超过SVAR；InternVL只有VP+G同时超过；LLaVA的VP+G为AUROC/AP取舍；Qwen2.5的VP+G两项均略低于SVAR。分类器预算并不相同：本方法参数继承先前24候选搜索，SVAR为固定原生头，因此这不是纯特征同预算比较。

## 复核文件

- 汇总：`outputs/v_vs_vpg_standardized_no_bn_811_v1/summary.json`
- 24个训练头：`outputs/v_vs_vpg_standardized_no_bn_811_v1/seed_metrics.csv`
- 每模型协议、checkpoint、训练历史和概率：`outputs/v_vs_vpg_standardized_no_bn_811_v1/<model>/`
- 复现实验：`scripts/evaluate_v_vs_vpg_standardized_no_bn_811.py`
