# 无 BatchNorm、最低验证损失 checkpoint：811 对照

## 实验设置

- 图片划分：3200 train / 400 validation / 400 test，split seed `20260912`，保留全部 mentions。
- 每个模型沿用原搜索在验证集选出的 champion 特征与超参数。
- 仅进行两项联合修改：`batch_norm=False`，`monitor=val_loss`。
- 单隐藏层 MLP，BCE（REAL=1），Adam；最多 150 epochs，early-stop patience 20，ReduceLROnPlateau factor 0.5 / patience 6 / min lr `1e-6`。
- seeds 43/44/45 独立训练；表中测试指标为三 seed 均值 ± 总体标准差，单位为 %，没有进行概率 ensemble。
- 这是测试集已经访问后的探索性消融，不是新的封闭模型选择。

## 结果

| 模型 | 特征 | 无BN最低loss最佳epoch 43/44/45 | 验证 AUROC / HALL-AUPR | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD | 相对原 champion ΔAUROC / ΔAP (pp) |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | True-RMS V | 30/31/21 | 88.46 / 50.90 | 87.96 ± 0.16 | 42.23 ± 0.81 | -0.16 / -1.82 |
| LLaVA-1.5-7B | True-RMS VP+G | 137/115/128 | 88.64 / 69.57 | 87.53 ± 0.11 | 67.42 ± 0.19 | -2.48 / -4.43 |
| Qwen3-VL-8B | True-RMS VP+G | 149/150/148 | 85.29 / 57.77 | 87.48 ± 0.17 | 62.28 ± 0.34 | -5.40 / -10.47 |
| InternVL2.5-8B | True-RMS VP+G | 83/93/59 | 85.59 / 58.50 | 88.14 ± 0.27 | 57.63 ± 0.63 | -0.85 / -2.44 |

原 champion 的测试 AUROC / HALL-AUPR 分别为 Qwen2.5 88.12/44.04、LLaVA 90.01/71.85、Qwen3 92.88/72.75、InternVL 88.99/60.08。

## 参数

| 模型 | width | 标准化 | 激活 | dropout | lr | wd | batch |
|---|---:|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 256 | 是 | GELU | 0.5 | 0.003 | 1e-6 | 256 |
| LLaVA-1.5-7B | 128 | 否 | ReLU | 0.3 | 0.001 | 0.001 | 512 |
| Qwen3-VL-8B | 128 | 否 | ReLU | 0.3 | 0.001 | 0.001 | 512 |
| InternVL2.5-8B | 248 | 是 | GELU | 0.1 | 0.0003 | 0.001 | 512 |

LLaVA 和 Qwen3 的原 champion 使用 `val_auroc` checkpoint，因此它们与原结果的差值同时包含去掉 BN 和改用最低验证损失两项改变。此前原有 BN 训练轨迹中，两种 checkpoint 几乎重合，三 seed 中各有两个 epoch 完全相同，另一个 seed 的验证 AUROC 只相差 0.007/0.023 pp；本次大幅下降主要与去掉 BN 后的训练轨迹相关，但严格的 BN 单因素效应需要另跑 `BN=False, monitor=val_auroc`。

Qwen3 三个最低-loss checkpoint 都位于 148–150 epoch，已经触及最大训练轮数；当前设置不能证明其无 BN 训练已经充分收敛。

## LLaVA/Qwen3：先标准化、再去掉 BN

补充设置保持 True-RMS VP+G、宽度128、ReLU、dropout 0.3、Adam lr `0.001` / wd `0.001`、batch 512不变。只在训练集拟合 StandardScaler，随后使用 `BatchNorm=False` 的单隐藏层 MLP；checkpoint、调度和早停仍依据最低验证 BCE loss。图片划分和三训练seed与上文相同。

| 模型 | 标准化 | BN | 最佳epoch 43/44/45 | 验证 AUROC / HALL-AUPR | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD |
|---|---|---|---|---:|---:|---:|
| LLaVA | 否 | 否 | 137/115/128 | 88.64 / 69.57 | 87.53 ± 0.11 | 67.42 ± 0.19 |
| LLaVA | 是 | 否 | 72/50/38 | 90.73 / 73.71 | **89.94 ± 0.07** | **71.76 ± 0.21** |
| LLaVA 原 champion | 否 | 是 | 61/53/70 | 91.16 / 73.97 | 90.01 ± 0.10 | 71.85 ± 0.39 |
| Qwen3 | 否 | 否 | 149/150/148 | 85.29 / 57.77 | 87.48 ± 0.17 | 62.28 ± 0.34 |
| Qwen3 | 是 | 否 | 74/69/67 | 90.22 / 67.57 | **92.97 ± 0.06** | **72.10 ± 0.14** |
| Qwen3 原 champion | 否 | 是 | 109/83/88 | 90.76 / 68.48 | 92.88 ± 0.17 | 72.75 ± 0.43 |

相对“不标准化、无BN”，StandardScaler使LLaVA测试 AUROC/HALL-AUPR提高 `+2.41/+4.34 pp`，Qwen3提高 `+5.49/+9.81 pp`。相对原有BN champion，LLaVA为 `-0.07/-0.08 pp`；Qwen3为 `+0.10/-0.66 pp`。因此这两个模型需要至少一种尺度归一化；本结果不支持“必须使用BN”，因为训练集StandardScaler在无BN时已经恢复了绝大部分性能。

这仍不是BN的严格单因素比较：LLaVA/Qwen3原 champion 的checkpoint依据为最高验证AUROC，本补充实验依据最低验证loss。测试集此前已经访问，Qwen3的 `+0.10 pp` AUROC不能作为新配置优于原配置的确认性结论。

## 四模型：block-wise 标准化、无 BN

对 True-RMS VP+G 的四个连续块 `AE_VP`、`log1p(S_V+S_P)`、`AE_G`、`log1p(S_G)` 分别计算一个训练集均值和总体标准差。每个统计量汇总全部训练mentions及该块全部decoder层；同一块的所有层共用该仿射变换。验证、测试只复用训练统计量。分类器和训练设置与上一节相同。

| 模型 | 标准化方式 | 最佳epoch 43/44/45 | 验证 AUROC / HALL-AUPR | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD |
|---|---|---|---:|---:|---:|
| LLaVA | 无 | 137/115/128 | 88.64 / 69.57 | 87.53 ± 0.11 | 67.42 ± 0.19 |
| LLaVA | 每列 Z-score | 72/50/38 | 90.73 / 73.71 | 89.94 ± 0.07 | 71.76 ± 0.21 |
| LLaVA | block-wise | 88/85/70 | 90.23 / 72.65 | **89.95 ± 0.05** | **72.57 ± 0.22** |
| Qwen3 | 无 | 149/150/148 | 85.29 / 57.77 | 87.48 ± 0.17 | 62.28 ± 0.34 |
| Qwen3 | 每列 Z-score | 74/69/67 | 90.22 / 67.57 | **92.97 ± 0.06** | **72.10 ± 0.14** |
| Qwen3 | block-wise | 109/126/86 | 89.46 / 66.37 | 92.08 ± 0.25 | 70.12 ± 0.36 |

LLaVA的block-wise测试结果相对逐列标准化为 `+0.01 AUROC/+0.80 HALL-AUPR pp`，但验证集为 `-0.50/-1.07 pp`，所以不能据测试集AP反向认定block-wise更好。Qwen3测试结果为 `-0.90/-1.97 pp`；其不同层的训练标准差在块内仍可相差约2.0至10.5倍，block-wise保留的层尺度差异对当前MLP更像干扰，逐列标准化更合适。这是结果解释，不是经过独立机制消融确认的因果结论。

Qwen2.5沿用验证选出的True-RMS V，使用两个块 `AE_V`、`log1p(S_V)`，每块28层；InternVL沿用True-RMS VP+G，使用四个32层块。其他设置不变。

| 模型 | 标准化方式 | 最佳epoch 43/44/45 | 验证 AUROC / HALL-AUPR | 测试 AUROC ± SD | 测试 HALL-AUPR ± SD |
|---|---|---|---:|---:|---:|
| Qwen2.5 | 每列 Z-score、无BN | 30/31/21 | 88.46 / 50.90 | **87.96 ± 0.16** | 42.23 ± 0.81 |
| Qwen2.5 | block-wise、无BN | 55/44/45 | 86.98 / 48.08 | 87.13 ± 0.05 | **42.53 ± 0.74** |
| Qwen2.5 原 champion | 每列 Z-score、有BN | 30/31/22 | 88.40 / 51.22 | 88.12 ± 0.07 | 44.04 ± 0.40 |
| InternVL | 每列 Z-score、无BN | 83/93/59 | 85.59 / 58.50 | **88.14 ± 0.27** | **57.63 ± 0.63** |
| InternVL | block-wise、无BN | 112/117/71 | 83.93 / 52.97 | 85.40 ± 0.10 | 51.72 ± 0.20 |
| InternVL 原 champion | 每列 Z-score、有BN | 48/62/62 | 87.06 / 61.23 | 88.99 ± 0.38 | 60.08 ± 0.77 |

相对逐列标准化，Qwen2.5 block-wise为 `-0.82 AUROC/+0.31 HALL-AUPR pp`，有指标取舍；InternVL为 `-2.74/-5.91 pp`，两项都明显更差。按验证AUROC优先规则，两者都应保留逐列标准化。InternVL块内不同层的训练标准差相差约1.8至5.9倍，保留这些尺度差异没有帮助当前分类器。

## 文件

- 汇总：`outputs/no_bn_val_loss_811_v1/summary.json`
- 三 seed 指标：`outputs/no_bn_val_loss_811_v1/seed_metrics.csv`
- 每模型协议、checkpoint 和概率：`outputs/no_bn_val_loss_811_v1/<model>/`
- 标准化无BN补充：`outputs/no_bn_standardized_val_loss_811_v1/`
- block-wise无BN补充：`outputs/no_bn_blockwise_val_loss_811_v1/`
- 复现实验：`scripts/ablate_no_bn_val_loss_811.py`
