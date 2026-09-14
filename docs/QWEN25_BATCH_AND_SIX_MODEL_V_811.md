# Qwen2.5 batch 消融与六模型 V 检测

## 统一说明

- 811 图片划分：3200 train / 400 validation / 400 test，split seed 20260912，保留全部 mentions。
- 检测数值是 seeds 43/44/45 各自指标的算术均值 ± 总体标准差（ddof=0），不是概率 ensemble。
- 单隐藏层搜索：24 个固定候选先跑 seed43，按 validation AUROC、HALL-AUPR、候选序号选 top3，再补 seeds44/45；按三种子 validation 均值冻结配置，之后才读取 test。无 train+validation refit。
- 旧 800 图此前已经查看，因此这些 400 图 test 结果属于探索性比较。

## Qwen2.5 的正式 test 结果

特征路径为真实 RMS all-attention `z-A_all→z`，局部 FP32 Gauss–Legendre K32。V=`[AE_V,log1p(S_V^all)]`；VP+G=`[AE_VP,log1p(S_V^all+S_P^all),AE_G,log1p(S_G^all)]`。

| 特征 | test AUROC | test HALL-AUPR | 已选单层 MLP |
|---|---:|---:|---|
| V | 88.12 ± 0.07 | 44.04 ± 0.40 | width256, GELU, StandardScaler, BN, dropout0.5, lr0.003, wd1e-6, batch256, val-loss checkpoint |
| VP+G | 87.68 ± 0.33 | 46.54 ± 1.05 | width128, ReLU, no scaler, BN, dropout0.3, lr0.001, wd0.001, batch512, val-AUROC checkpoint |

V 的 AUROC 高 0.44 个百分点；VP+G 的 HALL-AUPR 高 2.50 个百分点。

## Qwen2.5 batch size 受控消融

该消融只报告 validation，test 没有用于比较 batch。V 与 VP+G 分别固定上表配置中的所有其他参数，只替换 batch size；seeds43/44/45 均值 ± 总体标准差。训练仍按 epoch 与 patience 定义，因此 batch 改变时每 epoch 的 optimizer step 数也会变化。

| 特征 | batch | validation AUROC | validation HALL-AUPR |
|---|---:|---:|---:|
| V | 32 | 88.21 ± 0.29 | 49.87 ± 0.75 |
| V | 64 | 88.35 ± 0.47 | 50.55 ± 0.86 |
| V | 128 | 88.05 ± 0.17 | 49.75 ± 0.63 |
| V | 256 | **88.40 ± 0.19** | **51.22 ± 0.48** |
| V | 512 | 88.17 ± 0.27 | 50.95 ± 0.78 |
| VP+G | 32 | 86.76 ± 0.15 | 49.23 ± 0.65 |
| VP+G | 64 | 86.75 ± 0.12 | 49.08 ± 0.76 |
| VP+G | 128 | **86.96 ± 0.02** | **49.33 ± 0.67** |
| VP+G | 256 | 86.89 ± 0.16 | 49.26 ± 0.57 |
| VP+G | 512 | 86.75 ± 0.04 | 48.55 ± 0.46 |

大 batch 不是本方法有效的必要条件。V 的最优点是 256，但 64 仅低 0.05 AUROC 百分点；VP+G 的最优点是 128，比 512 高 0.21 AUROC 和 0.77 HALL-AUPR 百分点。现有差异更像优化轨迹与 checkpoint 选择的轻微变化，没有单调 batch 趋势。

## 四个原模型的 V test 结果

以下均为相同真实 RMS all-attention K32、相同 V 定义与相同 811 搜索协议。

| 模型 | test AUROC | test HALL-AUPR |
|---|---:|---:|
| Qwen2.5-VL-7B | 88.12 ± 0.07 | 44.04 ± 0.40 |
| LLaVA-1.5-7B | 89.77 ± 0.10 | 68.19 ± 0.52 |
| Qwen3-VL-8B | 89.62 ± 0.20 | 68.06 ± 0.28 |
| InternVL2.5-8B | 86.27 ± 0.07 | 53.17 ± 0.75 |

## MiniGPT-4 与 Shikra 第一轮结果

第一轮沿用它们已经完整审计和保存的 visual-only 路径 `z-A_V→z`，局部 FP32 Gauss–Legendre K4，特征为 `[AE_V,log1p(S_V)]`。分类器使用同一套 24 候选单隐藏层 811 搜索。原生基线也在同一 3200/400/400 mention 划分上重新训练。路径与上面的 all-attention K32 不同，因此不能据此做六模型严格横向排名。

| 模型 | 方法 | test AUROC | test HALL-AUPR |
|---|---|---:|---:|
| MiniGPT-4 | V（visual-only K4） | 90.53 ± 0.18 | 64.83 ± 1.54 |
| MiniGPT-4 | 原生 SVAR | 91.42 ± 0.16 | 70.67 ± 0.44 |
| MiniGPT-4 | MetaToken LR | 89.68 ± 0.00 | 64.22 ± 0.00 |
| MiniGPT-4 | MetaToken GB | 89.56 ± 0.04 | 61.18 ± 0.07 |
| Shikra | V（visual-only K4） | 86.25 ± 0.18 | 60.81 ± 0.30 |
| Shikra | 原生 SVAR | 88.03 ± 0.14 | 65.45 ± 0.42 |
| Shikra | MetaToken LR | 87.25 ± 0.00 | 64.14 ± 0.00 |
| Shikra | MetaToken GB | 87.10 ± 0.00 | 63.07 ± 0.01 |

MiniGPT-4 的 V 比原生 SVAR 低 0.89 AUROC、5.84 HALL-AUPR 百分点；Shikra 分别低 1.78、4.64 个百分点。V 在 MiniGPT-4 上超过两种 MetaToken，在 Shikra 上低于两种 MetaToken。

## 严格 all-attention K32 扩展状态

MiniGPT-4 与 Shikra 的 8 图 K32/K64 审计均通过。MiniGPT-4 的 gross 最大相对差为 `2.02e-7`、K32 最大闭合误差 `1.77e-5`；Shikra 分别为 `3.03e-7`、`1.19e-5`。两模型 4000 图提取已在服务器 32678 的两张 GPU 上运行；完成后将按完全相同的 811 单隐藏层协议更新正式六模型比较。
