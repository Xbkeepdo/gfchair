# 四模型：SVAR/MetaToken联合数据划分与训练种子

设置：每个seed将完整4000图重新划分为3200 train / 400 validation / 400 test；同一seed同时控制划分和分类器随机性。seeds43/44/45，全mentions。
SVAR：Linear(D,248)-ReLU-Linear(248,2)，Adam lr0.001、batch32、最多50 epochs、无标准化/BN/dropout/调度，validation loss checkpoint、patience5。
MetaToken LR：训练集StandardScaler+LogisticRegression(lbfgs,max_iter2000)；GB：训练集StandardScaler+GradientBoosting100。固定参数，无搜索，无train+validation refit。
每个seed在各自400图test上计算AUROC/HALL-AUPR；表中为算术均值±总体标准差(ddof=0)，不是概率ensemble。联合std同时包含测试图片构成与训练随机性。

| 模型 | 分类器 | 固定划分 AUROC | 联合种子 AUROC | 固定划分 HALL-AUPR | 联合种子 HALL-AUPR |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | SVAR | 87.34 ± 0.22 | 87.32 ± 0.87 | 47.06 ± 0.66 | 42.14 ± 3.26 |
| Qwen2.5-VL-7B | MetaToken LR | 83.39 ± 0.00 | 83.04 ± 1.63 | 44.08 ± 0.00 | 42.24 ± 1.93 |
| Qwen2.5-VL-7B | MetaToken GB | 82.73 ± 0.02 | 82.77 ± 0.69 | 38.98 ± 1.00 | 39.36 ± 2.56 |
| LLaVA-1.5-7B | SVAR | 90.44 ± 0.04 | 90.51 ± 0.77 | 70.96 ± 0.53 | 72.51 ± 3.39 |
| LLaVA-1.5-7B | MetaToken LR | 88.35 ± 0.00 | 88.76 ± 0.13 | 67.36 ± 0.00 | 69.66 ± 1.64 |
| LLaVA-1.5-7B | MetaToken GB | 88.61 ± 0.04 | 89.50 ± 0.24 | 66.88 ± 0.06 | 71.62 ± 0.50 |
| Qwen3-VL-8B | SVAR | 88.95 ± 0.23 | 86.60 ± 0.46 | 62.66 ± 0.39 | 58.29 ± 1.93 |
| Qwen3-VL-8B | MetaToken LR | 82.24 ± 0.00 | 83.28 ± 0.27 | 45.74 ± 0.00 | 47.70 ± 1.39 |
| Qwen3-VL-8B | MetaToken GB | 83.66 ± 0.02 | 84.50 ± 0.57 | 52.76 ± 0.23 | 54.54 ± 1.17 |
| InternVL2.5-8B | SVAR | 86.98 ± 0.61 | 86.90 ± 0.51 | 53.73 ± 1.14 | 54.09 ± 3.57 |
| InternVL2.5-8B | MetaToken LR | 83.38 ± 0.00 | 83.10 ± 1.47 | 46.96 ± 0.00 | 47.54 ± 3.69 |
| InternVL2.5-8B | MetaToken GB | 84.89 ± 0.02 | 84.68 ± 0.41 | 52.03 ± 0.14 | 50.89 ± 1.13 |

## 同一联合种子下与本方法比较

差值为本方法减原生SVAR，单位为百分点。V和VP+G均为预先定义的真实RMS all-attention K32特征；不根据本表测试结果选择特征。

| 模型 | 本方法特征 | 本方法 AUROC/AP | SVAR AUROC/AP | 差值 AUROC/AP |
|---|---|---:|---:|---:|
| Qwen2.5-VL-7B | visual | 85.56/49.41 | 87.32/42.14 | -1.76/+7.27 |
| Qwen2.5-VL-7B | vp_generation | 87.84/50.92 | 87.32/42.14 | +0.52/+8.78 |
| LLaVA-1.5-7B | visual | 89.43/71.01 | 90.51/72.51 | -1.08/-1.50 |
| LLaVA-1.5-7B | vp_generation | 89.95/71.40 | 90.51/72.51 | -0.56/-1.11 |
| Qwen3-VL-8B | visual | 88.14/60.39 | 86.60/58.29 | +1.54/+2.10 |
| Qwen3-VL-8B | vp_generation | 90.74/67.49 | 86.60/58.29 | +4.14/+9.20 |
| InternVL2.5-8B | visual | 86.20/53.49 | 86.90/54.09 | -0.71/-0.60 |
| InternVL2.5-8B | vp_generation | 86.98/55.95 | 86.90/54.09 | +0.07/+1.86 |

## 联合种子逐次测试结果

| 模型 | 分类器 | seed43 AUROC/AP | seed44 AUROC/AP | seed45 AUROC/AP |
|---|---|---:|---:|---:|
| Qwen2.5-VL-7B | SVAR | 86.53/38.03 | 86.89/42.39 | 88.53/46.00 |
| Qwen2.5-VL-7B | MetaToken LR | 83.81/40.49 | 80.78/41.31 | 84.53/44.92 |
| Qwen2.5-VL-7B | MetaToken GB | 82.97/35.76 | 81.84/40.74 | 83.49/41.57 |
| LLaVA-1.5-7B | SVAR | 90.78/72.59 | 91.28/76.62 | 89.46/68.32 |
| LLaVA-1.5-7B | MetaToken LR | 88.68/68.85 | 88.66/71.95 | 88.93/68.19 |
| LLaVA-1.5-7B | MetaToken GB | 89.54/71.46 | 89.18/72.29 | 89.77/71.11 |
| Qwen3-VL-8B | SVAR | 86.99/58.54 | 86.85/60.53 | 85.95/55.82 |
| Qwen3-VL-8B | MetaToken LR | 83.60/46.12 | 82.94/49.51 | 83.29/47.47 |
| Qwen3-VL-8B | MetaToken GB | 85.03/53.25 | 83.70/56.08 | 84.78/54.30 |
| InternVL2.5-8B | SVAR | 86.29/49.96 | 86.87/53.65 | 87.55/58.66 |
| InternVL2.5-8B | MetaToken LR | 83.38/50.38 | 81.18/42.33 | 84.74/49.91 |
| InternVL2.5-8B | MetaToken GB | 84.96/49.45 | 84.11/52.20 | 84.99/51.04 |

三个联合test集彼此不同；固定划分三次共享同一test集。MetaToken LR在固定划分下跨训练seed完全相同，是因为LBFGS流程确定；联合种子下的差异来自数据划分。只有三个新划分，不作显著性结论。
