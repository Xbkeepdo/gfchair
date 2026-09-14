# SVAR隐藏层248→256维对照

唯一改动：`Linear(D,248)-ReLU-Linear(248,2)`改为`Linear(D,256)-ReLU-Linear(256,2)`。SVAR输入、Adam lr0.001、batch32、max50、validation-loss patience5、无标准化/BN/dropout/调度/加权采样均不变。
固定划分为split seed20260912；联合种子中seed43/44/45分别重划全部4000图为3200/400/400，同时控制训练。每个seed单独计算test AUROC/HALL-AUPR；均值±总体std，不是ensemble。

| 模型 | 划分 | 248 AUROC/AP | 256 AUROC/AP | 256−248 AUROC/AP |
|---|---|---:|---:|---:|
| Qwen2.5-VL-7B | fixed_split | 87.34±0.22/47.06±0.66 | 87.30±0.30/47.35±0.63 | -0.04/+0.29 |
| Qwen2.5-VL-7B | joint_split_training | 87.32±0.87/42.14±3.26 | 87.36±0.83/43.74±4.52 | +0.04/+1.61 |
| LLaVA-1.5-7B | fixed_split | 90.44±0.04/70.96±0.53 | 90.33±0.08/70.53±0.54 | -0.11/-0.43 |
| LLaVA-1.5-7B | joint_split_training | 90.51±0.77/72.51±3.39 | 90.50±0.63/72.79±3.06 | -0.01/+0.28 |
| Qwen3-VL-8B | fixed_split | 88.95±0.23/62.66±0.39 | 88.68±0.02/60.99±0.52 | -0.27/-1.67 |
| Qwen3-VL-8B | joint_split_training | 86.60±0.46/58.29±1.93 | 86.53±0.32/57.32±1.17 | -0.06/-0.97 |
| InternVL2.5-8B | fixed_split | 86.98±0.61/53.73±1.14 | 87.80±0.30/55.09±0.45 | +0.82/+1.36 |
| InternVL2.5-8B | joint_split_training | 86.90±0.51/54.09±3.57 | 86.84±0.57/53.85±3.91 | -0.06/-0.24 |

## 结论

- 固定划分中，InternVL提升最明显：AUROC +0.82、HALL-AUPR +1.36个百分点；Qwen3则下降0.27和1.67个百分点。
- 联合种子中，Qwen2.5的HALL-AUPR提高1.61个百分点，但AUROC只提高0.04；LLaVA、Qwen3和InternVL的AUROC变化都在±0.06个百分点内。
- 四模型平均后，固定划分AUROC约+0.10、HALL-AUPR约−0.12个百分点；联合种子AUROC约−0.02、HALL-AUPR约+0.17个百分点。256维没有跨模型一致收益。

差值没有用于选模型或调参。248与256在同一regime/seed使用同一数据划分和minibatch顺序，但初始化张量形状变化，所以随机权重序列不可能逐参数配对；这里衡量完整训练配置的宽度替换效果。
