# Visual-only冻结单层MLP统一标准化结果（固定811，seeds42/43/44）

## 设置

- 特征为Visual-only条件路径`legacy_visual=[AE_V,log1p(S_E)]`；固定图片级3200 train / 400 validation / 400 test、全部mentions及seeds42/43/44。
- 保留上一轮由validation选出的单层MLP全部超参数，只强制逐列train-only z-score；不重新选参或合并train/validation。
- Qwen2.5和LLaVA的冻结配置原先`standardize=false`，本轮切换为true。Qwen3和InternVL原先已经是true，结果直接复用，所以它们的前后差为0不表示做了一次新的标准化消融。
- Qwen2.5/LLaVA的标准化配置分别对应原搜索网格中的同结构candidate5；复用其已有seed43并新增seeds42/44，共4次新训练。Qwen3/InternVL复用6个已有训练结果。

预设见`docs/LEGACY_VISUAL_STANDARDIZED_SEED_424344_811_PROTOCOL.md`。

## 标准化前后

三seed测试AUROC和HALL-AUPR均值±总体标准差，单位为%。

| 模型 | 统一标准化 AUROC / AP | 原冻结配置 AUROC / AP | 变化（AUROC / AP pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **88.21±0.20 / 43.31±0.46** | 87.90±0.16 / 45.75±0.83 | **+0.32 / −2.44** |
| LLaVA-1.5-7B | **89.97±0.02 / 68.73±0.40** | 89.66±0.10 / 67.57±0.87 | **+0.30 / +1.16** |
| Qwen3-VL-8B | **89.68±0.24 / 68.26±0.64** | 89.68±0.24 / 68.26±0.64 | 已标准化 |
| InternVL2.5-8B | **86.66±0.22 / 53.59±0.62** | 86.66±0.22 / 53.59±0.62 | 已标准化 |
| 四模型宏平均 | **88.63 / 58.47** | 88.47 / 58.80 | **+0.15 / −0.32** |

在实际发生开关变化的两个模型中，标准化均使AUROC约提高0.3个百分点。LLaVA的AP也提高1.16个百分点；Qwen2.5的AP反而下降2.44个百分点。因此统一标准化带来AUROC的小幅改善，但没有带来一致的AP收益。

## 与同样标准化的SVAR比较

| 模型 | Visual-only统一标准化 | SVAR标准化 | Visual-only−SVAR（AUROC / AP pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **88.21 / 43.31** | 85.19 / 39.46 | **+3.02 / +3.85** |
| LLaVA-1.5-7B | 89.97 / 68.73 | **90.21 / 70.39** | −0.24 / −1.66 |
| Qwen3-VL-8B | **89.68 / 68.26** | 89.26 / 63.29 | **+0.42 / +4.97** |
| InternVL2.5-8B | 86.66 / **53.59** | **87.11** / 53.21 | −0.45 / +0.39 |
| 四模型宏平均 | **88.63 / 58.47** | 87.94 / 56.59 | **+0.68 / +1.89** |

同样使用train-only逐特征标准化时，Visual-only在Qwen2.5和Qwen3两项指标都高于SVAR；LLaVA两项均低；InternVL表现为AUROC低、AP略高。宏平均两项均高于标准化SVAR，但提升并非跨模型一致。

## 边界与核验

本轮冻结原超参数，仅切换标准化。它回答的是现有冻结MLP加入标准化的效果，不等价于对“必须标准化”的搜索空间重新调参。固定test此前已访问，结果属于探索性对照。

- 输出：`outputs/legacy_visual_standardized_seed424344_811_v1/`
- 入口：`scripts/evaluate_legacy_visual_standardized_seed_424344_811.py`
- 12个头CPU重载和指标复算全部通过；4次新训练、8次精确配置复用。
- mean/scale相对train-only重算最大误差均为0；最大概率绝对误差`1.7881393e-7`；`validation.json`为`PASS`。
