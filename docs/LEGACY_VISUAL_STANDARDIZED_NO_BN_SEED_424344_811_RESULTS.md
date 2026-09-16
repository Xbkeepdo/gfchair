# Visual-only统一标准化后去BN结果（固定811，seeds42/43/44）

## 设置

- 特征仍为Visual-only条件路径`legacy_visual=[AE_V,log1p(S_E)]`；固定图片级3200 train / 400 validation / 400 test、全部mentions及seeds42/43/44。
- 对照基线为上一轮统一train-only逐列z-score的冻结单层MLP。
- 保留width、dropout、activation、learning rate、weight decay、batch size、monitor、scheduler及early stopping，仅把`batch_norm=true`改为false。
- Qwen2.5、LLaVA、Qwen3各重新训练3个种子；InternVL原本已经无BN，直接复用3个结果。

预设见`docs/LEGACY_VISUAL_STANDARDIZED_NO_BN_SEED_424344_811_PROTOCOL.md`。

## 去BN前后

三seed测试AUROC和HALL-AUPR均值±总体标准差，单位为%。

| 模型 | 标准化+无BN AUROC / AP | 标准化原配置 AUROC / AP | 去BN变化（AUROC / AP pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **88.14±0.16 / 41.53±0.63** | 88.21 / 43.31 | **−0.08 / −1.77** |
| LLaVA-1.5-7B | **89.74±0.09 / 68.45±0.06** | 89.97 / 68.73 | **−0.23 / −0.29** |
| Qwen3-VL-8B | **89.84±0.08 / 67.85±0.44** | 89.68 / 68.26 | **+0.16 / −0.42** |
| InternVL2.5-8B | **86.66±0.22 / 53.59±0.62** | 86.66 / 53.59 | 已无BN |
| 四模型宏平均 | **88.59 / 57.86** | 88.63 / 58.47 | **−0.04 / −0.62** |

三个实际去BN的模型AP全部下降。AUROC在Qwen2.5和LLaVA下降，Qwen3提高0.16个百分点。宏平均AUROC几乎不变，但AP下降0.62个百分点；因此当前冻结超参数下不支持统一去BN，保留原BN配置更好。

## 与标准化SVAR比较

| 模型 | Visual-only标准化+无BN | SVAR标准化 | Visual-only−SVAR（AUROC / AP pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **88.14 / 41.53** | 85.19 / 39.46 | **+2.94 / +2.08** |
| LLaVA-1.5-7B | 89.74 / 68.45 | **90.21 / 70.39** | −0.47 / −1.94 |
| Qwen3-VL-8B | **89.84 / 67.85** | 89.26 / 63.29 | **+0.58 / +4.55** |
| InternVL2.5-8B | 86.66 / **53.59** | **87.11** / 53.21 | −0.45 / +0.39 |
| 四模型宏平均 | **88.59 / 57.86** | 87.94 / 56.59 | **+0.65 / +1.27** |

无BN版本宏平均仍高于标准化SVAR，但模型方向不一致：Qwen2.5/Qwen3两项均胜，LLaVA两项均负，InternVL为AUROC低、AP略高。

## 边界与核验

本轮冻结其他超参数，仅切换BN；没有为无BN网络重新搜索学习率、正则化或宽度。固定test此前已访问，属于探索性消融。

- 输出：`outputs/legacy_visual_standardized_no_bn_seed424344_811_v1/`
- 入口：`scripts/evaluate_legacy_visual_standardized_no_bn_seed_424344_811.py`
- 12个头CPU重载和指标复算通过；9次新训练、3次原配置复用。
- mean/scale相对train-only重算最大误差均为0；最大概率绝对误差`1.7881393e-7`；`validation.json`为`PASS`。
