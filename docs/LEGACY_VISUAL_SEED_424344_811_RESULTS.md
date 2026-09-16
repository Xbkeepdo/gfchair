# Visual-only 条件路径：训练种子42/43/44

## 受控设置

- 特征仍为 `legacy_visual=[AE_V,log1p(S_E)]`，固定图片级3200/400/400划分及全部mentions。
- 四模型均保持上一轮由validation冻结的Torch单层MLP候选和完整参数不变，不重新选参、不合并train/validation。
- 只新增每模型seed42训练；seed43/44直接复用上一轮保存权重。新汇总采用seeds42/43/44，并与原seeds43/44/45比较。
- 因两组共享seed43/44，均值差只反映用seed42替换seed45，不是两组三次独立重复。

## 三seed汇总

| 模型 | seeds42/43/44 AUROC / HALL-AUPR | seeds43/44/45 | 均值变化（AUROC/AP pp） |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **87.90±0.16 / 45.75±0.83** | 87.95±0.09 / 45.73±0.80 | −0.05 / +0.02 |
| LLaVA-1.5-7B | **89.66±0.10 / 67.57±0.87** | 89.74±0.02 / 68.37±0.26 | −0.07 / −0.80 |
| Qwen3-VL-8B | **89.68±0.24 / 68.26±0.64** | 89.81±0.08 / 68.45±0.54 | −0.13 / −0.19 |
| InternVL2.5-8B | **86.66±0.22 / 53.59±0.62** | 86.69±0.25 / 53.70±0.67 | −0.03 / −0.11 |
| 四模型宏平均 | **88.47 / 58.80** | 88.55 / 59.06 | −0.07 / −0.27 |

## 逐seed结果

| 模型 | seed42 AUROC/AP | seed43 AUROC/AP | seed44 AUROC/AP |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 87.69 / 46.84 | 88.07 / 45.57 | 87.93 / 44.84 |
| LLaVA-1.5-7B | 89.52 / 66.34 | 89.71 / 68.15 | 89.76 / 68.22 |
| Qwen3-VL-8B | 89.36 / 67.76 | 89.76 / 67.86 | 89.92 / 69.17 |
| InternVL2.5-8B | 86.89 / 53.79 | 86.71 / 54.23 | 86.37 / 52.76 |

## 解读

- AUROC对这次种子替换较稳定，四模型绝对变化为0.03–0.13个百分点。
- Qwen2.5的HALL-AUPR基本不变；Qwen3和InternVL分别下降0.19和0.11个百分点。
- LLaVA的HALL-AUPR下降0.80个百分点且三seed标准差由0.26升到0.87。原因是新增seed42的AP为66.34%，低于原seed45的68.73%；其AUROC仍稳定。
- 相对SVAR的定性结论不变：Visual-only搜参单层在Qwen2.5、Qwen3同时超过SVAR，在LLaVA、InternVL仍低于SVAR。

同样使用seeds42/43/44重新训练的比例层原生SVAR及逐模型直接差值，见 [SVAR同种子结果](SVAR_PROPORTIONAL_SEED_424344_811_RESULTS.md)。

本实验使用此前已访问的固定400图test，结果仍是探索性的；三seed数量不足以估计完整随机种子分布。

## 产物与核验

- 汇总：`outputs/legacy_visual_single_mlp_seed424344_811_v1/summary.md`、`summary.json`、`comparison.csv`
- 每模型逐seed指标与最终概率：`outputs/legacy_visual_single_mlp_seed424344_811_v1/<model>/`
- 12个头CPU重载复算通过；4个seed42为新训练、8个seed43/44复用。最大概率绝对误差`2.3841858e-7`，`validation.json`为`PASS`
- 入口：`scripts/evaluate_legacy_visual_seed_424344_811.py`
