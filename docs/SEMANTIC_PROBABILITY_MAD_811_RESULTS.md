# ENDAC-811：目标词视觉概率的 MAD

复用四模型已保存的目标物体首 token 词表 softmax 概率矩阵 `p[l,v,y]`，没有重新运行 VLM。每个 mention、每层在**全部视觉 token** 上计算

`MAD_l = median_v |p[l,v,y] - median_u p[l,u,y]|`。

直接计算 raw（隐藏状态不经 final Norm）和 norm（经 final Norm）两版；分别测试逐层 MAD 单独输入、以及 `[逐层 MAD, 逐层 log1p(S_E)]`。这里没有先将概率沿视觉位置归一化，因此衡量的是原词表概率的视觉位置离散度，而非前一轮 JS 使用的条件空间分布。

固定既有 ENDAC-811 的 3200/400/400 图片划分与训练种子 43/44/45。四个输入组各用单隐藏层、无 BN、train-only 标准化的 PyTorch MLP 独立运行 24 次 Optuna 搜参；前三个 seed43 候选在 44/45 上复跑，参数和 checkpoint 按验证集选，测试集只作探索性特征比较。完整三种子均值±标准差、参数和预测见 [summary.md](../outputs/coco4000_512_endac_semantic_probability_mad_optuna_811/summary.md)。

## 四模型测试结果

各单元为 AUROC / HALL-AUPR（%，三种子均值）。AE 对照采用同一拆分、同样的 Optuna 单隐藏层设置。

| 模型 | raw MAD | norm MAD | raw MAD + logS | norm MAD + logS | AE + logS |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 61.51 / 18.87 | 81.54 / 47.40 | 84.57 / 44.93 | **85.91 / 49.32** | 87.18 / 54.77 |
| LLaVA-1.5-7B | 78.23 / 49.55 | 80.18 / 51.21 | 88.81 / 70.30 | **89.11 / 70.53** | 90.39 / 72.63 |
| Qwen3-VL-8B | 81.10 / 57.57 | 80.08 / 56.40 | 89.75 / **74.04** | **89.90** / 72.75 | 89.85 / 71.75 |
| InternVL-2.5-8B | 75.23 / 41.81 | 75.93 / 44.08 | **85.93 / 62.51** | 85.91 / 60.03 | 87.81 / 62.36 |

MAD 单独使用四模型均明显弱于 AE。拼接 logS 后，Qwen3 raw MAD 的 HALL-AUPR 达 74.04%，比 AE+logS 高 2.29 个百分点，但 AUROC 低 0.10 个百分点；其余模型没有对 AE 的一致收益。Qwen2.5 最好的 norm MAD+logS 在 AUROC/AP 上仍低于 AE 1.27/5.45 个百分点。已有 JS/全视觉乘积结果见 [前轮报告](SEMANTIC_JS_FULL_811_RESULTS.md)。这些是同一已多次查看测试集上的探索性点估计，未做显著性检验；拼接后的提升也不能单独归因于 MAD，需要同预算的 logS-only 对照。

## raw 概率的数值限制

Qwen2.5 raw MAD 为零的 mention×层行有 19,372/221,872（8.73%），涉及 7,923/7,924 个 mention；最后一层有 7,923 个 mention 为零。若多数位置的缓存概率已下溢为零，少数位置非零也无法改变“偏差的中位数”。norm MAD 没有零行。现有 trainer 对训练标准差 `<1e-12` 的列将缩放因子置为 1；Qwen2.5 raw 的 11/28 层因此实际上没有标准化，Qwen3 raw 为 4/36 层、InternVL raw 为 2/32 层，四模型 norm 均为 0。raw 尤其是 Qwen2.5 不能当作完整保真概率离散度解释。

四模型共 480 次验证拟合、48 个最终测试头完成。首次使用通用 CPU 重载门槛 `1e-6` 时，最大概率差 `5.11e-6` 未过；指标差仅 `3.33e-16`。本轮显式记录跨设备容差 `1e-5` 后，48/48 checkpoint、标准化统计和指标复算通过；最大差出现在 Qwen2.5 `raw_mad_logs/seed43`，详见 `validation.json`。
