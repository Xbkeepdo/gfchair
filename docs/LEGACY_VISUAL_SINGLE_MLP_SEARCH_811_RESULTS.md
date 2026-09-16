# Visual-only 条件路径单层 MLP 搜参结果（811）

## 设置

- 特征固定为 `legacy_visual=[AE_V, log1p(S_E)]`：`AE_V` 是视觉 attention mass，`S_E` 是真实 RMS Visual-only 条件路径 `z-A_V→z` 上各视觉 source token 的 FFN 响应范数和。
- 直接复用已有四模型缓存，没有重新生成回答、抽取 attention 或计算路径积分。
- 固定图片级 3200 train / 400 validation / 400 test，保留全部 mentions；训练 seeds 43/44/45。
- Torch 单层 MLP 使用此前 `SINGLE_MLP_SEARCH_811` 的同一组24候选。每模型先用 seed43 在 validation 上筛前三名，再补 seeds44/45，以三seed validation AUROC、HALL-AUPR冻结赢家；四模型全部冻结后才读取 test。
- XGB 是同一811特征上已经完成的18候选验证搜参结果，未重复训练。SVAR 是 LLaVA 第5–18层、其他模型按深度同比例映射的原生头结果。
- 以下均为三个训练seed的 test 均值 ± 总体标准差，单位为百分数，不是概率ensemble。

完整预注册设置见 [协议](LEGACY_VISUAL_SINGLE_MLP_SEARCH_811_PROTOCOL.md)。

## 结果

| 模型 | Torch单层搜参 AUROC / HALL-AUPR | 早期三层MLP | 早期sklearn单层 | 搜参XGB | 原生SVAR |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | **87.95±0.09 / 45.73±0.80** | 85.69±0.77 / 37.21±1.07 | 82.69±0.52 / 33.01±1.53 | 85.72±0.00 / 38.84±0.00 | 86.53±0.39 / 43.77±0.99 |
| LLaVA-1.5-7B | 89.74±0.02 / 68.37±0.26 | 90.09±0.14 / **69.52±0.12** | 88.76±0.46 / 66.22±0.84 | **90.24±0.00** / 69.13±0.00 | 90.40±0.01 / 71.15±0.36 |
| Qwen3-VL-8B | 89.81±0.08 / **68.45±0.54** | **90.12±0.14** / 66.95±0.64 | 87.91±0.39 / 62.01±1.85 | 89.36±0.00 / 63.89±0.00 | 89.31±0.08 / 63.66±0.82 |
| InternVL2.5-8B | 86.69±0.25 / 53.70±0.67 | **86.73±0.51** / 54.51±1.07 | 85.94±0.26 / **56.28±0.08** | 85.62±0.00 / 51.42±0.00 | 87.65±0.26 / 54.94±0.81 |
| 四模型宏平均 | **88.55 / 59.06** | 88.16 / 57.05 | 86.33 / 54.38 | 87.73 / 55.82 | 88.47 / 58.38 |

表内加粗只标记四种 Visual-only 分类头中的逐指标最高值；SVAR 单列作为外部方法对照。

## 差值与解读

| 模型 | Torch单层 − 早期三层（AUROC/AP pp） | Torch单层 − XGB | Torch单层 − SVAR |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | **+2.26 / +8.52** | **+2.23 / +6.89** | **+1.42 / +1.96** |
| LLaVA-1.5-7B | −0.36 / −1.15 | −0.50 / −0.77 | −0.67 / −2.78 |
| Qwen3-VL-8B | −0.31 / **+1.51** | **+0.45 / +4.57** | **+0.50 / +4.79** |
| InternVL2.5-8B | −0.05 / −0.81 | **+1.07 / +2.28** | −0.96 / −1.24 |
| 四模型宏平均 | **+0.39 / +2.02** | **+0.81 / +3.24** | **+0.07 / +0.68** |

- Qwen2.5 是明确受益的模型：新 Torch 单层头同时超过早期三层、XGB和SVAR。
- Qwen3 的新头提高HALL-AUPR，但早期三层头的AUROC仍高0.31个百分点；新头仍同时超过SVAR两项指标。
- LLaVA 不受益，已有XGB给出Visual-only最高AUROC，三层头给出最高HALL-AUPR，两者仍低于SVAR对应指标。
- InternVL的新头超过XGB，但与三层头基本持平且低于SVAR；早期sklearn单层的AP最高，同时AUROC较低。
- 宏平均上，新Torch单层比早期三层提高0.39 AUROC和2.02 HALL-AUPR个百分点，只略高于SVAR0.07/0.68个百分点。提升不跨模型一致，不能概括为单层头普遍更适合Visual-only条件路径。

## 验证集选择的参数

- Qwen2.5、LLaVA：width512、ReLU、BN、无标准化、dropout0.1、lr0.001、wd1e-5、batch256、监控validation loss。
- Qwen3：width128、ReLU、BN、训练集标准化、dropout0.3、lr0.001、wd1e-5、batch256、监控validation loss。
- InternVL：width128、ReLU、无BN、训练集标准化、dropout0.3、lr0.003、wd1e-4、batch256、监控validation AUROC。

本轮固定400图测试集此前已多次访问，属于探索性结果；不同分类器的搜参预算也不同。不能据此宣称独立测试集上的显著优势或把全部差异单独归因于分类器结构。

## 产物与核验

- 汇总：`outputs/legacy_visual_single_mlp_search_811_v1/summary.md`、`summary.json`、`comparison.csv`
- 每模型：`protocol.json`、`selection.json`、30个验证拟合、3个最终头及逐seed指标
- 独立核验：12个最终头均从保存权重在CPU重载；最大概率绝对误差 `2.3841858e-7`，所有测试指标复算一致；`validation.json` 为 `PASS`
- 代码：`scripts/search_legacy_visual_single_mlp_811.py`
