# 已废弃：错误解释的 SVAR 5%–55% 对照

> 这份结果把用户指定的 LLaVA 层 5–18 错误解释成了模型深度 5%–55%，不再用于汇报。修正结果见 `OURS_V_VPG_VS_SVAR_LLAVA_5_18_PROPORTIONAL_811_RESULTS.md`。

## 实验设置

- 数据：固定图片级 8:1:1 划分，3200/400/400 张图；划分种子 20260912；保留全部 mentions。
- 结果：测试集 AUROC 与 HALL-AUPR；表中为训练种子 43/44/45 的算术均值 ± 总体标准差，不是三模型概率集成结果。
- 我们的信号：True-RMS all-attention 路径 `z-A_all -> z`，局部 FP32，Gauss–Legendre K32。
  - V：逐层 `[AE_V, log1p(S_V^all)]`。
  - VP+G：逐层 `[AE_VP, log1p(S_V^all+S_P^all), AE_G, log1p(S_G^all)]`。
- 我们的分类器：各模型沿用验证集选择的单隐层 MLP 参数；仅用训练集统计量做逐列 StandardScaler；无 BatchNorm；训练完整 150 epochs，最终取最低 validation BCE loss 的 checkpoint。V 与 VP+G 在同一模型内使用完全相同的分类器参数。
- SVAR 特征：每个选中 decoder 层、每个 attention head 的视觉 attention mass，按层和 head 展平。
- SVAR 分类器：`Linear(D,248)-ReLU-Linear(248,2)`；Adam，lr=0.001，batch=32，最多 50 epochs，无输入标准化、BatchNorm、dropout、weight decay；最低 validation CE checkpoint，patience=5。
- SVAR 层区间：`start=floor(0.05L)`、`end=ceil(0.55L)`，采用零基半开区间 `[start,end)`。
- 本结果是在查看测试结果后的探索性跟进，不作为未触碰测试集的确认性比较。

## 主结果

括号中的差值为“该方法减 SVAR”；正值表示高于 SVAR。

| 模型 | 特征 | AUROC（均值 ± SD） | Δ AUROC | HALL-AUPR（均值 ± SD） | Δ HALL-AUPR |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | V | **0.8796 ± 0.0016** | **+0.0176** | 0.4223 ± 0.0081 | +0.0123 |
|  | VP+G | 0.8668 ± 0.0033 | +0.0049 | **0.4620 ± 0.0060** | **+0.0521** |
|  | SVAR 5%–55% | 0.8620 ± 0.0015 | — | 0.4099 ± 0.0065 | — |
| LLaVA-1.5-7B | V | 0.8971 ± 0.0013 | -0.0047 | 0.6843 ± 0.0025 | -0.0168 |
|  | VP+G | 0.8994 ± 0.0007 | -0.0025 | **0.7176 ± 0.0021** | **+0.0165** |
|  | SVAR 5%–55% | **0.9019 ± 0.0009** | — | 0.7011 ± 0.0022 | — |
| Qwen3-VL-8B | V | 0.9000 ± 0.0012 | +0.0050 | 0.6778 ± 0.0007 | +0.0551 |
|  | VP+G | **0.9297 ± 0.0006** | **+0.0348** | **0.7210 ± 0.0014** | **+0.0983** |
|  | SVAR 5%–55% | 0.8950 ± 0.0005 | — | 0.6226 ± 0.0079 | — |
| InternVL2.5-8B | V | 0.8545 ± 0.0014 | -0.0159 | 0.5305 ± 0.0022 | -0.0120 |
|  | VP+G | **0.8814 ± 0.0027** | **+0.0111** | **0.5763 ± 0.0063** | **+0.0338** |
|  | SVAR 5%–55% | 0.8703 ± 0.0046 | — | 0.5425 ± 0.0119 | — |
| 四模型宏平均 | V | 0.8828 | +0.0005 | 0.5787 | +0.0097 |
|  | VP+G | **0.8943** | **+0.0120** | **0.6192** | **+0.0502** |
|  | SVAR 5%–55% | 0.8823 | — | 0.5691 | — |

## SVAR 实际层范围

| 模型 | decoder 总层数 | 零基层范围 | 按自然序号表述 | 层数 | 输入维度 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 28 | `[1,16)` | 第 2–16 层 | 15 | 420 |
| LLaVA-1.5-7B | 32 | `[1,18)` | 第 2–18 层 | 17 | 544 |
| Qwen3-VL-8B | 36 | `[1,20)` | 第 2–20 层 | 19 | 608 |
| InternVL2.5-8B | 32 | `[1,18)` | 第 2–18 层 | 17 | 544 |

## 结论

- 按 AUROC，V 高于 SVAR 的模型是 Qwen2.5 和 Qwen3；VP+G 高于 SVAR 的模型是 Qwen2.5、Qwen3 和 InternVL。LLaVA 上 SVAR 最高，但只比 VP+G 高 0.0025。
- 按 HALL-AUPR，VP+G 在四个模型上都高于 SVAR；提升为 +0.0521、+0.0165、+0.0983、+0.0338。V 在 Qwen2.5 和 Qwen3 上高于 SVAR，在 LLaVA 和 InternVL 上较低。
- VP+G 相比 V 的收益具有模型差异：Qwen3 和 InternVL 两项指标都明显提升；LLaVA 主要提升 HALL-AUPR；Qwen2.5 的 AUROC 降低 0.0128，但 HALL-AUPR 提升 0.0398。因此 Qwen2.5 若以 AUROC 为首要指标应选 V；若更重视少数 HALL 类的排序质量，VP+G 更合适。
- 四模型宏平均上，VP+G 比 SVAR 高 0.0120 AUROC 和 0.0502 HALL-AUPR；V 与 SVAR 的 AUROC 基本相同（+0.0005），HALL-AUPR 高 0.0097。
- 该比较反映当前完整训练管线：我们的 MLP 参数按模型由验证集选择，而 SVAR 使用固定原生分类头。它能回答“当前方法整体效果是否优于原生 SVAR”，不能单独归因于特征本身。

原始结果位于：

- `outputs/v_vs_vpg_standardized_no_bn_811_v1/summary.json`
- `outputs/svar_fraction_5_55_811_v1/summary.json`
