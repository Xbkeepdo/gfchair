# V、VP+G 与 SVAR LLaVA 5–18 层比例对照（811）

## 实验设置

- 数据：固定图片级 8:1:1 划分，3200/400/400 张图；划分种子 20260912；保留全部 mentions。
- 结果：测试集 AUROC 与 HALL-AUPR；表中是训练种子 43/44/45 的算术均值 ± 总体标准差，不是概率集成结果。
- 我们的信号：True-RMS all-attention 路径 `z-A_all -> z`，局部 FP32，Gauss–Legendre K32。
  - V：逐层 `[AE_V, log1p(S_V^all)]`。
  - VP+G：逐层 `[AE_VP, log1p(S_V^all+S_P^all), AE_G, log1p(S_G^all)]`。
- 我们的分类器：各模型沿用验证集选出的单隐层 MLP 参数；训练集逐列 StandardScaler；无 BatchNorm；训练完整 150 epochs，保留最低 validation BCE loss checkpoint。Qwen2.5 的 VP+G 使用其独立19候选验证搜参赢家；其余表项沿用四模型 V/VP+G 受控实验参数。
- SVAR 特征：选中 decoder 层中各 attention head 的视觉 attention mass，按层和 head 展平。
- SVAR 分类器：`Linear(D,248)-ReLU-Linear(248,2)`；Adam，lr=0.001，batch=32，最多 50 epochs，无输入标准化、BatchNorm、dropout、weight decay；最低 validation CE checkpoint，patience=5。
- SVAR 参考范围：LLaVA 使用配置中的零基层号 5–18（含首尾），即 `[5,19)`。其他模型按 decoder 总深度缩放边界：`start=round(L*5/32)`、`end_exclusive=round(L*19/32)`。
- 本结果是在查看测试结果后的探索性跟进，不作为未触碰测试集的确认性比较。

## 主结果

差值为“该方法减 SVAR”；正值表示高于 SVAR。

| 模型 | 特征 | AUROC（均值 ± SD） | Δ AUROC | HALL-AUPR（均值 ± SD） | Δ HALL-AUPR |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | V | **0.8796 ± 0.0016** | **+0.0143** | 0.4223 ± 0.0081 | -0.0155 |
|  | VP+G（独立搜参） | 0.8720 ± 0.0004 | +0.0066 | **0.4776 ± 0.0079** | **+0.0399** |
|  | SVAR 比例层 | 0.8653 ± 0.0039 | — | 0.4377 ± 0.0099 | — |
| LLaVA-1.5-7B | V | 0.8971 ± 0.0013 | -0.0069 | 0.6843 ± 0.0025 | -0.0272 |
|  | VP+G | 0.8994 ± 0.0007 | -0.0046 | **0.7176 ± 0.0021** | **+0.0061** |
|  | SVAR 层 5–18 | **0.9040 ± 0.0001** | — | 0.7115 ± 0.0036 | — |
| Qwen3-VL-8B | V | 0.9000 ± 0.0012 | +0.0069 | 0.6778 ± 0.0007 | +0.0411 |
|  | VP+G | **0.9297 ± 0.0006** | **+0.0366** | **0.7210 ± 0.0014** | **+0.0843** |
|  | SVAR 比例层 | 0.8931 ± 0.0008 | — | 0.6366 ± 0.0082 | — |
| InternVL2.5-8B | V | 0.8545 ± 0.0014 | -0.0220 | 0.5305 ± 0.0022 | -0.0189 |
|  | VP+G | **0.8814 ± 0.0027** | **+0.0049** | **0.5763 ± 0.0063** | **+0.0270** |
|  | SVAR 比例层 | 0.8765 ± 0.0026 | — | 0.5494 ± 0.0081 | — |
| 四模型宏平均 | V | 0.8828 | -0.0019 | 0.5787 | -0.0051 |
|  | VP+G | **0.8956** | **+0.0109** | **0.6231** | **+0.0393** |
|  | SVAR 比例层 | 0.8847 | — | 0.5838 | — |

## 实际层范围

表中均为代码和保存特征采用的零基层号，首尾均包含。

| 模型 | decoder 总层数 | 层号 | 半开区间 | 层数 | SVAR 输入维度 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 28 | 4–16 | `[4,17)` | 13 | 364 |
| LLaVA-1.5-7B | 32 | 5–18 | `[5,19)` | 14 | 448 |
| Qwen3-VL-8B | 36 | 6–20 | `[6,21)` | 15 | 480 |
| InternVL2.5-8B | 32 | 5–18 | `[5,19)` | 14 | 448 |

## 结论

- AUROC：V 高于 SVAR 的是 Qwen2.5、Qwen3；VP+G 高于 SVAR的是 Qwen2.5、Qwen3、InternVL。LLaVA 上 SVAR 比 VP+G 高 0.0046。
- HALL-AUPR：VP+G 在四个模型上均高于 SVAR，提升分别为 +0.0399、+0.0061、+0.0843、+0.0270。V 只有 Qwen3 高于 SVAR。
- 四模型宏平均：VP+G 比 SVAR 高 0.0109 AUROC 和 0.0393 HALL-AUPR；V 比 SVAR低 0.0019 AUROC 和 0.0051 HALL-AUPR。
- Qwen2.5 仍表现为 AUROC 选 V、HALL-AUPR 选 VP+G。Qwen3 和 InternVL 均由 VP+G 给出更好的综合结果。LLaVA 的最高 AUROC来自 SVAR，最高 HALL-AUPR 来自 VP+G。
- 这是当前完整管线的比较：我们的 MLP 参数按模型由验证集选择，SVAR 使用固定原生分类器，因此不能把全部差异单独归因于特征。

原始结果：

- `outputs/v_vs_vpg_standardized_no_bn_811_v1/summary.json`
- `outputs/qwen25_vpg_standardized_no_bn_search_811_v1/summary.json`
- `outputs/svar_llava_5_18_proportional_811_v1/summary.json`
