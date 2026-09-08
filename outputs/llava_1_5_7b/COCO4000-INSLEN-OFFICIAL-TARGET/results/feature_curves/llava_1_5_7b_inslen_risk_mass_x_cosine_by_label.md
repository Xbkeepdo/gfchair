# llava_1_5_7b risk 与 mass × cosine 分标签逐层曲线

## 口径

- 标签：`0=hallucination`，`1=real/non-hallucination`。
- 上排曲线：每层类别均值，阴影为类别均值的 95% CI。
- 下排曲线：`Hall−Real`，阴影为两组独立均值差的近似 95% CI。
- `P=hmid/hpre` 只改变 risk 的 source marginal；同一 gate 的 `mass × cosine` 是 target-side EV，因此不随 P-source 或 transport cost 重复。
- 层号采用保存数组的 0-based decoder layer index。
- 输入分片：
  - `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：7796 rows
  - `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：7667 rows
- 逐层完整数值：`llava_1_5_7b_inslen_risk_mass_x_cosine_by_label.csv`。
- 图片/PDF：
  - `raw_risk`：`llava_1_5_7b_inslen_risk_mass_x_cosine_by_label_raw_risk.png` / `llava_1_5_7b_inslen_risk_mass_x_cosine_by_label_raw_risk.pdf`
  - `softmax_risk`：`llava_1_5_7b_inslen_risk_mass_x_cosine_by_label_softmax_risk.png` / `llava_1_5_7b_inslen_risk_mass_x_cosine_by_label_softmax_risk.pdf`
  - `mass_x_cosine`：`llava_1_5_7b_inslen_risk_mass_x_cosine_by_label_mass_x_cosine.png` / `llava_1_5_7b_inslen_risk_mass_x_cosine_by_label_mass_x_cosine.pdf`

## 汇总

| 特征 | Hall n | Real n | Hall 全层均值 | Real 全层均值 | 平均 Hall−Real | 最大差层 | 该层差值 | 最大 |d| 层/值 | 95% CI 不跨零层数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw logit · P=hmid · sqrt cost | 3487 | 11976 | 0.381927 | 0.367196 | +0.014730 | L20 | +0.048236 | L20 / +0.718 | 28/32 |
| Raw logit · P=hmid · cosine cost | 3487 | 11976 | 0.482931 | 0.461450 | +0.021481 | L7 | +0.061785 | L20 / +0.664 | 26/32 |
| Raw logit · P=hpre · sqrt cost | 3487 | 11976 | 0.379679 | 0.360938 | +0.018741 | L7 | +0.047575 | L20 / +0.712 | 29/32 |
| Raw logit · P=hpre · cosine cost | 3487 | 11976 | 0.479942 | 0.452499 | +0.027443 | L7 | +0.072465 | L20 / +0.654 | 29/32 |
| Softmax · P=hmid · sqrt cost | 3487 | 11976 | 0.340394 | 0.333912 | +0.006482 | L20 | +0.044010 | L20 / +0.651 | 30/32 |
| Softmax · P=hmid · cosine cost | 3487 | 11976 | 0.414357 | 0.405851 | +0.008506 | L25 | +0.058415 | L20 / +0.602 | 30/32 |
| Softmax · P=hpre · sqrt cost | 3487 | 11976 | 0.333521 | 0.322495 | +0.011026 | L20 | +0.042060 | L20 / +0.639 | 32/32 |
| Softmax · P=hpre · cosine cost | 3487 | 11976 | 0.404340 | 0.388996 | +0.015344 | L25 | +0.059724 | L21 / +0.587 | 29/32 |
| Raw logit · mass × cosine | 3487 | 11976 | 0.118743 | 0.139755 | -0.021012 | L28 | -0.050974 | L18 / -0.899 | 32/32 |
| Softmax · mass × cosine | 3487 | 11976 | 0.105377 | 0.128730 | -0.023353 | L28 | -0.050028 | L30 / -0.978 | 31/32 |

## 曲线观察

- Risk 的整体方向是 Hall 高于 Real，但不是每层同号。raw-logit risk 的正向分离更连续；softmax risk 在浅层和 L28–L31 有多次反向。八条 risk 的共同强区间是 L19–L27，最大标准化效应大多位于 L20/L21，`|d|≈0.59–0.72`。
- `P=hpre` 和 cosine cost 会放大 risk 的全层平均绝对差。例如 raw/cosine 的平均 Hall−Real 从 P=hmid 的 `+0.02148` 增至 P=hpre 的 `+0.02744`；但它也改变逐层尺度和协方差，不能仅凭均值差推出 MLP AUROC 一定更高。
- 两条 `mass × cosine` 曲线形状高度相似：L1–L5 是很小的 Hall>Real，L6 后方向翻转，L18–L30 的 Real>Hall 最明显；最大绝对均值差都在 L28，约 `-0.05`。标准化效应比单个 risk 层更强，raw 在 L18 为 `d=-0.899`，softmax 在 L30 为 `d=-0.978`。
- 因此 risk 和 `mass × cosine` 提供方向相反、层区间不同的信号：高 risk 倾向 Hall，高 target mass×cosine 倾向 Real。这与 `risk+EV` 比 risk-only 的 MLP 表现稳定提升一致。
- 样本量为 15,463，极小差异也可能让 95% CI 不跨零；解读时应优先看差值幅度、Cohen's d 和下游 AUROC，而不是只看显著层数。
