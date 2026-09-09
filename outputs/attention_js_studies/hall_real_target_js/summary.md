# Attention distribution / evidence：HALL–REAL 逐层 JS

主结果在同一图片内对唯一 HALL 与 REAL 目标 token 配对；每张图片先平均，再跨图片平均。JS 使用自然对数。Top-32 使用两侧各自 Top-32 的并集，并在并集内重新归一化。

| Model | Targets H/R | Images with H–R pairs | A H–R JS all / UTop32 | E H–R JS all / UTop32 | Same-token JS(A,E) H all / UTop32 | Same-token JS(A,E) R all / UTop32 |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | 3429/11500 | 1966 | 0.1210 / 0.1374 | 0.2122 / 0.2436 | 0.0481 / 0.0541 | 0.0438 / 0.0498 |
| Qwen2.5-VL-7B | 916/7738 | 703 | 0.2043 / 0.2438 | 0.2310 / 0.2635 | 0.0183 / 0.0137 | 0.0178 / 0.0127 |
| Qwen3-VL-8B | 2591/12111 | 1731 | 0.2681 / 0.3010 | 0.3265 / 0.3550 | 0.0231 / 0.0200 | 0.0210 / 0.0172 |
| InternVL2.5-8B | 1759/9864 | 1265 | 0.1981 / 0.2336 | 0.2568 / 0.2905 | 0.0280 / 0.0267 | 0.0271 / 0.0248 |

## 主要发现

- **LLaVA-1.5-7B**：E 相对 A 将 H–R JS 提高 `+0.0911`（全视觉）/`+0.1063`（Union-Top32）；A/E 全视觉峰值分别为 L20=`0.2495`、L20=`0.3256`。同 token gate-effect 的 H–R 差仅 `+0.0044`；plain attention 的 H–R entropy/Top-32 mass 差为 `-0.0143`/`+0.0105`。
- **Qwen2.5-VL-7B**：E 相对 A 将 H–R JS 提高 `+0.0267`（全视觉）/`+0.0197`（Union-Top32）；A/E 全视觉峰值分别为 L23=`0.3569`、L27=`0.3780`。同 token gate-effect 的 H–R 差仅 `+0.0004`；plain attention 的 H–R entropy/Top-32 mass 差为 `-0.0212`/`+0.0294`。
- **Qwen3-VL-8B**：E 相对 A 将 H–R JS 提高 `+0.0584`（全视觉）/`+0.0539`（Union-Top32）；A/E 全视觉峰值分别为 L22=`0.4481`、L22=`0.4890`。同 token gate-effect 的 H–R 差仅 `+0.0021`；plain attention 的 H–R entropy/Top-32 mass 差为 `-0.0236`/`+0.0091`。
- **InternVL2.5-8B**：E 相对 A 将 H–R JS 提高 `+0.0587`（全视觉）/`+0.0569`（Union-Top32）；A/E 全视觉峰值分别为 L25=`0.3165`、L26=`0.3767`。同 token gate-effect 的 H–R 差仅 `+0.0009`；plain attention 的 H–R entropy/Top-32 mass 差为 `-0.0148`/`+0.0126`。

同图、同标签目标对照（跨层均值，全视觉 token）：

| Model | A: H–R / H–H / R–R | E: H–R / H–H / R–R |
|---|---:|---:|
| LLaVA-1.5-7B | 0.1210 / 0.0690 / 0.1149 | 0.2122 / 0.1586 / 0.2032 |
| Qwen2.5-VL-7B | 0.2043 / 0.1876 / 0.1811 | 0.2310 / 0.2094 / 0.2105 |
| Qwen3-VL-8B | 0.2681 / 0.2444 / 0.2422 | 0.3265 / 0.2980 / 0.2994 |
| InternVL2.5-8B | 0.1981 / 0.1721 / 0.1800 | 0.2568 / 0.2235 / 0.2350 |

四模型的 H–R 均略高于 R–R，但 E 同时提高了 H–R、H–H 和 R–R 的不同目标 JS；因此 E 的更高 H–R JS 主要说明 gate 让目标条件分布更分化，不能单独解释为幻觉特异性。

## 每张图是什么意思

1. **`same_image_hall_real_js_attention_distribution`**：只画 A。横轴是 decoder layer，纵轴是在同一图片内 HALL token 与 REAL token 的平均 JS。实线使用全部视觉 token，虚线使用双方 Top-32 并集。越高表示两个目标词的 plain attention 空间分布越不同，但不表示哪一个更正确。
2. **`same_image_hall_real_js_attention_evidence`**：只画 E，坐标和配对方式与上一图相同。与 A 图相比整体升高，表示 gate 加权后不同目标词的证据分布更分化；这本身不等于更强的幻觉检测能力。
3. **`same_token_attention_to_evidence_js`**：比较同一个 token 的 A 与 E。红色是 HALL，绿色是 REAL；实线为全部视觉 token，虚线为 Union-Top32。值越高表示 semantic gate 对原 attention 的重排越强；红绿差才是这种重排是否与幻觉标签相关。
4. **`normalized_entropy_attention_distribution` / `normalized_entropy_attention_evidence`**：A、E 分开画。红色 HALL、绿色 REAL。值接近 1 表示在视觉 token 上更均匀，值越低表示越集中。
5. **`top32_mass_attention_distribution` / `top32_mass_attention_evidence`**：A、E 分开画，显示各分布自己的 Top-32 token 承载了多少概率质量；越高表示头部越集中。该指标受视觉 token 总数影响，适合在同一模型内比较层和标签，不宜直接比较模型间绝对值。

## 口径

- `A`/`attention_distribution` 使用主特征中的 `dgst_t_attention_support_per_layer`；它与 LLaVA/InternVL 二轮 shard 的同名量抽样误差不超过 FP32 舍入。Qwen 没有二轮 shard，因此使用这一同定义来源。
- `E`/`attention_evidence = normalize(A × hpre_raw_logit_gauss_gate)`。
- `same-image HALL–REAL JS` 衡量同图不同目标词的空间分布差异；它不是 HALL/REAL 分类器 AUROC。
- 归一化 attention 不保留原始 total visual-attention mass；本分析不能恢复该量。

## 产物

- `same_image_hall_real_js_layerwise.csv`：H–R/H–H/R–R 同图配对 JS。
- `same_token_attention_to_evidence_js_layerwise.csv`：语义 gate 对同一 token 注意力的改变量。
- `attention_signal_shape_layerwise.csv`：两种分布的 normalized entropy 与 own-Top32 mass。
- 七组分离 PNG/PDF：A/E 主 JS、同 token gate 改变量、A/E entropy、A/E Top-32 mass；旧 combined 图保留作追溯。
