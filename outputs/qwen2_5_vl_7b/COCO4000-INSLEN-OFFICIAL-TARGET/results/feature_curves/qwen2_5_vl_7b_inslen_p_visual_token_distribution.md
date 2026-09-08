# qwen2_5_vl_7b：P 在视觉 token 上的分布统计

- `P=hmid`：`softmax(cos(o_ffn(q_t), hmid(v_i)) / 0.07)`。
- `P=hpre`：`softmax(cos(o_ffn(q_t), hpre(v_i)) / 0.07)`。
- 每个目标对象、每层作为一个样本；`0=Hall`、`1=Real`。
- 原始 P 的 token 均值恒为约 `1/N_v`，因此同时统计 `N_v P_i`；其均匀分布基线为 1。
- 曲线统计对每个正式对象等权；直方图对每个视觉 token 等权。
- raw/softmax target gate 与 sqrt/cosine transport cost 不改变 P，故没有重复绘制。

## 输入与产物

- `outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：4,319 条。
- `outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：4,398 条。
- 逐层指标：`qwen2_5_vl_7b_inslen_p_visual_token_distribution_metrics.csv`。
- token 直方图：`qwen2_5_vl_7b_inslen_p_visual_token_distribution_histogram.csv`。
- 指标曲线：`qwen2_5_vl_7b_inslen_p_visual_token_distribution_metrics.png`。
- 分布热图：`qwen2_5_vl_7b_inslen_p_visual_token_distribution_histogram_heatmap.png`。

## 全层平均

| P source | 标签 | 样本数 | 平均视觉 token 数 | token 数范围 | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | hall | 916 | 354.49 | 99–529 | 0.015102 | 5.241017 | 0.198029 | 0.976763 | 0.886171 |
| P=hmid | real | 7,801 | 355.19 | 63–529 | 0.015838 | 5.521578 | 0.198136 | 0.975515 | 0.883759 |
| P=hpre | hall | 916 | 354.49 | 99–529 | 0.018977 | 6.618720 | 0.208026 | 0.969972 | 0.872370 |
| P=hpre | real | 7,801 | 355.19 | 63–529 | 0.020218 | 7.081852 | 0.208913 | 0.967937 | 0.869428 |

## Hall − Real

| P source | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 | 最大熵差层 |
|---|---:|---:|---:|---:|---:|---:|
| P=hmid | -0.000737 | -0.280561 | -0.000107 | +0.001248 | +0.002412 | L28 (+0.032359) |
| P=hpre | -0.001241 | -0.463132 | -0.000886 | +0.002035 | +0.002942 | L28 (+0.047782) |

注：Hall−Real 的熵为负、Top-K mass/Pmax 为正，均表示 Hall 的 P 更集中；反之表示 Real 更集中。
