# qwen3_vl_8b：P 在视觉 token 上的分布统计

- `P=hmid`：`softmax(cos(o_ffn(q_t), hmid(v_i)) / 0.07)`。
- `P=hpre`：`softmax(cos(o_ffn(q_t), hpre(v_i)) / 0.07)`。
- 每个目标对象、每层作为一个样本；`0=Hall`、`1=Real`。
- 原始 P 的 token 均值恒为约 `1/N_v`，因此同时统计 `N_v P_i`；其均匀分布基线为 1。
- 曲线统计对每个正式对象等权；直方图对每个视觉 token 等权。
- raw/softmax target gate 与 sqrt/cosine transport cost 不改变 P，故没有重复绘制。

## 输入与产物

- `outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：7,490 条。
- `outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：7,383 条。
- 逐层指标：`qwen3_vl_8b_inslen_p_visual_token_distribution_metrics.csv`。
- token 直方图：`qwen3_vl_8b_inslen_p_visual_token_distribution_histogram.csv`。
- 指标曲线：`qwen3_vl_8b_inslen_p_visual_token_distribution_metrics.png`。
- 分布热图：`qwen3_vl_8b_inslen_p_visual_token_distribution_histogram_heatmap.png`。

## 全层平均

| P source | 标签 | 样本数 | 平均视觉 token 数 | token 数范围 | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | hall | 2,593 | 270.29 | 70–400 | 0.020865 | 5.470815 | 0.270981 | 0.956365 | 0.836403 |
| P=hmid | real | 12,280 | 269.68 | 70–400 | 0.021213 | 5.554914 | 0.274851 | 0.955412 | 0.832252 |
| P=hpre | hall | 2,593 | 270.29 | 70–400 | 0.022149 | 5.838484 | 0.266801 | 0.956172 | 0.842276 |
| P=hpre | real | 12,280 | 269.68 | 70–400 | 0.022838 | 6.015635 | 0.271388 | 0.954752 | 0.837322 |

## Hall − Real

| P source | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 | 最大熵差层 |
|---|---:|---:|---:|---:|---:|---:|
| P=hmid | -0.000348 | -0.084099 | -0.003870 | +0.000953 | +0.004151 | L34 (+0.006746) |
| P=hpre | -0.000689 | -0.177150 | -0.004586 | +0.001420 | +0.004954 | L36 (+0.012429) |

注：Hall−Real 的熵为负、Top-K mass/Pmax 为正，均表示 Hall 的 P 更集中；反之表示 Real 更集中。
