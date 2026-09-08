# internvl_2_5_8b：P 在视觉 token 上的分布统计

- `P=hmid`：`softmax(cos(o_ffn(q_t), hmid(v_i)) / 0.07)`。
- `P=hpre`：`softmax(cos(o_ffn(q_t), hpre(v_i)) / 0.07)`。
- 每个目标对象、每层作为一个样本；`0=Hall`、`1=Real`。
- 原始 P 的 token 均值恒为约 `1/N_v`，因此同时统计 `N_v P_i`；其均匀分布基线为 1。
- 曲线统计对每个正式对象等权；直方图对每个视觉 token 等权。
- raw/softmax target gate 与 sqrt/cosine transport cost 不改变 P，故没有重复绘制。

## 输入与产物

- `outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：5,845 条。
- `outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：5,914 条。
- 逐层指标：`internvl_2_5_8b_inslen_p_visual_token_distribution_metrics.csv`。
- token 直方图：`internvl_2_5_8b_inslen_p_visual_token_distribution_histogram.csv`。
- 指标曲线：`internvl_2_5_8b_inslen_p_visual_token_distribution_metrics.png`。
- 分布热图：`internvl_2_5_8b_inslen_p_visual_token_distribution_histogram_heatmap.png`。

## 全层平均

| P source | 标签 | 样本数 | 平均视觉 token 数 | token 数范围 | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | hall | 1,766 | 256.00 | 256–256 | 0.010748 | 2.751404 | 0.229295 | 0.981288 | 0.904539 |
| P=hmid | real | 9,993 | 256.00 | 256–256 | 0.011964 | 3.062663 | 0.243101 | 0.977497 | 0.886396 |
| P=hpre | hall | 1,766 | 256.00 | 256–256 | 0.010916 | 2.794449 | 0.231109 | 0.980599 | 0.901920 |
| P=hpre | real | 9,993 | 256.00 | 256–256 | 0.012151 | 3.110675 | 0.245123 | 0.976748 | 0.883519 |

## Hall − Real

| P source | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 | 最大熵差层 |
|---|---:|---:|---:|---:|---:|---:|
| P=hmid | -0.001216 | -0.311259 | -0.013806 | +0.003792 | +0.018143 | L19 (+0.009075) |
| P=hpre | -0.001235 | -0.316226 | -0.014014 | +0.003851 | +0.018401 | L19 (+0.009414) |

注：Hall−Real 的熵为负、Top-K mass/Pmax 为正，均表示 Hall 的 P 更集中；反之表示 Real 更集中。
