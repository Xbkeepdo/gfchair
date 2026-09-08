# llava_1_5_7b：P 在视觉 token 上的分布统计

- `P=hmid`：`softmax(cos(o_ffn(q_t), hmid(v_i)) / 0.07)`。
- `P=hpre`：`softmax(cos(o_ffn(q_t), hpre(v_i)) / 0.07)`。
- 每个目标对象、每层作为一个样本；`0=Hall`、`1=Real`。
- 原始 P 的 token 均值恒为约 `1/N_v`，因此同时统计 `N_v P_i`；其均匀分布基线为 1。
- 曲线统计对每个正式对象等权；直方图对每个视觉 token 等权。
- raw/softmax target gate 与 sqrt/cosine transport cost 不改变 P，故没有重复绘制。

## 输入与产物

- `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：7,796 条。
- `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：7,667 条。
- 逐层指标：`llava_1_5_7b_inslen_p_visual_token_distribution_metrics.csv`。
- token 直方图：`llava_1_5_7b_inslen_p_visual_token_distribution_histogram.csv`。
- 指标曲线：`llava_1_5_7b_inslen_p_visual_token_distribution_metrics.png`。
- 分布热图：`llava_1_5_7b_inslen_p_visual_token_distribution_histogram_heatmap.png`。

## 全层平均

| P source | 标签 | 样本数 | 平均视觉 token 数 | token 数范围 | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | hall | 3,487 | 576.00 | 576–576 | 0.004205 | 2.422214 | 0.097633 | 0.991881 | 0.950895 |
| P=hmid | real | 11,976 | 576.00 | 576–576 | 0.004288 | 2.469915 | 0.099143 | 0.991282 | 0.947454 |
| P=hpre | hall | 3,487 | 576.00 | 576–576 | 0.004101 | 2.362225 | 0.096214 | 0.992386 | 0.954064 |
| P=hpre | real | 11,976 | 576.00 | 576–576 | 0.004235 | 2.439610 | 0.098201 | 0.991783 | 0.950602 |

## Hall − Real

| P source | Pmax | N_v Pmax | Top-32 mass | 归一化熵 | 有效 token 比例 | 最大熵差层 |
|---|---:|---:|---:|---:|---:|---:|
| P=hmid | -0.000083 | -0.047702 | -0.001510 | +0.000599 | +0.003441 | L26 (-0.009861) |
| P=hpre | -0.000134 | -0.077385 | -0.001987 | +0.000603 | +0.003463 | L26 (-0.009986) |

注：Hall−Real 的熵为负、Top-K mass/Pmax 为正，均表示 Hall 的 P 更集中；反之表示 Real 更集中。
