# 四模型：P 在视觉 token 上的分布统计

## 协议

- `P=hmid`：`softmax(cos(o_ffn(q_t), hmid(v_i)) / 0.07)`。
- `P=hpre`：`softmax(cos(o_ffn(q_t), hpre(v_i)) / 0.07)`。
- 使用 gfchair 的全部正式 InsLen 对象样本；每个目标对象、每一层作为一个样本，按 Hall/Real 分组。
- `P` 每层在视觉 token 上归一化。由于原始 token 均值恒为 `1/N_v`，跨模型主要比较 `N_v P_i`、Top-K mass、归一化熵和有效 token 比例。
- 曲线统计对每个目标对象等权；直方图对每个视觉 token 等权。raw/softmax target gate 与 sqrt/cosine transport cost 不改变 P，因此没有重复统计。

## 全层平均分布

| 模型 | P source | Hall `N_v Pmax` | Real `N_v Pmax` | Hall Top-32 | Real Top-32 | Hall 归一化熵 | Real 归一化熵 | 熵差 H−R |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | hmid | 2.4222 | 2.4699 | 0.0976 | 0.0991 | 0.9919 | 0.9913 | +0.0006 |
| LLaVA-1.5-7B | hpre | 2.3622 | 2.4396 | 0.0962 | 0.0982 | 0.9924 | 0.9918 | +0.0006 |
| InternVL2.5-8B | hmid | 2.7514 | 3.0627 | 0.2293 | 0.2431 | 0.9813 | 0.9775 | +0.0038 |
| InternVL2.5-8B | hpre | 2.7944 | 3.1107 | 0.2311 | 0.2451 | 0.9806 | 0.9767 | +0.0039 |
| Qwen2.5-VL-7B | hmid | 5.2410 | 5.5216 | 0.1980 | 0.1981 | 0.9768 | 0.9755 | +0.0012 |
| Qwen2.5-VL-7B | hpre | 6.6187 | 7.0819 | 0.2080 | 0.2089 | 0.9700 | 0.9679 | +0.0020 |
| Qwen3-VL-8B | hmid | 5.4708 | 5.5549 | 0.2710 | 0.2749 | 0.9564 | 0.9554 | +0.0010 |
| Qwen3-VL-8B | hpre | 5.8385 | 6.0156 | 0.2668 | 0.2714 | 0.9562 | 0.9548 | +0.0014 |

归一化熵都在 `0.95–0.99`，说明 P 整体仍然较分散，但明显不是完全均匀分布。最大 token 平均约为均匀概率的 `2.4–7.1` 倍；LLaVA 最平，Qwen2.5/Qwen3 的单 token 峰值更强。

## Hall 与 Real

| 模型 | P source | Hall 更分散的层数 | Hall 更集中的层数 | 最大绝对熵差层 | Hall−Real |
|---|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | hmid | 21 | 11 | L26 | -0.00986 |
| LLaVA-1.5-7B | hpre | 22 | 10 | L26 | -0.00999 |
| InternVL2.5-8B | hmid | 31 | 1 | L19 | +0.00907 |
| InternVL2.5-8B | hpre | 31 | 1 | L19 | +0.00941 |
| Qwen2.5-VL-7B | hmid | 15 | 13 | L28 | +0.03236 |
| Qwen2.5-VL-7B | hpre | 14 | 14 | L28 | +0.04778 |
| Qwen3-VL-8B | hmid | 29 | 7 | L34 | +0.00675 |
| Qwen3-VL-8B | hpre | 28 | 8 | L36 | +0.01243 |

- 八组全层平均熵差均为正，即总体上 Hall 的 P 略微更平、Real 略微更集中。
- InternVL 的方向最稳定：31/32 层都是 Hall 更分散。Qwen3 也以 Hall 更分散为主。
- LLaVA 与 Qwen2.5 有明显层间反转。LLaVA 在 L26 反而是 Hall 更集中；Qwen2.5 的 hpre 在 L28 是 Hall 更分散，但在 L14 又转为 Hall 更集中。
- 因此不能把 P 压成一个全层平均集中度标量；Hall/Real 信息主要是层局部的，保留逐层向量更合理。

## hpre 相对 hmid 的变化

- LLaVA：hpre 整体稍微更平，`N_v Pmax` 下降约 `0.03–0.06`，归一化熵提高约 `0.0005`。
- InternVL：hpre 稍微更尖，`N_v Pmax` 提高约 `0.04–0.05`，归一化熵降低约 `0.0007`。
- Qwen2.5：hpre 的变化最大，`N_v Pmax` 提高约 `1.38–1.56`，归一化熵降低约 `0.0068–0.0076`，明显强化了少数视觉 token。
- Qwen3：hpre 提高单个最大 token，但 Top-32 总质量略降，说明它不是简单地让整个头部一起变尖，而是改变了头部内部的质量分配。

## 产物

- [LLaVA 报告](llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/llava_1_5_7b_inslen_p_visual_token_distribution.md)
- [InternVL 报告](internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/internvl_2_5_8b_inslen_p_visual_token_distribution.md)
- [Qwen2.5 报告](qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/qwen2_5_vl_7b_inslen_p_visual_token_distribution.md)
- [Qwen3 报告](qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/qwen3_vl_8b_inslen_p_visual_token_distribution.md)

每个模型目录均包含指标曲线 PNG/PDF、`log10(N_vP_i)` 分布热图 PNG/PDF、逐层指标 CSV 和完整直方图 CSV。
