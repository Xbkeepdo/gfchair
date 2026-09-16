# Raw attention / AE Top-16 与 Top-32 的 REAL/HALL 曲线

定义：对每个目标、每层分别在视觉 token 内归一化 raw attention 与既有 `attention_evidence`（AE/T）分布；各自选择本分布权重最大的 K 个位置，曲线值为这些位置的概率质量之和。K 为16或32，不足K时保留全部视觉token。

统计使用四模型 COCO4000 正式缓存中的全部 train+test mentions，mention 等权。蓝色 REAL、红色 HALL；实线为中位数，阴影为25%–75%分位数，不是置信区间。各面板纵轴自适应，跨面板比较绝对高度时应读取刻度。没有重跑VLM、路径积分或检测器。

[Top-16图](top16_raw_attention_ae_real_hall.png) · [Top-32图](top32_raw_attention_ae_real_hall.png) · [四项总览](top16_top32_raw_attention_ae_real_hall.png) · [逐层数据](curves.csv) · [差值摘要](gap_summary.csv)

## 样本

| 模型 | 图片 | mentions (HALL / REAL) | 唯一目标 | 层 | 视觉token数 min / median / max |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 4000 | 8717 (916 / 7801) | 8654 | 28 | 63 / 345 / 529 |
| LLaVA-1.5-7B | 4000 | 15463 (3487 / 11976) | 14951 | 32 | 576 / 576 / 576 |
| Qwen3-VL-8B | 4000 | 14873 (2593 / 12280) | 14704 | 36 | 70 / 260 / 400 |
| InternVL2.5-8B | 4000 | 11759 (1766 / 9993) | 11630 | 32 | 256 / 256 / 256 |

## 逐层中位数方向

`H>R层数`只描述逐层中位数方向；`median Δ`是各层 `(HALL−REAL)` 中位数差的中位数，不是分类指标或显著性结论。

| 模型 | 信号 | K | H>R层数 | R>H层数 | median Δ(H−R) | 最大绝对差@层（带符号） |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | Normalized raw attention | 16 | 21/28 | 7/28 | +0.034973 | +0.111815@L22 |
| Qwen2.5-VL-7B | Normalized raw attention | 32 | 19/28 | 9/28 | +0.030520 | +0.094905@L22 |
| Qwen2.5-VL-7B | AE / T distribution | 16 | 21/28 | 7/28 | +0.039319 | +0.101788@L22 |
| Qwen2.5-VL-7B | AE / T distribution | 32 | 20/28 | 8/28 | +0.031675 | +0.079499@L22 |
| LLaVA-1.5-7B | Normalized raw attention | 16 | 23/32 | 9/32 | +0.008638 | +0.101963@L7 |
| LLaVA-1.5-7B | Normalized raw attention | 32 | 16/32 | 16/32 | +0.002595 | +0.069425@L7 |
| LLaVA-1.5-7B | AE / T distribution | 16 | 18/32 | 14/32 | +0.008416 | +0.089089@L7 |
| LLaVA-1.5-7B | AE / T distribution | 32 | 16/32 | 16/32 | -0.002231 | +0.057780@L7 |
| Qwen3-VL-8B | Normalized raw attention | 16 | 25/36 | 11/36 | +0.025690 | +0.076245@L24 |
| Qwen3-VL-8B | Normalized raw attention | 32 | 23/36 | 13/36 | +0.014195 | -0.064089@L7 |
| Qwen3-VL-8B | AE / T distribution | 16 | 25/36 | 11/36 | +0.024663 | -0.079147@L7 |
| Qwen3-VL-8B | AE / T distribution | 32 | 24/36 | 12/36 | +0.013085 | -0.075688@L7 |
| InternVL2.5-8B | Normalized raw attention | 16 | 22/32 | 10/32 | +0.013172 | +0.090372@L16 |
| InternVL2.5-8B | Normalized raw attention | 32 | 20/32 | 12/32 | +0.006247 | +0.073893@L16 |
| InternVL2.5-8B | AE / T distribution | 16 | 22/32 | 10/32 | +0.011409 | +0.085024@L16 |
| InternVL2.5-8B | AE / T distribution | 32 | 16/32 | 16/32 | -0.000700 | +0.062436@L16 |

注意：这里的 AE Top-K 是归一化 token 分布 T 的集中度，不是标量 `AE_K=AE×M_K`，也不是 raw attention 与 AE 的 Top-K 重合率。固定K还受视觉token数量与模型分辨率影响，不宜把不同模型的绝对高度直接作机制比较。

复现：`python scripts/plot_topk_raw_attention_ae_by_label.py`。
