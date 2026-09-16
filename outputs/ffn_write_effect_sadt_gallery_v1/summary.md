# SADT风格 Attention、AE、P_E 与 cosine Top16/Top32 图册

每模型10个main案例（HALL/REAL各5）和3个额外高差异HALL案例，共52张大图。所有图保留指定层，不作跨层mean。
每张图先给模型实际输入视图和完整生成原文，并用红色标出本次检测词；随后分别给Top16和Top32下的Attention、AE、P_E与softmax(cos/0.2)。
空间图仿照SADT作者实现：TopK之外置零，JET热图0.45+原图0.55；因此背景呈冷蓝色且没有patch框。
P_E来自all-attention K32逐token ||e_m||并在视觉token上归一化；cosine为cos(a_m,e_m)，softmax在每层全部视觉token上计算。softmax会抹去cosine正负号，本图只表达相对空间排序。
额外HALL案例按单层最大JS与最小Top16/32重合共同排序；完整候选逐层关系保存在CSV，避免只用挑图论证P_E不同于Attention。

[Attention–P_E全候选关系曲线](attention_pe_relationship.png)；[逐目标逐层CSV](attention_pe_relationship.csv)。

[未求和的局部差值 $|P_{E,m}-P_{W,m}|$：10个REAL/HALL案例](pe_write_local_amp_examples/summary.md)。

| 模型 | main | 额外差异HALL | 层 | 图册 |
|---|---:|---:|---|---|
| Qwen2.5-VL-7B | 10 | 3 | 10/15/20/25/28 | [qwen2_5_vl_7b/summary.md](qwen2_5_vl_7b/summary.md) |
| LLaVA-1.5-7B | 10 | 3 | 10/15/20/25/30/32 | [llava_1_5_7b/summary.md](llava_1_5_7b/summary.md) |
| Qwen3-VL-8B | 10 | 3 | 10/15/20/25/30/36 | [qwen3_vl_8b/summary.md](qwen3_vl_8b/summary.md) |
| InternVL2.5-8B | 10 | 3 | 10/15/20/25/30/32 | [internvl_2_5_8b/summary.md](internvl_2_5_8b/summary.md) |

## 全候选Attention–P_E关系

统计范围是四模型共享500图缓存中能与完整目标标注对齐的全部mention及上述指定层，不是只统计52个展示案例。

| 模型 | target-layer数 | JS中位数 | rank correlation中位数 | Top16 overlap中位数 | Top32 overlap中位数 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 4650 | 0.014 | 0.979 | 0.812 | 0.875 |
| LLaVA-1.5-7B | 9522 | 0.041 | 0.978 | 0.688 | 0.812 |
| Qwen3-VL-8B | 9384 | 0.019 | 0.974 | 0.812 | 0.844 |
| InternVL2.5-8B | 7308 | 0.015 | 0.976 | 0.875 | 0.875 |

### REAL/HALL分开

| 模型 | 标签 | target-layer数 | JS中位数 | rank correlation中位数 | Top16 overlap中位数 | Top32 overlap中位数 |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | REAL | 4040 | 0.014 | 0.979 | 0.812 | 0.875 |
| Qwen2.5-VL-7B | HALL | 610 | 0.015 | 0.980 | 0.875 | 0.875 |
| LLaVA-1.5-7B | REAL | 6870 | 0.040 | 0.978 | 0.688 | 0.812 |
| LLaVA-1.5-7B | HALL | 2652 | 0.042 | 0.977 | 0.750 | 0.812 |
| Qwen3-VL-8B | REAL | 7242 | 0.019 | 0.974 | 0.812 | 0.844 |
| Qwen3-VL-8B | HALL | 2142 | 0.019 | 0.974 | 0.812 | 0.844 |
| InternVL2.5-8B | REAL | 5994 | 0.015 | 0.976 | 0.875 | 0.875 |
| InternVL2.5-8B | HALL | 1314 | 0.015 | 0.975 | 0.875 | 0.875 |

运行：

```bash
python scripts/plot_sadt_style_write_effect_gallery.py --stage rank --models <model>
python scripts/plot_sadt_style_write_effect_gallery.py --stage plot --models <model>
python scripts/plot_sadt_style_write_effect_gallery.py --stage summarize
```

只读取既有COCO4000文本、K50紧凑attention/AE缓存和共享500图all-attention K32缓存，不运行模型或路径积分。
