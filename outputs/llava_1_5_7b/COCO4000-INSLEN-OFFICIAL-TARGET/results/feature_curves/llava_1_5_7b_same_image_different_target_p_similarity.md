# llava_1_5_7b：同图不同目标的 P 相似度

- 只比较同一图片中 `token_str` 不同的正式 InsLen 目标。
- 主结果先在每张图片内部对所有目标对取平均，再对图片等权平均，避免对象较多的 caption 主导结果。CSV 同时保存 pair-weighted 结果。
- 原始 cosine 会受到 P 的均匀公共分量影响，因此同时报告去均匀分量后的 centered cosine、归一化 JS、TV 和 Top-K 重合率。
- `pair JS / uniform JS` 比较目标间变化与 P 偏离均匀分布的幅度；越接近 0，越说明 P 主要是共享模板而非目标特异分布。

## 数据

- `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：7,796 条。
- `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：7,667 条。
- 二元标签记录：15,463。
- 至少含两个不同目标词的图片：3,457。
- 不同目标词总数：13,861。
- 同图不同目标 pair：26,782。
- 同词重复行排除：1,088。

## 全层平均（图片等权）

| P source | 图片数 | pair 数 | raw cosine | centered cosine | JS/ln2 | TV | Top-32 overlap | 随机 overlap | pair JS / uniform JS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | 3,457 | 26,782 | 0.948192 | 0.455250 | 0.018912 | 0.117292 | 0.284587 | 0.055556 | 1.098192 |
| P=hpre | 3,457 | 26,782 | 0.947152 | 0.401222 | 0.019169 | 0.117933 | 0.260239 | 0.055556 | 1.206529 |

## 按标签组合的 JS/ln2

| P source | Real–Real | Real–Hall | Hall–Hall |
|---|---:|---:|---:|
| P=hmid | 0.019581 | 0.019002 | 0.012631 |
| P=hpre | 0.019886 | 0.019168 | 0.012777 |

- 逐层完整统计：`llava_1_5_7b_same_image_different_target_p_similarity.csv`。
- 曲线图：`llava_1_5_7b_same_image_different_target_p_similarity.png`。
