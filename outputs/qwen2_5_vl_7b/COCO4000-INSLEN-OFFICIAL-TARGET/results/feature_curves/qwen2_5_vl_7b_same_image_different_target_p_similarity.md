# qwen2_5_vl_7b：同图不同目标的 P 相似度

- 只比较同一图片中 `token_str` 不同的正式 InsLen 目标。
- 主结果先在每张图片内部对所有目标对取平均，再对图片等权平均，避免对象较多的 caption 主导结果。CSV 同时保存 pair-weighted 结果。
- 原始 cosine 会受到 P 的均匀公共分量影响，因此同时报告去均匀分量后的 centered cosine、归一化 JS、TV 和 Top-K 重合率。
- `pair JS / uniform JS` 比较目标间变化与 P 偏离均匀分布的幅度；越接近 0，越说明 P 主要是共享模板而非目标特异分布。

## 数据

- `outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：4,319 条。
- `outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：4,398 条。
- 二元标签记录：8,717。
- 至少含两个不同目标词的图片：2,584。
- 不同目标词总数：7,622。
- 同图不同目标 pair：9,286。
- 同词重复行排除：0。

## 全层平均（图片等权）

| P source | 图片数 | pair 数 | raw cosine | centered cosine | JS/ln2 | TV | Top-32 overlap | 随机 overlap | pair JS / uniform JS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | 2,584 | 9,286 | 0.917857 | 0.481846 | 0.029737 | 0.144888 | 0.393005 | 0.094571 | 1.039211 |
| P=hpre | 2,584 | 9,286 | 0.917034 | 0.474412 | 0.030651 | 0.147160 | 0.394996 | 0.094571 | 1.054542 |

## 按标签组合的 JS/ln2

| P source | Real–Real | Real–Hall | Hall–Hall |
|---|---:|---:|---:|
| P=hmid | 0.029506 | 0.030693 | 0.025829 |
| P=hpre | 0.030431 | 0.031583 | 0.026607 |

- 逐层完整统计：`qwen2_5_vl_7b_same_image_different_target_p_similarity.csv`。
- 曲线图：`qwen2_5_vl_7b_same_image_different_target_p_similarity.png`。
