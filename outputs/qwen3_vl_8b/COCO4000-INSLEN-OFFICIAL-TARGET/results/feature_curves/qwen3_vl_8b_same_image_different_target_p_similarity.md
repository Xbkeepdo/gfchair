# qwen3_vl_8b：同图不同目标的 P 相似度

- 只比较同一图片中 `token_str` 不同的正式 InsLen 目标。
- 主结果先在每张图片内部对所有目标对取平均，再对图片等权平均，避免对象较多的 caption 主导结果。CSV 同时保存 pair-weighted 结果。
- 原始 cosine 会受到 P 的均匀公共分量影响，因此同时报告去均匀分量后的 centered cosine、归一化 JS、TV 和 Top-K 重合率。
- `pair JS / uniform JS` 比较目标间变化与 P 偏离均匀分布的幅度；越接近 0，越说明 P 主要是共享模板而非目标特异分布。

## 数据

- `outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：7,490 条。
- `outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：7,383 条。
- 二元标签记录：14,873。
- 至少含两个不同目标词的图片：3,466。
- 不同目标词总数：14,396。
- 同图不同目标 pair：28,952。
- 同词重复行排除：0。

## 全层平均（图片等权）

| P source | 图片数 | pair 数 | raw cosine | centered cosine | JS/ln2 | TV | Top-32 overlap | 随机 overlap | pair JS / uniform JS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | 3,466 | 28,952 | 0.955315 | 0.647851 | 0.019284 | 0.111989 | 0.537467 | 0.124001 | 0.705161 |
| P=hpre | 3,466 | 28,952 | 0.954791 | 0.617433 | 0.018202 | 0.110518 | 0.517226 | 0.124001 | 0.759818 |

## 按标签组合的 JS/ln2

| P source | Real–Real | Real–Hall | Hall–Hall |
|---|---:|---:|---:|
| P=hmid | 0.019082 | 0.020019 | 0.018491 |
| P=hpre | 0.018009 | 0.018960 | 0.017539 |

- 逐层完整统计：`qwen3_vl_8b_same_image_different_target_p_similarity.csv`。
- 曲线图：`qwen3_vl_8b_same_image_different_target_p_similarity.png`。
