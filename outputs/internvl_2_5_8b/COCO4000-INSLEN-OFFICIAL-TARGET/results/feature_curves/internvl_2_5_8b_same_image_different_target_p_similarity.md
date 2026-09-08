# internvl_2_5_8b：同图不同目标的 P 相似度

- 只比较同一图片中 `token_str` 不同的正式 InsLen 目标。
- 主结果先在每张图片内部对所有目标对取平均，再对图片等权平均，避免对象较多的 caption 主导结果。CSV 同时保存 pair-weighted 结果。
- 原始 cosine 会受到 P 的均匀公共分量影响，因此同时报告去均匀分量后的 centered cosine、归一化 JS、TV 和 Top-K 重合率。
- `pair JS / uniform JS` 比较目标间变化与 P 偏离均匀分布的幅度；越接近 0，越说明 P 主要是共享模板而非目标特异分布。

## 数据

- `outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part0.pkl`：5,845 条。
- `outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/features.part1.pkl`：5,914 条。
- 二元标签记录：11,759。
- 至少含两个不同目标词的图片：3,231。
- 不同目标词总数：11,063。
- 同图不同目标 pair：16,860。
- 同词重复行排除：0。

## 全层平均（图片等权）

| P source | 图片数 | pair 数 | raw cosine | centered cosine | JS/ln2 | TV | Top-32 overlap | 随机 overlap | pair JS / uniform JS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P=hmid | 3,231 | 16,860 | 0.938973 | 0.646263 | 0.023053 | 0.130842 | 0.496858 | 0.125000 | 0.693716 |
| P=hpre | 3,231 | 16,860 | 0.937536 | 0.638599 | 0.023477 | 0.132158 | 0.489805 | 0.125000 | 0.705459 |

## 按标签组合的 JS/ln2

| P source | Real–Real | Real–Hall | Hall–Hall |
|---|---:|---:|---:|
| P=hmid | 0.022927 | 0.023783 | 0.020704 |
| P=hpre | 0.023368 | 0.024172 | 0.020956 |

- 逐层完整统计：`internvl_2_5_8b_same_image_different_target_p_similarity.csv`。
- 曲线图：`internvl_2_5_8b_same_image_different_target_p_similarity.png`。
