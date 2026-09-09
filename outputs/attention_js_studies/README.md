# Attention JS experiments

两个问题拆成两个独立实验目录，避免图和统计口径混在一起：

1. [`hall_real_target_js/`](hall_real_target_js/summary.md)：同一图片内，不同 HALL/REAL 目标 token 之间的空间分布 JS，以及 A→E gate-effect、entropy、Top-32 mass。
2. [`interlayer_js/`](interlayer_js/summary.md)：同一个目标 token 在不同 decoder layer 之间的 JS。主图已经按模型、A/E、全部视觉 token/Union-Top32、HALL/REAL/差值完全拆开。
3. [`interlayer_js/detection/`](interlayer_js/detection/summary.md)：只使用相邻层 JS 的 standalone 幻觉检测；A/E × all/Top32 四组分别独立训练，不融合。

`interlayer_js/figures/`：

- `adjacent/{A,E}/{all_visual_tokens,union_top32}/{model}`：相邻层曲线，每张只有 HALL/REAL 两条线。
- `all_pair/{A,E}/{all_visual_tokens,union_top32}/{HALL,REAL,HALL_MINUS_REAL}/{model}`：全层两两热力图，每个文件只有一个模型和一个标签视图。
- `combined_overview/`：旧合并图，仅作版本追溯。

原始全视觉 attention mass 没有持久化，因此没有对应实验目录，也没有用逐层和恒为 1 的归一化 A 冒充 raw mass。
