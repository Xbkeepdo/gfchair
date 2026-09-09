# 四模型：同图不同目标的 P 相似度

## 统计口径

- 按 `image_id` 分组，只比较同图中 `token_str` 不同的正式 InsLen 目标。
- 主结果先在每张图片内对目标 pair 求均值，再让每张图片等权；CSV 同时保存 pair-weighted 统计。
- 原始 cosine 会被接近均匀的正公共分量推高，因此同时计算 `cos(P_a-u,P_b-u)`、`JS/ln2`、TV 和 Top-32 重合率，其中 `u_i=1/N_v`。
- `pair JS / uniform JS` 是目标间 JS 与目标 P 偏离均匀分布 JS 的比值。低值表示同图目标共享的 P 模板占比较强；接近或超过 1 表示目标条件变化与 P 自身的非均匀幅度相当。

## 全层平均

| 模型 | P source | 图片数 | pair 数 | raw cosine | centered cosine | JS/ln2 | TV | Top-32 overlap | 随机 overlap | pair JS / uniform JS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | hmid | 3,457 | 26,782 | 0.9482 | 0.4553 | 0.0189 | 0.1173 | 0.2846 | 0.0556 | 1.0982 |
| LLaVA-1.5-7B | hpre | 3,457 | 26,782 | 0.9472 | 0.4012 | 0.0192 | 0.1179 | 0.2602 | 0.0556 | 1.2065 |
| InternVL2.5-8B | hmid | 3,231 | 16,860 | 0.9390 | 0.6463 | 0.0231 | 0.1308 | 0.4969 | 0.1250 | 0.6937 |
| InternVL2.5-8B | hpre | 3,231 | 16,860 | 0.9375 | 0.6386 | 0.0235 | 0.1322 | 0.4898 | 0.1250 | 0.7055 |
| Qwen2.5-VL-7B | hmid | 2,584 | 9,286 | 0.9179 | 0.4818 | 0.0297 | 0.1449 | 0.3930 | 0.0946 | 1.0392 |
| Qwen2.5-VL-7B | hpre | 2,584 | 9,286 | 0.9170 | 0.4744 | 0.0307 | 0.1472 | 0.3950 | 0.0946 | 1.0545 |
| Qwen3-VL-8B | hmid | 3,466 | 28,952 | 0.9553 | 0.6479 | 0.0193 | 0.1120 | 0.5375 | 0.1240 | 0.7052 |
| Qwen3-VL-8B | hpre | 3,466 | 28,952 | 0.9548 | 0.6174 | 0.0182 | 0.1105 | 0.5172 | 0.1240 | 0.7598 |

## 解读

- raw cosine 的 `0.918–0.955` 确实很高，但它很大程度来自 P 的均匀公共分量，不能据此判断不同目标的 P 几乎相同。
- 去均匀后的 cosine 降到 `0.40–0.65`，说明 P 中同时存在明显的同图公共模式和中等强度的目标条件变化。
- Top-32 重合率是随机基线的约 `2.8–5.1` 倍，说明不同目标仍反复选择同一批视觉 token；InternVL/Qwen3 的公共模板最强。
- LLaVA/Qwen2.5 的 `pair JS / uniform JS≈1.0–1.2`，目标变化与 P 自身偏离均匀的幅度相当；InternVL/Qwen3 约 `0.69–0.76`，同图共享结构相对更强。
- 目标特异性强烈依赖层。例如 LLaVA centered cosine 在 L4 最低，Qwen2.5 在 L5 最低，Qwen3 的 hpre 在 L22 最低；全层平均会掩盖这些局部变化。

## 标签组合

| 模型 | P source | Real–Real JS | Real–Hall JS | Hall–Hall JS |
|---|---|---:|---:|---:|
| LLaVA-1.5-7B | hmid | 0.01958 | 0.01900 | 0.01263 |
| LLaVA-1.5-7B | hpre | 0.01989 | 0.01917 | 0.01278 |
| InternVL2.5-8B | hmid | 0.02293 | 0.02378 | 0.02070 |
| InternVL2.5-8B | hpre | 0.02337 | 0.02417 | 0.02096 |
| Qwen2.5-VL-7B | hmid | 0.02951 | 0.03069 | 0.02583 |
| Qwen2.5-VL-7B | hpre | 0.03043 | 0.03158 | 0.02661 |
| Qwen3-VL-8B | hmid | 0.01908 | 0.02002 | 0.01849 |
| Qwen3-VL-8B | hpre | 0.01801 | 0.01896 | 0.01754 |

四个模型的 Hall–Hall JS 都低于 Real–Real。该现象与“同图中的多个真实对象有各自不同视觉锚点，而多个幻觉对象更容易退化为相似的共享视觉/语言模板”一致，但仍不能证明 P 的高权重 token 落在正确对象区域；最终需要 COCO box-mass 验证。

## 产物

- [LLaVA 报告](llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/llava_1_5_7b_same_image_different_target_p_similarity.md)
- [InternVL 报告](internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/internvl_2_5_8b_same_image_different_target_p_similarity.md)
- [Qwen2.5 报告](qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/qwen2_5_vl_7b_same_image_different_target_p_similarity.md)
- [Qwen3 报告](qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/feature_curves/qwen3_vl_8b_same_image_different_target_p_similarity.md)

每个模型报告旁均保存逐层 CSV 及 PNG/PDF 曲线图。
