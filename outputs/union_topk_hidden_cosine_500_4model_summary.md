# 四模型 Union Top-K hidden cosine 诊断

支持集为 source Top-32 与 target Top-32 的并集；visual–visual 排除对角线和对称重复。vv/qv cosine 分布按全部有效 token 对 pooled；entropy 等标量先按图片聚合再等权平均。

## 覆盖

| 模型 | 图片 | 正式目标 | 唯一因果位置 | 重复 span | 标签冲突 |
|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | 500 | 1882 | 1815 | 67 | 5 |
| InternVL-2.5-8B | 500 | 1354 | 1346 | 8 | 0 |
| Qwen2.5-VL-7B | 500 | 1127 | 1120 | 7 | 0 |
| Qwen3-VL-8B | 500 | 1772 | 1753 | 19 | 0 |

## 全层平均（all 标签）

| 模型 | P/Q 分支 | vv cos | vv P90−P10 | centered vv | cosine-cost P90−P10 | qv cos | qv P90−P10 | qv entropy | qv max/uniform |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | hmid_cos__hpre_raw_logit_gauss | 0.3455 | 0.6975 | 0.1839 | 0.6626 | 0.2113 | 0.2469 | 0.8595 | 6.29 |
| LLaVA-1.5-7B | hmid_cos__hpre_softmax_prob_gauss | 0.3533 | 0.6834 | 0.1926 | 0.6463 | 0.2130 | 0.2425 | 0.8620 | 6.26 |
| LLaVA-1.5-7B | hpre_cos__hpre_raw_logit_gauss | 0.3489 | 0.6812 | 0.2048 | 0.6490 | 0.2133 | 0.2469 | 0.8626 | 6.19 |
| LLaVA-1.5-7B | hpre_cos__hpre_softmax_prob_gauss | 0.3577 | 0.6694 | 0.2156 | 0.6315 | 0.2150 | 0.2416 | 0.8650 | 6.16 |
| InternVL-2.5-8B | hmid_cos__hpre_raw_logit_gauss | 0.4964 | 0.3609 | 0.0962 | 0.3486 | 0.4270 | 0.2122 | 0.8771 | 6.08 |
| InternVL-2.5-8B | hmid_cos__hpre_softmax_prob_gauss | 0.4927 | 0.3631 | 0.0962 | 0.3508 | 0.4258 | 0.2141 | 0.8759 | 6.13 |
| InternVL-2.5-8B | hpre_cos__hpre_raw_logit_gauss | 0.4978 | 0.3606 | 0.1010 | 0.3483 | 0.4289 | 0.2116 | 0.8791 | 5.98 |
| InternVL-2.5-8B | hpre_cos__hpre_softmax_prob_gauss | 0.4943 | 0.3628 | 0.1011 | 0.3505 | 0.4277 | 0.2134 | 0.8778 | 6.02 |
| Qwen2.5-VL-7B | hmid_cos__hpre_raw_logit_gauss | 0.5605 | 0.3750 | 0.1121 | 0.3406 | 0.5024 | 0.2511 | 0.8776 | 5.64 |
| Qwen2.5-VL-7B | hmid_cos__hpre_softmax_prob_gauss | 0.5591 | 0.3829 | 0.1184 | 0.3469 | 0.5024 | 0.2536 | 0.8760 | 5.63 |
| Qwen2.5-VL-7B | hpre_cos__hpre_raw_logit_gauss | 0.5606 | 0.3761 | 0.1148 | 0.3401 | 0.5032 | 0.2511 | 0.8786 | 5.59 |
| Qwen2.5-VL-7B | hpre_cos__hpre_softmax_prob_gauss | 0.5592 | 0.3846 | 0.1213 | 0.3463 | 0.5031 | 0.2550 | 0.8771 | 5.57 |
| Qwen3-VL-8B | hmid_cos__hpre_raw_logit_gauss | 0.6097 | 0.3367 | 0.1160 | 0.3175 | 0.5290 | 0.2350 | 0.8964 | 5.39 |
| Qwen3-VL-8B | hmid_cos__hpre_softmax_prob_gauss | 0.6111 | 0.3364 | 0.1183 | 0.3164 | 0.5298 | 0.2361 | 0.8970 | 5.37 |
| Qwen3-VL-8B | hpre_cos__hpre_raw_logit_gauss | 0.6085 | 0.3372 | 0.1148 | 0.3170 | 0.5291 | 0.2353 | 0.8963 | 5.36 |
| Qwen3-VL-8B | hpre_cos__hpre_softmax_prob_gauss | 0.6099 | 0.3375 | 0.1168 | 0.3159 | 0.5299 | 0.2358 | 0.8969 | 5.34 |
