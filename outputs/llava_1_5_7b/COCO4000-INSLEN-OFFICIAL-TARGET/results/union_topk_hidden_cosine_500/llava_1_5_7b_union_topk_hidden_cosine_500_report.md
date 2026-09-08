# llava_1_5_7b：Union Top-K hidden cosine 诊断

支持集严格使用当前协议：source Top-32 与 target Top-32 的并集（最多 64 个视觉 token）。
对角线和对称重复项不进入 visual–visual 统计；`q_t` 是 InsLen 目标首 subtoken 的因果预测行。

## 覆盖

- 图片：500；正式 InsLen 目标：1882；唯一因果位置：1815。
- 分层抽样：has-real=461，hall-only=39。
- 合并同一因果位置的重复 span：67。

## 全层平均

`vv/qv cosine` 与 P90−P10 来自全部有效 token 对的 pooled 分布；`centered vv`、entropy 和 top−bottom 先按图片聚合，再对图片等权平均。

| P source / Q gate | vv cosine | vv P90−P10 | centered vv | qv cosine | qv P90−P10 | qv entropy | qv top−bottom |
|---|---:|---:|---:|---:|---:|---:|---:|
| hmid_cos__hpre_raw_logit_gauss | 0.3455 | 0.6975 | 0.1839 | 0.2113 | 0.2469 | 0.8595 | 0.3530 |
| hmid_cos__hpre_softmax_prob_gauss | 0.3533 | 0.6834 | 0.1926 | 0.2130 | 0.2425 | 0.8620 | 0.3466 |
| hpre_cos__hpre_raw_logit_gauss | 0.3489 | 0.6812 | 0.2048 | 0.2133 | 0.2469 | 0.8626 | 0.3523 |
| hpre_cos__hpre_softmax_prob_gauss | 0.3577 | 0.6694 | 0.2156 | 0.2150 | 0.2416 | 0.8650 | 0.3451 |

`centered vv` 仅用于判断公共各向异性方向，不参与当前 cost。完整逐层、Real/Hall 分组及置信区间见同目录 CSV。
