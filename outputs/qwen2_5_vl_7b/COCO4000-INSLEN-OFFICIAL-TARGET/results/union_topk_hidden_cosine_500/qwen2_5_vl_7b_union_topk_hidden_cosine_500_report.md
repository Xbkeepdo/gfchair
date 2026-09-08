# qwen2_5_vl_7b：Union Top-K hidden cosine 诊断

支持集严格使用当前协议：source Top-32 与 target Top-32 的并集（最多 64 个视觉 token）。
对角线和对称重复项不进入 visual–visual 统计；`q_t` 是 InsLen 目标首 subtoken 的因果预测行。

## 覆盖

- 图片：500；正式 InsLen 目标：1127；唯一因果位置：1120。
- 分层抽样：has-real=451，hall-only=49。
- 合并同一因果位置的重复 span：7。

## 全层平均

`vv/qv cosine` 与 P90−P10 来自全部有效 token 对的 pooled 分布；`centered vv`、entropy 和 top−bottom 先按图片聚合，再对图片等权平均。

| P source / Q gate | vv cosine | vv P90−P10 | centered vv | qv cosine | qv P90−P10 | qv entropy | qv top−bottom |
|---|---:|---:|---:|---:|---:|---:|---:|
| hmid_cos__hpre_raw_logit_gauss | 0.5605 | 0.3750 | 0.1121 | 0.5024 | 0.2511 | 0.8776 | 0.3420 |
| hmid_cos__hpre_softmax_prob_gauss | 0.5591 | 0.3829 | 0.1184 | 0.5024 | 0.2536 | 0.8760 | 0.3522 |
| hpre_cos__hpre_raw_logit_gauss | 0.5606 | 0.3761 | 0.1148 | 0.5032 | 0.2511 | 0.8786 | 0.3415 |
| hpre_cos__hpre_softmax_prob_gauss | 0.5592 | 0.3846 | 0.1213 | 0.5031 | 0.2550 | 0.8771 | 0.3516 |

`centered vv` 仅用于判断公共各向异性方向，不参与当前 cost。完整逐层、Real/Hall 分组及置信区间见同目录 CSV。
