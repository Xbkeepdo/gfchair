# internvl_2_5_8b：Union Top-K hidden cosine 诊断

支持集严格使用当前协议：source Top-32 与 target Top-32 的并集（最多 64 个视觉 token）。
对角线和对称重复项不进入 visual–visual 统计；`q_t` 是 InsLen 目标首 subtoken 的因果预测行。

## 覆盖

- 图片：500；正式 InsLen 目标：1354；唯一因果位置：1346。
- 分层抽样：has-real=444，hall-only=56。
- 合并同一因果位置的重复 span：8。

## 全层平均

`vv/qv cosine` 与 P90−P10 来自全部有效 token 对的 pooled 分布；`centered vv`、entropy 和 top−bottom 先按图片聚合，再对图片等权平均。

| P source / Q gate | vv cosine | vv P90−P10 | centered vv | qv cosine | qv P90−P10 | qv entropy | qv top−bottom |
|---|---:|---:|---:|---:|---:|---:|---:|
| hmid_cos__hpre_raw_logit_gauss | 0.4964 | 0.3609 | 0.0962 | 0.4270 | 0.2122 | 0.8771 | 0.3197 |
| hmid_cos__hpre_softmax_prob_gauss | 0.4927 | 0.3631 | 0.0962 | 0.4258 | 0.2141 | 0.8759 | 0.3227 |
| hpre_cos__hpre_raw_logit_gauss | 0.4978 | 0.3606 | 0.1010 | 0.4289 | 0.2116 | 0.8791 | 0.3178 |
| hpre_cos__hpre_softmax_prob_gauss | 0.4943 | 0.3628 | 0.1011 | 0.4277 | 0.2134 | 0.8778 | 0.3207 |

`centered vv` 仅用于判断公共各向异性方向，不参与当前 cost。完整逐层、Real/Hall 分组及置信区间见同目录 CSV。
