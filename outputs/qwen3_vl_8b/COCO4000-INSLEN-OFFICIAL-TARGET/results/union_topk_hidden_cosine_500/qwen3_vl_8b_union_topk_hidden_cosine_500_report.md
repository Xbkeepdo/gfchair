# qwen3_vl_8b：Union Top-K hidden cosine 诊断

支持集严格使用当前协议：source Top-32 与 target Top-32 的并集（最多 64 个视觉 token）。
对角线和对称重复项不进入 visual–visual 统计；`q_t` 是 InsLen 目标首 subtoken 的因果预测行。

## 覆盖

- 图片：500；正式 InsLen 目标：1772；唯一因果位置：1753。
- 分层抽样：has-real=472，hall-only=28。
- 合并同一因果位置的重复 span：19。

## 全层平均

`vv/qv cosine` 与 P90−P10 来自全部有效 token 对的 pooled 分布；`centered vv`、entropy 和 top−bottom 先按图片聚合，再对图片等权平均。

| P source / Q gate | vv cosine | vv P90−P10 | centered vv | qv cosine | qv P90−P10 | qv entropy | qv top−bottom |
|---|---:|---:|---:|---:|---:|---:|---:|
| hmid_cos__hpre_raw_logit_gauss | 0.6097 | 0.3367 | 0.1160 | 0.5290 | 0.2350 | 0.8964 | 0.3599 |
| hmid_cos__hpre_softmax_prob_gauss | 0.6111 | 0.3364 | 0.1183 | 0.5298 | 0.2361 | 0.8970 | 0.3545 |
| hpre_cos__hpre_raw_logit_gauss | 0.6085 | 0.3372 | 0.1148 | 0.5291 | 0.2353 | 0.8963 | 0.3599 |
| hpre_cos__hpre_softmax_prob_gauss | 0.6099 | 0.3375 | 0.1168 | 0.5299 | 0.2358 | 0.8969 | 0.3544 |

`centered vv` 仅用于判断公共各向异性方向，不参与当前 cost。完整逐层、Real/Hall 分组及置信区间见同目录 CSV。
