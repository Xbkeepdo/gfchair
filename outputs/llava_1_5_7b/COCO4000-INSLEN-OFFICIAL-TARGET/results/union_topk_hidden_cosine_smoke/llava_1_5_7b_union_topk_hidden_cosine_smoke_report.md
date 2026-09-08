# llava_1_5_7b：Union Top-K hidden cosine 诊断

支持集严格使用当前协议：source Top-32 与 target Top-32 的并集（最多 64 个视觉 token）。
对角线和对称重复项不进入 visual–visual 统计；`q_t` 是 InsLen 目标首 subtoken 的因果预测行。

## 覆盖

- 图片：2；唯一因果目标：8。
- 分层抽样：has-real=1，hall-only=1。
- 合并同一因果位置的重复 span：1。

## 全层平均（图片等权）

| P source / Q gate | vv cosine | vv P90−P10 | centered vv | qv cosine | qv P90−P10 | qv entropy | qv top−bottom |
|---|---:|---:|---:|---:|---:|---:|---:|
| hmid_cos__hpre_raw_logit_gauss | 0.3238 | 0.6713 | 0.1673 | 0.2100 | 0.2703 | 0.8485 | 0.3800 |
| hmid_cos__hpre_softmax_prob_gauss | 0.3346 | 0.6512 | 0.1808 | 0.2130 | 0.2647 | 0.8530 | 0.3706 |
| hpre_cos__hpre_raw_logit_gauss | 0.3266 | 0.6619 | 0.1892 | 0.2116 | 0.2712 | 0.8511 | 0.3793 |
| hpre_cos__hpre_softmax_prob_gauss | 0.3386 | 0.6416 | 0.2061 | 0.2147 | 0.2647 | 0.8553 | 0.3676 |

`centered vv` 仅用于判断公共各向异性方向，不参与当前 cost。完整逐层、Real/Hall 分组及置信区间见同目录 CSV。
