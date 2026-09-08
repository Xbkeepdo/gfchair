# TC-FVPA 预注册假设

冻结时间：2026-08-29 UTC，在本轮任何真实模型 TC-FVPA/Riesz/path/FP32 结果产生之前。本文档后续不得按测试结果修改；若协议必须变更，只能另建带时间戳的 deviation 文档。

固定目标协议为 InsLen official first subtoken、回答中第一次出现、目标 token 不进入 causal prefix；标签方向固定为 `HALL=0, REAL=1`。正式层为一基索引 `8,16,24,32` 或在层数不足时预先登记的等比例层，不按测试标签选层。主要 target functional 为 target log-probability，fixed-clean-competitor margin 为主消融，raw target logit 为补充。

## H0：WRITE dominance

`||a_m||_2` 解释 JFFN token ranking 的大部分变化。用 Pearson/Spearman、Top-K overlap、层内 residualization 和 paired spatial metrics 评估。

## H1：FFN 局部变换具有 WRITE 条件增量

在训练图像内拟合并在 held-out image 上应用 `log ||J_G a_m|| = beta_0 + beta_1 log ||a_m|| + e_m` 后，residualized local FFN response 能提高 frozen-write finite intervention、空间定位或 target specificity 的预测。若只改善 causal finite-effect prediction 而不改善定位，结论限定为 causal-faithfulness improvement。

## H2：Riesz target functional 具有行为效度

`grad_m S ^ T J_G a_m = (J_G^T grad_m S)^T a_m` 比 hidden-space energy 更好预测 target logit、fixed-competitor margin和 target log-probability 的 FP32 infinitesimal/finite changes。主要判断使用 log-probability，必须同时报告正负贡献。

## H3：finite path 优于 clean-point local Jacobian

相对 K=1 clean-endpoint local attribution，K=4/8/16/32 path attribution 在 preregistered current-block zero-write baseline 上降低 completeness error，并提高与 frozen-write LOO/region intervention 的相关和符号一致性。Gauss–Legendre 与 trapezoid 必须报告收敛，不能只选择较优者。

## H4：conditional value path 不等同完整视觉因果

Frozen-write/fixed-QK effects 与 full current-layer activation patching和 pixel-region counterfactual相关但不相同。若注意力重分配或 earlier-layer visual state 改变导致差异，报告估计量差异，不把 frozen-write attribution称为完整 patch removal effect。

## H5：target-path 特征增加幻觉检测价值

在相同 image split、相同 seeds、train-only标准化/threshold条件下，path 特征相对 `risk+AE/EV` 及 `risk+AE/EV+WRITE` 提高 held-out AUROC、REAL AUPRC或HALL AUPRC，并由10,000次paired image-cluster bootstrap给出不跨0的差值区间。若不能超过 WRITE，不声称更好的 FFN token attribution。

## H6：跨模型复现

一般性结论至少需要两个架构不同的正式模型按相同 protocol 复现。LLaVA/InternVL 结果不得外推到 Qwen；smoke 不计正式复现。

## 决策规则

- Path 改善 finite-effect prediction 但不改善 localization/detection：Outcome B。
- WRITE 解释 token ranking，FFN 特异信息主要存在于 aggregate direction/target consequence/interaction：Outcome C。
- Riesz/path 对 baseline、模型或干预 estimand 高度不稳：Outcome D。
- 只有同时得到 WRITE 条件增量、真正 FP32 causal support和跨模型 detection improvement 才允许 Outcome A。
