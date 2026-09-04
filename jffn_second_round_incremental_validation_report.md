# 1. Executive conclusion

## 2026-09-04 superseding addendum：Vector FFN Source Attribution

本报告以下正文记录的是旧 scalar-path/JFFN 与 bbox 路线，作为历史结果保留；其 source attribution 定义和 Outcome C **不再代表当前主方法**。当前定义已由

\[
e_m=\int_0^1J_G(z^0+\alpha A)a_m\,d\alpha
\]

的 vector path attribution 取代。Riesz 只解释 downstream target consequence，bbox 只作 auxiliary sanity check，不再用于判定 FFN source attribution 是否成立。

新路线已完成四模型正式实验，详见 [ffn_visual_source_attribution_report.md](ffn_visual_source_attribution_report.md)：Qwen2/LLaVA/Qwen3/InternVL 均完成 200 图 audit、全层 COCO4000、13×3 detector 和 100 图×4 层的三类因果干预。主模型四个预注册 `G−C` gate 的 10,000 次图片级 paired-bootstrap CI 下界全部大于 0，因此扩展与因果阶段为必跑且已完成。

更新后的结论是：vector source detector 增量成立；`P_FFN` 与 `P_WRITE` 在 fixed-QK、activation patching 和 pixel counterfactual 中都稳定强于确定性随机 region，但 `P_FFN` 没有跨模型、跨 intervention 稳定优于 `P_WRITE`。因此旧 bbox 的负结果不能否定 vector 方法，新的因果结果也不能被表述成 FFN ranking 全面替代 WRITE。

本轮只分析已经完成正式提取的 **LLaVA-1.5-7B** 和 **InternVL2.5-8B**。结论属于预注册决策树中的 **Outcome C**：

\[
P^{JFFN}\approx P^{WRITE}\ \text{（排序高度相似）},\qquad [I,S]\approx I\ \text{（检测增益不可信）}.
\]

更具体地说：

- **Jacobian 没有在 WRITE 之上增加空间定位价值。** 在固定的第 16–32 层，LLaVA 的 JFFN 相比 WRITE 的 bbox mass、patch AUPRC 分别下降 0.00117 和 0.00401；InternVL 分别下降 0.00190 和 0.00636，置信区间都完全低于 0。Top‑1 在 LLaVA 基本持平，在 InternVL 显著下降 0.00809。
- **Jacobian sensitivity \(S\) 没有在 \(I\) 之外提供稳定的幻觉检测价值。** LLaVA 的 \([I,S]-I\) AUROC 为 \(+0.0232\)，但 95% CI 为 \([-0.0030,+0.0492]\)；InternVL 为 \(+0.00150\)，CI 为 \([-0.00043,+0.00330]\)。两者均跨 0。
- **signed \(Q_j\) 是数学上成立的机制诊断，但不是已验证的检测特征。** \(\sum_jQ_j\approx R\) 的最大相对误差低于 \(4.50\times10^{-7}\)。\(P^{Q+}\) 在 LLaVA 提高空间质量，但在 InternVL 只提高 bbox mass、同时降低 Top‑1/AUPRC，未跨模型复现。
- **target-logit/margin alignment 没有得到跨模型支持。** \(A^{margin,total}\) 的 direction-free Hall AUROC 在 LLaVA 为 0.612，在 InternVL 仅 0.508；而且 REAL 并未稳定获得更强的正 margin support。\(P^{LOGIT+}\) 的空间定位在两个模型上都弱于 WRITE/JFFN。
- **因果干预的计算路径正确，但效果尚未被数值实验验证。** causal row、target exclusion 和 attribution additivity 均通过；然而预测效应远小于 FP16/BF16 logit 的可分辨步长，导致 LLaVA 在 \(\eta=0.05\) 时 63.5% 的观测变化为 0，InternVL 为 88.5%。因此方向验证仅为 PARTIAL，幅值匹配为 FAIL，不能把当前 attribution 称为已验证的 causal effect。

三类贡献必须分开：

| 贡献 | 本轮结论 |
| --- | --- |
| Mechanistic measurement：测量 FFN 对视觉写入方向的局部响应 | **是** |
| Spatial interpretability：Jacobian 是否优于 WRITE-only | **否** |
| Hallucination detection：Jacobian 是否带来独立检测信息 | **未发现** |

# 2. Exact experimental scope

正式 token-map/scalar 实验覆盖全部 32 个 decoder layer，预注册主分析固定为第 16–32 层，没有用 test label 选层。

| Model | Images | Unique positions | Mentions | REAL | REAL with COCO boxes | Grid mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | 3971 | 14951 | 15463 | 11976 | 11418 | 0 |
| InternVL2.5-8B | 3927 | 11630 | 11759 | 9993 | 9392 | 0 |

下游 split 完全沿用现有 image-level split：

| Model | Train images / mentions | Test images / mentions | Test REAL / HALL |
| --- | ---: | ---: | ---: |
| LLaVA-1.5-7B | 3174 / 12317 | 797 / 3146 | 2437 / 709 |
| InternVL2.5-8B | 3142 / 9378 | 785 / 2381 | 2008 / 373 |

昂贵的 logit-gradient/activation-intervention 使用预注册分层子集：每模型 25 张图、200 个 target-layer case（100 REAL、100 HALL），层为 8/16/24/32；其中 16 个 target-layer case 执行 5 种 token ranking strategy 加 aggregate response、3 个 \(\eta\)，共 288 条 intervention row。

Entropy matching 只使用训练 split 固定抽取的 500 张图；test image 使用数为 0。LLaVA 的逐层 \(\beta_l\) 范围为 0.8904–1.1479，InternVL 为 0.9126–1.0517，最大熵匹配误差分别为 \(1.31\times10^{-7}\) 和 \(1.61\times10^{-7}\)。这里匹配的是 WRITE entropy，不是第一轮匹配 old-hpre entropy 的另一组 \(\beta_l\)。

Qwen2.5-VL 和 Qwen3-VL 没有完成正式提取，**本报告不包含也不外推它们的结果**。

# 3. WRITE-only baseline

本轮新增的正确控制项为

\[
I_{l,t,j}=\lVert a^j_{l,t}\rVert_2,\qquad
P^{WRITE}_{l,t,j}=
\frac{I_{l,t,j}}{\sum_k I_{l,t,k}+\epsilon}.
\]

它保留 attention probability、value vector、head concatenation 和 output projection 的共同影响，但不经过 FFN Jacobian。它不是 raw attention probability，也不是 aggregate \(I=\lVert\sum_j a_j\rVert_2\)。

实现同时修复了旧审计字段 component_sum_relative_error 被硬编码为 0 的问题。现在会重建全部 source-token contribution，output projection bias 只加一次，并保存实际测量值：

| Model | Mean component error over layer means | Max observed component error | Mean attention reconstruction error |
| --- | ---: | ---: | ---: |
| LLaVA | 0.00120 | 0.00708 | 0.00116 |
| InternVL | 0.01242 | 0.09855 | 0.01209 |

InternVL 的极端误差较高，需要解释：正式路径为了节省内存不保存 \(o^{attn}\)，reference 使用 BF16 的 \(h^{mid}-h^{pre}\)，减法舍入会同时抬高 attention reconstruction 与 component-sum error；两者的层均值几乎一致。该审计不参与 \(a_j,E_j,P\) 或训练，且先前保存 \(o^{attn}\) 的 FP32/smoke 验证已通过，所以它没有改变现有特征。但它提醒我们：不能再把伪零当作全量精度证据。

# 4. WRITE versus JFFN spatial results

## Table 1 — Spatial incremental value

以下为相同 REAL-with-box cohort、第 16–32 层的 mention-weighted 均值。

| Model | Method | Top-1 pointing | Patch AUPRC | BBox mass |
| --- | --- | ---: | ---: | ---: |
| LLaVA | ATTN | 0.41035 | 0.43486 | 0.35693 |
| LLaVA | WRITE | **0.50197** | **0.44776** | **0.38778** |
| LLaVA | JFFN | 0.50177 | 0.44375 | 0.38660 |
| LLaVA | \(Q^+\) | 0.52405 | 0.45564 | 0.42558 |
| InternVL | ATTN | 0.63604 | 0.44202 | 0.37627 |
| InternVL | WRITE | **0.63298** | **0.45149** | **0.39812** |
| InternVL | JFFN | 0.62489 | 0.44513 | 0.39622 |
| InternVL | \(Q^+\) | 0.61853 | 0.43911 | 0.43751 |

严格的 image-level paired bootstrap（10000 次）直接比较 JFFN − WRITE：

| Model | Metric | Difference | 95% CI |
| --- | --- | ---: | --- |
| LLaVA | Top-1 | -0.00020 | [-0.00117, +0.00076] |
| LLaVA | Patch AUPRC | -0.00401 | [-0.00413, -0.00389] |
| LLaVA | BBox mass | -0.00117 | [-0.00129, -0.00106] |
| LLaVA | Patch AUROC | -0.00841 | [-0.00857, -0.00824] |
| InternVL | Top-1 | -0.00809 | [-0.00913, -0.00707] |
| InternVL | Patch AUPRC | -0.00636 | [-0.00652, -0.00620] |
| InternVL | BBox mass | -0.00190 | [-0.00208, -0.00171] |
| InternVL | Patch AUROC | -0.00703 | [-0.00716, -0.00691] |

因此第一项 falsification 成立：**WRITE 已经解释当前 JFFN 的全部空间增益，Jacobian 没有在 WRITE 上继续改善定位。**

## Table 2 — Jacobian reranking

相关/overlap 是第 16–32 层均值；correction/regression 使用全部 32 层的相同 box cohort，因此二者不能混成同一个分母。

| Model | Pearson \(I_j,E_j\) | Spearman | Top-1 agreement | Top-32 overlap | Corrections | Regressions | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA | 0.98061 | 0.97865 | 0.86680 | 0.92575 | 7300 | 5916 | +1384 |
| InternVL | 0.99008 | 0.99286 | 0.87564 | 0.94807 | 4430 | 5398 | -968 |

Jacobian 的确改变约 12–13% 的 Top‑1，所以不是逐 token 常数缩放；但 LLaVA 的全层 correction net gain 与主分析层的 bbox/AUPRC 下降并不矛盾：前者混合全部层且只看 Top‑1，后者固定 16–32 层并评价完整排序/质量。InternVL 在两个视角下都显示 regression 多于 correction。

同图异目标特异性只略有变化：

| Model | Method | Cosine ↓ | JS ↑ | TV ↑ | Top-32 overlap ↓ | Top-1 agreement ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA | WRITE | 0.51323 | 0.19423 | 0.48529 | 0.40095 | 0.21565 |
| LLaVA | JFFN | 0.50467 | 0.19541 | 0.48668 | 0.39887 | 0.19853 |
| InternVL | WRITE | 0.29817 | 0.30481 | 0.63782 | 0.34653 | 0.06157 |
| InternVL | JFFN | 0.29630 | 0.30521 | 0.63672 | 0.35276 | 0.06187 |

JFFN entropy-matched-to-WRITE 后 Top‑1/AUPRC 排名指标不变，且 bbox 仍不优于 WRITE；因此差异不是简单由 sharpness 导致。

![Figure 1 LLaVA spatial](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure1_spatial_localization.png)

![Figure 1 InternVL spatial](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure1_spatial_localization.png)

# 5. Jacobian token gain analysis

定义

\[
G_{l,t,j}=\frac{\lVert J_f(z)a_j\rVert_2}{\lVert a_j\rVert_2+\epsilon}.
\]

第 16–32 层结果：

| Model | Mean \(G_j\) | Within-layer CV | Robust CV | Mean minimum \(\lVert a_j\rVert\) | \(<10^{-4}\) fraction | \(<10^{-6}\) fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA | 0.38941 | 0.16452 | 0.16450 | \(3.04\times10^{-4}\) | 0.00309 | \(1.64\times10^{-7}\) |
| InternVL | 0.36875 | 0.15114 | 0.15114 | \(6.79\times10^{-4}\) | 0.000565 | 0 |

\(G_j\) 的 CV 为 15–16%，所以 Jacobian 并非完全等价于逐层常数 \(c_l\)；同时 robust CV 与原 CV 几乎一致，说明结果不是 tiny denominator 人为放大的。没有 token 小于 \(10^{-8}\)，数值稳定。

但这种方向选择性没有指向目标框：

| Model | Mean gain inside box | Mean gain outside box | Gain-only Top-1 pointing |
| --- | ---: | ---: | ---: |
| LLaVA | 0.37368 | **0.39407** | 0.29180 |
| InternVL | 0.35403 | **0.37015** | 0.28643 |

换句话说，Jacobian 确实对方向有中等程度的不同增益，但它没有优先放大真实物体 box 内的视觉写入；这与 JFFN 不优于 WRITE 的空间结果一致。

![Figure 2 LLaVA reranking](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure2_jacobian_reranking.png)

![Figure 2 InternVL reranking](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure2_jacobian_reranking.png)

# 6. Incremental hallucination-detection value

全部模型均为 balanced logistic regression；连续特征只用 train statistics 标准化；主层固定 16–32；以下 AUPRC 均以 HALL 为正类。

## Table 3 — Scalar hallucination value

| Model | Feature set | AUROC | Hall AUPRC |
| --- | --- | ---: | ---: |
| LLaVA | \(I\) | 0.57351 | 0.24700 |
| LLaVA | \(S\) | 0.55225 | 0.25356 |
| LLaVA | \(R\) | **0.67220** | **0.29147** |
| LLaVA | \([I,S]\) | 0.59668 | 0.25276 |
| LLaVA | \([I,S,D]\) | 0.62331 | 0.28718 |
| LLaVA | old risk | 0.79908 | 0.50283 |
| LLaVA | old risk + \(I\) | 0.79898 | 0.50139 |
| LLaVA | old risk + \(I,S,D\) | 0.80082 | 0.50742 |
| InternVL | \(I\) | 0.65930 | 0.29724 |
| InternVL | \(S\) | 0.53756 | 0.21090 |
| InternVL | \(R\) | 0.67778 | 0.30250 |
| InternVL | \([I,S]\) | 0.66079 | 0.29526 |
| InternVL | \([I,S,D]\) | 0.70100 | 0.32234 |
| InternVL | old risk | 0.67194 | 0.29273 |
| InternVL | old risk + \(I\) | 0.75022 | 0.33868 |
| InternVL | old risk + \(I,S,D\) | 0.75901 | 0.35760 |

## Table 4 — Jacobian incremental value conditional on \(I\)

| Model | Comparison | \(\Delta\) AUROC (95% CI) | \(\Delta\) Hall AUPRC (95% CI) |
| --- | --- | --- | --- |
| LLaVA | \([I,S]-I\) | +0.02317 [-0.00300,+0.04917] | +0.00576 [-0.01188,+0.02169] |
| InternVL | \([I,S]-I\) | +0.00150 [-0.00043,+0.00330] | -0.00198 [-0.00993,+0.00543] |
| LLaVA | old+\(I,S\) − old+\(I\) | -0.00090 [-0.00328,+0.00158] | +0.00112 [-0.00590,+0.00830] |
| InternVL | old+\(I,S\) − old+\(I\) | +0.00030 [-0.00157,+0.00223] | -0.00219 [-0.00700,+0.00375] |

因此 \(S\) 的独立增益在两模型上都失败。InternVL 的 old+\([I,S,D]\) 相对 old+\(I\) 有显著增益（AUROC +0.00878，CI [+0.00019,+0.01735]；AUPRC +0.01892，CI [+0.00191,+0.03747]），但这个组合同时加入 \(S,D\)，不能归因于 \(S\)。

Residualized \(R\) 使用 train-only 回归

\[
\log(R+\epsilon)=\beta_0+\beta_1\log(I+\epsilon)+e
\]

后得到：

| Model | \(\beta_1\) | Residual AUROC | Hall AUPRC | Test mean REAL | Test mean HALL |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA | 0.70019 | 0.75107 | 0.40645 | +0.01892 | -0.07427 |
| InternVL | 0.90234 | 0.58199 | 0.21915 | +0.00227 | -0.01882 |

这表明 \(R\) 中确有 \(I\) 不能解释的结构，LLaVA 很强、InternVL 较弱；但它与简单 \(S=R/I\) 不等价，也没有跨模型达到同等强度。

# 7. Signed \(Q_j\) and cancellation

定义

\[
u=\frac{\delta^{visual}}{\lVert\delta^{visual}\rVert_2+\epsilon},
\qquad Q_j=u^\top\delta_j.
\]

守恒检验 \(\sum_jQ_j\approx R\) 的最大相对误差为 LLaVA \(4.00\times10^{-7}\)、InternVL \(4.49\times10^{-7}\)，PASS。

## Table 5 — Cancellation（第 16–32 层）

| Model | Label | PositiveMass | NegativeMass | Neg/Pos | CancellationRatio | \(Q_j<0\) fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA | REAL | 3.02176 | 0.02357 | 0.00401 | 0.47810 | 0.06425 |
| LLaVA | HALL | 2.60809 | 0.01703 | 0.00390 | 0.45324 | 0.05189 |
| InternVL | REAL | 4.42410 | 0.01987 | 0.00334 | 0.49024 | 0.12333 |
| InternVL | HALL | 3.43183 | 0.01522 | 0.00315 | 0.48998 | 0.12171 |

LLaVA 的 HALL cancellation ratio 更低，表示 unsigned energy 相对 aggregate response 的相消更强；但 HALL 的 NegativeMass 和负 token 比例反而更低。InternVL 的 cancellation ratio 几乎相同。因此“幻觉具有更多负贡献/更多相消”没有一致证据，\(Q_j\) 当前应被视为机制诊断。

负 \(Q\) mass 与 box overlap 加权后的 bbox fraction 为 LLaVA 0.17068、InternVL 0.19880；正 support 的 \(P^{Q+}\) 在 LLaVA 定位更好，但 InternVL 的 Top‑1/AUPRC 更差，不能只展示正热图而隐藏负值。

![Figure 4 LLaVA cancellation](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure4_cancellation_real_hall.png)

![Figure 4 InternVL cancellation](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure4_cancellation_real_hall.png)

# 8. Target-logit / margin-aligned Jacobian

实验严格使用“生成目标首 subtoken 之前”的 causal query row，目标 token 不进入 input prefix；competitor 由 clean forward 固定。两个模型的 prefix_excludes_target 均为 100%。

Attribution additivity 最大相对误差：

| Model | Logit additivity | Margin additivity |
| --- | ---: | ---: |
| LLaVA | \(4.65\times10^{-6}\) | \(2.32\times10^{-4}\) |
| InternVL | \(1.33\times10^{-4}\) | \(1.28\times10^{-5}\) |

LLaVA custom attention capture 与标准 kernel 的 clean logit 最大绝对差为 0.015625；InternVL 使用 official eager full-attention capture 后为 0。这是 kernel/dtype 差异，attribution 的同一路径 additivity 仍通过。

## Table 6 — Logit-aligned attribution（每模型 200 cases）

| Model | Quantity | REAL mean | HALL mean | Direction-free Hall AUROC |
| --- | --- | ---: | ---: | ---: |
| LLaVA | \(A^{logit,total}\) | +0.00310 | -0.02660 | 0.6344 |
| LLaVA | \(A^{margin,total}\) | -0.01129 | +0.00599 | 0.6120 |
| LLaVA | Margin positive mass | 0.04107 | 0.03332 | 0.5120 |
| LLaVA | Margin negative mass | 0.05235 | 0.02734 | 0.6755 |
| InternVL | \(A^{logit,total}\) | -0.00170 | -0.01287 | 0.5054 |
| InternVL | \(A^{margin,total}\) | -0.00441 | -0.00867 | 0.5081 |
| InternVL | Margin positive mass | 0.05918 | 0.03809 | 0.5527 |
| InternVL | Margin negative mass | 0.06359 | 0.04676 | 0.5728 |

LLaVA 的 \(A^{margin,total}\) 虽有 direction-free 分离，但方向与“REAL 获得更强正支持”的原假设不一致；InternVL 基本随机，因此没有复现。

正 margin support 的空间子集为 LLaVA 92 个 case、InternVL 96 个 case：

| Model | Method | Top-1 | Patch AUPRC | BBox mass |
| --- | --- | ---: | ---: | ---: |
| LLaVA | WRITE | 0.60870 | 0.50513 | 0.49059 |
| LLaVA | JFFN | 0.59783 | 0.49658 | 0.48745 |
| LLaVA | \(Q^+\) | 0.61957 | 0.52095 | 0.54759 |
| LLaVA | \(LOGIT^+\) | 0.40217 | 0.43596 | 0.42951 |
| InternVL | WRITE | 0.77083 | 0.53674 | 0.49719 |
| InternVL | JFFN | 0.75000 | 0.52657 | 0.49108 |
| InternVL | \(Q^+\) | 0.75000 | 0.52318 | 0.54379 |
| InternVL | \(LOGIT^+\) | 0.58333 | 0.44319 | 0.43547 |

所以 logit alignment 没有产生更好的目标空间支持。

同一 200-case 的概念阶段相关矩阵已保存。几个关键 Pearson correlation：

| Model / label | corr(\(I,R\)) | corr(\(R,S\)) | corr(\(R,A^{margin}\)) | corr(\(S,A^{margin}\)) |
| --- | ---: | ---: | ---: | ---: |
| LLaVA REAL | 0.7917 | 0.8967 | -0.1078 | -0.0793 |
| LLaVA HALL | 0.5186 | 0.8907 | -0.1048 | -0.1048 |
| InternVL REAL | 0.9164 | 0.2588 | -0.0724 | +0.0063 |
| InternVL HALL | 0.9205 | -0.0618 | -0.2818 | +0.0534 |

原始 total visual-attention mass 没有在这轮分片中持久化，因此 unified table 中该项明确为 **NOT RUN**；不能用和为 1 的 attention_distribution 代替。

![Figure 5 LLaVA margin](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/logit_causal/figure5_margin_attribution_real_hall.png)

![Figure 5 InternVL margin](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/logit_causal/figure5_margin_attribution_real_hall.png)

# 9. Causal intervention

干预发生在 FFN output：

\[
m'=m-\eta\delta_j,\qquad
\Delta Margin\approx-\eta A^{margin}_j.
\]

它没有错误地改动 pre-FFN residual，因此不混入 identity path。每个 intervention 都使用相同 no-grad hook、\(\eta=0\) 的 baseline，避免拿 custom-capture clean forward 与 intervention kernel 直接相减。

## Table 7 — Prediction versus observation

| Model | \(\eta\) | Pearson | Spearman | All-row sign accuracy | Nonzero sign accuracy | Slope | MAE | Zero observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA | 0.05 | 0.232 | 0.121 | 0.188 | 0.514 | 2.217 | 0.00726 | 0.635 |
| LLaVA | 0.10 | 0.419 | 0.235 | 0.229 | 0.579 | 1.829 | 0.00734 | 0.604 |
| LLaVA | 0.25 | 0.509 | 0.390 | 0.333 | 0.627 | 1.143 | 0.00946 | 0.469 |
| InternVL | 0.05 | 0.039 | 0.038 | 0.063 | 0.545 | 2.897 | 0.02921 | 0.885 |
| InternVL | 0.10 | 0.072 | 0.065 | 0.063 | 0.353 | 3.250 | 0.04536 | 0.823 |
| InternVL | 0.25 | 0.150 | 0.072 | 0.115 | 0.579 | 2.937 | 0.05151 | 0.802 |

关键不是 all-row sign accuracy 本身，而是 dtype resolution：

- LLaVA 的最小非零 observed \(|\Delta Margin|\) 为 0.0078125，而 \(\eta=0.05\) 的 median \(|prediction|\) 仅 0.000123。
- InternVL 的最小非零 observed \(|\Delta Margin|\) 为 0.25，而 median prediction 仅 0.000201。

因此当前 scatter 呈明显水平量化带。LLaVA 在较大 \(\eta\) 下出现中等相关，但这不构成小扰动局部线性验证；InternVL 基本无法分辨。结论应是 **实验路径通过、科学因果验证未完成**，而不是 attribution 已失败或已成功。

![Figure 6 LLaVA causal](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/logit_causal/figure6_predicted_vs_observed_intervention.png)

![Figure 6 InternVL causal](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/logit_causal/figure6_predicted_vs_observed_intervention.png)

# 10. Per-layer findings

REAL/HALL 的 \(I,R,S\)、cancellation 和 \(A^{margin,total}\) 都保留逐层结果；图中的阴影是 mention/case-level normal-approximation 95% CI。它们是描述性 CI，不代替 image-cluster bootstrap 的主比较。

- 两个模型的 \(I\) 与 \(R\) 在中后层多数时候 REAL 高于 HALL；\(S\) 两条曲线几乎重合，验证了“响应大小主要随 incoming write magnitude 变化”。
- LLaVA 的末层 \(R,S\) 急剧上升，InternVL 的层间峰谷也明显；这说明把 \(S\) 当成跨层统一物理尺度并不安全。
- 第 16–32 层内 \(I_j,E_j\) 的 Pearson/Spearman 始终很高；最后一层下降，但没有形成稳定的 box correction。
- \(A^{margin,total}\) 在层间换符号。LLaVA 的 REAL/HALL 差异主要由层 16/24/32 驱动，InternVL 在层 16 的方向又不同，因此不能事后挑一个层作为主结论。
- cancellation 曲线在 LLaVA 有小差异，在 InternVL 基本重合。

![Figure 3 LLaVA I/R/S](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure3_I_R_S_real_hall.png)

![Figure 3 InternVL I/R/S](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure3_I_R_S_real_hall.png)

# 11. Failure cases

1. **Spatial falsification succeeded。** JFFN 没有超过 WRITE，InternVL correction 甚至少于 regression。
2. **Conditional sensitivity falsification succeeded。** \(S\) 加到 \(I\) 或 old risk+\(I\) 后没有可信增益。
3. **Logit alignment 没有复现。** LLaVA 有弱的方向无关分离，但 InternVL 约等于随机，而且 \(LOGIT^+\) 空间图明显变差。
4. **Causal observation 被 dtype 量化限制。** 预测变化比可观测 logit/margin step 小 1–3 个数量级。当前不能据此断言局部 attribution 的因果方向成立。
5. **Signed statistics 没有统一 label pattern。** \(Q_j\) 守恒，但 HALL 是否更 cancellation 在两模型上不一致。
6. **InternVL 全量 audit reference 精度较差。** component-sum 最大误差 9.85%，来源与 BF16 的 \(h^{mid}-h^{pre}\) reference 一致；未来应选择少量样本保留直接 \(o^{attn}\) 做 FP32 audit。
7. **Raw attention total mass：NOT RUN。** 当前分片只保存归一化 attention map；没有重跑全量模型来补这个解释性变量。
8. **Kendall rank：NOT RUN。** Pearson、Spearman 和 Top-K overlap 已足够回答当前 rank-change 问题，未为“方便项”额外增加全量计算。
9. **全 cohort logit gradient/intervention：NOT RUN。** 只完成明确标注的 25-image stratified subset，不能外推为全量 causal result。

# 12. Computational cost

| Model | Second-round shards | Disk | Peak extraction GPU memory | Logit+causal subset wall time |
| --- | ---: | ---: | ---: | ---: |
| LLaVA | 80 | 6.23 GiB | 14.73 GiB | 274.4 s |
| InternVL | 80 | 2.19 GiB | 17.25 GiB | 292.4 s |

- WRITE baseline 只在已构造的 \(a_j\) 上增加 norm 与归一化；signed \(Q_j\) 只增加 aggregate response unit-vector projection。二者没有单独 instrument wall time，不能给出伪精确耗时。
- 主要成本仍是全部视觉方向的 JVP；WRITE/Q 的存储随 \(L\times M\) 增长。
- Logit gradient 对每个 target-layer 需要额外 downstream backward；intervention 对每个策略/\(\eta\) 需要额外 forward。上表时间包含 25 图的 attribution、spatial analysis 和 288 次 intervention，不是单项微基准。
- 全量 extraction wall time 当时未写进 audit JSON，因此本报告只给持久化的显存、磁盘和 logit/causal 精确时间。
- 离线 token-map 分析逐 shard 流式读取，没有合并巨大根 pickle；这控制了 RAM，但全量排序、COCO mapping 和 10000 次 bootstrap 的 CPU 时间仍较高。

# 13. Revised scientific interpretation

当前最合理的机制链是：

\[
\text{attention retrieval}
\rightarrow a_j\ \text{(visual residual write)}
\rightarrow J_fa_j\ \text{(local FFN response)}
\rightarrow g^\top J_fa_j\ \text{(target consequence)}.
\]

数据支持第一步到第二步的可测量性，但不支持把第三步的 hidden norm 归一化为核心 probability map：

1. \(P^{WRITE}\) 已经带来空间定位；\(P^{JFFN}\) 与它的相关和 Top-K overlap 极高。
2. \(G_j\) 有 15–16% 的变异，但 box 内 gain 反而低于 box 外，说明 Jacobian 的方向选择性并非对象定位增益。
3. 归一化 \(P\) 删除了 aggregate magnitude、direction 和 cancellation；而检测上更有用的是 \(I,R\) 或 residualized \(R\)，不是 \(S\) 或 JFFN token probability。
4. target-margin attribution 是概念上更贴近任务的量，但本轮没有跨模型分离，也没有高精度 causal validation，暂时不能作为最终方法。

因此不能写“Jacobian 提高空间解释和幻觉检测”。更准确的论文表述应是：

> Jacobian-based JVP 提供了可信的局部 FFN response measurement；然而在当前两个完成模型上，token-level spatial ranking 的收益主要由 attention-mediated WRITE magnitude 解释，且未观察到 Jacobian sensitivity 对 object hallucination detection 的稳定增量价值。

# 14. Recommended final formulation

短期建议：

1. **空间可视化使用 \(P^{WRITE}\) 作为主基线/主图。** 它更简单、两模型均不差于 JFFN，并明确表示 attention-mediated residual write。
2. **检测保留 magnitude。** 优先比较 old risk+\(I\)、old risk+\(R\)、old risk+residualized-\(R\)、old risk+\([I,R,D]\)，不要只替换归一化 \(P\)。
3. **\(S\) 作为机制诊断，不单独宣称检测贡献。** InternVL 的 \([I,S,D]\) 增益需要拆分 \(D\) 与 interaction 才能归因。
4. **保留 signed quantities。** 建议的数据结构为

\[
\left(I,R,S,D,\text{Cancellation},A^{margin,total},\{A^{margin}_j\}_{j\in\mathcal V}\right),
\]

但 \(A^{margin}\) 在下一轮高精度 causal validation 前只作为 exploratory。
5. **下一轮因果验证先解决数值分辨率。** 以 FP32 保留 LM-head logits/margin 和 intervention delta；使用 same-hook \(\eta=0\) baseline；根据预测量级自适应选择仍处于局部区间但超过量化步长的 \(\eta\)；先在 aggregate \(\delta^{visual}\) 上验证，再做单 token。
6. 若需要完整概念阶段相关表，下一轮 extraction 应额外保存未归一化 visual attention total mass；不要从 normalized map 反推。

代表性热图显示 WRITE/JFFN/Q+ 很相似，而 LOGIT-margin+ 更分散；它是定性例子，不代替全量统计：

![Figure 7 representative heatmaps](jffn_second_round_representative_heatmaps.png)

# 15. PASS/FAIL scientific checklist

| Check | Status | Evidence |
| --- | --- | --- |
| WRITE baseline implemented correctly | **PASS** | \(a_j\) norm，含 attention/value/output projection、不含 Jacobian |
| WRITE vs JFFN paired spatial comparison completed | **PASS** | 两模型、相同 cohort、image bootstrap 10000 |
| Jacobian causes meaningful token reranking | **PARTIAL** | Top‑1 改变约 12–13%，但高度相关且无稳定收益 |
| Jacobian reranking improves bbox localization | **FAIL** | 两模型 bbox mass 均显著下降 |
| \(S\) adds predictive value beyond \(I\) | **FAIL** | 两模型 AUROC/AUPRC CI 均跨 0 |
| Residualized \(R\) contains label information | **PARTIAL** | LLaVA 强，InternVL 弱但高于随机 |
| Signed \(Q_j\) conservation holds | **PASS** | max error \(<4.50\times10^{-7}\) |
| Cancellation differs between REAL/HALL | **PARTIAL** | LLaVA 有差异，InternVL 几乎无差异，方向指标不统一 |
| Target-logit gradient uses correct causal row | **PASS** | target excluded from prefix，100% |
| Logit attribution additivity holds | **PASS** | max relative error \(2.32\times10^{-4}\) |
| Margin attribution computed correctly | **PASS** | fixed clean competitor + downstream gradient |
| Margin attribution separates REAL/HALL | **FAIL** | LLaVA 弱且方向反直觉，InternVL AUROC 0.508 |
| Causal intervention direction matches prediction | **PARTIAL** | LLaVA 大 \(\eta\) 有中等相关；小 \(\eta\)/InternVL 受量化 |
| Causal intervention magnitude matches prediction | **FAIL** | zero bands、量化步长远大于 prediction、slope 不稳定 |
| Results replicated on LLaVA | **PASS** | full cohort + logit subset 完成 |
| Results replicated on InternVL | **PASS** | full cohort + logit subset 完成 |
| No train/test leakage | **PASS** | image overlap 0；standardization/regression/entropy fit train-only |
| Previous component-sum audit bug fixed | **PASS** | 保存真实非零误差，不再硬编码 0 |
| Raw total visual-attention mass included | **NOT RUN** | 本轮分片未持久化该量 |

# What should I send to ChatGPT?

## 可直接转发的自包含摘要

只分析完成正式提取的 **LLaVA-1.5-7B**（3971 图、14951 positions、15463 mentions）和 **InternVL2.5-8B**（3927 图、11630 positions、11759 mentions）；Qwen 未完成，未纳入。

### 1. WRITE vs JFFN localization（层 16–32）

| Model | Method | Top-1 | Patch AUPRC | BBox mass |
| --- | --- | ---: | ---: | ---: |
| LLaVA | WRITE | 0.50197 | 0.44776 | 0.38778 |
| LLaVA | JFFN | 0.50177 | 0.44375 | 0.38660 |
| InternVL | WRITE | 0.63298 | 0.45149 | 0.39812 |
| InternVL | JFFN | 0.62489 | 0.44513 | 0.39622 |

JFFN−WRITE paired bootstrap：LLaVA Top‑1 -0.00020 CI [-0.00117,+0.00076]、AUPRC -0.00401 CI [-0.00413,-0.00389]、bbox -0.00117 CI [-0.00129,-0.00106]；InternVL Top‑1 -0.00809 CI [-0.00913,-0.00707]、AUPRC -0.00636 CI [-0.00652,-0.00620]、bbox -0.00190 CI [-0.00208,-0.00171]。

### 2. Reranking 与 \(G_j\)

LLaVA \(corr(I_j,E_j)\) Pearson/Spearman=0.98061/0.97865，Top‑1 agreement=0.86680，correction/regression=7300/5916；InternVL=0.99008/0.99286，Top‑1 agreement=0.87564，correction/regression=4430/5398。Jacobian 会改排名，但没有净改善主分析层空间定位。

\(G_j\) mean/CV：LLaVA 0.3894/0.1645，InternVL 0.3687/0.1511；tiny denominator 很少，数值稳定。box 内 gain 反而低于 box 外（0.3737<0.3941；0.3540<0.3701）。

### 3. Hallucination incremental value

| Model | \(I\) AUROC/AUPRC | \([I,S]\) | Difference AUROC 95% CI |
| --- | --- | --- | --- |
| LLaVA | 0.57351/0.24700 | 0.59668/0.25276 | +0.02317 [-0.00300,+0.04917] |
| InternVL | 0.65930/0.29724 | 0.66079/0.29526 | +0.00150 [-0.00043,+0.00330] |

\(S\) 没有可信的 conditional increment。Residualized-\(R\)：LLaVA AUROC/AUPRC=0.75107/0.40645，InternVL=0.58199/0.21915，说明 \(R\) 有部分 \(I\) 未解释结构，但跨模型强度不同。

### 4. Signed \(Q\) / cancellation

\(\sum_jQ_j\approx R\) 最大相对误差：LLaVA \(4.00e{-7}\)，InternVL \(4.49e{-7}\)。LLaVA cancellation ratio REAL/HALL=0.4781/0.4532；InternVL=0.4902/0.4900。HALL 并未跨模型稳定表现出更多 negative mass 或负 token，因此 cancellation 目前只是机制诊断。

### 5. Logit/margin attribution

每模型 25 图、200 target-layer cases（100 REAL/100 HALL；层 8/16/24/32）。LLaVA \(A^{logit,total}\) REAL/HALL=+0.00310/-0.02660，\(A^{margin,total}\)=-0.01129/+0.00599，direction-free AUROC=0.634/0.612；InternVL=-0.00170/-0.01287、-0.00441/-0.00867，AUROC=0.505/0.508。没有跨模型复现。\(LOGIT^+\) 的 Top‑1/AUPRC/bbox 在 LLaVA 为 0.402/0.436/0.430，在 InternVL 为 0.583/0.443/0.435，均弱于 WRITE。

### 6. Causal intervention

FFN-output intervention 使用 \(m'=m-\eta\delta_j\)，相同 hook 的 \(\eta=0\) baseline。LLaVA Pearson 在 \(\eta=.05/.1/.25\) 为 .232/.419/.509；InternVL 为 .039/.072/.150。小 \(\eta\) 时 zero-observed fraction 为 LLaVA 63.5%、InternVL 88.5%；最小非零 observed step 分别为 0.0078125/0.25，而 median prediction 仅 0.000123/0.000201。因此 causal row/additivity PASS，但方向只 PARTIAL、幅值 FAIL，主要受 FP16/BF16 resolution 限制。

### 7. Bugs / NOT RUN

- 修复 component_sum_relative_error 硬编码 0；真实 max 为 LLaVA 0.00708、InternVL 0.09855。该诊断不参与特征；InternVL 高值与 BF16 \(h^{mid}-h^{pre}\) reference 一致。
- 修复 intervention baseline：改为相同 no-grad hook 的 \(\eta=0\)，不再与不同 kernel 的 capture clean 相减。
- InternVL compact attention capture 不兼容 remote attention，改用 official eager full attention。
- Raw total visual-attention mass、Kendall、全 cohort causal intervention 均明确 **NOT RUN**。

### 8. 最终判断

**Outcome C。** Jacobian JVP 是可信的 mechanistic measurement，但当前 \(P^{JFFN}\) 的空间收益主要来自 attention-mediated WRITE magnitude；Jacobian 没有在 WRITE 上改善 bbox/ranking，\(S\) 没有在 \(I\) 上增加稳定幻觉检测信息。signed/logit-aligned quantities 值得作为结构化诊断继续研究，但本轮没有跨模型检测证据，causal attribution 还需要 FP32-resolution intervention 才能验证。

最重要的图：

- [LLaVA WRITE/JFFN spatial](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure1_spatial_localization.png)
- [InternVL WRITE/JFFN spatial](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/figure1_spatial_localization.png)
- [Representative WRITE/JFFN/Q+/LOGIT+ heatmaps](jffn_second_round_representative_heatmaps.png)
- [LLaVA causal prediction vs observation](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/logit_causal/figure6_predicted_vs_observed_intervention.png)
- [InternVL causal prediction vs observation](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/logit_causal/figure6_predicted_vs_observed_intervention.png)
