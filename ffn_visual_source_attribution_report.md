# Vector FFN Source Attribution 正式实验报告

日期：2026-09-04
基线提交：`b11cfc0f`；实验在现有未提交工作树上增量执行，未提交、未覆盖旧实验。

> 当前执行状态：**FORMAL COMPLETE / PASS**。四模型的 audit、COCO4000 全层提取、13 组 detector、预注册 gate 和三类正式因果干预均已完成；所有必需 shard、状态、bootstrap 与 checksum 通过验收。

## 1. 方法与预注册决策

主 source attribution 定义为

\[
e_m=\int_0^1 J_G(z^0+\alpha A)a_m\,d\alpha,
\qquad
z^0=z-\sum_m a_m,
\qquad
A=\sum_m a_m.
\]

其中 `FFN_PATH_GROSS_m=||e_m||₂`，`P_FFN` 是 gross 的归一化分布；`PATH_SIGNED_Q_m` 是 `e_m` 在总有限 FFN effect 方向上的投影；gross strength、net strength 和 `kappa=net/gross` 分别记录总响应、净响应和 cancellation。Riesz 只保留为 downstream target consequence，不作为 FFN source attribution 定义。旧 bbox 指标只作 auxiliary sanity check，不参与方法成立或 gate 判定。

数值规则预注册为：以 Gauss–Legendre K64 作数值参考而非真值，从 K={4,8,16,32} 选同时通过两模型所有代表层五项阈值的最小值；`torch.func.linearize` 只有在两模型 FP32 synthetic/real 检查均通过误差、显存和速度条件时才启用。OOM 只允许按 token chunk `256/128/64/32` 回退，耗尽后必须 FAIL。

Detector 使用固定 image 8:2 split。`WRITE` block 只指 `D_EW` 轨迹；JS 与 OT 分开训练，共 13 组：共享 A/B/F，以及各距离的 C/D/E/G/H。三 seed 为 43/44/45，MLP `[128,64,32]`、dropout 0.3、batch 256、最多 100 epochs、无标准化、无重采样和无类别权重、minimum-train-loss checkpoint、train-REAL-F1 threshold，并同时报告固定 0.5。

条件 gate 是两模型 × JS/OT 四个 `G-C` seed-ensemble AUROC 差中，任一 10,000 次图片级 paired-bootstrap 95% CI 下界大于 0。只有通过才扩展 Qwen3/InternVL 并运行四模型正式因果实验。

## 2. 实现与验证

- `features/ffn_visual_path_attribution.py` 新增分块、批量、默认不保存 `[M,T,D]` 的 `streaming_vector_path_statistics()`；保留 `vector_path_components()` 为完整向量 parity oracle。零 gross 返回 uniform P，零 net 返回 zero signed-Q/kappa，并显式记录 degenerate flag。
- `features/ffn_visual_interactions.py` 新增 vector-valued sampled Shapley，保存 `[region,D]` attribution、标准误、running estimates、总 effect 与 vector completeness。
- 四个独立 CLI 分别负责 audit、全量提取、分析/gate 和 gated counterfactual；全部使用原子 shard、resume、状态、失败、环境、命令与 checksum 记录。
- 真实 full-path smoke 已在四模型各完成 1 图全层：Qwen2/LLaVA/Qwen3/InternVL 分别有 4/3/5/3 个 unique targets；所有主张量 shape 正确、有限，三个非负分布逐层归一化。扩展模型另完成独立 audit smoke。
- 核心路径、Shapley、JS/OT、feature registry、主模型 gate 隔离、resume、counterfactual selection/coverage 与 bootstrap 的定向测试通过；最终完整测试数见第 7 节。

## 3. 200 图 vector audit

### 3.1 覆盖与验收

| Model | Processed images | Target-layer cases | FP32/scaling/Shapley cases | Regions | Permutations | Remaining failures |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen2.5-VL-7B | 200 | 1,316 | 50 | 8/16 | 128 | 0 |
| LLaVA-1.5-7B | 200 | 1,420 | 50 | 8/16 | 128 | 0 |
| Qwen3-VL-8B | 200 | 1,456 | 50 | 8/16 | 128 | 0 |
| InternVL2.5-8B | 200 | 1,380 | 50 | 8/16 | 128 | 0 |

Qwen2 audit 发生过两次可恢复 CUDA OOM：第一次来自复制整层做 FP32，改为只把当前 norm+FFN 临时转 FP32 后恢复；第二次来自 FP32 K64 chunk 256，随后把预注册的 `256/128/64/32` 回退统一用于整个 audit path。两次均在首个失败处停止、保留已完成 shard、修复后 resume；没有跳过 case。最终两模型 audit 状态均为 PASS，checksum 完整。

Gate 通过后，Qwen3/InternVL 使用已经冻结的 K4，未参与重新选 K。Qwen3 image `150779` 的首次 FP32 K64 在全部 chunk 都 OOM；根因是仍持有 36 层 capture，而 audit 只需四个代表层。释放非代表层 capture 后原 case resume 成功。扩展 audit 最终同样为 200 图、50 个详细 case、0 remaining failure。Qwen3/InternVL 的 FP32 completeness median/p90/max 分别为 `3.41e-6/1.07e-5/2.98e-5` 与 `7.83e-6/3.86e-5/1.01e-4`；8-region Shapley/path median cosine 为 `0.9999934/0.9999979`，median norm relative error 为 `0.00372/0.00214`。

### 3.2 K 冻结

正式选择为 **Gauss–Legendre K=4**。K4 相对 K64 的逐代表层结果如下；全部明显通过 `rho≥.99 / JS≤.005 / Top32≥.95 / p90 gross≤1% / p90 |delta kappa|≤.01`。

| Model | Layer | Median Spearman | Median JS | Median Top-32 overlap | p90 gross rel. error | p90 abs(delta kappa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2 | 7 | 0.99999942 | 2.58e-9 | 1.000 | 6.47e-4 | 3.86e-4 |
| Qwen2 | 14 | 0.99999943 | 2.80e-9 | 1.000 | 7.69e-4 | 3.41e-4 |
| Qwen2 | 21 | 0.99999954 | 3.05e-9 | 1.000 | 8.19e-4 | 3.75e-4 |
| Qwen2 | 28 | 0.99999825 | 3.49e-8 | 1.000 | 1.75e-3 | 9.42e-4 |
| LLaVA | 8 | 0.99999975 | 2.57e-10 | 1.000 | 2.84e-4 | 1.57e-4 |
| LLaVA | 16 | 0.99999987 | 9.10e-11 | 1.000 | 2.71e-4 | 1.27e-4 |
| LLaVA | 24 | 0.99999975 | 1.71e-10 | 1.000 | 2.76e-4 | 1.33e-4 |
| LLaVA | 32 | 0.99999969 | 7.64e-10 | 1.000 | 2.97e-4 | 2.13e-4 |

### 3.3 FP32、quadrature 与 Shapley

Native K64 与 FP32 K64 的 median P Spearman 为 Qwen2 `0.99999934`、LLaVA `0.99999975`，median JS 为 `3.23e-9/2.21e-10`，Top-32 overlap 均为 1.0。p90 gross relative error 为 `0.00270/0.000111`，p90 |delta kappa| 为 `0.00801/0.000777`。FP32 trapezoid K64 与 FP32 Gauss K64 的 median Spearman/Top-32 均为 1.0，median JS 约 `1e-15`。

Vector Shapley 与 path region component 高度一致：Qwen2 8/16-region 的 case-level median cosine 为 `0.9999953/0.9999966`，median norm relative error 为 `0.00318/0.00308`；LLaVA 为 `0.9999981/0.9999985` 和 `0.00205/0.00178`。全部 Shapley vector completeness 为有限且保存的真实值；本次观测最大值为 0（每个 permutation 的 telescoping vector sum 在保存精度下精确闭合）。

FP32 Gauss K64 的 path completeness relative error median/p90/max 为 Qwen2 `4.41e-6/1.14e-5/2.25e-5`、LLaVA `4.24e-6/1.02e-5/2.78e-5`，支持积分与有限 FFN effect 的向量闭合。模型原生低精度 K4 的同一误差为 Qwen2 `0.0400/0.1244/1.0750`、LLaVA `0.00493/0.01568/0.17993`；这是小 net effect 下 BF16/FP16 endpoint subtraction 与 JVP 量化的真实相对误差，未硬编码成零，也未被用于 K 选择或 detector 特征。因而 source 排名的数值稳定性与原生精度下的 vector completeness 必须分别陈述。

Source scaling 呈连续且按 ranking 分离。相对 lambda=1 的有限 effect norm，中位 lambda=0 比例在 Qwen2 的最高 P_FFN token/最高 8-region/随机 region 为 `0.932/0.704/0.911`，LLaVA 为 `0.953/0.747/0.915`；lambda=1.25 时对应为 Qwen2 `1.017/1.076/1.023`、LLaVA `1.012/1.071/1.023`。

`linearize` 没有同时满足“相对 L2≤5e-3、峰值显存不增加、耗时至少下降 10%”的全部条件，因此正式 backend 冻结为现有 `vmap(jvp)`。

## 4. COCO4000 全层 source attribution

所有模型均使用冻结的 Gauss–Legendre K4 与 `vmap(jvp)`。无目标图片也进入 processed-image manifest；它们不伪造 position row。

| Model | Decoder layers | Processed | With target | No target | Unique targets | Mentions | Train / test mentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 28 | 4,000 | 3,679 | 321 | 8,654 | 8,717 | 6,985 / 1,732 |
| LLaVA-1.5-7B | 32 | 4,000 | 3,971 | 29 | 14,951 | 15,463 | 12,317 / 3,146 |
| Qwen3-VL-8B | 36 | 4,000 | 3,943 | 57 | 14,704 | 14,873 | 11,846 / 3,027 |
| InternVL2.5-8B | 32 | 4,000 | 3,927 | 73 | 11,630 | 11,759 | 9,378 / 2,381 |

全局 loader 验证了跨 shard 图片/target 无重复、position 集合与 mention target 集合精确一致、固定 8:2 image split 对齐、所有层的 `ATTENTION_EVIDENCE/P_WRITE/P_FFN/PATH_SIGNED_Q` shape 与 finite，以及三个非负分布的逐层归一化。正式 shard 只保存 compact 轨迹；完整 `e_m` 仅存在于 50-case audit 子集。固定、按 image/response 排序而非结果筛选的 REAL/HALL 示例图位于各模型 `figures/fixed_examples.png/.pdf`；旧 bbox 数值只保留为 auxiliary sanity check。

## 5. Detector、bootstrap 与预注册 gate

每个模型均完成 13 feature sets × seeds 43/44/45。下表为 test 上三 seed 概率平均后的 AUROC；所有固定 0.5 与 train-REAL-F1 threshold 的 AUROC、REAL/HALL AUPR、precision、recall、F1 位于各模型的 `tables/detector_metrics.csv`。

| Model | A | F | C_JS | G_JS | H_JS | C_OT | G_OT | H_OT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 | .8341 | .8694 | .7998 | .8718 | .8824 | .8054 | .8723 | .8912 |
| LLaVA | .8948 | .9010 | .8912 | .9026 | .9053 | .8900 | .9046 | .9068 |
| Qwen3 | .8789 | .8855 | .8650 | .8846 | .8945 | .8639 | .8919 | .8974 |
| InternVL | .8474 | .8697 | .8537 | .8742 | .8796 | .8489 | .8770 | .8842 |

预注册 gate 的四个比较全部通过，而不只是“任一通过”：

| Model | Comparison | AUROC delta | Image-level paired-bootstrap 95% CI | Gate |
| --- | --- | ---: | ---: | --- |
| Qwen2.5 | G_JS − C_JS | +.07192 | [.04113, .10490] | PASS |
| Qwen2.5 | G_OT − C_OT | +.06685 | [.03775, .09814] | PASS |
| LLaVA | G_JS − C_JS | +.01139 | [.00228, .02090] | PASS |
| LLaVA | G_OT − C_OT | +.01456 | [.00505, .02374] | PASS |

因此条件阶段状态为 `RUN_REQUIRED`。每个 detector 比较均恰有 10,000 个有效图片级 resample。扩展模型的描述性 `G-C` 也为正且 CI 下界大于 0，但没有被反向用于 gate。主要单块消融显示：`G-D/G-E` 大多显著为正；主模型的 `G-F` CI 均跨 0，而扩展模型只有 OT 的 `G-F` 明确为正。这意味着 gate 的大增益主要是相对单独 `D_EW` 加入 `S/kappa` 与其余距离，不能归因于“把三种距离拼在一起”本身。四模型两种距离的 `H-A` CI 均大于 0。

### 5.1 Strength、kappa 与 R_cos 事后探索性消融

这五组不属于预注册 13 组，也不参与 gate。它们复用完全相同的 image 8:2 split、seeds 43/44/45、MLP、checkpoint 和 threshold 协议；输入分别为全层 `gross_strength`、全层 `kappa=net_strength/gross_strength`、两者拼接、全层旧 `R_cos`，以及三者直接拼接。联合组不含 AE、JS 或 OT path-distance blocks。

| Model | Feature | Ensemble AUROC | Mean HALL-AUPR | Mean HALL-F1 |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5 | strength | .8569 | .3727 | .1613 |
|  | kappa | .7950 | .3312 | .2175 |
|  | strength+kappa | .8637 | .3970 | **.2379** |
|  | R_cos | .7988 | .3061 | .2399 |
|  | strength+kappa+R_cos | **.8696** | **.3988** | .2193 |
| LLaVA | strength | .8860 | .6828 | .6191 |
|  | kappa | .8693 | .6274 | .5633 |
|  | strength+kappa | .8912 | .6925 | .6367 |
|  | R_cos | .8735 | .6266 | .5787 |
|  | strength+kappa+R_cos | **.8998** | **.6983** | **.6390** |
| Qwen3 | strength | .8704 | .5655 | .5010 |
|  | kappa | .8152 | .4669 | .4234 |
|  | strength+kappa | .8721 | .5682 | .5329 |
|  | R_cos | .8292 | .5119 | .4298 |
|  | strength+kappa+R_cos | **.8785** | **.5820** | **.5394** |
| InternVL | strength | .8443 | .5054 | .3030 |
|  | kappa | .8151 | .4484 | .3619 |
|  | strength+kappa | .8594 | .5212 | .3971 |
|  | R_cos | .7771 | .3782 | .3254 |
|  | strength+kappa+R_cos | **.8648** | **.5321** | **.4855** |

`strength+kappa − strength` 的 seed-ensemble AUROC 增益及 10,000 次图片级 paired-bootstrap 95% CI 为：Qwen2 `+.00683 [.00120,.01254]`、LLaVA `+.00523 [.00083,.00948]`、Qwen3 `+.00170 [-.00319,.00654]`、InternVL `+.01511 [.00859,.02182]`。因此 kappa 在三个模型提供明确的小幅增益，在 Qwen3 没有显著增益；`strength+kappa − kappa` 四模型的 CI 下界均大于 0。再加入 R_cos 相对 `strength+kappa` 在 Qwen2、LLaVA、Qwen3 有明确 AUROC 增益，InternVL 的 CI 跨 0。总体上 strength 仍是跨模型最稳定、贡献最大的单信号。

全正式 cohort（train+test，仅作描述、不作为 test 指标）按 mention 绘制了逐层 median 与 IQR，四模型多数层均为 REAL strength 高于 HALL：Qwen2 `24/28`、LLaVA `23/32`、Qwen3 `33/36`、InternVL `29/32`。单层 raw strength 预测 REAL 的最高 AUROC 分别为 Qwen2 `.6965`（layer 7）、LLaVA `.7807`（layer 11）、Qwen3 `.7354`（layer 13）、InternVL `.7369`（layer 9）。曲线和逐层原始统计位于各模型 `figures/strength_real_hall_curve.png` 与 `tables/strength_real_hall_curve.csv`。

### 5.2 A 组来源及与旧实验的口径差异

正式表中的 A 是本方案内部 reference group，不是 `baseline/results/comparison/` 中的 SVAR/native baseline。其定义严格为 `R_cos + AE strength`：`R_cos` 从原 `features.pkl` 的 `dgst_t_hpre_raw_logit_gauss_source_hpre_cos_risk_sqrt_hpre_per_layer` 原样读取；`AE strength` 是每层非负 `attention × hpre_raw_logit_gauss gate` 在 support token 上求和。旧 selected-feature 实验最接近的一组则是 `R_cos + hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine`，第二个 block 是 EV 而不是 AE。

| Model | Formal A: ensemble AUROC | Formal A: per-seed mean±std | Old R_cos+EV: per-seed mean±std |
| --- | ---: | ---: | ---: |
| Qwen2.5 | .8341 | .8234±.0032 | .8286±.0068 |
| LLaVA | .8948 | .8876±.0024 | .8890±.0017 |
| Qwen3 | .8789 | .8633±.0112 | .8581±.0025 |
| InternVL | .8474 | .8386±.0032 | .8482±.0055 |

表面差异还来自两个训练/汇总口径：正式 A 使用三 seed 概率平均后再算一次 AUROC，旧 summary 报三个 seed AUROC 的均值±总体标准差；正式 A 的 `drop_last=False`，旧 selected-feature/YAML shared MLP 使用 `drop_last=True`。两边的图片 8:2 split、mention 数和标签 cohort 完全一致，AUROC 也不受 threshold 选择影响。因此不能把正式 A 的 ensemble 数直接与旧表的 per-seed mean、SVAR 原方法或 shared-MLP SVAR 横向当成同一实验。

### 5.3 AE strength 公式与替换为旧 mass×cosine 的全组对照

对层 \(\ell\)、当前 target response 位置 \(p\)、视觉 support \(V\) 和目标词 \(y\)，先取当前 target query row 的 post-softmax 多头平均注意力，并只在视觉 support 上重新归一化：

\[
a_{\ell j}=\frac{H^{-1}\sum_h A_{\ell h}[p,j]}
{\sum_{u\in V}H^{-1}\sum_h A_{\ell h}[p,u]},\qquad j\in V.
\]

令 \(r_{\ell j}=\operatorname{LMHead}_y(h^{pre}_{\ell j})\) 为视觉位置对目标词的 hpre raw logit，\(m_\ell=\operatorname{median}_{j\in V}r_{\ell j}\)，则当前名为 `hpre_raw_logit_gauss` 的 gate 实际实现为带 Gaussian-consistent MAD scale 的 sigmoid：

\[
\hat\sigma_\ell=1.4826\operatorname{median}_{j\in V}|r_{\ell j}-m_\ell|+\epsilon,
\qquad
g_{\ell j}=\operatorname{sigmoid}\!\left(\frac{r_{\ell j}-m_\ell}{\hat\sigma_\ell}\right).
\]

非归一化 attention evidence、AE strength 和用于 JS/OT 的 target distribution 分别为

\[
u_{\ell j}=\max(a_{\ell j}g_{\ell j},0),\qquad
AE_\ell=\sum_{j\in V}u_{\ell j},\qquad
T_{\ell j}=\frac{u_{\ell j}}{AE_\ell}.
\]

零和时实现返回 uniform \(T\)。由于 \(a\) 已归一化且 \(g\in(0,1)\)，`AE strength` 本质上是 attention-weighted mean gate，通常位于 `[0,1]`；它不是 FFN vector path 的 `gross_strength`。

旧 mass×cosine EV 使用相同 \(T_\ell\)，取稳定 Top-32 集合 \(R_\ell\)（不足 32 时取全部），再定义

\[
M_\ell=\sum_{j\in R_\ell}T_{\ell j},\qquad
C_\ell=|R_\ell|^{-1}\sum_{j\in R_\ell}
\cos(h^{pre}_{\ell p},h^{pre}_{\ell j}),\qquad
EV_\ell=M_\ell C_\ell.
\]

所以 AE 是全视觉 support 上的非负门控注意力总量；旧 EV 是 Top-32 concentration 与 hpre state alignment 的有符号乘积。以下实验只把正式 13 组中的 AE block 替换成旧 EV，`R_cos/S/kappa/JS/OT`、固定图片 split、seeds、MLP、`drop_last=False` 和其他训练协议均保持不变。表中为旧 EV 版本的 seed-ensemble AUROC，括号为 `old EV − AE`；粗体表示 10,000 次图片级 paired-bootstrap CI 不含 0。

| Feature | Qwen2.5 | LLaVA | Qwen3 | InternVL |
| --- | ---: | ---: | ---: | ---: |
| A | .8468 (+.0127) | .8953 (+.0005) | .8725 (-.0065) | .8555 (+.0081) |
| B | .7931 (+.0053) | .8679 (-.0086) | .8350 (-.0135) | .8430 (+.0083) |
| F | .8692 (-.0002) | **.8949 (-.0061)** | **.8786 (-.0069)** | .8722 (+.0024) |
| C_JS | .8093 (+.0094) | .8908 (-.0004) | .8653 (+.0003) | .8499 (-.0039) |
| D_JS | .8223 (+.0005) | .8805 (-.0051) | .8461 (-.0087) | .8566 (+.0025) |
| E_JS | .8139 (+.0085) | .8857 (+.0011) | .8702 (+.0057) | .8579 (+.0042) |
| G_JS | .8771 (+.0053) | .9024 (-.0002) | .8835 (-.0011) | .8713 (-.0030) |
| H_JS | .8846 (+.0022) | .9045 (-.0008) | **.8880 (-.0066)** | .8755 (-.0041) |
| C_OT | .8107 (+.0053) | **.8805 (-.0095)** | .8656 (+.0017) | .8483 (-.0006) |
| D_OT | .8354 (+.0102) | **.8842 (-.0127)** | .8584 (-.0086) | .8596 (+.0061) |
| E_OT | .8035 (-.0082) | .8806 (-.0076) | .8645 (+.0073) | .8521 (+.0036) |
| G_OT | .8726 (+.0003) | .9039 (-.0007) | **.8851 (-.0068)** | .8697 (-.0073) |
| H_OT | .8899 (-.0013) | .9053 (-.0015) | **.8864 (-.0110)** | .8763 (-.0079) |

52 个直接比较中没有任何一个显著支持旧 EV；显著下降共有 7 项：LLaVA 的 F `-.0061 [-.0106,-.0018]`、C_OT `-.0095 [-.0187,-.0004]`、D_OT `-.0127 [-.0230,-.0029]`，以及 Qwen3 的 F `-.0069 [-.0114,-.0025]`、H_JS `-.0066 [-.0115,-.0019]`、G_OT `-.0068 [-.0120,-.0015]`、H_OT `-.0110 [-.0161,-.0061]`。Qwen2/InternVL 的全部 CI 跨 0。结论是旧 EV 可以在 A 等个别点估计上升，但没有可靠增益；对含 `S/kappa` 的完整组，保留 AE 更稳妥。

### 5.4 AE 与 Top-32 mass 的区别，以及 AE×Cosine 对照

`AE` 的求和范围是当前图片的全部视觉 token support（不含文本 token），每个视觉 token 对应模型视觉网格中的 patch/merged patch，而不是人工 bbox。它是每层一个绝对证据强度标量：

\[
AE_\ell=\sum_{j\in V}u_{\ell j}.
\]

`M` 则先用 AE 把同一批 token 归一化成 \(T_{\ell j}=u_{\ell j}/AE_\ell\)，再只累加 Top-32：

\[
M_\ell=\sum_{j\in R_\ell}T_{\ell j}
=\frac{\sum_{j\in R_\ell}u_{\ell j}}{AE_\ell}.
\]

因此 `AE` 衡量“全图视觉证据有多强”，`M` 衡量“已经归一化的证据有多集中”；若对全部视觉 token 求 `M`，它恒等于 1。按用户指定的新对照固定使用旧 Top-32 区域 cosine，但用 `AE` 替代 `M`：

\[
AE\!C_\ell=AE_\ell C_\ell,
\qquad
C_\ell=|R_\ell|^{-1}\sum_{j\in R_\ell}
\cos(h^{pre}_{\ell p},h^{pre}_{\ell j}).
\]

这与旧 \(MC\) 的区别只在乘数；它也不同于 Top-32 的绝对证据 \(AE\,M\,C=(\sum_{j\in R}u_j)C\)。实验将 13 组中的 AE block 原位替换为 \(AE C\)，其余 block 与训练协议保持不变。四模型各完成 13 组×3 seeds，共 156 个 checkpoint。遵照用户要求，本实验不做 bootstrap；下表只给 seed-ensemble AUROC 点估计，括号为 `AE×Cosine − AE`，不得解释为显著差异。

| Feature | Qwen2.5 | LLaVA | Qwen3 | InternVL |
| --- | ---: | ---: | ---: | ---: |
| A | .8377 (+.0036) | .8983 (+.0035) | .8819 (+.0030) | .8610 (+.0136) |
| B | .7951 (+.0074) | .8689 (-.0076) | .8482 (-.0002) | .8293 (-.0053) |
| F | .8659 (-.0035) | .9012 (+.0002) | .8838 (-.0017) | .8699 (+.0001) |
| C_JS | .8144 (+.0146) | .8865 (-.0047) | .8655 (+.0004) | .8510 (-.0027) |
| D_JS | .8316 (+.0097) | .8810 (-.0046) | .8506 (-.0042) | .8526 (-.0015) |
| E_JS | .8235 (+.0181) | .8861 (+.0014) | .8686 (+.0041) | .8480 (-.0058) |
| G_JS | .8669 (-.0048) | .9048 (+.0023) | .8861 (+.0016) | .8727 (-.0016) |
| H_JS | .8799 (-.0025) | .9072 (+.0019) | .8927 (-.0019) | .8790 (-.0006) |
| C_OT | .8101 (+.0046) | .8838 (-.0063) | .8630 (-.0009) | .8369 (-.0120) |
| D_OT | .8399 (+.0146) | .8930 (-.0039) | .8667 (-.0003) | .8496 (-.0038) |
| E_OT | .8121 (+.0004) | .8847 (-.0035) | .8574 (+.0002) | .8420 (-.0066) |
| G_OT | .8736 (+.0013) | .9066 (+.0020) | .8903 (-.0016) | .8719 (-.0051) |
| H_OT | .8873 (-.0040) | .9097 (+.0029) | .8945 (-.0030) | .8807 (-.0036) |

52 个点差中 23 个为正，平均差仅 `+0.00007`。Qwen2 平均 `+.00459`，但其完整 G/H 组有升有降；LLaVA/Qwen3/InternVL 的跨组平均分别为 `-.00127/-.00034/-.00269`。因此 `AE×Cosine` 没有形成跨模型一致增益，不能替代 AE；其较清楚的用途是作为“证据强度是否需要 signed state-alignment 调制”的探索性敏感性分析。

## 6. 条件扩展与因果阶段

Gate 通过后，Qwen3/InternVL 完成了不重选 K 的 audit、全层 COCO4000 与 detector；随后四模型各固定选择 100 张图、每图一个 label-balanced target、四个代表层、regular 8-region，共 **1,600 target-layer cases**。每个 case 对最高 `P_FFN`、最高 `P_WRITE` 和确定性随机 region 建立策略映射；相同 region 只执行一次。

三类干预均正式运行：fixed-QK 只减当前 target row 的 clean-QK visual value contribution；activation patching 把对应 mean-filled pixel counterfactual 的 selected-region visual residual states patch 到 clean 当前层输入；pixel counterfactual 保持尺寸并以图像均值填充输入矩形后完整重跑。所有 shard 保存 current-layer vector delta、target logit、固定 clean competitor margin 与 log-probability delta。

下表给出最可比的 log-probability 绝对效应排名差，定义为图片内四层平均的 `|Δlog p|_left-|Δlog p|_right`，再按图片 paired bootstrap。每项均为 10,000 个有效 resample。

| Model | Intervention | P_FFN − random, mean [95% CI] | P_FFN − P_WRITE, mean [95% CI] |
| --- | --- | ---: | ---: |
| Qwen2.5 | fixed-QK | .0271 [.0163, .0396] | .00025 [-.00143, .00199] |
| Qwen2.5 | activation patch | .1933 [.1182, .2823] | .00868 [-.00597, .02564] |
| Qwen2.5 | pixel CF | .7084 [.4401, 1.0142] | -.00990 [-.05023, .03338] |
| LLaVA | fixed-QK | .0103 [.00688, .01434] | .00033 [-.00011, .00100] |
| LLaVA | activation patch | .1114 [.0571, .1835] | .01259 [.00102, .02914] |
| LLaVA | pixel CF | .2482 [.1443, .3757] | .00717 [-.01614, .03198] |
| Qwen3 | fixed-QK | .0100 [.00528, .01530] | .00044 [-.00095, .00180] |
| Qwen3 | activation patch | .2608 [.1243, .4214] | .01215 [-.00308, .04095] |
| Qwen3 | pixel CF | .5337 [.2582, .8449] | -.04903 [-.11050, .00712] |
| InternVL | fixed-QK | .0111 [.00581, .01665] | -.00000 [-.00045, .00047] |
| InternVL | activation patch | .1299 [.0755, .1923] | -.00173 [-.00467, .00005] |
| InternVL | pixel CF | .4464 [.2473, .6709] | -.02646 [-.05941, -.00274] |

结论是：`P_FFN` 与 `P_WRITE` 在三种因果层级上都稳定强于随机 region；但 `P_FFN` 并未跨模型、跨 intervention 稳定优于 `P_WRITE`。唯一明确支持 `P_FFN>P_WRITE` 的 log-probability 项是 LLaVA activation patch；InternVL pixel CF 反而明确支持 `P_WRITE>P_FFN`，其余 CI 跨 0。因此 vector path attribution 获得了“非随机因果定位”的支持，但没有获得“普遍替代 WRITE ranking”的支持。

不同 estimand 的差距同样不能混为一谈：

| Model | `e_R` vs FFN removal branch delta: mean norm rel. error | `a_R+e_R` vs block residual delta |
| --- | ---: | ---: |
| Qwen2.5 | .3211 | .2411 |
| LLaVA | .0818 | .0438 |
| Qwen3 | .4230 | .2886 |
| InternVL | .4450 | .2903 |

LLaVA 最接近，其他模型差距较大，符合 Aumann–Shapley shared-path allocation 与单 region removal effect 在非线性/交互下并非同一 estimand。完整 27 个 bootstrap 比较/模型（3 intervention × 3 score × 3 ranking contrast）保存在 `metrics/counterfactual_results.json`。

## 7. 可复现性与最终验收

正式命令：

```bash
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_visual_source_audit.py --model MODEL --device CUDA --num-images 200 --audit-cases 50 --permutations 128 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_study.py --stage freeze
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_visual_source_attribution.py --model MODEL --device CUDA --k 0 --jvp-backend auto --num-images 0 --shard-images 10 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_study.py --stage detectors --training-device cuda:0 --bootstrap-resamples 10000 --resume
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_visual_source_counterfactuals.py --model MODEL --device CUDA --num-images 100 --bootstrap-resamples 10000 --resume
```

每个模型的固定输出根为：

`outputs/<model>/COCO4000-INSLEN-OFFICIAL-TARGET/results/ffn_visual_source_attribution_v1/`

最终验收：四模型 audit 均为 200/50/0，full extraction 均为 4000 processed images，13×3 detector seed 完整，所有 detector/counterfactual bootstrap 均恰为 10,000，四模型各 100 个 counterfactual shard/400 cases，且无缺层、缺 family/strategy、重复映射或非有限值。独立 checksum 审计逐字节复算 26 份 manifest、2,823 个唯一文件、32,562,887,917 bytes，结果全部 PASS。定向回归测试最终为 64/64 PASS，`py_compile` 与 `git diff --check` PASS。

运行中所有可恢复问题均保留证据：Qwen2 两类 audit OOM、Qwen3 非代表层 capture OOM、LLaVA 全量双 rank resume、Qwen3 全量单卡转双卡 resume，以及 InternVL counterfactual image `120777` 的完整 pixel capture cache OOM。最后一项改为只缓存四个所需层的 visual `h_prev`、target row 与 logits 后，原失败图片重试并完成；没有 case 被跳过。旧 scalar-path Qwen3、VQA、neuron intervention 和 bbox 优化均未运行，符合预注册范围。

最终状态：**FORMAL COMPLETE / PASS**。科学结论不是“FFN path 全面优于 WRITE”，而是：vector attribution 数值定义与 detector 增量成立，`P_FFN/P_WRITE` 都具有显著的非随机因果定位能力；两者之间没有跨模型、跨 intervention 的稳定胜者。
