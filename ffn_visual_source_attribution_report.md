# Vector FFN Source Attribution 正式实验报告

日期：2026-09-04
基线提交：`b11cfc0f`；实验执行时在现有未提交工作树上增量开发，未覆盖旧实验。正式研究随后发布于 `d6f0728`，本次增补结果截至 2026-09-06。

GitHub 发布范围（2026-09-07）：实验代码、主报告、紧凑 JSON/CSV 指标和 PNG/PDF 曲线。原始特征、完整向量、训练预测 `.pt` 与模型 checkpoint 仍保留在本地，不包含在 Git 仓库中；checksum 清单记录的是本地完整验收集合，不表示其中全部文件均已上传。

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

### 5.5 JS 改为 pairwise Union-Top32 的对照

正式 JS 在全部视觉 token 上计算；本对照只改 JS support。对每个分布对 \(P,Q\in\{(E,W),(W,F),(E,F)\}\) 分别定义

\[
U_{PQ}=\operatorname{Top32}(P)\cup\operatorname{Top32}(Q),\qquad |U_{PQ}|\le64,
\]

随后把 \(P,Q\) 截取到各自的 \(U_{PQ}\)、分别重新归一化，再计算 JS。三对使用三个独立 union，不共享区域。实验仅重训 `C_JS/D_JS/E_JS/G_JS/H_JS`，AE、`R_cos`、S、kappa、OT、固定图片 split 和三 seed 训练协议均不变，共完成 4 模型×5 组×3 seeds=60 个 checkpoint。按用户要求不做 bootstrap；下表是 seed-ensemble AUROC 的“全 support → Union-Top32（差值）”。

| Feature | Qwen2.5 | LLaVA | Qwen3 | InternVL |
| --- | ---: | ---: | ---: | ---: |
| C_JS | .7998 → .7969 (-.0030) | .8912 → .8869 (-.0043) | .8650 → .8630 (-.0021) | .8537 → .8562 (+.0025) |
| D_JS | .8218 → .8159 (-.0060) | .8856 → .8851 (-.0005) | .8548 → .8632 (+.0084) | .8542 → .8483 (-.0058) |
| E_JS | .8054 → .8012 (-.0042) | .8847 → .8810 (-.0037) | .8645 → .8624 (-.0021) | .8537 → .8498 (-.0039) |
| G_JS | .8718 → .8712 (-.0005) | .9026 → .9020 (-.0006) | .8846 → .8906 (+.0060) | .8742 → .8762 (+.0020) |
| H_JS | .8824 → .8853 (+.0029) | .9053 → .9072 (+.0019) | .8945 → .8931 (-.0015) | .8796 → .8749 (-.0047) |

20 个模型×组点差中只有 6 个为正，平均差 `-.00097`。`E_JS` 四模型全部下降，平均 `-.00350`；`G_JS` 平均 `+.00171`，但只在 Qwen3/InternVL 上升；`H_JS` 在 Qwen2/LLaVA 小升、在 Qwen3/InternVL 下降。Union-Top32 因而没有跨模型一致改善，主方法继续保留全视觉 support JS；它可以作为稀疏-support 敏感性对照，但当前点估计不支持替换正式 JS。

### 5.6 C/Q 使用边界、Snet 与 D_OT+strength

`C_m` 与 `Q_m` 不是同一信号。`C_m` 是带 downstream target gradient 的 target consequence：

\[
C_m^{target}=\int_0^1\nabla S_y(\alpha)^\top
J_G(z(\alpha))a_m\,d\alpha.
\]

它保留在既有 Riesz/path 基础设施和旧解释层中，但本轮正式全层提取按预注册明确不运行 downstream gradient，因此 **C_m 没有进入 COCO4000 compact shard、13 组 detector、gate 或因果 region ranking**。正式 A/H 中的 `R_cos` 是旧 hpre-cos/sqrt-matched-state OT risk，不是 `C_m`。

`Q_m=\hat\Delta^\top e_m` 则已在四模型所有正式 target-layer 保存为 `PATH_SIGNED_Q`，并通过 shape、finite、chunk parity、零 net 和 vector completeness 相关检查，也用于固定 signed-Q 示例图。但预注册 detector 只使用 gross 分布、gross strength、kappa 和三组距离；因果 ranking 只比较 `P_FFN/P_WRITE/random`，所以 **Q_m 已计算和验证，但没有作为 detector 输入或 region ranking**。

追加实验直接使用已保存的净强度

\[
S_{net}=\left\|\sum_m e_m\right\|_2
=\|G(z)-G(z^0)\|_2,
\]

并把 `D_OT+strength` 精确定义为 `AE + D_WF^OT + gross_strength`；这里的 `D_OT` 沿用正式 D 组，而不是三种 OT 的合称。两组均复用相同 split、三 seeds 和 MLP，不做 bootstrap。

| Model | Snet AUROC | Gross S AUROC | Δ | Snet HALL-AUPR / F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 | .8586 | .8569 | +.0017 | .3986 / .2614 |
| LLaVA | .8799 | .8860 | -.0061 | .6591 / .5818 |
| Qwen3 | .8670 | .8704 | -.0034 | .5695 / .4952 |
| InternVL | .8393 | .8443 | -.0050 | .4945 / .3284 |

| Model | D_OT+strength AUROC | D_OT AUROC | Δ | HALL-AUPR / F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 | .8697 | .8252 | +.0445 | .3990 / .2972 |
| LLaVA | .8987 | .8970 | +.0017 | .7021 / .6506 |
| Qwen3 | .8811 | .8669 | +.0141 | .5997 / .5506 |
| InternVL | .8606 | .8534 | +.0071 | .5295 / .4850 |

Snet 单独相对 gross strength 仅 Qwen2 小升，其余三模型下降，说明把 gross transformation 和 cancellation 压成一个乘积没有稳定收益。`D_OT+strength` 四模型均高于 D_OT 单独，但仍低于各模型正式 `G_OT`；strength 提供了清楚的增量，完整的三距离与 kappa 仍保留额外信息。

全 cohort 的逐层 median/IQR 曲线如下：

| Model | Snet: REAL>HALL layers | Snet 最强单层 REAL-AUROC | kappa: REAL>HALL layers | kappa 单层 AUROC 范围 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 | 26/28 | L7: .6924 | 14/28 | [.3444, .6211] |
| LLaVA | 26/32 | L7: .7805 | 25/32 | [.2330, .7413] |
| Qwen3 | 36/36 | L13: .7298 | 10/36 | [.3746, .5796] |
| InternVL | 32/32 | L9: .7359 | 14/32 | [.3843, .6176] |

Snet 曲线整体随深度呈锯齿式放大，REAL median 在绝大多数层高于 HALL，主要间隔出现在早中层，但 IQR 仍有重叠。kappa 曲线不是单向的类别偏移：不同层频繁交叉，Qwen2/Qwen3/InternVL 很多层反而是 HALL 更高，LLaVA 多数中后层 REAL 更高；其检测力来自整条层轨迹，而不是“kappa 越高越真实”的单调规则。完整曲线与逐层数值位于各模型 `figures/{net_strength,kappa}_real_hall_curve.png` 和对应 CSV。

### 5.7 Bounded strength 对照（2026-09-06）

按后续问题追加软饱和变换，只作用于 detector 输入，不修改 `e_m`、`P_FFN`、原始 strength 或 kappa：

\[
S_{bounded,\ell}=\frac{S_{gross,\ell}}
{S_{gross,\ell}+\tau_\ell+\epsilon},\qquad
\tau_\ell=\operatorname{median}_{\text{training mentions}}S_{gross,\ell}.
\]

`tau` 由每个模型的训练集逐层独立估计，测试集只复用训练值；不使用 label，也不读取测试统计。只训练回答问题所需的 `bounded_strength` 与 `D_OT+bounded_strength` 两组，后者仍精确定义为 `AE+D_WF^OT+bounded_strength`。split、三 seeds、MLP、checkpoint 和 threshold 均不变，不做 bootstrap。

| Model | Bounded S | Raw S | delta | D_OT+bounded S | D_OT+raw S | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 | .8476 | .8569 | -.0093 | .8881 | .8697 | +.0184 |
| LLaVA | .8933 | .8860 | +.0073 | .9064 | .8987 | +.0078 |
| Qwen3 | .8812 | .8704 | +.0108 | .9027 | .8811 | +.0217 |
| InternVL | .8550 | .8443 | +.0106 | .8681 | .8606 | +.0076 |

组合组的 HALL-AUPR/HALL-F1 从 raw 到 bounded 分别为：Qwen2 `.3990/.2972 → .4816/.4680`、LLaVA `.7021/.6506 → .6989/.6598`、Qwen3 `.5997/.5506 → .6519/.5974`、InternVL `.5295/.4850 → .5240/.5075`。因此 AUROC 和 HALL-F1 在四模型组合中全部提高；HALL-AUPR 在 Qwen2/Qwen3 提高、LLaVA/InternVL 略降。

变换后的全 train/test 实际范围为 Qwen2 `[.0427,.9651]`、LLaVA `[.0292,.9605]`、Qwen3 `[.0145,.9763]`、InternVL `[.0153,.9420]`；各层训练 median 按定义映射到 0.5。`tau` 的逐层范围分别为 `[.419,101.442]`、`[.493,18.197]`、`[.0876,38.142]`、`[.124,16.278]`。完整 calibration、曲线和指标位于各模型 `tables/bounded_strength_calibration.csv`、`figures/bounded_strength_real_hall_curve.png` 与 `tables/bounded_strength_feature_metrics.csv`。

结果支持“尺度约束有利于 strength 与 D_OT 拼接”的优化解释，因为 `D_OT+bounded strength` 四模型一致优于 raw 组合；但 bounded strength 单独在 Qwen2 下降，说明变换不是无条件提升信号本身。由于这是无 bootstrap 的事后消融，暂不替换预注册正式 G/H 结果。

### 5.8 提取时相对 strength 对照（2026-09-06）

进一步测试完全不依赖训练集标定的 0--1 特征。对每个 target-layer 直接使用提取阶段已有量：

\[
R_\ell=\sum_m\lVert e_{m,\ell}\rVert_2,
\qquad
W_\ell=\sum_m\lVert a_{m,\ell}\rVert_2,
\qquad
S_{relative,\ell}=\frac{R_\ell}{R_\ell+W_\ell}.
\]

其中 `R` 是原 gross FFN path response，`W` 是同一视觉 source writes 的 gross input energy。`R+W` 退化为 0 时返回 0；实现将结果限制在 `[0,1)`。该比值无量纲、逐样本逐层计算，不读取 label、训练集或测试集统计，也没有可拟合的 `tau`。它表达的是“FFN path response 占 response+input-write 的相对比例”，不再等同于原始绝对 response strength。

只训练 `relative_strength` 和 `D_OT+relative_strength`；后者仍是 `AE+D_WF^OT+relative_strength`。split、三 seeds、MLP、checkpoint 与 train-REAL-F1 threshold 全部复用正式协议，不做 bootstrap。

| Model | Relative S | Raw S | delta | D_OT+relative S | D_OT+raw S | delta | vs D_OT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 | .8237 | .8569 | -.0331 | .8575 | .8697 | -.0123 | +.0322 |
| LLaVA | .8830 | .8860 | -.0030 | .9064 | .8987 | +.0077 | +.0094 |
| Qwen3 | .8871 | .8704 | +.0167 | .8928 | .8811 | +.0118 | +.0259 |
| InternVL | .8671 | .8443 | +.0227 | .8697 | .8606 | +.0091 | +.0162 |

`D_OT+relative_strength` 的 HALL-AUPR/HALL-F1 分别为 Qwen2 `.4117/.3870`、LLaVA `.6974/.6432`、Qwen3 `.6215/.5639`、InternVL `.5295/.5030`。全 train/test 实际范围为 Qwen2 `[.1930,.6893]`、LLaVA `[.1355,.7487]`、Qwen3 `[.0977,.8834]`、InternVL `[.1502,.8466]`，均已在特征构造时落入 `[0,1)`。

REAL median 高于 HALL median 的层数依次为 `15/28、24/32、30/36、16/32`；最佳单层 REAL-AUROC 为 Qwen2 L21 `.6616`、LLaVA L19 `.7231`、Qwen3 L21 `.6620`、InternVL L11 `.6473`。曲线有层结构但大量重叠和交叉，不能解释为“relative strength 越大就一定越真实”。逐层 CSV、曲线和训练指标位于各模型 `tables/relative_strength_real_hall_curve.csv`、`figures/relative_strength_real_hall_curve.png` 与 `tables/relative_strength_feature_metrics.csv`。

结论是相对化对 Qwen3/InternVL 有帮助、对 LLaVA 联合组基本持平并略升，但对 Qwen2 明显有害；它也没有像 train-median bounded strength 那样让四模型联合 AUROC 一致优于 raw。因此可把该量保留为无标定的独立候选特征，不能直接替换 raw strength 或正式 G/H。四模型均复核为 3,200/800 image split、交集 0；实现层面没有 train/test 统计或 label 泄漏。但本对照是在已多次查看同一 test cohort 后追加，存在后验特征选择偏差，只能作为探索性结果，真正确认需冻结公式后换独立 held-out cohort。

### 5.9 Plain log1p strength 对照（2026-09-06）

按后续问题直接对 gross strength 做逐元素自然对数压缩：

\[
S_{log1p,\ell}=\ln(1+S_{gross,\ell}).
\]

这里没有加入 `tau`、标准化、裁剪或 train-set calibration；`D_OT+log1p_strength` 精确定义为 `AE+D_WF^OT+log1p_strength`。其余 split、三 seeds、MLP、checkpoint 与 train-REAL-F1 threshold 不变，不做 bootstrap。

| Model | log1p S | Raw S | delta | D_OT+log1p S | D_OT+raw S | delta | vs bounded combo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 | .8484 | .8569 | -.0084 | .8831 | .8697 | +.0134 | -.0050 |
| LLaVA | .8921 | .8860 | +.0061 | .9061 | .8987 | +.0074 | -.0003 |
| Qwen3 | .8774 | .8704 | +.0071 | .8937 | .8811 | +.0126 | -.0090 |
| InternVL | .8548 | .8443 | +.0105 | .8694 | .8606 | +.0088 | +.0012 |

四个 `D_OT+log1p_strength` 都高于 D_OT 单独，增量依次为 `+.0579/+.0091/+.0267/+.0159`。组合 HALL-AUPR/HALL-F1 为 Qwen2 `.4600/.4582`、LLaVA `.7045/.6612`、Qwen3 `.6268/.5594`、InternVL `.5385/.5018`。相对 bounded 组合，只有 InternVL AUROC 略高，其余三模型略低；因此 plain log1p 是比 raw 更稳定的简单压缩，但没有取代此前 bounded 方案。

变换后 train/test 合并实际范围为 Qwen2 `[.0573,5.7246]`、LLaVA `[.1250,4.2965]`、Qwen3 `[.00615,5.1331]`、InternVL `[.0155,4.1988]`。它显著缩小原始动态范围，但并不把数值限制到 `[0,1]`。由于 `log1p` 严格单调，逐层排序、单层 AUROC 和 REAL/HALL median 的相对方向与 raw strength 相同；检测器差异反映固定有限训练下的数值条件变化，而不是新增信息。完整曲线和指标位于各模型 `figures/log1p_strength_real_hall_curve.png`、`tables/log1p_strength_real_hall_curve.csv` 与 `tables/log1p_strength_feature_metrics.csv`。

该变换不读取 label 或任何 train/test 统计，因而没有预处理统计泄漏；但与 5.7/5.8 相同，它是在已查看同一 test cohort 后追加的事后消融，只能作为探索性结果。

### 5.10 D_OT 的 REAL/HALL 逐层曲线（2026-09-06）

为避免组名和单个信号混淆，本节画的是纯距离 `D_WF^OT=OT(P_WRITE,P_FFN)`；此前 detector 的 `D_OT` 组实际输入为 `AE+D_WF^OT`。距离沿用双方各 Top-32 的 union（最多 64 token），截取后分别重新归一化，cost 为 `sqrt_matched_state`。图中没有 AE，也不是检测器预测概率或 R_cos。

复用四模型正式 COCO4000 的全部 train+test mentions，每个 mention 等权；蓝线为 REAL 中位数，红线为 HALL 中位数，阴影为 25%–75% 分位区间（IQR，不是置信区间）。仅做描述性分析，不训练、不重提取、不做 bootstrap，不据此报告显著性或选择最优层。

![四模型 D_WF OT 的 REAL/HALL 逐层曲线](outputs/ffn_visual_source_d_ot_real_hall.png)

上图各子图纵轴独立缩放，以看清层间差异；跨模型绝对尺度比较可看[统一 0–1 纵轴版](outputs/ffn_visual_source_d_ot_real_hall_full_01.png)。另存有[矢量 PDF](outputs/ffn_visual_source_d_ot_real_hall.pdf)。

| Model | REAL 逐层中位数的平均值 | HALL 逐层中位数的平均值 | HALL 中位数较高的层数 |
| --- | ---: | ---: | ---: |
| Qwen2.5 | .01717 | .01668 | 12/28 |
| LLaVA | .02691 | .02629 | 12/32 |
| Qwen3 | .02238 | .02217 | 13/36 |
| InternVL | .02075 | .02011 | 12/32 |

表中先在每层、每类内取中位数，再跨层平均，不是全部 observations 的均值。两类曲线多次交叉，IQR 大量重叠，不支持“幻觉的 D_OT 始终更大”这一简单解释。多数层中位数约在 `.01–.04`；LLaVA 最后一层 REAL/HALL 为 `.16841/.16870`，Qwen3 为 `.21034/.21238`，两类一起升高，不能把末层峰值本身当成幻觉特异信号。逐层边际分布接近也不排除跨层组合包含检测信息；此前 `D_OT` 组的 AUROC 包含 AE，不能解释为纯 `D_WF^OT` 单独训练的表现。

这也说明理论范围 `[0,1]` 不等于实际特征铺满该区间；与 strength 拼接时应比较实际分布，而不能仅按理论上下界判断尺度已匹配。各模型完整逐层表、单模型图和统计分别保存于 `ffn_visual_source_attribution_v1/tables/d_ot_real_hall_curve.csv`、`figures/d_ot_real_hall_curve.png`、`metrics/d_ot_curve_results.json`；总统计为 `outputs/ffn_visual_source_d_ot_real_hall.json`。

复现命令：`CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py`。脚本主流程耗时 89.6 秒（不含初始模块导入）；四模型均核验 4000 processed images，合计 256 行逐层/标签统计，有限值、分位数顺序、AE/距离拆分与所有图表产物检查通过。

### 5.11 JS 的 REAL/HALL 逐层曲线（2026-09-06）

与 5.10 使用同一对 WRITE/FFN 分布，本节展示纯 `D_WF^JS=JS(P_WRITE,P_FFN)`，不包含 AE；此前 `D_JS` detector 组输入仍是 `AE+D_WF^JS`。实际实现为自然对数 JS 散度：

\[
JS(P,Q)=\tfrac12 KL(P\Vert M)+\tfrac12 KL(Q\Vert M),\qquad M=\tfrac12(P+Q).
\]

没有取平方根，也没有除以 `ln(2)`；理论范围为 `[0,ln(2)]≈[0,.6931]`。正式版本在全部视觉 token 上计算。统计沿用四模型全部正式 train+test mentions，每个 mention 等权，蓝色 REAL、红色 HALL，实线为中位数，阴影为 IQR（不是置信区间）；不重训、不做 bootstrap。

![四模型全视觉 token JS 的 REAL/HALL 逐层曲线](outputs/ffn_visual_source_d_js_real_hall.png)

| Model | REAL 逐层中位数的平均值 | HALL 逐层中位数的平均值 | HALL 中位数较高的层数 | 最后一层 REAL / HALL 中位数 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 | .002155 | .001962 | 7/28 | .009280 / .008258 |
| LLaVA | .002890 | .002822 | 9/32 | .039749 / .041679 |
| Qwen3 | .004535 | .004541 | 15/36 | .100795 / .103195 |
| InternVL | .001983 | .001968 | 12/32 | .011712 / .014745 |

表中平均值是先逐层、逐类取中位数，再跨层平均。两类曲线仍多次交叉、IQR 大量重叠，没有统一的“JS 越大越幻觉”方向；LLaVA/Qwen3 末层仍是两类同时升高。多数层中位数为 `10^-3` 量级，JS 数值比此前 OT 更小，但两种距离定义不同，不能据数值大小判断检测优劣，也不能把单层边际重叠解释为跨层组合无信息。

上图各模型纵轴独立缩放；另见[统一理论范围纵轴版](outputs/ffn_visual_source_d_js_real_hall_full_ln2.png)和[矢量 PDF](outputs/ffn_visual_source_d_js_real_hall.pdf)。四模型的逐层数值、单图、统计分别为各自 `ffn_visual_source_attribution_v1/tables/d_js_real_hall_curve.csv`、`figures/d_js_real_hall_curve.png`、`metrics/d_js_curve_results.json`。

为对齐后续的稀疏-support 实验，同时从已有分布调用原 `union_topk_js_matrices()` 生成[双方 Top-32 并集版曲线](outputs/ffn_visual_source_d_js_union_topk_real_hall.png)及[统一理论范围版](outputs/ffn_visual_source_d_js_union_topk_real_hall_full_ln2.png)。该版本先取 `Top32(P_WRITE)∪Top32(P_FFN)`，在并集内分别重新归一化，再计算相同的自然对数 JS；不是取全量 JS 的 Top-K 项，也不使用 OT cost。

| Model | Union JS：REAL 逐层中位数的平均值 | Union JS：HALL 逐层中位数的平均值 | HALL 中位数较高的层数 |
| --- | ---: | ---: | ---: |
| Qwen2.5 | .001700 | .001567 | 10/28 |
| LLaVA | .002887 | .002827 | 11/32 |
| Qwen3 | .004619 | .004712 | 14/36 |
| InternVL | .001708 | .001633 | 12/32 |

Union 版同样存在多次交叉和大量重叠，没有出现跨模型一致的单向分离。Qwen3 末层 REAL/HALL 中位数为 `.11460/.11937`，LLaVA 为 `.05188/.05314`，仍都是两类共同升高。对应逐层 CSV/单图/JSON 文件名使用 `d_js_union_topk` 前缀；两版总统计分别为 `outputs/ffn_visual_source_d_js_real_hall.json` 与 `outputs/ffn_visual_source_d_js_union_topk_real_hall.json`。这些只是描述性曲线，不是新训练或新显著性结论；5.5 的旧训练对照保持不变。

复现命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py --distance js
CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py --distance js_union_topk
```

复用同一绘图脚本，默认 `--distance ot` 保持可用，本次未重写已有 OT 图。两版 CPU 主流程耗时分别为 83.7 秒、228.2 秒（均不含初始模块导入）；合计 512 行统计，正式 cohort 与旧 OT 的 counts 完全一致。所有输入 finite/范围、AE 与距离拆分、分位数顺序、逐层标签去重和 JSON/CSV/PNG/PDF 产物检查通过；四张总图已目视检查。原有 JS 对称性/恒等性、union-support 与 feature 拼接两个定向单测通过，`py_compile` 与 `git diff --check` 通过。

### 5.12 JS(P_FFN,T) 的 REAL/HALL 逐层曲线（2026-09-06）

本节将 5.11 的分布对从 WRITE/FFN 换成 FFN/目标证据：`JS(P_FFN,T)=D_EF^JS`。其中 `T=ATTENTION_EVIDENCE=normalize(attention_support×hpre_raw_logit_gauss_gate)`，是视觉 token 上的归一化目标证据分布，不是标量 `AE strength`、词表概率或 bbox。JS 对称，因此复用已保存的 `JS(T,P_FFN)`；此前 `E_JS` detector 组输入为 `AE+D_EF^JS`，本图仅画纯距离，不含 AE。

正式全视觉 token 版如下；蓝色为 REAL、红色为 HALL，实线为中位数，阴影为 IQR（不是置信区间）。样本仍是各模型 COCO4000 的全部正式 train+test mentions，每 mention 等权；自然对数 JS、不取平方根、不除以 ln(2)，理论范围 `[0,ln(2)]`。

![四模型 JS(P_FFN,T) 的 REAL/HALL 逐层曲线](outputs/ffn_visual_source_d_ef_js_real_hall.png)

| Model | REAL 逐层中位数的平均值 | HALL 逐层中位数的平均值 | HALL 中位数较高的层数 |
| --- | ---: | ---: | ---: |
| Qwen2.5 | .032865 | .032794 | 14/28 |
| LLaVA | .069637 | .079370 | 28/32 |
| Qwen3 | .034091 | .036449 | 30/36 |
| InternVL | .038259 | .039504 | 23/32 |

表中先逐层、逐类取中位数，再跨层平均。与此前 `JS(P_WRITE,P_FFN)` 相比，本分布对在 LLaVA/Qwen3/InternVL 更常出现 HALL 中位数较高；可描述为这些模型的多数层中，幻觉样本的 FFN source 分布与目标证据分布偏离更多。Qwen2 没有统一方向，不能推广成四模型通用的单调规律。两类 IQR 仍明显重叠，层数计数不等于独立重复实验，也不能据此宣称显著性或纯 JS 检测器准确率。

上图各模型纵轴独立缩放；另有[统一理论范围版](outputs/ffn_visual_source_d_ef_js_real_hall_full_ln2.png)和[矢量 PDF](outputs/ffn_visual_source_d_ef_js_real_hall.pdf)。各模型 `ffn_visual_source_attribution_v1/` 下的 `tables/d_ef_js_real_hall_curve.csv`、`figures/d_ef_js_real_hall_curve.png`、`metrics/d_ef_js_curve_results.json` 保存完整逐层统计及单图。

同时完成[Union-Top32 版](outputs/ffn_visual_source_d_ef_js_union_topk_real_hall.png)及[统一理论范围版](outputs/ffn_visual_source_d_ef_js_union_topk_real_hall_full_ln2.png)：取 `Top32(T)∪Top32(P_FFN)`，分别重归一化后计算 JS，不沿用 WRITE/FFN 的并集。

| Model | Union：REAL 逐层中位数的平均值 | Union：HALL 逐层中位数的平均值 | HALL 中位数较高的层数 |
| --- | ---: | ---: | ---: |
| Qwen2.5 | .028888 | .029242 | 16/28 |
| LLaVA | .077656 | .090307 | 27/32 |
| Qwen3 | .030700 | .033709 | 31/36 |
| InternVL | .037290 | .038694 | 21/32 |

Union 版的方向计数与全 token 版大体一致，LLaVA/Qwen3 多数层 HALL 较高，Qwen2 仍较混合；但此描述不等于 union 能提高检测 AUROC，既有训练结果仍以 5.5 为准。并集版逐模型产物使用 `d_ef_js_union_topk` 前缀；两版总图/统计使用 `outputs/ffn_visual_source_d_ef_js_real_hall.*` 和 `outputs/ffn_visual_source_d_ef_js_union_topk_real_hall.*`。

复现命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py --pair ef --distance js
CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py --pair ef --distance js_union_topk
```

本次仅复用已有特征及绘图/union JS 函数，不训练、不重跑 VLM、不做 bootstrap；默认 `--pair wf` 及旧图保持不变。两版主流程耗时 105.7/254.5 秒（不含初始模块导入），退出码均为 0。四模型 counts 与前一轮完全一致；512 行统计、finite/范围、标签数/层数/去重/分位数顺序、JSON/CSV/PNG/PDF 全部核验通过。运行时验证 `E_JS` 的 AE 拆分和距离与 `G_JS` 的 `D_EF` block 精确一致；原 union feature 单测新增 `E_JS` 对应断言，连同 JS 对称性/恒等性测试共 `2/2 PASS`。四张总图目视检查、`py_compile` 和 `git diff --check` 通过。

### 5.13 WRITE/gross/net/core 消融与距离的条件增量（2026-09-06）

按新增优先级，只复用已保存紧凑特征，不运行 VLM forward。符号固定为：

\[
I_\ell=\sum_m\|a_{m,\ell}\|_2,\qquad
S_\ell=\sum_m\|e_{m,\ell}\|_2,\qquad
N_\ell=\|G(z_\ell)-G(z^0_\ell)\|_2,\qquad
\kappa_\ell=N_\ell/S_\ell.
\]

`I` 是保存的 `write_mag` 在视觉 token 上求和，**不是** `||sum_m a_m||`；`S/N` 分别取原始 `gross_strength/net_strength`。数值积分的 `sum_m e_m` 不一定与有限端点 effect 完全相等，本次不替换保存的 N。AE 仍是 `hpre_raw_logit_gauss` attention×gate 的总和，R_cos 仍取正式 old-hpre-cos source/相同 target/sqrt-matched-state OT。

全部 I/S/N 使用原始尺度，不混入此前的 bounded、relative 或 log1p。U 的四条轨迹是 `AE+S+kappa+R_cos`，实际拼接顺序为 `[R_cos,AE,S,kappa]`，严格保持 H 删除全部三条距离后的 block 顺序。H_JS 使用正式**全视觉 token JS**，不是 Union-Top32 对照。

新增训练 `AE+I、AE+S、AE+I+S、AE+N、AE+S+N、U` 六组；`AE、AE+S+kappa、H_OT、H_JS` 分别精确复用正式 `B/F/H_OT/H_JS` 的三个 checkpoint/预测，核验原配置及当前 cohort 后复用，不重训、不覆盖。共 72 个新 checkpoint、48 个复用 checkpoint。所有组沿用固定 image 8:2 split、seeds 43/44/45、MLP `[128,64,32]`（原 Linear–BatchNorm–ReLU–Dropout）、dropout `.3`、batch 256、最多 100 epochs、BCE、无输入标准化/重采样/类别加权、minimum-train-loss checkpoint 和 train-REAL-F1 threshold；另报固定 0.5 指标。**按用户要求不做 bootstrap，均为探索性点估计。**

#### 5.13.1 三种子概率集成 AUROC

下面是先平均三个种子的预测概率、再计算 AUROC，不是 per-seed AUROC 的均值。AE 一行是正式 B，而非旧 A 或 baseline comparison 中的 SVAR。

| Feature | Qwen2.5 | LLaVA | Qwen3 | InternVL |
| --- | ---: | ---: | ---: | ---: |
| AE（复用 B） | .7877 | .8765 | .8484 | .8346 |
| AE+I | .8444 | .8927 | .8655 | .8464 |
| AE+S | .8670 | .8990 | .8789 | .8543 |
| AE+I+S | .8616 | .9004 | .8969 | .8644 |
| AE+N | .8679 | .8941 | .8800 | .8545 |
| AE+S+kappa（复用 F） | .8694 | .9010 | .8855 | .8697 |
| AE+S+N | .8800 | .8972 | .8886 | .8654 |
| U=AE+S+kappa+R_cos | .8812 | .9055 | .8906 | .8768 |
| H_OT（复用） | .8912 | .9068 | .8974 | .8842 |
| H_JS（复用） | .8824 | .9053 | .8945 | .8796 |

#### 5.13.2 回答各项消融问题

1. **WRITE 强度有用，但未解释全部 FFN 收益。** `AE+I−AE` 为 Qwen2/LLaVA/Qwen3/InternVL `+.05667/+.01626/+.01707/+.01175`。在已含 I 时再加 S，`AE+I+S−AE+I` 为 `+.01727/+.00760/+.03137/+.01806`；四模型、全部 12 个配对 seed 差均为正。因此在当前固定训练协议下，S 的预测增量不能完全由 gross WRITE 强度替代。
2. **I 与 S 的互补存在模型差异。** `AE+I+S−AE+S` 为 `-.00540/+.00136/+.01796/+.01012`；Qwen2 加 I 后下降，Qwen3/InternVL 增益较大。不能仅从 I 与 S 的相关性推断应删除其中之一。
3. **净 effect 单独不能稳定替代总强度与比例。** `AE+N−AE+S` 为 `+.00086/-.00491/+.00104/+.00021`，与 gross 单独总体接近；但 `AE+N−(AE+S+kappa)` 四模型均为负：`-.00147/-.00692/-.00554/-.01523`。
4. **κ 没有跨模型独占的优势。** `AE+S+kappa−(AE+S)` 为 `+.00233/+.00200/+.00659/+.01544`；把 κ 换成 N 后，`AE+S+kappa−(AE+S+N)` 为 `-.01059/+.00376/-.00306/+.00435`。Qwen2/Qwen3 更偏向 S+N，LLaVA/InternVL 更偏向 S+κ。对于本轮所有 S>0 的记录，`(S,κ)` 与 `(S,N)` 可相互确定；不同分数反映有限 MLP 的参数化/优化差异，**不是 κ 引入了独立于 S/N 的额外信息，也不能单凭其提升断言学会了复杂的语义冲突机制。**
5. **U 已接近 H，但不能统一删除距离。** R_cos 在 F 之上的增量 `U−F` 为 `+.01180/+.00448/+.00514/+.00710`。距离在 U 之上的条件增量如下：

| Comparison：ensemble AUROC delta | Qwen2.5 | LLaVA | Qwen3 | InternVL |
| --- | ---: | ---: | ---: | ---: |
| H_OT−U | +.01006 | +.00133 | +.00680 | +.00740 |
| H_JS−U | +.00123 | −.00015 | +.00390 | +.00275 |

OT 在 Qwen2/Qwen3/InternVL 的三个单种子差也全部为正；**LLaVA 是需要明确区分集成与单种子的例外**：H_OT 的 seed-mean AUROC 为 `.900563`，低于 U 的 `.901817`，三个配对 seed 差分别为 `-.000326/-.002515/-.000921`，仅概率集成后反超。因此不能把 LLaVA 的 `+.00133` 表述为稳定的单模型距离收益，更不能在未做 bootstrap 时称其显著。JS 增量整体较小；Qwen2 的 seed 方向也有混合。

#### 5.13.3 数值/恢复验收与结果文件

四模型合计 `1,650,608` 个 mention-layer 条目，全部 S>0；保存的 N 与 `S*kappa` 最大相对偏差低于 `5.96e-8`，最大绝对偏差 `8.56e-6`。这是存储字段的代数一致性，**不是新的 vector completeness 证明**。原始 κ 大于 1 的条目数分别为 `28/244076、0/494816、2427/535428、76/376288`，最大 κ 分别为 `1.5420/.9630/3.2319/1.5140`。由于这里分子是有限端点 effect，分母来自数值积分 gross，不能把所有保存的 κ 当成严格 `[0,1]` 的精确抵消率。本轮如实保留原值，没有裁剪或重跑积分来改变对照；对应数值边界不能被机制解释掩盖。

新增和相关单测 `16/16 PASS`，`py_compile`、`git diff --check` 通过。独立从 120 个新/复用 MLP checkpoint 在当前 test 特征上重算概率，与保存预测的最大绝对偏差低于 `3e-6`；40 行组汇总、240 行 seed×threshold 指标和 52 行条件比较均 finite。四模型恢复检查中禁止任何 trainer 调用，并逐文件对比 hash/mtime，验证没有重训或改写完成产物。验收清单包含 382 个文件的 checksum，GPU 已释放，无实验或验收失败。

各模型 `ffn_visual_source_attribution_v1/` 下新增：

- `metrics/core_signal_feature_results.json`：公式、I/S/N 范围、恒等式误差、全部组指标、13 个配对 seed/集成差和旧 checkpoint 来源。
- `metrics/core_signal_feature_training_progress.pt`：30 组 seed 预测及完整 threshold reports；新 checkpoint 在 `metrics/probes_core_signal_features/`，复用的仍在原 `metrics/probes/`。
- `tables/core_signal_feature_metrics.csv`：10 组集成 AUROC/AUPR、seed-mean±std、mean HALL precision/recall/F1。标准差采用 `ddof=0`；F1/precision/recall 是按各 seed 的训练集 REAL-F1 阈值评估后取均值，**不是集成 F1**。
- `tables/core_signal_feature_seed_metrics.csv`：全部 3 seeds×固定 0.5/train-F1 的 AUROC、REAL/HALL AUPR/P/R/F1 和阈值。
- `tables/core_signal_feature_comparisons.csv`：13 个条件比较的集成 AUROC/HALL-AUPR 点差；逐 seed 差保存在 JSON。

跨模型[机器可读汇总](outputs/ffn_visual_source_core_signal_ablation_summary.json)与[独立验收/checksum](outputs/ffn_visual_source_core_signal_ablation_audit.json)已保存。首次模型主流程耗时依次 `144.1/280.2/261.7/228.7` 秒（不含初始模块导入，含读取/检查/18 个新 MLP 训练）；两 GPU 分别串行执行 Qwen2→Qwen3、LLaVA→InternVL，因此这些秒数不是应直接相加的墙钟总耗时。复现命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_signal_ablation.py --study core_signals --models qwen2_5_vl_7b,qwen3_vl_8b --training-device cuda:0 --bootstrap-resamples 0 --resume
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_signal_ablation.py --study core_signals --models llava_1_5_7b,internvl_2_5_8b --training-device cuda:1 --bootstrap-resamples 0 --resume
```

该阶段没有从测试集拟合预处理或选择 checkpoint/threshold，但使用了此前已反复查看的 test cohort，属于事后探索；不据此更改预注册 gate 或追认因果机制。确认泛化增益仍需冻结方案后的独立 held-out 评估。

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
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study bounded_strength --models MODEL --training-device cuda:0 --bootstrap-resamples 0 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study relative_strength --models MODEL --training-device cuda:0 --bootstrap-resamples 0 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study log1p_strength --models MODEL --training-device cuda:0 --bootstrap-resamples 0 --resume
```

每个模型的固定输出根为：

`outputs/<model>/COCO4000-INSLEN-OFFICIAL-TARGET/results/ffn_visual_source_attribution_v1/`

最终验收：四模型 audit 均为 200/50/0，full extraction 均为 4000 processed images，13×3 detector seed 完整，所有 detector/counterfactual bootstrap 均恰为 10,000，四模型各 100 个 counterfactual shard/400 cases，且无缺层、缺 family/strategy、重复映射或非有限值。独立 checksum 审计逐字节复算 26 份 manifest、2,823 个唯一文件、32,562,887,917 bytes，结果全部 PASS。定向回归测试最终为 64/64 PASS，`py_compile` 与 `git diff --check` PASS。

运行中所有可恢复问题均保留证据：Qwen2 两类 audit OOM、Qwen3 非代表层 capture OOM、LLaVA 全量双 rank resume、Qwen3 全量单卡转双卡 resume，以及 InternVL counterfactual image `120777` 的完整 pixel capture cache OOM。最后一项改为只缓存四个所需层的 visual `h_prev`、target row 与 logits 后，原失败图片重试并完成；没有 case 被跳过。旧 scalar-path Qwen3、VQA、neuron intervention 和 bbox 优化均未运行，符合预注册范围。

最终状态：**FORMAL COMPLETE / PASS**。科学结论不是“FFN path 全面优于 WRITE”，而是：vector attribution 数值定义与 detector 增量成立，`P_FFN/P_WRITE` 都具有显著的非随机因果定位能力；两者之间没有跨模型、跨 intervention 的稳定胜者。
