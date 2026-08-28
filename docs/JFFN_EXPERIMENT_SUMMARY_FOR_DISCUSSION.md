# GF-CHAIR / JFFN 实验总结（供进一步讨论）

## 1. 这份文档回答什么问题

本项目研究视觉语言模型生成物体词时，视觉 token 经 decoder attention 写入当前生成位置、再经过 FFN 局部变换后的响应，能否用于物体幻觉检测。本文整合以下两份主报告及其后的增量实验：

1. `jacobian_visual_ffn_validation_report.md`：JFFN-P 的公式、实现、数值正确性、分布特性、空间定位和第一轮检测结果。
2. `jffn_second_round_incremental_validation_report.md`：WRITE-only 控制、Jacobian 增量价值、signed attribution、target-logit attribution 和因果干预。
3. 后续实验：`risk+EV+S`、`S×risk+EV`、JFFN–target Union-TopK JS、Union 区域 S、不同聚合 S、分类头敏感性和 LLaVA 500 图 aggregate-S 实验。

需要先强调：完整 JFFN 正式实验只完成了 **LLaVA-1.5-7B** 和 **InternVL2.5-8B**。Qwen2.5-VL、Qwen3-VL 只完成过前期 smoke 或旧特征实验，本文不把它们外推为 JFFN 正式结果。

## 2. 实验协议

### 2.1 目标对象与标签

- 使用 GF-CHAIR 中复现的 InsLen official first-token protocol。
- CHAIR 负责物体词归一化和 REAL/HALL 标签。
- 按官方启发式找到目标物体首 subtoken ID，并使用该 ID 在回答中的第一次出现。
- 特征取生成目标首 subtoken 之前的因果行，即目标位置 `toke_idx-1`。
- 未解析目标直接跳过；相同 detected word 按官方规则去重。
- caption、图片级 train/test split 和目标 cohort 在各项对照中保持不变。

正式样本：

| 模型 | 图片 | 唯一目标位置 | mention | train mention | test mention | test REAL/HALL |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | 3971 | 14951 | 15463 | 12317 | 3146 | 2437/709 |
| InternVL2.5-8B | 3927 | 11630 | 11759 | 9378 | 2381 | 2008/373 |

### 2.2 JFFN 的定义

在 decoder 第 `l` 层、生成位置 `t`：

\[
z_{l,t}=h^{pre}_{l,t}+o^{attn}_{l,t}.
\]

视觉 token `j` 对该位置的 attention residual write 为：

\[
a^j_{l,t}=W_l^O\operatorname{concat}_h
\left(\alpha_{l,t,h,j}v_{l,h,j}\right).
\]

令：

\[
f_l(z)=\operatorname{FFN}_l(\operatorname{Norm}_l(z)),
\]

通过 forward-mode JVP 计算：

\[
\delta^j_{l,t}=J_{f_l}(z_{l,t})a^j_{l,t},
\qquad E_{l,t,j}=\|\delta^j_{l,t}\|_2.
\]

新视觉分布为：

\[
P^{JFFN}_{l,t,j}=\frac{E_{l,t,j}}
{\sum_kE_{l,t,k}+\epsilon}.
\]

这里没有 softmax 和温度。`P_JFFN` 只是归一化的局部 FFN 响应能量，不能称为物体存在概率、信息量或因果概率。

用于解释的聚合量为：

\[
I=\|a^{visual}\|_2,
\quad R=\|J_f(z)a^{visual}\|_2,
\quad S=\frac{R}{I+\epsilon},
\quad D=\cos(J_f(z)a^{visual},a^{visual}).
\]

其中 `I` 是视觉写入强度，`R` 是经过 FFN 后的响应强度，`S` 是单位视觉写入对应的局部 FFN gain。

### 2.3 旧方法与正式下游头

- 旧 source P：基于视觉 token hidden state 与当前生成位置 hidden state 的 cosine，经归一化得到。
- target Q：hpre raw-logit Gaussian 或 hpre softmax-prob Gaussian gate。
- support：`Top32(P) ∪ Top32(Q)`，即 Union Top-K，最多 64 个视觉 token。
- cost：主设置为 `sqrt_matched_state`，另有 `cosine_matched_state`。
- 原 EV：逐层 `mass×cosine`。
- 正式三层 MLP：`[128,64,32]`，batch 256，最多 100 epochs，seeds 43/44/45；使用 minimum-train-loss checkpoint 和 train-F1 threshold，不做特征标准化。

## 3. 数值与实现验证

### 3.1 已通过的检查

- LLaVA separate Q/K/V 与 InternVL packed `wqkv`、GQA repeat、`o_proj/wo` 均有独立适配。
- output projection bias 只进入完整 attention 重建一次，不重复分配给每个视觉 token。
- `Norm` 在被求导函数内，FFN residual identity skip 不在 Jacobian 内。
- 目标首 subtoken 之前的因果行、视觉 token 范围和 causal mask 均已核对。
- LLaVA 576 个方向、InternVL 256 个方向可通过 `vmap(jvp)` 并行；全并行与分块结果一致到约 `10^-6–10^-3`（取决于 dtype/kernel batch shape）。
- LLaVA 独立 FP32 审计中，JVP 标量线性最大误差约 `7.82e-7`，逐 token JVP 加和误差最大约 `3.65e-4`。
- LLaVA central finite-difference 呈正常 U 形，合适 epsilon 下 relative error 约 `10^-4`。
- 两模型 smoke 峰值显存均低于 22 GiB。

### 3.2 已发现并修复/澄清的问题

- 第一版 `component_sum_relative_error` 曾被硬编码为 0；第二轮已改为真实重建并记录误差。该字段不参与 `a_j/E_j/P` 或训练，所以不改变旧实验结果。
- InternVL 全量路径用 BF16 的 `hmid-hpre` 作为 attention reference，会因两个大残差相减放大 relative error；保留真实 `o_attn` 的 smoke 明显更准。
- 完整删除视觉写入时，局部 Jacobian 近似的 relative error 中位数约 0.084，因此 JFFN 只能解释局部一阶响应，不能当作大幅 intervention 的精确预测器。
- target-logit intervention 的预期变化小于 FP16/BF16 logit 的可分辨步长，大量观测为 0；所以当前只能确认计算路径，不能宣称已经验证因果幅值。

## 4. 第一轮核心结果：P_JFFN 更像“空间解释分布”，不是更好的幻觉检测 P

### 4.1 相比旧 cosine-P，P_JFFN 明显更有结构

- 风险层的归一化熵：LLaVA 约 `0.764–0.766`，InternVL 约 `0.710–0.727`，不再接近均匀。
- 同图不同目标 JS：LLaVA/InternVL 约 `0.195/0.305`，显著高于 old-hpre P 的约 `0.016`。
- COCO REAL box 上 patch AUPRC：old-hpre `0.349/0.316`，P_JFFN `0.451/0.445`。
- Top-1 pointing：old-hpre `0.336/0.279`，P_JFFN `0.502/0.625`。
- entropy matching 后空间排序优势仍存在，说明不是简单因为分布变尖。

### 4.2 但直接替换旧 P 的检测结果为负

预注册 `raw-logit gate + sqrt cost + risk+EV`：

| 模型 | old-hpre P AUROC | P_JFFN AUROC | 差值 | 图片级 95% CI |
|---|---:|---:|---:|---|
| LLaVA | 0.8934 | 0.8832 | -0.0102 | [-0.0198,-0.0012] |
| InternVL | 0.8567 | 0.8506 | -0.0060 | [-0.0197,+0.0077] |

LLaVA 显著下降，InternVL 没有可靠提升；entropy-matched JFFN 同样没有优于旧 P。因此“定位更准”不等于“更能识别幻觉”。

## 5. 第二轮核心结果：JFFN 排序的大部分来自 attention WRITE magnitude

第二轮增加严格 WRITE-only 控制：

\[
P^{WRITE}_{l,t,j}=\frac{\|a^j_{l,t}\|_2}
{\sum_k\|a^k_{l,t}\|_2+\epsilon}.
\]

第 16–32 层空间结果：

| 模型 | 方法 | Top-1 | Patch AUPRC | BBox mass |
|---|---|---:|---:|---:|
| LLaVA | WRITE | **0.50197** | **0.44776** | **0.38778** |
| LLaVA | JFFN | 0.50177 | 0.44375 | 0.38660 |
| InternVL | WRITE | **0.63298** | **0.45149** | **0.39812** |
| InternVL | JFFN | 0.62489 | 0.44513 | 0.39622 |

JFFN−WRITE 的 Patch AUPRC/BBox mass 在两模型上均显著小于 0。与此同时：

- `I_j=||a_j||` 与 `E_j=||Ja_j||` 的 Pearson 为 `0.9806/0.9901`。
- Top-1 agreement 为 `0.8668/0.8756`。
- Top-32 overlap 为 `0.9258/0.9481`。
- token gain `G_j=||Ja_j||/(||a_j||+eps)` 的层内 CV 约 15–16%，说明 Jacobian 不是完全常数缩放，但 gain 在真实 box 内反而低于 box 外。

因此当前最稳妥的解释是：`P_JFFN` 的空间质量主要来自 attention-mediated visual write magnitude；Jacobian 确实重排序了一部分 token，但没有产生更好的目标空间排序。

## 6. 聚合 Jacobian 标量仍能作为旧特征的补充

这里出现了最值得继续讨论的结果：**替换 P 失败，不代表聚合标量 S 没有补充信息。**

将逐层 `[old risk_32, EV_32, S_32]` 拼接为 96 维，使用完全相同的三层 MLP：

| 模型 | 特征 | AUROC | Hall-AUPR | Hall-F1 |
|---|---|---:|---:|---:|
| LLaVA | risk+EV | 0.888141 | 0.656175 | 0.644332 |
| LLaVA | risk+EV+aggregate S | **0.893087** | **0.675664** | **0.658950** |
| InternVL | risk+EV | 0.848825 | 0.502827 | 0.466311 |
| InternVL | risk+EV+aggregate S | **0.866745** | **0.546172** | **0.510817** |

seed-ensemble 图片级 paired bootstrap：

- LLaVA AUROC 增量 `+0.00948`，95% CI `[+0.00190,+0.01734]`。
- InternVL AUROC 增量 `+0.01806`，95% CI `[+0.00474,+0.03156]`。

这与第二轮 balanced logistic regression 中 `[I,S]-I` 没有稳定增益并不冲突：

1. 第二轮问题是“在 I 条件下，固定 16–32 层的线性 S 是否有独立信息”；
2. 三层 MLP 问题是“所有层 S 是否能与非线性的 old risk 和 EV 交互”；
3. 两者的输入、分类头和假设不同。

所以目前可以说：**S 不是一个强单变量，也没有证明独立于 I 的普适机制增益；但作为旧 risk+EV 的多层非线性补充，在两个正式模型上都有可重复提升。**

## 7. S 曲线和不同 S 定义

### 7.1 REAL > HALL 的解释

多数 LLaVA 层上 `S_REAL>S_HALL`。这表示控制视觉写入模长后，真实物体方向平均获得更高的局部 FFN gain。它是类别条件均值差，不意味着每个 REAL 样本都大于每个 HALL，也不能单独证明 FFN 导致幻觉。

### 7.2 三种主要 S

1. 全视觉 aggregate：
   \[
   S_{all,agg}=\frac{\|\sum_jJa_j\|}{\|\sum_ja_j\|+\epsilon}.
   \]
   保留 token 之间的方向相消。
2. 全视觉 tokenwise：
   \[
   S_{all,tok}=\frac{\sum_j\|Ja_j\|}{\sum_j\|a_j\|+\epsilon}.
   \]
   是逐 token gain 的 `I_j` 加权平均，不保留相消。
3. Union-TopK tokenwise：只在 `Top32(P_JFFN)∪Top32(Q)` 上计算第二种比值。

完整 4000 图结果：

| 模型 | risk+EV | +all aggregate | +all tokenwise | +Union tokenwise |
|---|---:|---:|---:|---:|
| LLaVA AUROC | 0.888141 | 0.893087 | **0.894717** | 0.891793 |
| InternVL AUROC | 0.848825 | **0.866745** | 0.863213 | 0.866605 |

Union 区域效果具有模型依赖性：

- LLaVA：Union tokenwise 低于 all tokenwise，ensemble 差 `-0.00415`，CI `[-0.00806,-0.00027]`。
- InternVL：Union tokenwise 高于 all tokenwise，差 `+0.00830`，CI `[+0.00191,+0.01498]`；但与旧 aggregate S 基本打平。

因此不支持用一个 Union S 统一替代所有模型；LLaVA 的信息更分布式，InternVL 更能从排除区外 token 中受益。

### 7.3 LLaVA 500 图 Union aggregate 实验

新增：

\[
S_{union,agg}=\frac{\|\sum_{j\in U}Ja_j\|}
{\|\sum_{j\in U}a_j\|+\epsilon}.
\]

公平 500 图 cohort 只有 395/105 train/test images、1513/404 mentions，因此只用于快速筛选：

| 特征 | AUROC | Hall-AUPR | Hall-F1 |
|---|---:|---:|---:|
| risk+EV | 0.821245 | 0.549276 | 0.610406 |
| +all aggregate S | 0.829718 | 0.570312 | 0.603090 |
| +all tokenwise S | 0.840155 | **0.615408** | 0.596042 |
| +Union tokenwise S | **0.845628** | 0.603987 | **0.627679** |
| +Union aggregate S | 0.839762 | 0.576353 | 0.612775 |

500 图与 4000 图的 S 内部排序不一致；这说明 500 图能判断“有 S 信号”，不能可靠选择最优 S。Union aggregate 还没有 4000 图正式结果。

## 8. 其他构造的结果

### 8.1 `S×risk + EV`

逐层相乘会把原本多数层 `S_REAL>S_HALL` 的方向调制成许多层 `S×risk_HALL>S×risk_REAL`，具有直观机制意义。但硬乘法丢失了 risk 与 S 的独立幅度：

| 模型 | risk+EV AUROC | S×risk+EV AUROC | risk+EV+S AUROC |
|---|---:|---:|---:|
| LLaVA | 0.888141 | 0.890893 | **0.893087** |
| InternVL | 0.848825 | 0.851277 | **0.866745** |

`S×risk+EV` 对 baseline 的小幅提升置信区间跨 0，且显著弱于分别拼接 `[risk,EV,S]`。因此它适合作为解释性消融，不适合作为主特征。

### 8.2 `P_JFFN`–target Union-TopK JS + EV

在 Union support 内重新归一化 P/Q，计算自然对数 Jensen–Shannon divergence，再与 EV 拼接。结果没有稳定优于 OT risk+EV，并且 JFFN source 通常低于 old-hpre source：

- LLaVA raw gate：JFFN `0.87606`，old-hpre `0.88955`。
- InternVL raw gate：JFFN `0.83873`，old-hpre `0.84881`。

结论：JS 只比较概率错位，丢掉 OT cost 中的 hidden-state 几何，不能稳定替代当前 risk。

### 8.3 分类头敏感性

把相同 64/96 维 Ours 特征放入 S-VAR 原生单隐藏层头、MetaToken Logistic Regression/Gradient Boosting：

- 加入 S 在两模型×三种原生头中都提高 AUROC，说明 S 信息不是三层 MLP 独占。
- 但总体上 `[128,64,32]` 三层 MLP 仍最好。
- S-VAR 隐藏宽度从 248 改成 128 后，两模型的 64/96 维输入均下降。

因此当前 Ours 不需要寻找一个“论文名称对应的原生头”；应把分类头作为验证协议的一部分，并报告同一输入在多个合理头下的稳健性。主结果继续使用预注册三层 MLP，原生头作为敏感性消融。

## 9. Signed、logit-aligned 与因果实验

### 9.1 Signed contribution

定义 aggregate response 方向：

\[
u=\frac{\delta^{visual}}{\|\delta^{visual}\|+\epsilon},
\qquad Q_j=u^T\delta_j.
\]

`sum_j Q_j≈R` 的最大相对误差低于 `4.5e-7`。但 REAL/HALL 的负质量和 cancellation pattern 没有跨模型一致方向；`Q+` 空间图只在 LLaVA 有部分改善，所以当前只应作为机制诊断。

### 9.2 Target-logit/margin alignment

- 使用目标首 subtoken 之前的 causal row，目标 token 不进入 prefix。
- attribution additivity 在两个模型上通过。
- direction-free Hall AUROC：LLaVA margin total `0.612`，InternVL `0.508`，没有跨模型复现。
- `P_LOGIT+` 的空间定位弱于 WRITE/JFFN。

### 9.3 Intervention

因果路径和 eta=0 hook baseline 已核对，但预测变化远小于模型 dtype 的 logit 量化步长。`eta=0.05` 时零观测比例为 LLaVA `63.5%`、InternVL `88.5%`。当前结论只能是：计算方向合理，因果 effect magnitude 尚未得到实验证实。

## 10. 当前最稳妥的科学结论

### 可以声称

1. JFFN 正确测量了 FFN 对 attention-mediated visual write 方向的局部一阶响应。
2. P_JFFN 比旧 hidden-cosine P 更尖锐、更具目标特异性，并具有更好的物体空间定位。
3. WRITE magnitude 已解释 P_JFFN 的大部分空间排序，Jacobian reranking 没有超过 WRITE-only。
4. 直接用 P_JFFN 替换旧 P 不会改善幻觉检测，在 LLaVA 上反而显著下降。
5. 多层 aggregate/tokenwise S 作为旧 risk+EV 的附加块，在 LLaVA 与 InternVL 三层 MLP 上均提高 AUROC；最佳聚合方式具有模型依赖性。

### 不能声称

1. P_JFFN 是物体存在概率或视觉 grounding probability。
2. 空间定位提升证明了幻觉检测提升。
3. Jacobian 比 attention WRITE 本身提供了更好的空间解释。
4. S 已被证明是独立于 I 的普适幻觉因果变量。
5. 当前低精度 intervention 已验证 attribution 的因果幅值。
6. LLaVA/InternVL 结论已经在 Qwen2.5/Qwen3 上成立。

## 11. 推荐下一步

1. **把 S 定位为辅助特征，不再替换 P。** 主线优先研究 `[risk,EV,S]` 的跨模型稳定性。
2. **使用训练集选择模型特定 S 聚合。** 在 nested validation 中从 all-aggregate、all-tokenwise、Union-tokenwise 中选择，避免用 test 决定变体。
3. **分解 S 的来源。** 比较 `log R`、`log I`、`log R−α_l log I` 和 `R/I`；`α_l` 只在 train 拟合，检验 residualized response 是否比固定比值稳定。
4. **做 matched-norm 方向控制。** 把真实视觉方向与随机/背景/框内方向比较，判断 S 是 FFN 层尺度还是对象特异方向响应。
5. **提高因果干预可分辨性。** FP32 局部 FFN、larger-but-local eta sweep、多次重复及 logit margin 累积，先证明预测量级可观测，再讨论 causal agreement。
6. **完成 Qwen 正式验证。** 在相同 InsLen cohort、split、首 subtoken、三 seeds 下验证 `[risk,EV,S]`，而不是把已有 smoke 当正式结果。

## 12. 供 ChatGPT 重点讨论的问题

1. 为什么 `P_JFFN` 的空间定位明显优于 old cosine-P，却在幻觉检测中更差？这是否意味着“grounding map quality”和“model uncertainty”本质不同？
2. 当 `P_JFFN≈P_WRITE` 时，是否还有更合理的 Jacobian 标量，能去除 write magnitude 而保留方向选择性？
3. `risk+EV+S` 的提升应如何与线性 `[I,S]-I` 的负结果共同解释？需要什么条件互信息或交互检验？
4. `S=R/I` 是否应改为逐层 train-fitted residual `log R−α_l log I`，或使用 matched-norm Jacobian gain？
5. LLaVA 偏好全视觉 tokenwise S、InternVL 偏好 Union S，这种差异可能来自视觉 token 数、GQA、动态分辨率还是 attention 分布结构？
6. 如何设计在 FP16/BF16 下仍可辨认、同时保持局部线性有效的 causal intervention？

## 13. 代码与报告入口

核心实现：

- `features/visual_ffn_jacobian.py`
- `features/jffn_experiment.py`
- `features/dgst_t.py`
- `models/dgst_capture.py`
- `scripts/run_jffn_p_comparison.py`
- `scripts/run_jffn_second_round_extraction.py`

主报告：

- `jacobian_visual_ffn_validation_report.md`
- `jffn_second_round_incremental_validation_report.md`

后续实验脚本：

- `scripts/train_old_risk_ev_s_mlp.py`
- `scripts/train_s_times_risk_ev_mlp.py`
- `scripts/train_jffn_union_topk_js_ev_mlp.py`
- `scripts/train_union_topk_region_s_mlp.py`
- `scripts/train_union_topk_aggregate_s_mlp.py`
- `scripts/train_ours_with_native_baseline_heads.py`
- `scripts/plot_s_clear_visualizations_500.py`

所有正式大规模 features、checkpoint、图片和预测保存在本地 `outputs/`，不应提交到代码仓库。
