# C 目标后果归因：定义、实现与待审阅问题（2026-09-09）

这是本轮用户与助手讨论的技术整理，不是逐字聊天记录，也不是 C 检测结果报告。用于外部 GPT 审阅；公式以当前代码为准。

## 1. 本次讨论及实验状态

用户要求补提取 C，目标分数同时使用 logit、margin、log-probability，并保存每层每个视觉 token 的原始 C_m，再比较 C、Q、B_Q 及其与既有特征的组合。用户随后询问为何提取慢、要求优化，并追问为何需要后续层以及 C 的精确定义。

- Q/B_Q 四模型 96 个检测头已经完成，见[总表](../outputs/ffn_target_consequence_cqb_v1/q_bq_summary.md)和[完整指标](../outputs/ffn_target_consequence_cqb_v1/q_bq_summary.json)。
- C 仍在提取，不能报告四模型 C 检测已完成。GPU0 按 Qwen2→LLaVA，GPU1 按 Qwen3→InternVL 续跑。
- 当前运行 K4，当前 Norm+FFN 和 suffix 使用 FP32，原始状态/WRITE 来自原生精度捕获；不声称整模型 FP32。
- 原 v2 数值 FAIL 保留；近期训练属于旧 800 图上的探索性比较。没有 bootstrap，也没有独立 2000 图确认。
- 本次上传文档、表格、紧凑 JSON、代码和测试；图片、原始张量、checkpoint、完整日志仅保留本地。

## 2. 当前实现的精确定义

固定一个生成目标 token y、其预测位置以及当前层 l。以下省略层下标。

- z：clean post-attention / pre-FFN residual。
- a_m：当前预测行在 clean QK/attention 权重下，从视觉 token m 获得的 value write 向量。
- A = sum_m a_m，z0 = z − A。
- G(x) = FFN(Norm(x))，不包括 residual identity。

路径为

\[
z(\alpha)=z^0+\alpha A,\quad u(\alpha)=G(z(\alpha)),\quad 0\le\alpha\le1.
\]

**当前层 skip 固定 clean**，所以送入后续层的是

\[
r(\alpha)=z+u(\alpha),
\]

而不是 z(alpha)+u(alpha)。因此这里只归因经过当前 FFN 分支的视觉影响，不包括直接经过当前 skip 的增量；更早层的视觉影响也没有被删除。

令 F_s(u) 表示把当前 FFN 输出替换为 u、加上固定 clean skip，再运行后续 decoder、最终 Norm 和 LM head 得到的标量分数。则

\[
\boxed{C_m^{(s)}=\int_0^1
\nabla_u F_s(u(\alpha))^\top J_G(z(\alpha))a_m\,d\alpha.}
\]

等价实现是每个积分点求复合函数 F_s(G(x)) 对 x 的梯度，再与 a_m 点乘。无需显式构造大 Jacobian，也无需每个视觉 token 单独反向。

最终 logits 记为 b_v，三种分数分别是

\[
s_{logit}=b_y,\quad s_{margin}=b_y-b_c,\quad
s_{logprob}=b_y-\log\sum_v e^{b_v}.
\]

c 为原 clean 输出中最高的非目标 token，沿整条路径固定。每个 target 保存 C_m[3,L,M]，三种 score 不混合；M 是视觉 token 数，L 是 decoder 层数。

C_m>0 表示该 source 的路径归因总体提高所选目标分数，C_m<0 表示总体降低。正负不是 REAL/HALL 标签，也不是语义正确性的判定。

## 3. 为什么要经过后续所有层

目标分数定义在最后的 LM head，当前层的变化可能被后续 attention、FFN、Norm 和残差放大、抵消或改变方向。若 H_l 把当前 block 输出映射到最终归一化隐藏状态，则最终分数对当前输出的梯度包含 J_H_l 的链式反向作用。直接对早期状态使用 LM head 会变成另一种 local/logit-lens 分数，并非同一个 C。

后续“层”不是后续生成的“词”。每个路径点不重新运行视觉编码器，也不重新生成文本。预测行之前的 token 不能被该行的干预影响，故其后续层 prefix K/V 可以缓存；节点相关计算主要重跑预测行。捕获采用与旧 v2 一致的 full-caption causal 前向以保持原生数值 WRITE 对齐；suffix 只使用裁剪后的目标前 prefix，并验证预测行对目标/未来 token 的 attention 为零。

所有层都提取 C 时，早期层需要更长 suffix；按层独立路径累计，suffix 层计算量随 L 近似二次增长。K4 需要四个路径点，各点分数梯度通常不同。批处理能提高吞吐，但不能把 K64 的计算量等同 K4。

## 4. 与 e_m、Q 及 clean-gradient 近似的区别

\[
e_m=\int_0^1J_G(z(\alpha))a_m\,d\alpha.
\]

e_m 是当前 FFN 的向量更新归因，C_m 是指定最终分数的标量归因。当前 Q 复用旧 v2 endpoint-direction path_signed_q，它描述相对局部净 FFN 方向的投影，不直接表示支持/反对目标词。

当前 C **不是**只在 clean 点求一次梯度再乘已积分的 e_m：

\[
\nabla_uF_s(G(z))^\top e_m.
\]

该表达式是可另行验证的冻结梯度近似，除非梯度沿路径恒定或满足特殊条件，否则不等于本轮 C。用一次 clean backward 获得所有层梯度也不能替代各层独立路径下的梯度。

## 5. 完整性与解释边界

在固定上下文、相同 suffix 和精确积分下，链式法则给出

\[
\sum_m C_m^{(s)}=F_s(G(z))-F_s(G(z-A)).
\]

K4 实际记录左右两边及闭合绝对/相对误差；零端点效应单列退化标记，不把相对误差虚构为通过。端点属于同一 FP32 replay 定义，不保证逐位等于原生 clean logits。

这不是自然像素删除效应，不是跨所有层的全局 source 分解，也不是逐 source leave-one-out 差值；非线性交互按共同路径分配。不同层的 C 不能直接相加宣称整模型贡献。各 source 的正负可以抵消：只有总和可由两个端点恢复，端点差不能恢复每个 C_m 或正负总量。

检测器每层使用 C_positive=sum max(C_m,0)、C_negative=sum max(-C_m,0)，分别 log1p 后拼接。Q 同理；B_Q=Q_negative/S。F=全 AE+log1p(raw S)，K=κ_vec。固定原3200/800图、seeds43/44/45、旧三隐藏层协议，不新增超参搜索。

## 6. 已采用的加速不改变上述定义

新后端在每个实际路径点，把三种分数的 suffix VJP 合并为 batched VJP，保留该点真实函数值和一阶导数，再复用原路径聚合。不是只用 clean 梯度，也不省略后续层；不保证高阶导数等价。

四模型各在 image283/599、first/last target、first/middle/last 层做匹配基准，共48个 target-layer；每个 case warmup 后交替计时三对。

| 模型 | 测得速度比（旧/新） | 最大 C 相对 L2 差 |
|---|---:|---:|
| Qwen2.5 | 1.2760 | 1.3478e-5 |
| LLaVA | 1.2493 | 3.7555e-6 |
| Qwen3 | 1.2918 | 3.5088e-6 |
| InternVL | 1.2936 | 6.5451e-5 |

节点/端点分数最大差均为0，显存峰值均下降。不是全4000图端到端吞吐保证。切换时2319张已完成图片保留，SHA和mtime复核通过。新目标行显式标记后端；原冻结源码和旧shard不覆盖。

## 7. 请外部审阅重点检查

1. 本研究是否需要这个经过当前 FFN、skip 固定的最终 target consequence，还是更适合另行研究 clean-gradient×e_m？两者不能混称同一个 C。
2. K4 对 C（不是仅对 P_FFN）的闭合误差，尤其早期层和低端点效应的尾部，是否足以支持后续解释？实现 parity 通过不能代替数值求积充分性。
3. 原生状态捕获 + 局部/suffix FP32 replay 的 estimand 边界是否清楚？缓存、mask、Qwen3 frozen effective additions 是否需要更多实模型覆盖？
4. 固定 clean 竞争词的 margin、log-probability 的全词表竞争与 logit，能否提供不同于局部强度的增量？只能等待完整检测结果回答。
5. 正/负总量是否丢失过多空间信息？原始 C_m 已保留供后续处理，本轮不事后搜索大量聚合方式。
6. 旧800图已多轮探索，不能把点估计改善视作独立泛化或显著性证据。

代码入口：[定义与suffix](../features/ffn_target_consequence.py)、[提取](../scripts/run_ffn_target_consequence.py)、[检测分析](../scripts/analyze_ffn_target_consequence.py)、[联合VJP](../features/ffn_target_consequence_fast.py)、[加速基准](../scripts/benchmark_cqb_optimization.py)、[恢复与后端来源验证](../scripts/run_cqb_optimized.py)。
