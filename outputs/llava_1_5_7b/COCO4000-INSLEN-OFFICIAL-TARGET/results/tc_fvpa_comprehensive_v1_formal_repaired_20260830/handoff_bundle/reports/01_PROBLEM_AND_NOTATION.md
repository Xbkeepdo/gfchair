# 问题、估计量与符号

研究问题是：当前 causal text-prediction position 的 source-wise MHSA visual residual writes 经 `G_l(z)=FFN_l(Norm_l^ffn(z))` 非线性变换后，如何形成 FFN residual-update components，以及这些 component 如何支持或抑制目标 token 决策。它不是不同 Transformer 深度的一般比较。

一基 decoder block 为 `l`，目标首 subtoken位置为 `t`，预测它的因果行是 `q_t`，视觉 source token为 `m`。`a_{l,t,m}` 是在观测 clean attention pattern 条件下的 value-path residual write；output projection bias 只加入完整 attention output一次，不分配给 token。

`z=h_pre+o_attn`，`A=sum_m a_m`，`m_0=G(z)`。局部 response 为 `delta_m=J_G(z)a_m`；`I_m=||a_m||`、`E_m=||delta_m||`、`gain_m=E_m/(I_m+eps)`。aggregate response方向的 signed score 为 `Q_m=u^T delta_m`，所有报告必须同时给出正质量、负质量、net、negative fraction及 vector/signed cancellation。

Branch-isolated downstream functional保持 clean residual skip `z`，只替换这一层、这一行的 FFN branch output，再执行全部 downstream decoder、final norm和LM head。三个 scalar为 target logit、固定 clean competitor margin及 target log-probability。

FFN-output Riesz representative为 `g=grad_m S(m_0)`；pre-FFN pullback为 `r=J_G(z)^T g`。局部 target attribution满足 `C_m=g^T(J_Ga_m)=r^Ta_m`。Riesz理论只给出 differential 的向量表示，不证明视觉 attribution。

Current-block zero-write baseline为 `z0=z-A`。它不是 no-image state，不删除 earlier blocks积累的视觉信息。Vector path component为 `e_m^path=int_0^1 J_G(z0+alpha A)a_m d alpha`；target path同时积分 downstream gradient。完整性只针对这个明确 baseline和branch-isolated functional。

Finite estimands保持分离：frozen-write LOO、fixed-QK value zeroing、full current-layer activation patching、end-to-end pixel counterfactual。前两者是 clean attention decomposition条件效应；后两者包含 attention redistribution或整网视觉变化。
