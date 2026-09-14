# 下一阶段实验方案：All-attention Path 与 All-attention + Residual Path

## 1. 当前研究状态与已验证结论

当前方法已经围绕视觉 attention write（注意力写入）完成了较完整的数值与几何验证。

对当前层 \(l\)、当前生成位置 \(t\)，定义真实 FFN 分支为

\[
G_l(z)=\mathrm{FFN}_l(\mathrm{Norm}_l(z)).
\]

对视觉 token \(m\)，其 current-layer attention write（当前层注意力写入）记为

\[
a_m.
\]

当前 visual-only path（仅视觉路径）定义为

\[
A_V=\sum_{m\in V}a_m,
\qquad
z_0=z-A_V,
\]

\[
e_m^{V}
=
\int_0^1
J_G(z_0+\alpha A_V)a_m\,d\alpha.
\]

现有实验已经验证：

1. \(J_G\) 不是标量，而是一个 \(D\times D\) 的 linear operator（线性算子）。
2. path-integrated operator（路径积分算子）
   \[
   \hat J=\int_0^1J_G(\gamma(\alpha))d\alpha
   \]
   在 visual-write subspace（视觉写入子空间）上不接近 \(cI\)。
3. \(\cos(a_m,e_m)\) 在四个模型中整体为负，说明 FFN 对 write 存在明显的 direction change（方向改变），而不是简单缩放。
4. 完整 visual-write subspace（视觉写入子空间）的 singular spectrum（奇异值谱）并不平坦，说明 FFN 存在 anisotropic transformation（各向异性变换）。
5. 但目前这些“FFN 是否复杂”的全局几何量在 REAL/HALL 间没有跨模型统一趋势，因此下一步不应该继续证明 \(J\) 是否非平凡，而应该研究：
   - FFN 对不同 source type（来源类型）的处理是否不同；
   - prompt / generation context（提示词/生成文本上下文）是否改变同一视觉证据的 FFN transformation（FFN变换）；
   - current attention write（当前注意力写入）与 historical residual representation（历史残差表示）在 FFN 中是合作还是抵消；
   - 这些现象是否能稳定提升 hallucination detection（幻觉检测）。

---

# 2. 总体研究主线

下一阶段建议围绕以下三层展开：

\[
\boxed{
\text{Attention Routing}
\rightarrow
\text{Source-specific FFN Transformation}
\rightarrow
\text{Residual Integration}
}
\]

分别回答：

1. **Attention Routing（注意力路由）**
   - visual / prompt / generation 分别写入了多少信息？

2. **Source-specific FFN Transformation（来源特定 FFN 变换）**
   - FFN 如何分别放大、缩小、旋转 visual / prompt / generation 的 write？

3. **Residual Integration（残差整合）**
   - 当前新写入的信息，与历史 residual 表示之间是支持、竞争还是抵消？

最终再观察这些信号对 REAL/HALL 的差异以及 hallucination detection（幻觉检测）性能。

---

# 3. 实验 A：All-attention Path（全注意力路径）

## 3.1 来源定义

当前层当前生成位置的 pre-FFN residual state（FFN前残差状态）写成

\[
z
=
r
+
A_V
+
A_P
+
A_G
+
b_O,
\]

其中：

- \(r=h^{pre}_{l,t}\)：historical residual representation（历史残差表示）；
- \(A_V\)：所有视觉 token 的 attention write 之和；
- \(A_P\)：所有 prompt token 的 attention write 之和；
- \(A_G\)：所有 generation-prefix token（生成前缀 token）的 attention write 之和；
- \(b_O\)：attention output projection bias（注意力输出投影偏置）。

所有 token-level write 均来自 clean attention pattern（干净注意力模式）：

\[
a_m
=
W_O
\operatorname{Concat}_h
\left(
\alpha^h_{t,m}V^h_m
\right).
\]

---

## 3.2 All-attention Path 定义

令

\[
A_{\mathrm{all}}
=
A_V+A_P+A_G.
\]

baseline（基线）：

\[
z_0^{A}
=
z-A_{\mathrm{all}}
=
r+b_O.
\]

路径：

\[
\boxed{
\gamma_A(\alpha)
=
r+b_O
+
\alpha(A_V+A_P+A_G)
}
\]

其中

\[
\gamma_A(0)=r+b_O,
\qquad
\gamma_A(1)=z.
\]

这条路径回答：

> 在 historical residual representation（历史残差表示）固定的条件下，当前 Attention 子层刚刚写入的 visual / prompt / generation 信息被 FFN 如何处理？

---

# 4. All-attention Path 中每个 token 的贡献

对所有 causal attention source（因果注意力来源）中的每个 token \(m\)，定义

\[
\boxed{
e_m^{A}
=
\int_0^1
J_G(\gamma_A(\alpha))a_m\,d\alpha
}
\]

其中

\[
G(z)=FFN(RMSNorm(z)).
\]

所有来源共享同一条 path（路径），因此

\[
\sum_{m\in A}e_m^{A}
=
G(z)-G(r+b_O).
\]

这保证 All-attention source attribution（全注意力来源归因）具有 completeness（完备性）。

---

# 5. 正式提取时需要保存什么？

下一轮正式提取建议保留 **每个 attention token 的标量统计**，但不保存完整 \(D\)-维 \(e_m\) 向量。

对每个 source token \(m\)，保存：

## 5.1 输入 write 范数

\[
\boxed{
A_m=\|a_m\|_2
}
\]

## 5.2 FFN path response 范数

\[
\boxed{
E_m=\|e_m^{A}\|_2
}
\]

也就是每个 \(e_m\) 的 norm（范数）。

这部分与现有 visual-only 正式实验中的

```text
ffn_path_gross
```

概念一致，只是下一轮扩展到：

- prompt；
- visual；
- generation-prefix；

全部 attention sources（注意力来源）。

## 5.3 Source gain（来源增益）

\[
\boxed{
g_m
=
\frac{\|e_m^{A}\|}
{\|a_m\|}
}
\]

不需要额外保存，可以由 \(A_m,E_m\) 离线计算。

## 5.4 Direction cosine（方向余弦）

\[
\boxed{
r_m
=
\cos(a_m,e_m^{A})
}
\]

这个需要在提取时在线计算，因为只保存范数无法恢复 cosine。

## 5.5 不保存完整 \(e_m\)

正式 4000 图不保存

\[
e_m\in\mathbb R^D
\]

完整向量。

计算流程建议：

```text
JVP/path component
        ↓
在线计算 ||e_m||、cos(a_m,e_m)
        ↓
保存标量
        ↓
释放 e_m
```

这样能保留主要几何信息，同时避免巨大存储开销。

---

# 6. Group-level（组级）统计

虽然保存每个 token 的 \(\|e_m\|\)，但 source-level（来源级）研究不能只看 token norm 相加。

对

\[
g\in\{V,P,G\}
\]

定义：

\[
E_g
=
\sum_{m\in g} e_m.
\]

正式提取时建议在线累积：

\[
E_V,\qquad E_P,\qquad E_G.
\]

不必长期保存完整向量，但在 GPU 上当前 layer/target 内保留用于计算 group geometry（组几何）。

---

## 6.1 Gross source strength（来源总响应强度）

\[
\boxed{
S_g^{gross}
=
\sum_{m\in g}\|e_m\|
}
\]

它表示来源内部所有 token 一共触发了多大的 FFN response（FFN响应），不考虑相互抵消。

---

## 6.2 Net source strength（来源净响应强度）

\[
\boxed{
S_g^{net}
=
\|E_g\|
=
\left\|
\sum_{m\in g}e_m
\right\|
}
\]

---

## 6.3 Within-source cancellation（来源内部抵消）

\[
\boxed{
\kappa_g
=
\frac{
\|E_g\|
}{
\sum_{m\in g}\|e_m\|
}
}
\]

解释：

- \(\kappa_g\approx1\)：该来源内部 token response 方向较一致；
- \(\kappa_g\ll1\)：该来源内部存在较强 cancellation（抵消）。

分别计算：

\[
\kappa_V,\quad
\kappa_P,\quad
\kappa_G.
\]

---

# 7. FFN 对不同来源的放大与旋转

## 7.1 Group gain（组增益）

输入 group write：

\[
A_g=\sum_{m\in g}a_m.
\]

定义：

\[
\boxed{
G_g
=
\frac{
\|E_g\|
}{
\|A_g\|
}
}
\]

分别得到：

\[
G_V,\qquad G_P,\qquad G_G.
\]

研究：

> FFN 是否对 visual / prompt / generation 信息采用不同的 effective gain（有效增益）？

---

## 7.2 Group rotation（组旋转）

\[
\boxed{
R_g
=
\cos(A_g,E_g)
}
\]

分别为：

\[
R_V,\quad R_P,\quad R_G.
\]

研究：

> FFN 对三类来源的 direction transformation（方向变换）是否不同？

---

# 8. 来源之间的合作与冲突

这是下一阶段重点。

## 8.1 FFN 前的来源关系

例如：

\[
C_{VG}^{in}
=
\cos(A_V,A_G)
\]

\[
C_{VP}^{in}
=
\cos(A_V,A_P)
\]

\[
C_{PG}^{in}
=
\cos(A_P,A_G)
\]

---

## 8.2 FFN 后的来源关系

\[
C_{VG}^{out}
=
\cos(E_V,E_G)
\]

\[
C_{VP}^{out}
=
\cos(E_V,E_P)
\]

\[
C_{PG}^{out}
=
\cos(E_P,E_G)
\]

---

## 8.3 FFN 对来源关系的改变

\[
\boxed{
\Delta C_{VG}
=
C_{VG}^{out}
-
C_{VG}^{in}
}
\]

同样：

\[
\Delta C_{VP},
\qquad
\Delta C_{PG}.
\]

这些量直接回答：

> FFN 是让 visual / generation / prompt 的表示更加一致，还是更加冲突？

---

# 9. Visual context modulation（视觉上下文调制）

这是下一轮最值得关注的信号之一。

当前已有 visual-only path：

\[
E_V^{vis}.
\]

All-attention path 中可得到：

\[
E_V^{all}.
\]

注意二者使用同一个

\[
A_V,
\]

改变的只是 path context（路径上下文）。

定义：

\[
\boxed{
C_V^{ctx}
=
\cos(E_V^{vis},E_V^{all})
}
\]

以及：

\[
\boxed{
M_V^{ctx}
=
\frac{
\|E_V^{all}-E_V^{vis}\|
}{
\|E_V^{vis}\|
}
}
\]

物理意义：

> prompt / generation attention context 的存在，改变了 FFN 如何解释同一视觉 evidence（视觉证据）多少。

这是非常适合 hallucination mechanism（幻觉机制）研究的信号。

---

# 10. All-attention 的组间 cancellation（组间抵消）

定义：

\[
\boxed{
\kappa_{\mathrm{source}}^{A}
=
\frac{
\|E_V+E_P+E_G\|
}{
\|E_V\|+\|E_P\|+\|E_G\|
}
}
\]

它与原来的 visual-token cancellation 不同：

- 原来的 \(\kappa\)：视觉 token 内部；
- 新的 \(\kappa_{\mathrm{source}}^{A}\)：visual / prompt / generation 三个来源组之间。

---

# 11. 实验 B：All-attention + Residual

这一部分要区分两种定义。

---

# 12. B1：True All-attention + Residual Path（真实全注意力+残差路径）

定义：

\[
S
=
r+A_V+A_P+A_G.
\]

如果 bias 单独固定，则：

\[
\boxed{
\gamma_{AR}(\alpha)
=
b_O+\alpha S
}
\]

对应各来源：

\[
E_R^{AR}
=
\int_0^1
J_G(\gamma_{AR}(\alpha))r\,d\alpha,
\]

\[
E_V^{AR}
=
\int_0^1
J_G(\gamma_{AR}(\alpha))A_V\,d\alpha,
\]

\[
E_P^{AR},
\qquad
E_G^{AR}.
\]

满足：

\[
E_R^{AR}
+
E_V^{AR}
+
E_P^{AR}
+
E_G^{AR}
=
G(z)-G(b_O).
\]

---

## 12.1 该版本的问题

如果

\[
b_O\approx0,
\]

路径接近：

\[
0\rightarrow z.
\]

RMSNorm（均方根归一化）具有近似 scale invariance（尺度不变性），因此：

\[
RMSNorm(\alpha z)
\]

在绝大部分 \(\alpha>0\) 区域变化较小，而主要变化集中在 \(\alpha\approx0\)。

所以：

- 原 visual-only path 已验证的 K4 不能直接复用；
- 必须重新做 numerical quadrature audit（数值求积审计）；
- 该 path 暂时不建议直接作为 4000 图主实验。

---

# 13. B2：Frozen-endpoint Full-source Decomposition（冻结端点完整来源分解）

这建议作为 residual analysis（残差分析）的主版本。

先计算：

\[
n=RMSNorm(z).
\]

clean endpoint RMS scale（干净端点RMS缩放）：

\[
D(z)
=
\operatorname{Diag}
\left(
\frac{\gamma}
{\sqrt{\operatorname{mean}(z^2)+\epsilon}}
\right).
\]

然后：

\[
\tilde R=D(z)r,
\]

\[
\tilde V=D(z)A_V,
\]

\[
\tilde P=D(z)A_P,
\]

\[
\tilde G=D(z)A_G,
\]

\[
\tilde B=D(z)b_O.
\]

满足 exact endpoint decomposition（精确端点分解）：

\[
\boxed{
n
=
\tilde R+\tilde V+\tilde P+\tilde G+\tilde B
}
\]

然后只对纯 FFN：

\[
F(n)=FFN(n)
\]

走

\[
\gamma_F(\alpha)=\alpha n.
\]

定义：

\[
C_R
=
\int_0^1
J_F(\alpha n)\tilde R\,d\alpha,
\]

\[
C_V,
\quad
C_P,
\quad
C_G,
\quad
C_B.
\]

满足：

\[
\boxed{
FFN(n)
=
FFN(0)
+
C_R+C_V+C_P+C_G+C_B.
}
\]

---

# 14. Residual 部分重点分析指标

## 14.1 Residual dominance（残差主导度）

令

\[
C_A=C_V+C_P+C_G.
\]

定义：

\[
\boxed{
D_R
=
\frac{
\|C_R\|
}{
\|C_R\|+\|C_A\|
}
}
\]

解释：

> 当前 FFN 的响应更主要来自历史 residual，还是来自本层刚写入的新 attention 信息？

注意：

\[
\boxed{
Residual \neq Language\ Prior
}
\]

因为 residual 中同时包含前面层累积的：

- visual information（视觉信息）；
- prompt information（提示词信息）；
- generated-text information（生成文本信息）；
- previous FFN updates（历史FFN更新）。

论文中应称：

> historical residual representation（历史残差表示）。

---

## 14.2 Visual–Residual alignment（视觉-残差对齐）

\[
\boxed{
R_{RV}
=
\cos(C_R,C_V)
}
\]

---

## 14.3 Generation–Residual alignment（生成文本-残差对齐）

\[
\boxed{
R_{RG}
=
\cos(C_R,C_G)
}
\]

---

## 14.4 Prompt–Residual alignment（提示词-残差对齐）

\[
R_{RP}
=
\cos(C_R,C_P).
\]

---

## 14.5 Residual visual-generation balance（残差视觉-生成文本平衡）

定义：

\[
\boxed{
B_{RVG}
=
\cos(C_R,C_G)
-
\cos(C_R,C_V)
}
\]

解释：

- \(B_{RVG}>0\)：历史 residual 的 FFN 响应方向更接近 generation；
- \(B_{RVG}<0\)：更接近 visual。

该量很适合作为 hallucination candidate signal（幻觉候选信号），但实验前不预设 HALL 应该更高还是更低。

---

# 15. Attention–Residual cooperation（注意力-残差协作）

定义：

\[
\boxed{
\kappa_{A,R}
=
\frac{
\|C_A+C_R\|
}{
\|C_A\|+\|C_R\|
}
}
\]

解释：

- 接近 1：current attention information（当前注意力信息）与 historical residual（历史残差）同向合作；
- 较低：两者存在明显 cancellation（抵消）。

这是下一阶段非常值得关注的量。

---

# 16. FFN 前后 residual-source geometry（残差-来源几何）变化

Frozen-RMS input space（冻结RMS输入空间）：

\[
C_{RV}^{in}
=
\cos(\tilde R,\tilde V)
\]

FFN 后：

\[
C_{RV}^{out}
=
\cos(C_R,C_V)
\]

定义：

\[
\boxed{
\Delta C_{RV}
=
C_{RV}^{out}
-
C_{RV}^{in}
}
\]

同理：

\[
\Delta C_{RG}.
\]

这样可以区分：

> residual 和 visual / generation 的冲突本来就存在，

还是：

> FFN 进一步放大或改变了这种冲突。

---

# 17. 下一轮实验分三阶段

## Stage 1：Numerical Audit（数值审计）

数据量：

- 每模型约 50 个 representative target-layer cases（代表目标-层案例）。

### 17.1 All-attention Path

比较：

\[
K=4,8,16,32,64
\]

以 FP32 K64 作为 numerical reference（数值参考）。

验证：

1. completeness（完备性）；
2. K4/K8/K16 与 K64 的误差；
3. group JVP 与逐 token 累加一致；
4. factored backend（因式分解后端）与 direct `vmap(jvp)` 一致；
5. prompt / visual / generation token 分区正确；
6. causal future tokens（未来token）贡献严格为 0。

### 17.2 All-attention + Residual True Path

额外检查：

\[
K=4,8,16,32,64
\]

必要时小规模增加：

\[
K=128.
\]

重点观察：

- 是否在 \(\alpha\approx0\) 附近变化剧烈；
- K4 是否失效；
- true path 与 frozen-RMS decomposition 的差异。

若 true residual path 数值不稳定，则只保留为 appendix ablation（附录消融），不进入正式 4000 图主实验。

---

# 18. Stage 2：Mechanism Study（机制研究）

数据量：

- 四模型共享 500 图；
- 保持现有 400/100 train/test image split（图片划分）；
- 全部目标和全部 decoder layers（解码器层）。

不训练检测器，只看 REAL/HALL 的 descriptive mechanism（描述性机制）。

---

## 18.1 All-attention 重点曲线

每层画：

\[
G_V,G_P,G_G
\]

\[
R_V,R_P,R_G
\]

\[
C_{VG}^{out},
C_{VP}^{out},
C_{PG}^{out}
\]

\[
\Delta C_{VG},
\Delta C_{VP},
\Delta C_{PG}
\]

\[
C_V^{ctx},
M_V^{ctx}
\]

\[
\kappa_V,\kappa_P,\kappa_G
\]

\[
\kappa_{\mathrm{source}}^{A}.
\]

---

## 18.2 Full-source + Residual 重点曲线

\[
D_R
\]

\[
R_{RV},R_{RG},R_{RP}
\]

\[
B_{RVG}
\]

\[
\kappa_{A,R}
\]

\[
\Delta C_{RV},
\Delta C_{RG}.
\]

---

## 18.3 统计口径

继续沿用现有规范：

1. 先对 target-layer（目标-层）统计；
2. 再按 mentions 等权汇总；
3. REAL/HALL 分开；
4. 报告 mean（均值）、median（中位数）、IQR（四分位区间）；
5. 保留 all/train/test；
6. 做 same-image paired comparison（同图配对比较）；
7. 排除 label-conflicting targets（标签冲突目标）；
8. 机制阶段只做描述，不根据 500 图结果挑 detector features。

---

# 19. Stage 3：Hallucination Detection（幻觉检测）

数据量：

- 原 4000 图；
- 原 3200 / 800 image split；
- 全部 mentions；
- 四模型：
  - Qwen2.5-VL-7B
  - LLaVA-1.5-7B
  - Qwen3-VL-8B
  - InternVL2.5-8B

固定训练协议，不重新调 classifier（分类器）结构。

---

# 20. 固定的 feature families（特征族）

建议提前冻结以下 7 组：

## F0：现有 attention baseline（注意力基线）

使用当前最稳定的：

\[
U_P,U_G,U_V
\]

即 attention × gate（注意力×门控）。

---

## F1：All-attention magnitude（全注意力强度）

例如：

\[
S_V^{gross},
S_P^{gross},
S_G^{gross}
\]

\[
S_V^{net},
S_P^{net},
S_G^{net}
\]

\[
G_V,G_P,G_G.
\]

---

## F2：All-attention geometry（全注意力几何）

包括：

\[
R_V,R_P,R_G
\]

\[
C_{VG}^{out},C_{VP}^{out},C_{PG}^{out}
\]

\[
\Delta C_{VG},\Delta C_{VP},\Delta C_{PG}
\]

\[
C_V^{ctx},M_V^{ctx}
\]

\[
\kappa_V,\kappa_P,\kappa_G,\kappa_{\mathrm{source}}^{A}.
\]

---

## F3：Full-source magnitude/share（完整来源强度/份额）

包括：

\[
\|C_R\|,
\|C_V\|,
\|C_P\|,
\|C_G\|
\]

以及 residual share（残差份额）。

---

## F4：Residual geometry（残差几何）

包括：

\[
R_{RV},R_{RG},R_{RP}
\]

\[
B_{RVG}
\]

\[
\kappa_{A,R}
\]

\[
\Delta C_{RV},\Delta C_{RG}.
\]

---

## F5：Geometry only（纯几何）

\[
F5=F2+F4.
\]

用来检验：

> direction / interaction geometry（方向/交互几何）本身是否有检测价值。

---

## F6：Attention + Geometry（注意力+几何）

\[
F6=F0+F5.
\]

这是最重要的最终融合组。

---

# 21. Detector protocol（检测器协议）

保持当前固定配置：

- MLP hidden layers（隐藏层）：`[128,64,32]`
- BatchNorm（批归一化）
- dropout = 0.3
- Adam
- learning rate = 0.001
- weight decay = \(10^{-5}\)
- batch size = 256
- maximum epochs = 100
- seeds = 43 / 44 / 45
- 原 image split 不变
- 不重新调参

主指标：

\[
AUROC
\]

和

\[
HALL\text{-}AUPR.
\]

最终主要比较：

\[
F5\ vs\ F0
\]

\[
F6\ vs\ F0
\]

以及：

\[
F5/F6
\]

对已有 source-strength baseline（来源强度基线）的提升。

---

# 22. Length / Position Control（长度/位置控制）

generation-related features（生成文本相关特征）必须增加控制变量，因为 hallucinated objects 往往可能出现在更晚的位置。

至少保存并评估：

\[
N_G
=
\text{generated-prefix token count}
\]

以及：

\[
response\_index.
\]

训练 trivial control（简单控制）：

\[
[N_G,response\_index].
\]

正式结论只有在新 source geometry features 明显超过这个控制后才更可信。

对 generation source strength，建议同时报告：

### Raw gross（原始总量）

\[
S_G^{gross}
\]

### Per-token mean（每token均值）

\[
\frac{S_G^{gross}}{N_G}.
\]

---

# 23. 正式提取时建议保存的数据结构

每个 target、每层：

## 23.1 Token-level scalars（token级标量）

对每个 causal attention source：

```text
source_type            # prompt / visual / generation
source_position
write_norm             # ||a_m||
effect_norm            # ||e_m||
direction_cosine       # cos(a_m, e_m)
```

不保存完整 `a_m`、`e_m` 向量。

---

## 23.2 Group-level scalars（组级标量）

对 visual / prompt / generation：

```text
gross_norm
net_norm
group_gain
group_rotation
within_source_kappa
token_count
```

---

## 23.3 Cross-source geometry（跨来源几何）

```text
cos_vg_in
cos_vg_out
delta_cos_vg

cos_vp_in
cos_vp_out
delta_cos_vp

cos_pg_in
cos_pg_out
delta_cos_pg

visual_context_cos
visual_context_relative_change

source_group_kappa
```

---

## 23.4 Residual geometry（残差几何）

```text
residual_norm
attention_total_norm
residual_share

cos_residual_visual
cos_residual_prompt
cos_residual_generation

residual_visual_generation_balance
attention_residual_kappa

delta_residual_visual_cos
delta_residual_generation_cos
```

---

# 24. 实现建议

一次 clean model capture（干净模型捕获）完成所有实验。

流程：

```text
capture current layer
        ↓
reconstruct all_token_writes
        ↓
split positions:
prompt / visual / generation
        ↓
构造 A_P / A_V / A_G / residual / bias
        ↓
Branch A:
All-attention true RMSNorm path
        ↓
逐token在线计算 ||e_m|| / cosine
        ↓
聚合 E_P / E_V / E_G
        ↓
计算 source geometry

Branch B:
Visual-only path
        ↓
得到 E_V^vis
        ↓
计算 visual context modulation

Branch C:
Frozen-endpoint full-source decomposition
        ↓
得到 C_R / C_P / C_V / C_G / C_B
        ↓
计算 residual geometry
```

---

# 25. 计算优化

## 25.1 每个 token 的 \(\|e_m\|\) 保留

正式提取中建议保留：

\[
\boxed{
\|e_m\|
}
\]

因为后续可以自由计算：

- source gross；
- per-token mean；
- source gain；
- Top-k token；
- distribution；
- token-level detector；
- visual/prompt/generation 内部统计。

---

## 25.2 但完整 \(e_m\) 不落盘

完整向量只在数值审计子集保存。

正式 4000 图只保存 scalar summaries（标量摘要）。

---

## 25.3 Group vector 在线累计

在计算 \(e_m\) 时：

```text
E_V += e_m  if visual
E_P += e_m  if prompt
E_G += e_m  if generation
```

当前 target/layer 结束后计算所有 group cosine / norm，再释放 group vectors。

---

# 26. 预期论文故事

如果实验支持，可以形成以下叙事：

### 第一层

Attention routing（注意力路由）决定：

> 哪些 visual / prompt / generated sources 被写入当前 target residual。

### 第二层

FFN transformation（FFN变换）决定：

> 这些已写入的信息如何被放大、旋转、混合和抵消。

### 第三层

Residual integration（残差整合）决定：

> 当前新信息如何与 historical residual representation（历史残差表示）结合。

最终可以研究 hallucination 是否对应于：

- visual context modulation 更强；
- visual-generation alignment 更差；
- historical residual 更偏向 generation；
- attention 与 residual cancellation 更强；
- 或其他稳定的 source interaction pattern（来源交互模式）。

注意以上都必须由数据验证，不能提前作为结论。

---

# 27. 最终执行顺序

## Step 1

实现 All-attention path：

- 保留每个 token 的 \(\|a_m\|\)
- 保留每个 token 的 \(\|e_m\|\)
- 保留每个 token 的 \(\cos(a_m,e_m)\)
- 在线累计 \(E_V,E_P,E_G\)

## Step 2

做 50-case numerical audit：

- K4/8/16/32/64
- K64 FP32 reference
- completeness
- direct JVP parity
- group/individual sum parity

## Step 3

做 All-attention + residual true path 小规模审计：

- 重点检查 RMSNorm near-zero path 问题
- 必要时 K128
- 不稳定则不作为主正式路径

## Step 4

扩展 frozen-RMS full-source composition：

- 保留 group vectors 的在线统计
- 加入 residual geometry

## Step 5

共享 500 图机制实验：

- REAL/HALL 曲线
- same-image paired analysis
- 不挑 detector feature

## Step 6

冻结 F0–F6 feature families

## Step 7

原 4000 图固定 MLP / 3 seeds 跑 hallucination detection

## Step 8

最终固定比较做 image-level paired bootstrap（图片级配对自助法）

重点比较：

\[
F5/F6
\]

相对于：

\[
F0
\]

以及已有 source-strength baselines（来源强度基线）。

---

# 28. 当前最值得优先关注的信号

如果需要控制实验规模，我建议优先级如下：

1. \[
   \boxed{
   C_V^{ctx}
   =
   \cos(E_V^{vis},E_V^{all})
   }
   \]

2. \[
   \boxed{
   \cos(E_V,E_G)
   }
   \]

3. \[
   \boxed{
   \cos(C_R,C_V)
   }
   \]

4. \[
   \boxed{
   \cos(C_R,C_G)
   }
   \]

5. \[
   \boxed{
   B_{RVG}
   =
   \cos(C_R,C_G)-\cos(C_R,C_V)
   }
   \]

6. \[
   \boxed{
   \kappa_{A,R}
   }
   \]

7. source-specific gain / rotation：
   \[
   G_V,G_P,G_G,
   \qquad
   R_V,R_P,R_G.
   \]

---

# 29. 核心原则

下一阶段不要再围绕：

> “\(\hat J\) 是否只是 scalar scaling（标量缩放）？”

因为现有完整子空间实验已经否定这一点。

下一阶段真正应该回答的是：

\[
\boxed{
\text{FFN 的非平凡变换如何依赖 information source（信息来源）与 residual context（残差上下文）？}
}
\]

以及：

\[
\boxed{
\text{这种 source-dependent transformation（来源依赖变换）是否能够解释或检测 hallucination？}
}
\]

这将比继续优化单一的 \(P_{FFN}\)、JS 或 OT 更有研究价值。
