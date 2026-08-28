# JFFN-P：视觉注意写入经 FFN Jacobian 响应的完整验证报告

## 0. 范围与结论

本报告只分析已经完整完成的两个模型：

- LLaVA-1.5-7B：3971 张图，14951 个唯一目标位置，15463 个官方 mention；
- InternVL2.5-8B：3927 张图，11630 个唯一目标位置，11759 个官方 mention。

没有续跑、补跑或推断 Qwen2.5-VL/Qwen3-VL 的结果。下文凡是“独立深度验证”，指 LLaVA 的 3 图、6 目标、5 层实验；InternVL 只具有正式 5 图 smoke、完整 cohort 统计和下游实验，因此不会把 LLaVA 的所有控制实验外推为 InternVL 已经验证。

### Executive verdict

一句话结论是：**JFFN-P 的核心数值实现是可信的，它是一个比旧 cosine-P 更尖锐、更有目标特异性、空间定位也更好的“局部 FFN 响应能量分布”；但是它没有成为更好的幻觉检测 P。**

具体而言：

1. **代码正确性：基本 PASS。** Attention 分解、视觉 token 范围、因果行、Norm+FFN 函数、JVP 方向、GQA/packed-QKV、输出投影方向均与公式一致。当前发现一个不影响 (E/P) 的审计字段 bug：`component_sum_relative_error` 被硬编码成 `0.0`。
2. **数学正确性：有条件 PASS。** 计算的确是
   \[
   \delta^j_{l,t}=J_{f_l}(z_{l,t})a^j_{l,t},\qquad
   E_{l,t,j}=\|\delta^j_{l,t}\|_2.
   \]
   但 (E) 是局部响应向量的模长，不是概率、信息量、因果效应或“物体存在置信度”。
3. **数值可靠性：LLaVA PASS，InternVL PASS/有限。** 两模型 smoke 均通过有限性、分布归一化、线性、分块一致性和有限差分；LLaVA 还通过了独立 FP32 epsilon sweep。InternVL 尚未做同等规模的独立 FP32 sweep、随机方向和图像扰动。
4. **科学有效性：部分成立。** 新 P 对目标和空间位置明显更敏感，且这种排序优势在熵匹配后仍然存在；因此不是单纯“softmax 变尖”造成的假象。
5. **幻觉检测有效性：当前主假设不成立。** 预注册主设置下，LLaVA 显著差于旧 P；InternVL 没有可靠提升。定位更准不等于更能判断模型是否在幻觉。

汇总图：

![JFFN-P 两模型汇总](jacobian_visual_ffn_validation_summary.png)

## 1. 当前实现到底计算了什么

一层 decoder 的残差路径被捕获为：

\[
h^{pre}_{l,t}\xrightarrow{\mathrm{Attn}}o^{attn}_{l,t},\qquad
z_{l,t}=h^{pre}_{l,t}+o^{attn}_{l,t}.
\]

对视觉位置 (j)，代码从该层、该 query 行的 attention 权重和 V 向量重建：

\[
a^j_{l,t}=W_l^O\operatorname{concat}_h
\left(\alpha_{l,t,h,j}v_{l,h,j}\right).
\]

然后对真实的 FFN 子函数

\[
f_l(z)=\operatorname{FFN}_l(\operatorname{Norm}_l(z))
\]

执行精确 forward-mode JVP：

\[
\delta^j_{l,t}=J_{f_l}(z_{l,t})a^j_{l,t},
\quad E_{l,t,j}=\|\delta^j_{l,t}\|_2,
\quad P_{l,t,j}=\frac{E_{l,t,j}}{\sum_kE_{l,t,k}+\epsilon}.
\]

这里没有 softmax 和温度。熵匹配控制才使用：

\[
P^{match}_{l,t}=\operatorname{softmax}
\left(\frac{\log(E_{l,t}+\epsilon)}{\beta_l}\right),
\]

其中 (β_l) 只用训练 split 中固定 500 张图拟合，两个模型的测试泄漏均为 0。

### 代码路径

- Decoder/attention/FFN 适配：[features/visual_ffn_jacobian.py](features/visual_ffn_jacobian.py#L43)
- 精确 `FFN(Norm(z))`：[features/visual_ffn_jacobian.py](features/visual_ffn_jacobian.py#L137)
- V 投影、GQA repeat、InternLM2 packed V 槽：[features/visual_ffn_jacobian.py](features/visual_ffn_jacobian.py#L149)
- 每个视觉 token 的 attention residual write：[features/visual_ffn_jacobian.py](features/visual_ffn_jacobian.py#L186)
- `vmap(jvp)`：[features/visual_ffn_jacobian.py](features/visual_ffn_jacobian.py#L301)
- (E)、P、I/R/S/D：[features/visual_ffn_jacobian.py](features/visual_ffn_jacobian.py#L498)
- hpre、hmid、FFN 输出和 attention 行的 hook：[models/dgst_capture.py](models/dgst_capture.py#L318)
- 显式 causal mask、模型自身 scaling、FP32 attention softmax：[models/dgst_capture.py](models/dgst_capture.py#L15)
- 目标 token 前一因果行定位：[models/dgst_capture.py](models/dgst_capture.py#L1237)

## 2. 张量形状与模型适配

| 模型 | (L) | (D) | FFN 中间维 | (H) | (H_{kv}) | (D_h) | 视觉 token (M) | 正式 dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LLaVA-1.5-7B | 32 | 4096 | 11008 | 32 | 32 | 128 | 576 | FP16 |
| InternVL2.5-8B | 32 | 4096 | 14336 | 32 | 8 | 128 | 256 | BF16 |

关键中间量：

- attention row：`[T,H,S]`；
- repeated V：`[H,S,Dh]`；
- 每视觉 token 写入：`a_tokens [M,T,D]`；
- 聚合视觉写入：`a_visual [T,D]`；
- JVP 响应：`responses [M,T,D]`；
- energy/P：`[T,M]`。

LLaVA 是标准 `v_proj/o_proj`；InternVL 的 InternLM2 `wqkv` 布局为每个 KV head 的 `[query groups..., K, V]`，代码取最后一个槽作为 V，再按 4 个 GQA group repeat。两条路径均通过真实模型 smoke；synthetic 单元测试还覆盖了 GQA、packed-QKV 和非零 output bias 的处理。

输出投影 bias 的处理是正确的：完整 attention 重建包含一次 bias；每个 token 的 (a^j) 不分摊 bias，否则把同一个常数重复加 (M) 次。当前 LLaVA/InternVL 的 attention output projection 本身无 bias。

## 3. 首先记录原实现中发现的问题

### 3.1 `component_sum_relative_error` 被硬编码为 0

位置：[features/visual_ffn_jacobian.py](features/visual_ffn_jacobian.py#L283)，当前第 289 行。

当前返回值是：

```python
"component_sum_relative_error": 0.0
```

独立重算“视觉 token + 非视觉 token + output bias”的真实分解误差后，30 个案例为：

- mean：(5.02\times10^{-4})
- median：(5.17\times10^{-4})
- max：(6.01\times10^{-4})

严重性：**中等审计风险，低算法风险。** 它不会进入 (a^j,E,P) 或训练，只会让审计日志错误地显示完美的 0。正确实现应真正累加全序列 token contribution 和 bias，再与捕获的 attention output 比较。按照审计要求，本轮没有悄悄修改核心文件。

### 3.2 正式全量 shard 中的若干验证值是占位 0

正式抽取为了速度设置 `validation_computed=False`。因此未执行的 finite-difference、response-sum 等字段里可能保存 0；这些值只有与 `validation_computed` 一起读取才有意义。它们没有参与特征或训练，但不能被当成“全量 4000 图误差等于 0”。

### 3.3 BF16 下用 `h_mid-h_prev` 当重建参考会放大相消误差

全量 shard 没保留 `o_attn`，因此参考值来自 `h_mid-h_prev`：

| 模型 | 全量平均 relative error | median | q90 | mean cosine |
|---|---:|---:|---:|---:|
| LLaVA | 0.00116 | 0.00086 | 0.00211 | 0.999999 |
| InternVL | 0.01209 | 0.00915 | 0.02328 | 0.999902 |

InternVL 的 1.2% 看起来较大，但独立保留 `o_attn` 的 5 图 smoke 中最大重建误差只有 0.00315、最小 cosine 0.999996。主要原因是 BF16 的两个大残差相减，而不是 (a^j) 公式错误。建议后续 smoke 保留 `o_attn`；正式抽取无需为此增加巨大存储。

## 4. Attention 分解、视觉索引与因果位置

### 4.1 Attention reconstruction

LLaVA 独立 30 案例：

| 指标 | mean | median | max |
|---|---:|---:|---:|
| `FFN(Norm(z))` 重算 vs 捕获 FFN 输出 relative error | 0.000574 | 0.000610 | 0.000738 |
| attention 完整重建 relative error | 0.000429 | 0.000440 | 0.000576 |
| 全 token component sum relative error | 0.000502 | 0.000517 | 0.000601 |
| 独立聚合 (a^{visual}) vs 正式实现 | 0.000129（聚合均值） | — | 0.000388（聚合 max） |

单 token 相对误差最高约 1%，发生在模长极小的 token 上，主要来自 FP16 中不同 GEMM batch shape 的舍入；聚合方向误差远小于它。

### 4.2 LLaVA 视觉 token 范围

三张图的真实 processor/capture 审计完全一致：

- 序列位置 0 是 BOS，token id 1；
- 视觉范围是半开区间 `[5,581)`；
- 恰好 576 个 `<image>` 展开位置；
- processor 后 input length 与模型实际 capture length 一致；
- 三图完整序列长度分别为 716、725、675。

因此本实验不是误把单个 placeholder 当 576 个 hidden state，也没有 off-by-one。目标 response index 被映射到“生成它的前一因果行”；例如 image 408805 的 response index 24/100 对应 decoder query row 616/692。

### 4.3 InternVL 视觉 token 范围

InternVL wrapper 直接构造 `IMG_START + 256 × IMG_CONTEXT + IMG_END`，正式 smoke 的 `visual_end-visual_start=256`，并使用真实 `<IMG_CONTEXT>` token 区间。它不依赖 LLaVA 的 placeholder 展开逻辑。

### 4.4 首 subtoken 与目标行

独立样例确实在首 subtoken 上测量：

| 物体词 | 首 subtoken decode | response index | 标签 |
|---|---|---:|---|
| stove | `st` | 24 | REAL |
| cup | `cup` | 100 | HALL |
| toilet | `to` | 85 | REAL |
| bottle | `bott` | 106 | HALL |
| cat | `cat` | 9 | REAL |
| bird | `bird` | 28 | HALL |

这与 gfchair 的 InsLen official first-token protocol 一致。JFFN 使用的 query row 是该首 subtoken 生成前的行，不是目标 token 自己已经进入模型后的行。

## 5. Autoregressive、causal mask 与 KV cache

正式抽取明确 `use_cache=False`，因此正式结果不存在“把 cache 中历史 visual K/V 当成当前层重新生成 hidden state”的问题。每张图只做一次完整 caption teacher-forced forward，但自定义 attention 使用模型传入的 scaling、加性 mask，并显式屏蔽未来位置；未来 answer token 不可见。

LLaVA 对两个目标、五层做了 full-caption 与真实 prefix 的比较：

| 指标 | mean | max |
|---|---:|---:|
| logits relative error | 0.000921 | 0.000934 |
| hmid relative error | 0.00111 | 0.00168 |
| (a^{visual}) relative error | 0.00148 | 0.00321 |
| P 的 JS | (8.66\times10^{-7}) | (1.80\times10^{-6}) |

所有 logits argmax 一致。这些小差异来自 FP16 下 full/prefix 不同矩阵形状的 kernel 舍入，而不是因果泄漏。

同一个 prefix 使用 native DynamicCache 与 `use_cache=False` 的 logits relative error 为精确 0，argmax 全一致。缓存第一层 K/V 形状示例为 `[1,32,617,128]` 或 `[1,32,693,128]`，视觉历史位置包含在 cache 长度内。结论是：**cache 等价性通过，但正式抽取根本不依赖 cache。**

## 6. 被求导的函数、图断开与 residual skip

### 6.1 Norm 在 Jacobian 内，FFN residual skip 不在

`ffn_from_residual()` 对两类模型分别执行：

```python
layer.mlp(layer.post_attention_layernorm(z))
layer.feed_forward(layer.ffn_norm(z))
```

因此 Jacobian 包含 RMSNorm/Norm 和 FFN，符合 (f_l(z)=FFN_l(Norm_l(z)))。它没有返回 `z + FFN(Norm(z))`，所以没有把 identity residual 错算进 (J_fa)。此项 PASS。

### 6.2 `no_grad`/detach 没有破坏 JVP

模型 caption forward 在 `torch.no_grad()` 中，捕获张量也被 detach；这是有意避免保留 32 层的反向图。随后 JVP 把 detach 后的 (z) 作为新的 primal 输入，在 `torch.enable_grad()`/forward AD 中穿过仍然存在的 Norm+FFN 模块。

实测：`z.requires_grad=False`，最终保存 response 也 detach，但 `torch.func.jvp` 返回非零、能通过 finite difference 的方向导数。它不需要回传到图像编码器或之前 decoder 层，因此这里不是 accidental graph detachment。

## 7. JVP、有限差分与非线性有限效应

### 7.1 直接线性与可加性

LLaVA FP32 临时 Norm+FFN 副本：

- 标量线性 (J(ca)=cJ(a))：mean relative error (7.01\times10^{-7})，max (7.82\times10^{-7})；
- per-token 加和 (∑_jJa_j=J(∑_ja_j))：mean (1.60\times10^{-4})，max (3.65\times10^{-4})；
- signed projection contribution 的和：mean 1.000002，范围 `[0.999721,1.000288]`。

两模型正式 smoke：

| 模型 | P sum max error | max JVP linearity error | full/chunk max error | median FD cosine |
|---|---:|---:|---:|---:|
| LLaVA | (2.38\times10^{-7}) | (2.02\times10^{-6}) | (2.42\times10^{-6}) | 1.000000 |
| InternVL | (2.38\times10^{-7}) | (1.10\times10^{-6}) | (1.74\times10^{-6}) | 1.000000 |

严格说 full/chunk 在 FP16/BF16 不会逐 bit 相同，因为 GEMM batch 形状不同，但误差远低于正式阈值。

### 7.2 Central finite-difference epsilon sweep

LLaVA 用 FP32 Norm+FFN 对 9 个 epsilon 做了 270 个比较。中位数如下：

| (ε) | (ε\|a\|/\|z\|) | central rel. error | central cosine | forward rel. error |
|---:|---:|---:|---:|---:|
| (10^{-4}) | (7.51\times10^{-6}) | 0.01642 | 0.99987 | 0.04207 |
| (10^{-3}) | (7.51\times10^{-5}) | 0.00173 | 0.999999 | 0.00408 |
| (10^{-2}) | (7.51\times10^{-4}) | 0.000191 | 1.000000 | 0.00129 |
| (3\times10^{-2}) | 0.00225 | 0.000102 | 1.000000 | 0.00258 |
| (10^{-1}) | 0.00751 | 0.000073 | 1.000000 | 0.00840 |
| (3\times10^{-1}) | 0.02252 | 0.000552 | 1.000000 | 0.02512 |
| 1 | 0.07507 | 0.00608 | 0.999997 | 0.08272 |

小 epsilon 端受减法消去影响，大 epsilon 端受非线性影响，呈正常 U 形；局部 JVP 在约 (10^{-2}) 到 (10^{-1}) 区间高度可靠。

### 7.3 真正删除视觉写入不是完全线性的

比较

\[
f(z)-f(z-r a^{visual})
\]

和 (rJ_f(z)a^{visual})：

| 删除比例 (r) | median relative error | median cosine |
|---:|---:|---:|
| 0.10 | 0.00843 | 0.999965 |
| 0.25 | 0.02105 | 0.999780 |
| 0.50 | 0.04215 | 0.999116 |
| 0.75 | 0.06328 | 0.998002 |
| 1.00 | 0.08443 | 0.996434 |

完整删除时 mean error 0.1527、max 0.5989、最低 cosine 0.8012。因此 JFFN 是可信的**局部**响应，不应被描述成“完整移除图像后的精确 FFN 改变量”。

## 8. I/R/S/D、随机方向与方向几何

定义：

\[
I=\|a^{visual}\|_2,\quad
R=\|J_f(z)a^{visual}\|_2,\quad
S=R/(I+\epsilon),\quad
D=\cos(J_f(z)a^{visual},a^{visual}).
\]

LLaVA 独立实验为每个层–目标生成 32 个与 (a^{visual}) 等模长的随机方向：

| 层 | visual (S) | random median | ratio | visual percentile |
|---:|---:|---:|---:|---:|
| 1 | 1.233 | 0.319 | 3.853 | 1.000 |
| 8 | 0.804 | 0.540 | 1.489 | 1.000 |
| 16 | 0.429 | 0.410 | 1.046 | 0.672 |
| 24 | 0.249 | 0.241 | 1.032 | 0.620 |
| 32 | 2.032 | 0.384 | 5.298 | 1.000 |

解释：视觉方向在早层和最终层具有明显特殊的 FFN sensitivity，但 16/24 层仅比 matched-norm random 高约 3%–5%。所以不能笼统声称每层 FFN 都“专门放大视觉方向”。(S) 也受层本身 Jacobian scale 影响，跨层直接比较必须配 random baseline 或逐层标准化。

独立 3 图的小样本中 REAL/HALL 的平均 (S) 分别 0.942/0.957，方向 cosine 分别 -0.183/-0.217；样本太少，不能做类别结论。

## 9. Per-token (E_j)：attention、输入模长、gain 与 cancellation

LLaVA 30 个案例中：

- attention weight 与 (E_j)：Pearson 0.764，Spearman 0.866；
- (‖a_j‖) 与 (E_j)：Pearson mean 0.951，median 0.988；
- 层 16/24 的 attention–(E) Spearman 仍为 0.970/0.987。

这说明新 P 并非纯 attention map，但大部分 token 排序仍由“attention 权重 × value/output 写入模长”驱动。Jacobian 对排序做了二次调制，而不是完全替代 attention retrieval。

token-level gain

\[
G_j=\frac{\|J_f(z)a_j\|}{\|a_j\|+\epsilon}
\]

可以把 attention/write magnitude 和 FFN sensitivity 分开，但当 (‖a_j‖) 极小时会不稳定，不能未经 shrink/threshold 就直接归一化成 P。

有符号贡献取总响应方向 (u=J(a^{visual})/\|J(a^{visual})\|)：

\[
C_j=\frac{u^\top J(a_j)}{\|J(a^{visual})\|}.
\]

(∑_jC_j) 数值上等于 1，但 (∑_j|C_j|) mean 1.0088、max 1.0754，存在负贡献。更重要的是：

\[
\frac{\|\sum_jJ(a_j)\|}{\sum_j\|J(a_j)\|}
\]

的 mean 0.588、median 0.557、min 0.351。也就是向量方向存在明显分散/相消。当前 (P_j\propto\|J(a_j)\|) 丢弃符号，会把互相抵消的 token 也都当作高“能量”。这是重要的数学局限。

## 10. 图像扰动与负控制

固定 LLaVA caption/目标，仅替换视觉输入为黑图、8×8 patch shuffle、无关图。选取层的 P-JS：

| 控制 | L1 | L8 | L16 | L24 | L32 |
|---|---:|---:|---:|---:|---:|
| black | 0.036 | 0.368 | 0.396 | 0.445 | 0.142 |
| patch shuffle | 0.044 | 0.328 | 0.394 | 0.500 | 0.121 |
| unrelated | 0.033 | 0.363 | 0.357 | 0.468 | 0.131 |

中层 P cosine 经常降到 0.06–0.23，说明新 P 确实随图像内容/空间排列改变，不是只由固定文本 query 产生。与此同时 aggregate (S) 的比值多在约 0.89–1.36：**空间 token 排名比总 gain 更图像敏感。**

本轮没有完成真正的“移除全部视觉 token 后重新生成”或严格 text-only forward，因为这会改变序列结构和模型输入分布；也没有在 InternVL 重复图像扰动。因此 text-only/visual-mask 项标为未完成，而不是 PASS。

## 11. P 分布本身是否合理

### 11.1 新 P 很尖，旧 P 近乎均匀

风险层 16–32 平均：

| 模型/标签 | old-hpre entropy | new entropy | matched entropy | old effective M | new effective M | new Top-32 mass |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA REAL | 0.9891 | 0.7641 | 0.9891 | 539.0/576 | 147.7/576 | 0.5637 |
| LLaVA HALL | 0.9902 | 0.7658 | 0.9894 | 542.5/576 | 151.3/576 | 0.5575 |
| InternVL REAL | 0.9691 | 0.7269 | 0.9695 | 217.1/256 | 64.2/256 | 0.7310 |
| InternVL HALL | 0.9735 | 0.7098 | 0.9703 | 222.3/256 | 60.9/256 | 0.7408 |

因此你之前“视觉 token 概率都差不多”的担心适用于旧 cosine-P；JFFN-P 并不平。它甚至非常尖。问题反而变成：这种尖锐性是否是有意义的排序，还是仅仅由 attention magnitude 放大。

### 11.2 同图不同目标的 P 更不相同

| 模型/source | raw cosine | JS | TV | Top-32 overlap |
|---|---:|---:|---:|---:|
| LLaVA old-hpre | 0.936 | 0.0163 | 0.1287 | 0.2820 |
| LLaVA new | 0.505 | 0.1954 | 0.4867 | 0.3989 |
| LLaVA new matched | 0.960 | 0.0088 | 0.0953 | 0.3989 |
| InternVL old-hpre | 0.940 | 0.0162 | 0.1314 | 0.5664 |
| InternVL new | 0.296 | 0.3052 | 0.6367 | 0.3528 |
| InternVL new matched | 0.835 | 0.0373 | 0.2035 | 0.3528 |

熵匹配保持 token 排名，所以 Top-32 overlap 与 raw new 完全一致；概率差异被摊平后 cosine/JS 大幅回到旧 P 附近。由此可分解出两个效应：raw JS 的一大部分来自尖锐性，但 Top-K 排名的目标特异性是真实存在的。

### 11.3 COCO 空间定位明显改善

仅在有同类别真实 COCO box 的 REAL mention 上：LLaVA 11418 个，InternVL 9392 个；网格不匹配为 0。

| 模型/source | bbox mass | uniform enrichment | Top-1 pointing | patch AUPRC |
|---|---:|---:|---:|---:|
| LLaVA old-hpre | 0.2581 | 1.080 | 0.3360 | 0.3494 |
| LLaVA new | 0.3866 | 4.775 | 0.5018 | 0.4507 |
| LLaVA new matched | 0.2732 | 1.300 | 0.5018 | 0.4507 |
| InternVL old-hpre | 0.2389 | 1.010 | 0.2786 | 0.3165 |
| InternVL new | 0.3962 | 4.128 | 0.6248 | 0.4451 |
| InternVL new matched | 0.2763 | 1.524 | 0.6248 | 0.4451 |

Top-1 pointing 和 patch AUPRC 只依赖排序，因此 raw new 与 entropy-matched 完全相同；两者都大幅优于旧 P。这是最强的正面证据：**Jacobian/attention-write 组合产生了更好的物体相关 token 排序，不只是把分布做尖。**

## 12. REAL/HALL 与下游幻觉检测

### 12.1 I/R/S/D 的全量类别差异

风险层 16–32 平均：

| 模型/标签 | I | R | S | D | P entropy | Top-32 mass |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA REAL | 6.746 | 2.998 | 0.4394 | -0.0982 | 0.7641 | 0.5637 |
| LLaVA HALL | 6.355 | 2.588 | 0.4331 | -0.1020 | 0.7658 | 0.5573 |
| InternVL REAL | 10.019 | 4.404 | 0.4333 | -0.1244 | 0.7268 | 0.7311 |
| InternVL HALL | 7.846 | 3.410 | 0.4417 | -0.1158 | 0.7096 | 0.7410 |

最稳定的模式不是 (S)，而是 HALL 的视觉 write (I) 和响应模长 (R) 更小。单变量最佳方向 AUROC：

| 模型 | I | R | S | D |
|---|---:|---:|---:|---:|
| LLaVA | 0.5736 | **0.6734** | 0.5529 | 0.5740 |
| InternVL | 0.6602 | **0.6787** | 0.5378 | 0.6183 |

这说明“有多少视觉写入/响应”比“单位写入被 FFN 放大多少”更像幻觉信号。(I) 与 (R) 的风险层平均相关也很高：LLaVA REAL/HALL 约 0.861/0.881，InternVL 约 0.920/0.937。

### 12.2 预注册主比较

主设置固定为 raw-logit Q gate + sqrt cost + risk+EV；AUROC 是三个 seed 的预测先平均后计算，并做 10000 次图片级 paired bootstrap：

| 模型 | old-hpre | new JFFN | Δ | 95% CI | 结论 |
|---|---:|---:|---:|---|---|
| LLaVA | 0.8934 | 0.8832 | -0.0102 | `[-0.0198,-0.0012]` | 新 P 显著更差 |
| InternVL | 0.8567 | 0.8506 | -0.0060 | `[-0.0197,+0.0077]` | 无可靠差异 |

熵匹配控制：

| 模型 | old-hpre | new matched | Δ | 95% CI | 结论 |
|---|---:|---:|---:|---|---|
| LLaVA | 0.8934 | 0.8872 | -0.0062 | `[-0.0142,+0.0015]` | 仍无提升 |
| InternVL | 0.8567 | 0.8552 | -0.0014 | `[-0.0151,+0.0116]` | 基本持平 |

因此不能把新 P 的空间优势写成幻觉检测优势。更合理的解释是：JFFN-P 找到了“模型当前从哪些 patch 形成 attention-mediated FFN 响应”，而幻觉还取决于语言先验、目标 token logit margin、视觉证据是否足够、对象存在性和跨层竞争。一个 hallucinated object 也可能对某些 patch 形成非常集中的内部响应。

### 12.3 后验辅助模型：I/R/S/D 值得继续研究

用相同 image split、标准化 balanced logistic 做探索性分析（去除 label-conflict position；不是预注册 MLP，不能和上表直接横比）：

| 模型 | old risk | new risk | old+new | I/R/S/D | old+I/R/S/D |
|---|---:|---:|---:|---:|---:|
| LLaVA | 0.8010 | 0.7480 | 0.8271 | 0.8678 | 0.8795 |
| InternVL | 0.6723 | 0.6650 | 0.7133 | 0.8366 | 0.8450 |

这个结果不是新的 headline，但给出明确方向：不要只把 (E_j) 归一化成 P；保留总量 (I,R,S,D) 可能更有检测价值，特别是 (R)。

## 13. 效率、显存与分片

LLaVA 单层、2 个目标、576 个方向的 warmed timing：

| 路径 | 时间 |
|---|---:|
| 全 576 vmap | 0.00426 s |
| chunk=64 | 0.01592 s |
| 顺序 8 个 | 0.01818 s |
| 顺序 576 外推 | 1.3086 s |

全并行约比顺序外推快 307 倍，比 chunk=64 快 3.7 倍。两模型 5 图 smoke 峰值显存分别：

- LLaVA：15.30 GiB；
- InternVL：18.08 GiB。

都低于 22 GiB 验收线。正式路径会按估算显存选择全并行或 chunk，OOM 时回退；chunk 不改变公式。

### 为什么是许多 `.pt` shard，而不是一个 `features.pkl`

`.pt` 不是数学要求，`pickle` shard 也能保存相同张量。这里分片的原因是每个唯一目标位置保存 `[L,M]` 的 old/new P、E 和诊断量，LLaVA 单个旧根 `features.pkl` 已约 5.8 GB。一个大 pkl 会在合并、反序列化、重写时产生很高的 CPU 内存峰值，而且中断后往往需要整文件重来。每 50 张图一个原子 `.pt`：

- 可以流式训练/分析；
- 中断只损失当前 shard；
- 两 GPU 各自写文件，不争同一个 pickle；
- 避免再次合并一个数 GB 根文件。

所以关键是“分片和张量原生存储”，不是 `.pt` 后缀本身。如果更喜欢 pkl，可以改成 pkl shards，结果不会因此变化。

## 14. PASS/FAIL 检查表

| 检查项 | LLaVA | InternVL | 证据/说明 |
|---|---|---|---|
| 视觉 attention decomposition | PASS | PASS | 独立 30 案例；两模型 smoke |
| visual token 区间 | PASS | PASS | `[5,581)`/576；IMG_CONTEXT/256 |
| 首 subtoken 前一因果行 | PASS | PASS | wrapper 共用定位函数，真实样例 |
| causal mask/scaling/softmax | PASS | PASS | 显式 future mask、模型 scaling、FP32 softmax |
| 正式 KV-cache 语义 | N/A | N/A | 正式 `use_cache=False` |
| cache on/off 等价控制 | PASS | 未独立做 | LLaVA logits 精确一致 |
| (f=FFN(Norm(z))) | PASS | PASS | 两类适配分支 |
| FFN residual skip 未混入 | PASS | PASS | `ffn_from_residual` 只返回 FFN 输出 |
| JVP tangent 是 (a_j) | PASS | PASS | 代码审计、smoke、FD |
| per-token JVP 可加性 | PASS | PASS-smoke | LLaVA 独立 FP32 + 两模型 smoke |
| central FD 局部匹配 | PASS | PASS-smoke | LLaVA epsilon sweep；两模型 median cosine=1 |
| full/chunk 一致 | PASS | PASS | 误差 (<2.5\times10^{-6}) |
| accidental detach | PASS | PASS | fresh-primal forward AD |
| output bias/head layout/GQA | PASS | PASS | 真实模型 + synthetic tests |
| P 有限且逐层和为 1 | PASS | PASS | max sum error (2.38\times10^{-7}) |
| random matched-norm control | PASS | 未做 | LLaVA 32 directions/case |
| text/nonvisual direction control | 部分 | 未做 | LLaVA 有 causal nonvisual direction，不等价于 text-only 模型 |
| image corruption control | PASS | 未做 | LLaVA 黑图/shuffle/无关图 |
| 真正 visual-mask/text-only | 未做 | 未做 | 会改变输入分布，需另设实验 |
| end-to-end causal logit intervention | 未做 | 未做 | 当前只验证本层 FFN finite effect |
| 审计字段真实记录 component error | **FAIL** | **FAIL** | 当前硬编码 0，不影响 P |
| 空间/目标有效性 | PASS | PASS | COCO box、same-image target |
| 幻觉检测优于旧 P | **FAIL** | **FAIL/无差异** | 预注册 bootstrap |

## 15. 更深层的数学风险

1. **局部性。** Jacobian 只在当前 (z) 附近有效；完整移除视觉写入时误差明显增大。
2. **表示纠缠。** 当前 (z) 已包含前面层和前面 token 携带的视觉信息。减掉当前层 (a^{visual}) 不等于“模型没看图”。
3. **attention write 不等于语义视觉证据。** (a_j) 是当前层通过 attention 搬运的 residual 写入，可能含背景、格式、位置或语言相关成分。
4. **模长不等于信息量。** (‖J a_j‖) 大只表示 hidden vector 改变大，未说明目标 logit 变得更正确。
5. **JVP 不是因果结论。** 它是局部导数；需要 activation intervention/patching 验证对 logit 和输出的真实影响。
6. **坐标依赖。** L2 norm 依赖模型 hidden basis/scale，跨模型、跨层不能裸比较。
7. **相消。** P 使用每个响应的非负模长，丢掉方向和负贡献；实测 cancellation ratio median 只有 0.557。
8. **attention/input magnitude 主导。** (‖a_j‖) 与 (E_j) Pearson 0.951，JFFN 排名很大部分仍是 retrieval strength。
9. **定位与真实性不同。** 模型能把 hallucinated query 对准某个区域，不代表该对象真的存在。

## 16. 建议的下一轮实验

按优先级：

1. **把总量诊断加入正式模型。** 预注册 `old risk + I/R/S/D`，先做 (R) 和 (I+R)；当前辅助结果强烈提示它们比归一化 P 更有用。
2. **把 magnitude 与 sensitivity 分开。** 同时保存/训练 attention mass、(I_j=‖a_j‖)、(G_j=E_j/(I_j+ε))，对小 (I_j) 使用阈值或 empirical-Bayes shrink，避免爆炸。
3. **改用目标 logit 对齐的标量 Jacobian。** 例如
   \[
   A_j=\nabla_z \operatorname{logit}(y_t)^\top a_j
   \]
   或对正确/竞争 token 的 logit margin 求 signed attribution。它比 full-hidden L2 更贴近幻觉问题。
4. **保留 signed/cancellation 信息。** 尝试 (C_j)、正负 mass、cancellation ratio，而不是只用非负 (E_j)。signed 值不应直接冒充概率分布。
5. **逐层 random-normalized sensitivity。** 使用 (S/S_{random,median}) 或逐层 z-score，控制纯 Jacobian layer scale。
6. **做真正的因果 intervention。** 在若干层对选中 visual-token write 做 10%/25%/50% patch/ablation，测目标 logit、margin 和生成概率；与 JVP 预测的 signed logit change 比较。
7. **扩大 image corruption 与 counterfactual。** 对同对象存在/不存在的配对图片、bbox 内外 patch replacement 做验证，比黑图更接近对象因果证据。
8. **不要把 raw JFFN-P 单独替换旧 P 作为当前主方法。** 它可以作为空间解释图或与旧 risk 的补充特征，但两模型现有结果不支持“它是更好的 hallucination P”。

## 17. 验证与可复现文件

### 汇总

- 机器可读总表：[jacobian_visual_ffn_validation_summary.json](jacobian_visual_ffn_validation_summary.json)
- 扁平 CSV：[jacobian_visual_ffn_validation_summary.csv](jacobian_visual_ffn_validation_summary.csv)
- 两模型总图：[jacobian_visual_ffn_validation_summary.png](jacobian_visual_ffn_validation_summary.png)

### LLaVA 独立深度验证

目录：[results/jacobian_visual_ffn_validation](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jacobian_visual_ffn_validation)

- 总 metrics：`jacobian_visual_ffn_validation_metrics.json`
- 30 个 layer-target 案例：`jacobian_visual_ffn_cases.csv`
- 17280 个视觉 token 诊断：`jacobian_visual_ffn_token_diagnostics.csv`
- epsilon sweep：`jacobian_visual_ffn_epsilon_sweep.csv`
- nonlinear removal curve：`jacobian_visual_ffn_removal_curve.csv`
- prefix/cache：`jacobian_visual_ffn_prefix_cache.csv`
- image controls：`jacobian_visual_ffn_image_controls.csv`
- timing：`jacobian_visual_ffn_efficiency.csv`
- 诊断图：`jacobian_visual_ffn_validation_diagnostics.png`

### 两模型正式结果

- LLaVA：[llava_1_5_7b_jffn_p_comparison_report.md](outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_p_comparison/llava_1_5_7b_jffn_p_comparison_report.md)
- InternVL：[internvl_2_5_8b_jffn_p_comparison_report.md](outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_p_comparison/internvl_2_5_8b_jffn_p_comparison_report.md)
- 每个目录还包含 `training_results.json`、`p_analysis.json`、三个 P 分析 CSV、六张 P 分析图和 5 图 `smoke_5images/extraction_audit.json`。

### 本轮校验状态

- `python -m unittest tests.test_visual_ffn_jacobian tests.test_jffn_experiment tests.test_visual_support_ranges`
- 结果：17 tests，全部通过。
- 核心文件和两个新分析脚本均通过 `py_compile`。
- 项目按约定没有 `.git`，因此没有 commit/diff 可引用；没有修改原 JFFN 核心公式，只新增独立审计、汇总脚本和本报告。

## 18. What should I send to ChatGPT?

如果要把这项工作交给另一个 ChatGPT/研究者复核，不要发送 80 多个大 `.pt` shard。最小充分材料是：

1. 本报告 `jacobian_visual_ffn_validation_report.md`；
2. `jacobian_visual_ffn_validation_summary.json` 和 `.csv`；
3. 核心代码：
   - `features/visual_ffn_jacobian.py`；
   - `models/dgst_capture.py`；
   - `models/llava_wrapper.py`；
   - `models/internvl_wrapper.py`；
4. 独立验证原始表：cases、epsilon sweep、removal curve、token diagnostics、prefix/cache、image controls、efficiency；
5. 两模型 `training_results.json`、`p_analysis.json` 和正式 Markdown report；
6. 两个 `smoke_5images/extraction_audit.json`；
7. 当前配置中 `jffn_p_comparison` 段、模型路径/版本、PyTorch/Transformers 版本与 GPU 型号；
8. 若需要逐样本复算，只附一个 LLaVA shard 和一个 InternVL shard，不必发送全量 shard。

最重要的数值应直接写在消息里：PASS/FAIL 表、attention reconstruction error、JVP additivity、finite-difference epsilon 曲线、full-removal error、random-direction ratio、attention/input-vs-E 相关、cancellation ratio、空间验证、预注册 AUROC 差值和 bootstrap CI。这样复核者能区分“实现正确”“空间解释有效”和“幻觉检测没有提升”这三个不同结论。
