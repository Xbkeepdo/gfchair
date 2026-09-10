# FFN 完整来源分解验证实验

## 这次做了什么

本实验在 Qwen2.5-VL-7B、LLaVA-1.5-7B、Qwen3-VL-8B、InternVL2.5-8B 的原 COCO4000 图片上，增加一个 Kobayashi-style 完整来源分解，与原来的视觉条件路径 `e_m` 比较。目标定位、3200/800 图片划分和全部 mentions 沿用已有实验。

旧对照具体来自 v2 production_k4 的 local-FP32 K4 特征；不混用 v1 原生精度的检测成绩。数值统计按唯一目标×层计算，检测与类别曲线保留全部 mentions。

这是验证性实验，不是原论文的完全复现。没有新图像确认、超参数搜索、bootstrap、SHA256 验证或独立产物审计。Qwen2 的两个旧 K4 案例及原 `e_m` 保持原样。生成文本和已有 `e_m` 都复用，仅新分解需要重新前向。

计算代码是 `features/ffn_source_composition.py`，入口为 `scripts/run_ffn_source_composition.py`；attention 重建仅增加一个默认关闭的 `return_all_sources` 参数。结果统一在 `outputs/ffn_source_composition_v1/`。

## a_m、e_m、n_m、c_m 分别是什么

固定当前 decoder 层和正在预测的目标位置 t：

- `a_m`：视觉位置 m 向 t 写入的 attention 向量，已经包含各头注意力系数、value 和输出投影。保留原 concat 写法。
- `e_m`：恢复本层视觉写入的过程中，分配给 m 的 Norm+FFN 输出变化。
- `n_m`：固定真实输入的 Norm 缩放系数后，视觉来源 m 在 FFN 输入空间的分量。
- `c_m`：所有来源沿共同路径构成 FFN 输出时，分配给 m 的输出贡献向量。

旧方法定义 `G(z)=FFN(Norm(z))`、`A=sum_visual a_m`：

\[
e_m=\int_0^1J_G(z-A+\alpha A)a_m\,d\alpha,
\qquad \sum_m e_m\approx G(z)-G(z-A).
\]

这里 `G(z-A)` 是其余上下文的条件背景，并不是纯语言或完全无视觉状态。

新方法先将来源分成全部 token 的 attention write、目标位置自身 residual skip、attention 输出偏置：

\[
z\approx\sum_{j\in\mathrm{tokens}}a_j+h_{\mathrm{prev},t}+b_O.
\]

value bias 已包含在真实 value 中，不再重复分配。自身 attention 来源仍在 token 集合里，与 residual skip 分开。非视觉来源按原序列位置保存，包含特殊 token；因果 query 之后的位置注意力为零。

在实际 z 上固定 Norm 缩放系数，得到每个来源的 n_j。令 `n=Norm(z)`：

\[
c_j=\int_0^1J_{\mathrm{FFN}}(\alpha n)n_j\,d\alpha,
\qquad
\mathrm{FFN}(n)\approx\mathrm{FFN}(0)+\sum_jc_j.
\]

FFN没有另行计算原生跨token attention；P_C是由归因向量范数构造的图。`c_m` 是一个 hidden-size 维向量，不是概率。只有把视觉 `||c_m||` 在视觉内部归一化，才得到 `P_C`。新旧方法的交互分摊规则不同，`S_C` 和 `S_E` 不应当相等，也不能只凭谁大判断谁更正确。

“全部来源”指当前层输入位置的来源。深层文本向量和 residual 本身可能包含此前的视觉信息，因此本分解不是对原始图像像素的全部路径归因。

例如 `F(x)=x²`，背景1、视觉2。旧条件路径分解为背景输出1、视觉效应8；共同零点路径分解为背景贡献3、视觉贡献6。两者都加回9，但视觉份额不同。

## 四模型的 Norm 和 FFN 适配

检查的是本机实际运行的文本 decoder 模块，而不是视觉编码器 Norm 或 Q/K Norm。

| 模型 | FFN 前模块 | 本地 Norm 类 | 当前 epsilon |
|---|---|---|---:|
| Qwen2.5 | post_attention_layernorm | Qwen2RMSNorm | 1e-6 |
| Qwen3 | post_attention_layernorm | Qwen3VLTextRMSNorm | 1e-6 |
| LLaVA | post_attention_layernorm | LlamaRMSNorm | 1e-5 |
| InternVL | ffn_norm | InternLM2RMSNorm | 1e-5 |

四个本地实现都是相同形式的 RMSNorm，但模块类型、权重和 epsilon 取自各模型实际层：

\[
n_j=\gamma\odot s_j\,/\sqrt{\mathrm{mean}(z^2)+\epsilon}.
\]

这里固定真实 z 的分母；这个操作不是 Norm Jacobian，也不是对单个来源分别做 RMSNorm。Attention 前 Norm 仍调用模型原生模块。

Norm 和 FFN 在当前层局部提升到 FP32，前缀、attention 捕获及 WRITE 仍保持原精度；不是整模型 FP32 重跑。源向量求和与实际 residual 加法存在的舍入差异直接记录，不制造额外来源补齐输出。

FFN 包含完整 SwiGLU：`down(act(gate(n))*up(n))`。直接 JVP 是数值参考。正式提取把积分中的两项乘积法则系数预先求和，再对每个来源执行 gate/up/down 线性投影：

\[
c_j=W_d\{\bar A\odot(W_g n_j)+\bar B\odot(W_u n_j)\},
\]

其中 `Abar=sum_k w_k act'(gate(alpha_k*n))*up(alpha_k*n)`，`Bbar=sum_k w_k act(gate(alpha_k*n))`。模型自身的激活函数提供导数，偏置参与真实路径点；方向投影不再添加偏置。节点通过 `alpha*(W*n)+bias` 广播，一次批量计算全部节点激活与导数；来源方向继续分块矩阵运算，两张GPU各跑独立模型。这样只重排同一个完整 FFN JVP 的线性运算，不更改积分节点或归因规则。Qwen/LLaVA 使用 gate_proj/up_proj/down_proj，InternVL 使用 w1/w3/w2。

## 保存了哪些信号

| 字段 | 含义 |
|---|---|
| raw_attention_mean | `[层,视觉token]`，目标 query 的跨头平均注意力；FP32，视觉范围内不再归一化 |
| raw_attention_mass | 每层 raw attention 在视觉 token 上的总和 |
| ae_token_weight | 既有目标证据逐 token 加权量，按保存的 `T*AE` 重建 |
| AE | 视觉内部归一化 attention 对目标语义 gate 的加权均值 |
| p_write / I | `||a_m||` 的视觉分布 / 模长总和 |
| p_ffn / S | 原 `||e_m||` 的视觉分布 / 模长总和，即 P_E / S_E |
| p_n | 固定 Norm 系数后的视觉来源分布 P_N |
| c_mag / p_c / S_C | `||c_m||` / 归一化视觉分布 P_C / 模长总和 |
| N_C | `||sum_visual c_m||`，视觉贡献的净强度 |
| kappa_C | `N_C/S_C`，视觉贡献向量一致程度；较小表示方向分散或相消较强 |
| other_token_mag | 非视觉 token 逐来源贡献范数，按原序列排除视觉范围后的顺序 |
| residual_mag | 目标 residual skip 分配到 FFN 输出的贡献范数 |
| attention_bias_mag | attention 输出偏置来源分配到 FFN 输出的贡献范数 |
| ffn_zero_norm | FFN(0) 的范数，作为单独零输入输出项 |
| gross_all / visual_fraction | 全部逐来源范数总和（含零输入输出项）/ S_C占该总和的份额 |
| gain_E / gain_C（曲线） | S_E/I、S_C/I，两种规则下的相对响应倍率 |
| closure_relative / closure_absolute | 来源贡献加 FFN(0) 与真实局部 FP32 FFN 输出的相对/绝对差 |
| quadrature_relative | 直接把完整 n 作为方向积分的闭合误差，用于观察求积本身的不足 |
| source_output_relative | 来源和与完整 n 方向积分的输出差，用于区分来源重建误差的影响 |

Raw attention 的均值与既有某些代码中的跨头求和不同。原始向量不乘 value 或语义 gate。只在 `raw_AE` 分布比较时临时对 raw attention 做视觉内部归一化，不覆盖原始向量。热图的 raw attention 和 AE_token 保留原始量纲，各自色条显示范围。

现有 AE 的精确口径尤其需要区分：设 `R=sum_visual raw_attention_mean`，`A_vis=raw_attention_mean/R`，则 `u=A_vis*gate`、`AE=sum_visual u`、`T=u/AE`。原 `features/dgst_t.py` 在跨头均值后已调用视觉范围的 `_renormalize`，然后才乘 gate。因此 AE 并不包含 R；它是条件于视觉来源的 gate 加权均值。不能用原始 raw attention 与 u 的数值比值判断 gate 是否放大注意力，因为两者之间还隔着除以 R。新保存的 raw attention 和 R 保留了这一原始视觉质量；形状比较则用归一化 raw attention 与 T。

非视觉 gross 必须逐来源取范数再求和，不能先合并向量再取范数。`visual_fraction` 是此分解下的范数份额，不是严格信息占比。任何来源图都不是物体存在概率。

差图包含 `P_E-P_W`、`P_C-P_W`、`P_C-P_N`。最后一项针对固定 Norm 来源分解后的 FFN 变化。负值表示相对份额下降，并不表示对目标 logit 有负贡献。正负份额在内部证据 T 的 Top-32 区域分别累加，T 是模型内部代理，不是真实框标注。

曲线包含 JS、OT、Spearman、Top-32 overlap、TV；使用所有 mentions（包括原 train/test），按 REAL/HALL 给出中位数和 IQR，CSV 另存均值。IQR 不是置信区间。固定热图按 image_id/response_index 选首个 REAL/HALL 案例，不挑选最好看的结果。

零视觉总强度时分布未定义，距离计算使用均匀分布占位并统计数量，展示时注明未定义。diff 不作为 OT 输入。

零强度下的 kappa 或倍率数值也只作计算占位，以 degenerate 标记区分，不能解释成强烈相消。

## 检测特征（原9组，追加F_C后共10组）

令 `L(x)=log1p(x)`，公共底座 `B=[AE,L(I),L(S_E)]`。

| 组名 | 输入 |
|---|---|
| AE_I | AE + L(I) |
| B | AE + L(I) + L(S_E) |
| AE_I_SC | AE + L(I) + L(S_C) |
| B_SC | B + L(S_C) |
| B_JS_TE | B + JS(T,P_E) |
| B_JS_TC | B + JS(T,P_C) |
| B_OT_TE | B + OT(T,P_E) |
| B_OT_TC | B + OT(T,P_C) |
| F | **AE + L(S_E)**，近期定义，不含 kappa |
| F_C（追加） | **AE + L(S_C)**，仅视觉贡献强度，不含I |

`B-AE_I` 检查已有 FFN 强度在 WRITE 强度之外的增量；`AE_I_SC-B` 比较两种强度；`B_SC-B` 检查新强度的额外作用。新旧热图在相同 B 底座上分别比较，避免把强度增益误写成热图增益。`B-F` 检查在近期 F 上再加 WRITE 强度的影响。

这里的增量只针对固定底座和训练协议，不证明与全部 attention 信息独立。

## 追加方法：QE / QC 的 softmax(温度0.2) 与各自F融合

用户确认先计算每层JS，再拼接AE+S，不把变长视觉softmax向量直接送入MLP。

\[
d_E=G(z)-G(z-A),\quad Q^E_m=\langle e_m,d_E/\|d_E\|\rangle,
\]
\[
d_C=\mathrm{FFN}(n)-\mathrm{FFN}(0),\quad Q^C_m=\langle c_m,d_C/\|d_C\|\rangle.
\]

这里QC沿**整体FFN端点差**方向投影，不使用视觉净向量sum_visual c_m的方向。Q保留正负号，是投影长度，不除以分量范数变成cosine；正负号表示对该FFN响应方向的支持/抵消，不是目标logit贡献的正负。

两版分别构造 `P_E^Q=softmax(QE/0.2)`、`P_C^Q=softmax(QC/0.2)`，在全视觉support上计算自然对数JS(P,T)，再使用：

- `F_E_Q02=[AE,log1p(S_E),JS(P_E^Q,T)]`，对照已有F。
- `F_C_Q02=[AE,log1p(S_C),JS(P_C^Q,T)]`，对照已有F_C。

应分别看两组相对各自底座的增量。跨组比较F_C_Q02与F_E_Q02时，S和Q同时变化，因此该差值不能单独归因于Q的定义。

QE直接复用v2 K4的path_signed_q；QC使用K50及原生前缀/局部FP32。QC通过同一个积分线性算子的VJP计算：`Q_m=<n_m,Jbar_FFN^T unit(d_C)>`，无需再次形成全部高维c_m。四模型的首/中/末层投影与已保存c_m的直接内积对照，最大相对误差约6.34e-7。零参考方向记为退化并令Q=0（softmax为均匀分布）。

本轮仅重训两组×3seeds×4模型=24个头，F/F_C直接复用。原4000图、3200/800划分、全部mentions和MLP保持，不做SHA、bootstrap或新图像确认。QC需要补一次前向，QE及两版强度不重算。

入口 `scripts/run_ffn_qe_qc_tau02.py`：`--stage pipeline --models <模型> --device cuda:<设备>` 自动串联QC提取、JS/矩阵和训练，四模型完成后 `--stage summarize`。输出在 `outputs/ffn_source_composition_v1/q_softmax_tau02/`，与原10组分开。每图保存原始QC、两版JS、Pmax和退化标记；原始QE仍在旧v2源文件。softmax/JS使用FP64，检测输入FP32；不裁剪Q或改为绝对值。

JS 是全视觉 support 上的自然对数 JS。OT 复用既有双方 Top-32 union、h_prev 来源状态及 sqrt-cosine cost、精确 EMD 求解。raw attention 本轮不另增检测头。

原composition及F_C对照共120头，QE/QC融合另加24头。所有组使用同一套原 MLP[128,64,32]、BatchNorm、dropout.3、Adam(lr=.001,weight_decay=1e-5)、batch256、最多100epochs、train-loss早停/学习率调度/minimum-train-loss checkpoint。无额外标准化、类别权重或重采样。报告逐seed及概率集成 AUROC/HALL-AUPR、固定0.5和训练REAL-F1双阈值。

## 数值检查与来源干预

合成测试检查偏置、来源加法、完整 SwiGLU JVP、零来源和九组特征。四模型各选原训练集按图片ID排序的前8图，全部目标和层比较直接 JVP K4/K16，并在首个有目标图片上核对投影复用版本与直接 JVP。

最初按K4/K16选择了K16；用户随后明确上限K50，因此在同8图上追加K16/K32/K50对照。四模型共同选择闭合最大误差≤1%、S_C相对K50差异P90≤1%的最小K；如果无共同通过者，使用上限K50并明确报告误差。全量不做自适应修复。GL节点和权重直接使用已安装NumPy的 `numpy.polynomial.legendre.leggauss`，从[-1,1]变换到[0,1]，无需新包。K50仍不是真值，新路径不能沿用旧e_m的收敛结论。

fixed-QK 复用每模型旧100张图片及目标清单，层为 Qwen2[7,14,21,28]、Qwen3[9,18,27,36]、LLaVA/InternVL[8,16,24,32]。每层划分8区域，按 WRITE、conditional、composition、固定随机四种策略各选一个区域。在原生捕获上减去所选区域的 attention write，观察目标 log-probability 变化。策略选同一区域时复用前向。

主要比较图片内四层平均的绝对 log-probability 变化，再报告图片间配对差均值、中位数和胜率；同时保存有符号变化。较大绝对效应表示对预测有影响，不自动意味着正确物体证据。没有重跑 pixel 或 activation patching，也没有 bootstrap。

删除 attention write 会同时影响 residual 路径和后续 FFN，因此该干预不是纯 FFN 分支的因果效应；本轮只用它比较区域排序，不要求共同路径的 c_m 等于单区域删除效应。

## 如何运行与查看

在项目根目录使用现有 vicr Python；建议 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

```bash
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage audit --models qwen2_5_vl_7b llava_1_5_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage audit --models qwen3_vl_8b internvl_2_5_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage select-k
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage audit50 --models qwen2_5_vl_7b llava_1_5_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage audit50 --models qwen3_vl_8b internvl_2_5_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage select-k50
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage verify-fast --models qwen2_5_vl_7b llava_1_5_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage verify-fast --models qwen3_vl_8b internvl_2_5_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage pipeline --models qwen2_5_vl_7b llava_1_5_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage pipeline --models qwen3_vl_8b internvl_2_5_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_source_composition.py --stage summarize
```

`pipeline` 顺序执行提取、分析、训练、fixed-QK及少量展示向量。也可单独使用 `extract/analyze/train/intervene/examples`。重复命令按已存在的逐图文件和训练result跳过；如主动改变公式或参数，请使用新的结果目录，脚本不会替你验证旧文件来源。

每模型目录包含 `shards/k<K>`（选中的节点数）、`progress.json`、`analysis.json`、`curves.csv/png/pdf`、固定案例、`matrices.pt`、`heads/`、`detection.json`、`fixed_qk.csv`、`fixed_qk_summary.json`。全局 `numerics.json` 记录小样本选择；`summary.md` 和 `detection.csv` 在四模型完成后汇总结果。

`example_vectors_k50.pt` 保存各模型首个有目标训练图片的第一个目标，在首/中/末三个层的 sources、n_j、c_j、真实z/n/FFN输出及零输入输出；只保留这个目标的高维向量。sources/c 的前面部分按原序列位置排列，最后两行依次为 residual skip 和 attention 输出偏置。视觉来源用 position.visual_range 切片。全量shard只保留紧凑统计和raw attention，不保存所有高维c_j。

读取raw attention示例：

```python
import torch
row = torch.load("outputs/ffn_source_composition_v1/qwen2_5_vl_7b/shards/k50/image_000000000283.pt",
                 map_location="cpu", weights_only=False)["positions"][0]
raw = row["raw_attention_mean"]  # [28, visual_tokens]，未在视觉内部归一化
visual_mass, ae = row["raw_attention_mass"], row["AE"]
ae_token = row["ae_token_weight"]
```

## 实际结果（2026-09-10，已完成）

四模型各4000图全部完成，保留原3200/800划分及全部50,812个mentions。原阶段共108个检测头（后续F_C追加12头）、400图fixed-QK（1600个图像×层case），以及四份K50高维展示样本。两路pipeline正常退出；没有追加新图像、bootstrap、分类器搜索或SHA验证。

### 数值结果

本轮统一K50。下表的总闭合包含来源重建误差；纯积分列单独检查完整n方向的积分与实际FFN端点。数值统计按唯一目标×层计算。

| 模型 | 闭合中位数 | P90 | 最大值 | >1%条目 / 总条目 | 纯积分最大相对误差 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 | 0.1359% | 0.5403% | 9.5132% | 7847/242312 | 1.05e-06 |
| LLaVA | 0.0128% | 0.0165% | 0.0545% | 0/478432 | 2.38e-07 |
| Qwen3 | 0.1209% | 0.2366% | 11.6194% | 21295/529344 | 1.78e-06 |
| InternVL | 0.0968% | 0.1261% | 0.3384% | 0/372160 | 2.41e-07 |

K50的纯积分闭合误差在四模型都很小；两种Qwen的完整来源重建仍有尾部误差，不能称为所有样本精确闭合。这里没有用额外bias或数值来源把残差补回去。LLaVA和InternVL全量总闭合均未超过1%。四模型均无零视觉总强度条目。这些误差不能解释成特殊语义机制，也不能据此认定检测差异全部由数值误差造成。

### 检测结果

每格为三seed概率集成 **AUROC / HALL-AUPR**。F明确使用AE+log1p(S_E)，距离组全部含底座B，不是距离单独检测。

| 组 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| AE_I | 0.870320 / 0.444710 | 0.902676 / 0.718400 | 0.886481 / 0.632260 | 0.863149 / 0.549897 |
| B | 0.881034 / 0.460093 | 0.906132 / 0.717329 | 0.896747 / 0.664920 | 0.870476 / 0.537155 |
| AE_I_SC | 0.874516 / 0.453304 | 0.907223 / 0.720366 | 0.897139 / 0.673012 | 0.867782 / 0.543936 |
| B_SC | 0.881013 / 0.450900 | 0.904614 / 0.718221 | 0.900019 / 0.669176 | 0.869596 / 0.555812 |
| B_JS_TE | 0.875093 / 0.449509 | 0.904857 / 0.718454 | 0.904018 / 0.685049 | 0.877491 / 0.586110 |
| B_JS_TC | 0.882667 / 0.460138 | 0.906856 / 0.726737 | 0.890310 / 0.645600 | 0.873851 / 0.550485 |
| B_OT_TE | 0.879743 / 0.447930 | 0.905254 / 0.723580 | 0.899828 / 0.674214 | 0.870790 / 0.550492 |
| B_OT_TC | 0.872666 / 0.447456 | 0.907956 / 0.723416 | 0.881986 / 0.630541 | 0.867541 / 0.557991 |
| F | 0.882138 / 0.457740 | 0.904960 / 0.719276 | 0.896510 / 0.649342 | 0.863812 / 0.546154 |

主要AUROC差值如下，单位为百分点；逐seed及双阈值指标另见完整CSV。

| 比较 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| 增加旧S_E | +1.071 | +0.346 | +1.027 | +0.733 |
| 用S_C替换S_E | -0.652 | +0.109 | +0.039 | -0.269 |
| 在B上增加S_C | -0.002 | -0.152 | +0.327 | -0.088 |
| 新图替换旧图：JS | +0.757 | +0.200 | -1.371 | -0.364 |
| 新图替换旧图：OT | -0.708 | +0.270 | -1.784 | -0.325 |
| 在F上增加I | -0.110 | +0.117 | +0.024 | +0.666 |

- **旧响应强度仍值得保留。** B相对AE_I的集成AUROC四模型都提高，但逐seed并非全部为正，LLaVA/InternVL的集成HALL-AUPR反而下降。本次采用log1p口径，不能照搬旧raw强度报告中的“12个配对seed全正”。
- **S_C没有统一替代S_E或提供AUROC增量。** 替换时LLaVA/Qwen3略升、Qwen2/InternVL下降；额外加入S_C后仅Qwen3的集成AUROC上升。不过额外加入S_C的HALL-AUPR在LLaVA、Qwen3、InternVL提高，指标间存在取舍。
- **新热图的检测效果依模型和距离而变。** JS匹配对照在Qwen2/LLaVA更好，在Qwen3/InternVL更差；OT匹配对照仅LLaVA的AUROC更好，且其HALL-AUPR略降。Qwen3两种新图距离的三个配对seed AUROC均低于旧图。因此当前结果不支持把composition图全面替换原conditional图。
- **相对B的增量也要单独看。** 新图JS在Qwen2/LLaVA/InternVL的集成AUROC小幅增加、Qwen3下降；新图OT仅LLaVA的集成AUROC增加。以上是同一旧测试集的探索性点估计，不声明统计显著或跨数据集泛化。

### fixed-QK与区域排序

下面是图像内四层先平均后的绝对log-probability变化之差；正值表示composition选择的区域产生更大的绝对变化。

| 模型 | C−WRITE | C−conditional | C−random | C与conditional同区 | C与WRITE同区 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 | -0.000033 | -0.000285 | +0.026851 | 97.25% | 94.00% |
| LLaVA | +0.000372 | +0.000043 | +0.010313 | 97.50% | 82.00% |
| Qwen3 | +0.000304 | -0.000131 | +0.009877 | 98.75% | 82.00% |
| InternVL | -0.000034 | -0.000032 | +0.011099 | 99.75% | 94.50% |

composition相对随机区域的平均绝对效应四模型均更大，但相对WRITE/conditional的差异很小且方向不统一，配对差中位数均为0。尤其在本次8区域划分下，composition和conditional有97.25%–99.75%的case选中同一区域。因此胜率很低通常伴随大量平局，不能直接读成几乎总是更差；该粗区域实验也不能证明逐token热图完全相同。当前没有普遍优于WRITE或conditional的区域排序证据。

### 已执行的检查与产物

- 11项定向测试通过：`OMP_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_source_composition tests.test_visual_ffn_jacobian -q`。
- 四模型真实direct-JVP与向量化版本对照通过；已保存检查结果。K50展示向量的视觉范数与正式紧凑特征最大相对差约1.78e-7。没有做全量checkpoint独立审计。
- [完整结果总表](../outputs/ffn_source_composition_v1/summary.md)、[检测CSV](../outputs/ffn_source_composition_v1/detection.csv)、[逐seed双阈值指标](../outputs/ffn_source_composition_v1/seed_metrics.csv)、[配对检测增量](../outputs/ffn_source_composition_v1/paired_detection.csv)。
- [数值统计](../outputs/ffn_source_composition_v1/numerical_summary.csv)、[fixed-QK配对结果](../outputs/ffn_source_composition_v1/fixed_qk_summary.csv)、[区域选择一致率](../outputs/ffn_source_composition_v1/region_agreement.csv)。
- 每模型目录的`curves.csv/png/pdf`保存强度、raw attention、分布变化及证据区份额变化；`example_label0/1`为固定REAL/HALL图；`example_vectors_k50.pt`保存少量真实高维示例。

本轮完成了完整来源分解这一解释视角的验证，但没有发现它在检测和区域定位上普遍优于旧路径。保留已有S_E，并把c_m作为另一种来源归因规则的对照，是当前结果支持的使用方式。

## 追加：AE + 视觉S_C（2026-09-10，完成）

用户追加要求查看不含I的AE+S(c_m)。沿用近期F的尺度处理，新增 **F_C=[AE,log1p(S_C)]**，其中S_C=sum_visual ||c_m||；不含I、kappa、文本/residual/bias贡献，也不是||sum_visual c_m||。对照F=[AE,log1p(S_E)]，两者输入维度相同。

复用K50紧凑特征和原3200/800划分、全部mentions、原MLP及seeds43/44/45，只训练12个新头。原108头直接复用；没有VLM前向、重算路径或干预、SHA检查或bootstrap。代码build_groups现含10组；本次从缓存AE_I_SC的首/末两个块直接构造F_C，未重新计算log1p。

每格为三seed概率集成AUROC / HALL-AUPR（%）；差值单位为百分点。

| 模型 | 原F：AE+L(S_E) | F_C：AE+L(S_C) | AUROC差 | HALL-AUPR差 |
|---|---:|---:|---:|---:|
| Qwen2.5 | 88.214 / 45.774 | 88.151 / 47.162 | -0.063 | +1.388 |
| LLaVA | 90.496 / 71.928 | 90.482 / 72.008 | -0.014 | +0.081 |
| Qwen3 | 89.651 / 64.934 | 89.422 / 64.920 | -0.229 | -0.014 |
| InternVL | 86.381 / 54.615 | 85.520 / 52.621 | -0.861 | -1.994 |

F_C的集成AUROC四模型均低于F，其中Qwen2/LLaVA差距很小，InternVL下降较多。Qwen2的HALL-AUPR提高约1.388个百分点，LLaVA小幅提高；Qwen3基本持平，InternVL下降约1.994个百分点。逐seed AUROC在前三模型方向混合，InternVL三个seed均下降。此对照仍不支持用S_C统一替换S_E；它是旧测试集上的探索性点估计。

3项定向测试通过，检查F/F_C公式与特征块。两路训练正常退出，新增12个checkpoint及逐seed/双阈值/集成指标已保存。当前总计10组、120个头。[F_C直接对照CSV](../outputs/ffn_source_composition_v1/fc_comparison.csv)，全量结果见更新后的summary.md、detection.csv、seed_metrics.csv和paired_detection.csv。

## 追加结果：QE / QC softmax温度0.2与对应F融合（完成）

四模型原4000图、全部50,812 mentions及3200/800划分。两个新组合各3个seed，共24个新头，原F/F_C对照复用。QE来自原v2 K4，QC使用K50；二者分别投影到各自端点差的单位方向，不除以分量范数，不取绝对值。用户已确认softmax后先算JS，再拼接AE+log1p(S)。

令J_E=JS(softmax(QE/0.2),T)，J_C=JS(softmax(QC/0.2),T)。每格为三seed概率集成 **AUROC / HALL-AUPR（%）**。

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| F_E：AE+L(S_E) | 88.214 / 45.774 | 90.496 / 71.928 | 89.651 / 64.934 | 86.381 / 54.615 |
| F_E+J_E | 87.893 / 46.021 | 90.876 / 72.727 | 88.686 / 64.128 | 85.469 / 52.996 |
| F_C：AE+L(S_C) | 88.151 / 47.162 | 90.482 / 72.008 | 89.422 / 64.920 | 85.520 / 52.621 |
| F_C+J_C | 87.814 / 48.172 | 91.000 / 72.751 | 89.941 / 66.726 | 87.140 / 56.663 |

相对各自底座的增量，单位为百分点：

| 模型 | QE融合 ΔAUROC / ΔHALL-AUPR | QC融合 ΔAUROC / ΔHALL-AUPR |
|---|---:|---:|
| Qwen2.5 | -0.320 / +0.247 | -0.337 / +1.010 |
| LLaVA | +0.380 / +0.799 | +0.518 / +0.743 |
| Qwen3 | -0.965 / -0.806 | +0.519 / +1.806 |
| InternVL | -0.912 / -1.619 | +1.620 / +4.042 |

- **QC融合在LLaVA、Qwen3、InternVL均提高集成AUROC和HALL-AUPR**，其中InternVL分别提高约1.620和4.042个百分点。Qwen2的AUROC下降约0.337点，但HALL-AUPR提高约1.010点。
- **QE融合仅LLaVA两项均提高。** Qwen2的AUROC下降、AP小升；Qwen3和InternVL两项均下降。逐seed AUROC上，QE融合在LLaVA三seed均正，在另外三模型均负。
- QC融合的逐seed AUROC增量在LLaVA和InternVL三seed均正，Qwen3两正一负，Qwen2一正两负。组合F_C+J_C相对F_E+J_E在LLaVA/Qwen3/InternVL的三seed AUROC均更高，但此跨组比较同时改变S与Q，不能单独归因于Q。
- 本次QC融合的表现比此前只用c_m模长构造P_C的距离更有希望，但本轮底座和分布公式不同，不是严格单因素对照；不据此宣称QC是普遍更好的归因或新机制。仍是原测试集上的探索性比较，没有bootstrap或独立图像确认。

两个softmax分布的集中度不同。以下是唯一目标×层条目中，最大单token概率超过0.99的比例：

| 模型 | QE | QC |
|---|---:|---:|
| Qwen2.5 | 10.553% | 0.918% |
| LLaVA | 0.307% | 0.013% |
| Qwen3 | 19.980% | 2.646% |
| InternVL | 4.434% | 0.202% |

QE在温度0.2下更容易形成尖锐分布；这里只是描述性观察，未通过额外控制实验确认其是否导致性能差异。四模型QC参考方向退化条目均为0。K50来源重建的既有数值限制继续保留。

四项定向测试通过，真实QC投影与保存c_m直接内积相对差max6.34e-7；两路pipeline均退出0，无新模型提取或训练失败。没有全量checkpoint重载或SHA审计。新增训练入口为scripts/run_ffn_qe_qc_tau02.py，阶段pipeline/summarize；原base.train只增加可选结果目录以复用训练器。

[本轮完整总表](../outputs/ffn_source_composition_v1/q_softmax_tau02/summary.md)、[检测CSV](../outputs/ffn_source_composition_v1/q_softmax_tau02/detection.csv)、[配对增量](../outputs/ffn_source_composition_v1/q_softmax_tau02/paired.csv)、[逐seed双阈值指标](../outputs/ffn_source_composition_v1/q_softmax_tau02/seed_metrics.csv)。每模型目录另有curves PNG/PDF/CSV、原始QC、两版JS/Pmax及settings。

## 追加：Prompt / generation 贡献曲线（2026-09-10，完成）

用户要求查看两类文本来源曲线。新增 `scripts/plot_ffn_text_sources.py`，从四模型原4000图K50分片的 `other_token_mag` 离线汇总；保留原3200/800划分和全部50,812 mentions，训练集与测试集分别绘图。没有重跑模型、积分或检测器，也没有补提非视觉attention。

设目标的query位置为q，目标在生成文本中的索引为r，则生成前缀起点g=q-r+1。原序列中非视觉的[0,g)位置记为prompt，[g,q+1)记为generation。缓存已删去视觉区间，切分索引为g-(visual_end-visual_start)；generation恰含r个token，包括当前query位置，不含待预测目标及未来。prompt包含system、聊天模板和特殊token，不是纯instruction字符串。各模型prompt token数固定为25/17/14/28。

绘制三种尺度：S_group=sum_group ||c_j||；每token强度S_group/N_group（先逐mention相除再统计）；来源份额S_group/gross_all。gross_all含全部token来源、residual、attention输出偏置和FFN(0)范数；图中只画prompt/generation/visual，因此份额不必相加为1。空generation总量为0、每token均值未定义而排除；此次全部mentions均没有空generation。

实线均值、虚线中位数、阴影IQR（非置信区间），蓝色REAL、橙色HALL。测试集结果如下；训练集对应层数完全一致。

| 模型 | prompt总量REAL>HALL层数 | generation总量HALL>REAL层数 | generation每token强度REAL>HALL层数 | REAL/HALL平均前缀长度 |
|---|---:|---:|---:|---:|
| Qwen2.5 | 28/28 | 28/28 | 28/28 | 33.16 / 58.46 |
| LLaVA | 32/32 | 31/32 | 32/32 | 32.37 / 71.46 |
| Qwen3 | 36/36 | 34/36 | 36/36 | 105.59 / 149.01 |
| InternVL | 32/32 | 32/32 | 32/32 | 72.64 / 118.21 |

Prompt范数份额也在四模型全部层REAL更高。Generation份额在Qwen2/LLaVA/InternVL全部层、Qwen3的35/36层HALL更高。但其每token强度方向反转，且HALL前缀明显更长，不能把总量或份额直接解释成幻觉更依赖语言历史。长度相除是描述性尺度对照，不是长度/位置匹配实验；各分组仍有分布重叠，不推断显著性或因果。贡献按本层来源位置归因，文本内部可能已融合视觉；原K50的来源重建误差限制仍适用。

[完整曲线与说明](../outputs/ffn_source_composition_v1/text_source_curves/summary.md)、[测试总强度](../outputs/ffn_source_composition_v1/text_source_curves/test_gross.png)、[测试每token强度](../outputs/ffn_source_composition_v1/text_source_curves/test_per_token.png)、[测试来源份额](../outputs/ffn_source_composition_v1/text_source_curves/test_fraction.png)。同目录保存train版本、全部PNG/PDF、4608条逐层统计curves.csv与counts.json。

运行命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/plot_ffn_text_sources.py`。合成边界检查（含空generation和非零FFN(0)）、全量未来位置范数为0、分组加residual/bias/FFN(0)范数重建gross_all均通过；全量运行退出0，py_compile和git diff --check通过。未增加哈希或产物审计。

## 追加：4000图合并曲线与两种归一化的区别（2026-09-10，完成）

用户要求直接查看4000图汇总。原脚本增加all范围，重新从各mention数据计算合并均值、中位数和IQR，主图改为all_gross、all_per_token、all_fraction，保留train/test版本。范围为每模型4000图的全部mentions，各mention等权，非图片等权；四模型合计50,812 mentions。曲线不是train/test曲线的简单平均。

对某层某目标，令S_g=sum_{j属于g} ||c_j||，N_g为该组来源token数。每token强度为S_g/N_g，回答组内平均每个来源token贡献多大；来源份额为S_g/gross_all，回答该组占全部来源范数总量多少。gross_all=sum_all ||c_j||+||FFN(0)||，包含residual与attention输出偏置来源。它不是||sum_all c_j||，也不是attention概率。每个mention先计算比值，再对REAL/HALL分别统计。

例如同样gross_all=100，一组S=20、N=10，每token强度2、份额20%；另一组S=40、N=40，每token强度1、份额40%。因此平均每token较弱仍可能有较高整体份额。每token归一化直接除以长度，来源份额不除长度；前者仍不等于控制了所有长度/位置差异。

4000图合并后，四模型prompt总强度和每token强度均在全部层REAL更高；generation总强度HALL更高层数分别28/28、31/32、34/36、32/32，每token强度则全部层REAL更高。合并REAL/HALL平均生成前缀长度分别33.00/58.67、31.97/70.74、101.39/154.09、70.13/116.65。解释边界与上一节相同。

[4000图每token强度](../outputs/ffn_source_composition_v1/text_source_curves/all_per_token.png)、[4000图来源份额](../outputs/ffn_source_composition_v1/text_source_curves/all_fraction.png)、[4000图总强度](../outputs/ffn_source_composition_v1/text_source_curves/all_gross.png)。summary.md默认展示合并曲线；当前9组PNG/PDF、6912条CSV统计，counts.json新增all。运行同一plot_ffn_text_sources.py命令退出0；原边界/总量检查、合并样本数与加权均值核对、py_compile、git diff --check通过。两张合并主图已实际查看。

## 追加：三类来源份额独立检测（2026-09-10，完成）

用户要求训练曲线中的三组来源份额。新增scripts/train_ffn_source_shares.py，复用分组统计collect及原base.train。三组分别为各层sum_prompt ||c_j||/gross_all、sum_generation ||c_j||/gross_all、sum_visual ||c_j||/gross_all；保留原始0–1份额，不取log，不拼AE、强度或互相拼接。Prompt仍包含system/模板/特殊token；generation为已生成因果前缀。每组维度Qwen2/LLaVA/Qwen3/InternVL为28/32/36/32。

每模型原4000图、原3200/800划分与全部mentions不变，曲线合并4000图不改变检测评估范围：检测指标只在原800图holdout计算。复用原MLP隐藏层128/64/32、BN、dropout .3、Adam lr .001、weight_decay 1e-5、batch256、最多100epochs、train-loss调度和早停、最低train-loss checkpoint。三组×三seeds43/44/45×四模型=36头，两路均退出0；不搜索超参数、不重提模型/积分、不加SHA。已有F_E/F_C只引用同源基线。

以下为三seed均值±总体标准差，单位%；HALL-F1采用各seed训练集REAL-F1阈值，非ensemble。

| 模型 | 特征 | AUROC | HALL-AUPR | HALL-F1 |
|---|---|---:|---:|---:|
| Qwen2.5-VL | prompt | 81.395 ± 0.444 | 35.102 ± 0.890 | 23.051 ± 1.574 |
| Qwen2.5-VL | generation | 80.054 ± 0.474 | 29.805 ± 1.029 | 15.107 ± 3.044 |
| Qwen2.5-VL | visual | 83.846 ± 0.299 | 38.037 ± 1.483 | 24.288 ± 4.380 |
| Qwen2.5-VL | F_E | 87.449 ± 0.085 | 43.982 ± 1.678 | 38.401 ± 3.469 |
| Qwen2.5-VL | F_C | 87.182 ± 0.306 | 45.568 ± 0.204 | 42.087 ± 4.250 |
| LLaVA-1.5 | prompt | 88.273 ± 0.086 | 66.457 ± 0.054 | 59.197 ± 1.986 |
| LLaVA-1.5 | generation | 88.054 ± 0.088 | 65.186 ± 0.749 | 58.323 ± 2.859 |
| LLaVA-1.5 | visual | 88.544 ± 0.028 | 68.785 ± 0.156 | 62.970 ± 0.265 |
| LLaVA-1.5 | F_E | 90.015 ± 0.272 | 70.815 ± 0.619 | 66.788 ± 0.073 |
| LLaVA-1.5 | F_C | 90.101 ± 0.081 | 71.224 ± 0.251 | 66.731 ± 0.435 |
| Qwen3-VL | prompt | 85.310 ± 0.271 | 54.436 ± 0.320 | 46.601 ± 1.413 |
| Qwen3-VL | generation | 85.906 ± 0.304 | 54.447 ± 1.360 | 45.837 ± 3.601 |
| Qwen3-VL | visual | 86.190 ± 0.164 | 56.497 ± 0.714 | 49.680 ± 2.230 |
| Qwen3-VL | F_E | 88.655 ± 0.305 | 62.832 ± 0.353 | 58.839 ± 1.088 |
| Qwen3-VL | F_C | 88.607 ± 0.323 | 62.627 ± 0.770 | 56.987 ± 1.453 |
| InternVL-2.5 | prompt | 84.434 ± 0.137 | 49.463 ± 0.580 | 35.788 ± 3.395 |
| InternVL-2.5 | generation | 83.912 ± 0.270 | 45.992 ± 0.196 | 34.900 ± 3.455 |
| InternVL-2.5 | visual | 84.106 ± 0.117 | 48.846 ± 0.207 | 32.084 ± 4.349 |
| InternVL-2.5 | F_E | 85.825 ± 0.587 | 53.034 ± 1.201 | 51.378 ± 0.362 |
| InternVL-2.5 | F_C | 84.895 ± 0.600 | 50.890 ± 0.702 | 48.848 ± 1.210 |

本轮视觉份额在Qwen2.5/LLaVA/Qwen3的三组中平均AUROC与HALL-AUPR最高；InternVL两项均以prompt份额最高。四模型三组单独输入的平均AUROC与HALL-AUPR都低于各自F_E和F_C。因此这些份额有检测信息，但本次结果不支持单独替代AE+S；尚未检验与AE+S拼接后的增益，不声称显著性。

生成前缀长度/位置混杂仍在，检测关联不能证明因果；原K50来源重建误差限制仍适用。原汇总函数的detection.json保留ensemble字段以兼容，本轮主表全部使用逐seed统计。

[完整结果](../outputs/ffn_source_composition_v1/source_share_detection/summary.md)、[均值与std CSV](../outputs/ffn_source_composition_v1/source_share_detection/detection.csv)、[逐seed双阈值](../outputs/ffn_source_composition_v1/source_share_detection/seed_metrics.csv)。每模型另有输入matrices.pt、protocol.json、9个head的model.pt/result.pt及概率，根目录gpu0.log/gpu1.log。

运行：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /opt/conda/private/envs/vicr/bin/python -u scripts/train_ffn_source_shares.py --models qwen2_5_vl_7b llava_1_5_7b --device cuda:0`；另一路为qwen3_vl_8b internvl_2_5_8b及cuda:1。完成后同入口`--summarize`。

复用的合成边界/完整来源gross检查通过；正式图片划分与份额有限性/范围检查通过，四模型各9头共36头完成，汇总退出0；脚本py_compile、git diff --check通过。未做哈希或全量checkpoint独立审计。

## 追加：三类来源份额拼接检测（2026-09-10，完成）

用户要求把prompt、generation、visual三组信号拼接。沿用同一入口scripts/train_ffn_source_shares.py，新组prompt_generation_visual=[prompt全层份额, generation全层份额, visual全层份额]，各模型84/96/108/96维。直接拼接已有FP32矩阵，不重新归一化、不加AE/强度、不重提来源；原三组36头复用，只训练4模型×3seed=12个新头。原4000图/3200-800/全部mentions、MLP优化调度早停与checkpoint选择、seeds43/44/45均不变。

每格为三seed均值±总体标准差，单位%；增量为百分点。最佳单组按同一测试集三seed平均AUROC作描述性比较，前三模型为visual，InternVL为prompt；没有据此选择本轮训练配置。

| 模型 | 拼接AUROC | 拼接HALL-AUPR | 相对最佳单组ΔAUROC | 相对最佳单组ΔAP |
|---|---:|---:|---:|---:|
| qwen2_5_vl_7b | 85.773 ± 0.110 | 42.661 ± 1.120 | +1.927 | +4.624 |
| llava_1_5_7b | 89.607 ± 0.084 | 68.778 ± 0.271 | +1.062 | -0.007 |
| qwen3_vl_8b | 89.404 ± 0.405 | 64.711 ± 1.060 | +3.214 | +8.214 |
| internvl_2_5_8b | 86.474 ± 0.132 | 51.972 ± 1.778 | +2.041 | +2.509 |

拼接结果：四模型平均AUROC均高于最佳单组，三seed的AUROC增量也均为正。HALL-AUPR在Qwen2.5/Qwen3/InternVL提高，LLaVA基本持平（-0.007点）。对照AE+S：Qwen3的AUROC/AP均高于F_E和F_C；InternVL的AUROC高于两者，AP低于F_E但高于F_C；Qwen2.5/LLaVA两项仍低于两种基线。本次不作显著性声明。

这说明三组联合输入在本轮固定训练条件下比单独输入更有效，但未检验与AE+S拼接后的增益；输入从L维变成3L维，没有做维度匹配对照，不据此宣称因果机制或显著性。原长度/位置混杂与K50数值限制继续保留。

[更新总表](../outputs/ffn_source_composition_v1/source_share_detection/summary.md)、[拼接相对单组CSV](../outputs/ffn_source_composition_v1/source_share_detection/concat_comparison.csv)。每模型heads/prompt_generation_visual/seed43、seed44、seed45保存model.pt/result.pt及概率；总计48个来源份额头。原双阈值和逐seed指标在seed_metrics.csv。

命令仍为scripts/train_ffn_source_shares.py --models <模型列表> --device cuda:<0/1>，然后--summarize；两路28576/58235退出0，日志concat_gpu0.log/concat_gpu1.log。矩阵三个block与原单组逐元素一致、维度3L及每模型3个新result检查通过；脚本编译和git diff --check通过，无SHA/重提模型/调参/bootstrap/提交上传。

## 追加：原始S_g与log1p(S_g)五组检测及4000图曲线（2026-09-10，完成）

用户确认采用log(1+S_g)，并同时看原始强度与视觉+prompt相加。S_g=sum_{j属于g} ||c_j||，g为prompt/generation/visual；不除gross_all或token数。visual就是完整分解的S_C，非旧e_m的S_E。Prompt包含system/模板/特殊token，generation仅包括当前目标前已有的生成前缀。

新增scripts/train_ffn_source_strengths.py，复用既有collect/split_norms与base.train。两套尺度各5组：三组单独输入、[prompt全层, generation全层, visual全层]拼接、S_V+S_P逐层相加。log组分别对各单组取log1p后拼接；合计组严格为log(1+S_V+S_P)，不是两个log相加，也不是把c_j向量相加后取范数。单组/合计组L维，拼接3L维，L分别28/32/36/32。

四模型原4000图、原3200/800划分、全部50,812 mentions和原MLP/seeds43/44/45保持，10×4×3=120个新头全部完成；原FE/FC和份额拼接作为已有基线引用。输入无额外缩放/标准化，原128/64/32隐藏层、BN、dropout .3、Adam、batch256、最多100epochs、训练loss调度/早停/最低训练loss checkpoint不变。没有超参数搜索、bootstrap、VLM/积分重提或SHA校验。


raw AUROC，三seed均值±总体std（%）：

| 输入 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| Prompt | 81.539 ± 0.190 | 88.888 ± 0.035 | 84.085 ± 0.184 | 85.129 ± 0.193 |
| Generation | 80.050 ± 0.314 | 87.785 ± 0.069 | 85.771 ± 0.241 | 83.304 ± 0.358 |
| Visual | 84.391 ± 0.301 | 88.421 ± 0.075 | 86.399 ± 0.275 | 83.847 ± 0.119 |
| 三组拼接 | 85.536 ± 0.146 | 90.052 ± 0.112 | 90.202 ± 0.111 | 87.040 ± 0.122 |
| 视觉＋prompt相加 | 83.165 ± 0.282 | 88.917 ± 0.160 | 86.031 ± 0.382 | 84.462 ± 0.253 |

log1p AUROC，三seed均值±总体std（%）：

| 输入 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| Prompt | 82.457 ± 0.084 | 89.055 ± 0.141 | 85.896 ± 0.248 | 85.438 ± 0.279 |
| Generation | 80.975 ± 0.249 | 87.895 ± 0.183 | 85.681 ± 0.801 | 83.746 ± 0.263 |
| Visual | 84.774 ± 0.414 | 88.944 ± 0.095 | 86.906 ± 0.664 | 84.723 ± 0.269 |
| 三组拼接 | 85.617 ± 0.103 | 89.974 ± 0.308 | 90.141 ± 0.323 | 86.787 ± 0.263 |
| 视觉＋prompt相加 | 83.525 ± 0.365 | 89.353 ± 0.158 | 86.312 ± 0.704 | 84.303 ± 0.585 |

raw HALL_AUPR，三seed均值±总体std（%）：

| 输入 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| Prompt | 33.831 ± 1.033 | 70.422 ± 0.293 | 53.125 ± 0.895 | 51.070 ± 0.286 |
| Generation | 32.304 ± 0.972 | 64.061 ± 0.048 | 55.224 ± 0.527 | 46.676 ± 1.371 |
| Visual | 37.307 ± 1.573 | 68.218 ± 0.247 | 55.704 ± 0.640 | 50.505 ± 0.374 |
| 三组拼接 | 42.039 ± 1.280 | 70.899 ± 0.559 | 66.351 ± 0.625 | 57.071 ± 0.630 |
| 视觉＋prompt相加 | 35.174 ± 0.638 | 69.741 ± 0.314 | 56.857 ± 0.874 | 50.090 ± 0.191 |

log1p HALL_AUPR，三seed均值±总体std（%）：

| 输入 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| Prompt | 35.145 ± 1.039 | 70.617 ± 0.390 | 56.117 ± 1.144 | 51.488 ± 1.130 |
| Generation | 33.713 ± 0.983 | 63.604 ± 0.261 | 55.594 ± 2.444 | 47.964 ± 0.271 |
| Visual | 38.485 ± 0.936 | 68.824 ± 0.123 | 56.957 ± 1.625 | 51.843 ± 0.255 |
| 三组拼接 | 42.018 ± 0.309 | 70.339 ± 0.626 | 65.740 ± 0.752 | 55.029 ± 0.971 |
| 视觉＋prompt相加 | 35.152 ± 0.553 | 70.097 ± 0.176 | 58.570 ± 0.838 | 49.607 ± 1.407 |

结果解读：四模型在raw/log1p两套中，concat平均AUROC均高于其余四组。log1p使单独prompt和visual的AUROC/AP在四模型都提高，但concat仅Qwen2的AUROC略升，另外三模型下降；concat的AP均略降或下降。V+P相加没有跨模型一致超过单独visual：raw在LLaVA/InternVL更高、两个Qwen更低；log1p仅LLaVA更高（按AUROC）。
原始concat对照AE+S：Qwen3/InternVL的AUROC/AP均超过F_E/F_C；LLaVA与F_E接近且略低于F_C；Qwen2仍低于两基线。相对份额concat，原始concat在LLaVA/Qwen3/InternVL双指标提高，Qwen2下降。以上均为点估计，不宣称显著性或因果，concat输入维度为单组3倍。

两套4000图曲线均已生成并实际查看，四列为prompt/generation/visual/visual+prompt。每条mention等权，先计算原值或log1p再按REAL/HALL统计；实线均值、虚线中位数、阴影IQR。log1p压缩末层大值，便于查看中间层，但不是按前缀长度归一化，也不能据曲线分离直接推断检测或因果。

在原值与log1p两套中，prompt均为四模型全部层REAL均值更高；generation除少数层外HALL更高。V+P在Qwen2/Qwen3/InternVL两套全部层REAL更高，LLaVA分别27/32、26/32层。已有前缀长度/位置混杂和原K50来源重建误差限制仍适用。

[原始强度曲线](../outputs/ffn_source_composition_v1/source_strength_detection/all_raw.png)、[log1p曲线](../outputs/ffn_source_composition_v1/source_strength_detection/all_log1p.png)、[完整结果与基线](../outputs/ffn_source_composition_v1/source_strength_detection/summary.md)。输出还包括各模型matrices.pt、protocol.json、30个head及概率；根目录detection.csv（52组含12旧基线）、seed_metrics.csv（312条双阈值记录）、comparisons.csv（44条差值）、curves.csv（2048条）及两张PDF。

运行命令：OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /opt/conda/private/envs/vicr/bin/python -u scripts/train_ffn_source_strengths.py --models <模型列表> --device cuda:<0/1>。GPU0顺序Qwen2/LLaVA，GPU1顺序Qwen3/InternVL；两路32984/21519退出0。完成后--summarize退出0；日志gpu0.log/gpu1.log。

合成来源边界、拼接、先加再log和零值log1p检查通过；原缓存全量来源记账/因果未来零、图片划分、特征有限性非负检查通过；40组各3seed完整，脚本编译和diff检查通过。未做哈希/全量checkpoint独立审计、提交或上传。
