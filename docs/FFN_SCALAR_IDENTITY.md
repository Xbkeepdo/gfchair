# visual-write directions 上的 FFN 是否接近标量恒等映射

沿用 `docs/FFN_INPUT_GEOMETRY.md` 的四模型共享500图、全部目标和层、原400/100
图片划分及局部FP32 K4路径积分。用户已明确要求子空间检验也使用完整500图。
结果根：`outputs/ffn_scalar_identity_20260912/`。

## 1. 逐token检验与共享标量拟合

令 `G=FFN(Norm)`，`z0=z-sum a_m`，`Jhat=sum_k w_k J_G(z0+alpha_k sum a_m)`。
每个目标、每一层有自己的固定Jhat，所有该目标的visual writes共享这条路径。
它不含residual identity，也不是单点Jacobian。

已有cosine与范数足够计算

\[
C_m=\cos(a_m,e_m),\quad
g_m=C_m\frac{\|e_m\|}{\|a_m\|},\quad
R_m=\sqrt{1-C_m^2}.
\]

所以三个量不相互独立；R是cosine的确定函数。本次重用已保存的逐源cosine
和同一K4定义的v2范数，既有两条实现的范数相对误差最大约2.16e-6。
零向量的cosine/R记为未定义；非零a对应零e时g=0。没有把零分母填成有效cosine。
这里的g是有符号平行投影增益，区别于上一轮非负的范数比 `||e||/||a||`。

保留负g：若Jhat=cI且c<0，C应接近−1而不是+1；R仍应接近0。
同时g_m接近常数本身也不够，必须检查剩余的正交分量。

对同一个目标层的全部writes拟合共享标量（c允许正或负）：

\[
c_A=\frac{\sum_m a_m^\top e_m}{\sum_m\|a_m\|^2},\qquad
\epsilon_A=\frac{\|E-c_AA\|_F}{\|E\|_F}.
\]

这是按原WRITE能量加权的最佳拟合，不是g_m的token等权均值。残差满足

\[
\epsilon_A^2=
\frac{\sum_m\|e_m-g_ma_m\|^2}{\sum_m\|e_m\|^2}
+\frac{\sum_m\|a_m\|^2(g_m-c_A)^2}{\sum_m\|e_m\|^2}.
\]

两项分别量化正交变换和增益不一致，避免把旋转全部误记为增益变化。
先按目标层对视觉token统计，再按REAL/HALL的mentions等权汇总，最后跨层等权。
另保留原train/test与同图配对；gain std先在目标层内部计算，不把层间变化混入。

## 2. 500图逐token结果（完成，6496 mentions）

表内均为REAL / HALL的跨层类内均值：

| 模型 | 平均C | parallel gain g | token内g标准差 | 正交比例R | 共享c拟合残差epsilon_A |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 | −.266319 / −.266558 | −.175378 / −.174854 | .046857 / .048244 | .944489 / .944736 | .957634 / .957442 |
| LLaVA | −.201412 / −.204931 | −.117408 / −.120908 | .044724 / .043870 | .961037 / .960409 | .969987 / .969737 |
| Qwen3 | −.224211 / −.225117 | −.152654 / −.151146 | .050494 / .051030 | .955810 / .955223 | .966576 / .964601 |
| InternVL | −.207447 / −.205750 | −.128577 / −.126737 | .040336 / .040958 | .965866 / .966468 | .974680 / .974602 |

实际视觉writes上的scalar identity拟合很差，REAL/HALL都如此。即使每个token
使用各自最佳g_m，合并后的归一化残差仍约 .956–.973；共享c的失败主要来自
正交分量，而不只是不同token的g不一致。这里数值是**范数比例**，不是能量比例；
不能把平均epsilon=.96直接写成“96%能量残差”。

曲线的后段常见C和g接近0、R接近1。这表示平行分量趋小、正交分量占比高，
不能解释为FFN作用消失。即使把c取0，非零的e仍全部成为scalar拟合残差。

[三指标图](../outputs/ffn_scalar_identity_20260912/tokens500/all_metrics.png) ·
[共享标量拟合](../outputs/ffn_scalar_identity_20260912/tokens500/all_scalar_fit.png) ·
[汇总CSV](../outputs/ffn_scalar_identity_20260912/tokens500/summary.csv) ·
[逐层CSV](../outputs/ffn_scalar_identity_20260912/tokens500/curves.csv)

## 3. 完整数值子空间检验

对每个目标层，取非零a_m并仅为求基底做列归一化（不中心化；理论span不变）。
FP64计算Gram矩阵、特征分解，保留所有 `sigma>1e-5*sigma_max` 的方向，再用
薄QR正交化。没有预设Top32、能量99%截断或随机子空间。保存1e-4/1e-5/1e-6
三个阈值的秩计数、全部输入奇异值、实际WRITE重建误差。基底是针对已捕获
有限精度writes的完整**数值**子空间，不声称恢复阈值以下的精确实数秩。

Gram会平方条件数，因此必须使用FP64，且smoke在固定首/中/末层与直接FP64
SVD的秩和span对照；每个目标层检查Q正交性。阈值以下方向不参与主算子检验，
秩与重建误差将随正式结果披露。

保持原始path不变，将JVP探测方向换成Q的列：

\[
Y=\hat JQ,\quad B=Q^\top Y,\quad L=Y-QB.
\]

`B`一般只是投影后的压缩矩阵。除非L=0，不能把B单独当成完整限制算子。
完整子空间作用由Y表示。最佳共享标量及主检验为

\[
c_Q=\operatorname{tr}(B)/r,\quad
\epsilon_Q=\|Y-c_QQ\|_F/\|Y\|_F,\quad
\ell=\|L\|_F/\|Y\|_F.
\]

c_Q对正交基方向等权，与原WRITE能量加权的c_A一般不同。另报告
`||B-c_Q I||/||B||`、非对角Frobenius比例及以下恒等式的误差：

\[
\|Y-c_QQ\|_F^2=\|B-c_QI\|_F^2+\|Y-QB\|_F^2.
\]

正式用FP64的B^T B及Y^T Y特征值平方根计算奇异值谱，smoke与直接FP64 SVD(B)
比较。保存逐个奇异值，报告CV和P90/P10，避免仅凭
不稳定的最小奇异值或条件数下结论。

非对角项依赖基底；cI拟合残差和奇异值不依赖正交基的选择。奇异值相等只能
说明等比例伸缩，不能排除旋转。纯旋转可以有大非对角项，因此方向混合和
增益各向异性分开报告；完整Y的奇异值用于检验后者。
例如统一缩放乘一个正交旋转也会保持P_WRITE/P_FFN的范数分布，却不一定是cI。
因此归一化来源分布相近不能单独推出scalar identity。

## 4. 完整500图子空间结果（完成）

四模型各500图、全部目标和层完成：共2,000个image-runs、6,371个唯一目标、
206,980个target-layer；按原提及表统计为6,496 mentions。每个目标、每层均
单独拟合最优c_Q，允许负值；以下仍为 **REAL / HALL** 的跨层类内均值。

| 模型 | 完整误差 epsilon_Q | 投影后误差 ||B-cI||/||B|| | 子空间外比例 ell |
|---|---:|---:|---:|
| Qwen2.5 | .936482 / .936819 | .765639 / .766809 | .859473 / .859570 |
| LLaVA | .959864 / .958857 | .864093 / .862235 | .859856 / .859093 |
| Qwen3 | .951351 / .950788 | .778285 / .778289 | .888971 / .887707 |
| InternVL | .965951 / .966461 | .814072 / .816003 | .912167 / .912634 |

**Jhat在visual-write子空间上不接近cI。** 即使逐目标、逐层重新选择最佳
正/负c，完整相对残差仍约 .936–.966。所有target-layer中最小的完整残差
分别为 .7224/.8581/.7510/.8490，因此这个结论不是跨层平均掩盖少量近似cI
情况造成的。原始writes加权的epsilon_A与正交基等权的epsilon_Q数值不同，
但二者均不支持scalar identity。

子空间外分量很大，说明只查看B会漏掉大量响应；同时投影后的B也明显偏离cI。
ell是对正交基探测响应Y定义的Frobenius范数比例，不是直接对原始e_m做token
等权平均的比例。个别层的ell比跨层均值低，例如Qwen3末层；这些层仍有很大的
子空间内混合，不能把所有非cI现象都归为子空间外响应。

### 奇异值是否近似相等

P90/P10指奇异值**数值**的90%分位除以10%分位，均先在目标层内计算。

| 模型 | B的P90/P10 | 完整Y的P90/P10 | 当前Q基底下B非对角比例 |
|---|---:|---:|---:|
| Qwen2.5 | 9.963 / 9.921 | 2.780 / 2.775 | .7620 / .7632 |
| LLaVA | 11.664 / 11.516 | 2.812 / 2.798 | .8615 / .8597 |
| Qwen3 | 10.603 / 10.712 | 2.551 / 2.584 | .7731 / .7729 |
| InternVL | 9.367 / 9.382 | 2.150 / 2.160 | .8095 / .8113 |

谱并不平坦。完整Y的P90/P10平均约2.15–2.81、奇异值CV约.374–.474，支持
方向依赖的增益变化；因此也不是单纯的统一缩放乘一个等距旋转。B的谱更宽，
但B还受到投影几何的影响，不能把B的9–12倍直接当成完整响应的增益跨度。
非对角项大说明当前正交基下的方向混合；真正的增益各向异性依据完整Y的谱。

### REAL/HALL之间的差异

完整epsilon_Q的H−R依次为 `+.000337/−.001007/−.000563/+.000510`，
没有统一的“HALL更偏离scalar identity”。同图配对的差值也只有
`+.000549/−.000279/−.000973/+.000832`。两个Qwen的全量与100图test
非配对差值还会改变符号，不把这些小差异包装成稳定的幻觉机制。

完整Y的P90/P10 H−R为 `−.00510/−.01414/+.03364/+.00940`，同图配对为
`−.00518/−.01374/+.02405/+.00888`，方向仍因模型而异。非cI、方向变化和
子空间外响应首先是REAL/HALL共有的层结构；它们本身不足以解释幻觉发生。
没有进行新的检测器训练、显著性检验或因果干预。

### 秩与数值核对

| 模型 | 唯一targets | target-layers | 实际r范围 | 旧e范数最大相对误差 |
|---|---:|---:|---:|---:|
| Qwen2.5 | 1115 | 31220 | 63–529 | 5.93e-7 |
| LLaVA | 1889 | 60448 | 574–576 | 9.69e-7 |
| Qwen3 | 1892 | 68112 | 80–400 | 2.62e-6 |
| InternVL | 1475 | 47200 | 256 | 6.50e-7 |

全部case在主阈值1e-5与更宽松的1e-6下秩相同。Qwen2/Qwen3/InternVL在
1e-4下也相同；LLaVA有3,619/60,448个case在更严格的1e-4下会少保留方向，
正式计算使用较宽的1e-5，没有采用1e-4结果。主阈值仅LLaVA有3个case的秩
小于非零列数；原始WRITE投影重建误差在四模型全部case最大仅2.20e-15。

七项定向测试通过，包括负scalar、纯旋转、不同token不同增益、B=cI但有
leakage、纯B=0外泄、秩亏基底、固定原积分路径的基底JVP与奇偶续跑覆盖。
四模型所有层的真实smoke包含520个target-layer：FP64 Gram谱与直接FP64
SVD最大相对差7.56e-12，基底JVP与直接JVP最大相对差1.03e-6；首/中/末层
的直接SVD基底与Gram基底span差最大1.15e-13。

正式全量检查通过：原500图集合、mention/目标ID/标签/全部层完全对齐，
全部谱长度等于r、非负且降序，Q最大正交误差1.60e-14；由保存谱能量独立
重建epsilon与leakage的恒等式最大误差1.78e-15。未出现零投影响应case。

### 产物及运行记录

- [完整子空间：cI误差、子空间外响应、Y谱](../outputs/ffn_scalar_identity_20260912/subspace500/all_subspace.png)
- [投影B：cI误差、非对角、B谱](../outputs/ffn_scalar_identity_20260912/subspace500/all_projected.png)
- [汇总CSV](../outputs/ffn_scalar_identity_20260912/subspace500/summary.csv)
- [逐层REAL/HALL统计](../outputs/ffn_scalar_identity_20260912/subspace500/curves.csv)
- [同图配对统计](../outputs/ffn_scalar_identity_20260912/subspace500/paired_images.csv)

各模型目录保存`metrics.npz`、`mentions.json`、`spectra.csv`、`audit.json`；
每个`extraction/image_*.pt`保存全部目标层的逐个B/Y奇异值、输入谱、统计和
检查结果。两张子空间主图及逐token主图均已查看。B矩阵仅在smoke保存FP32
示例，直接FP64 SVD检查在量化保存前的B上运行；未声称FP32示例矩阵能复现
1e-12级谱审计误差。

首次真实smoke在Qwen2/Qwen3的FP32 SVD上失败：相对FP64谱误差分别
4.115e-5/2.932e-5，超过既定1e-5；原日志保留为`smoke_gpu{0,1}.log`。
未启动当时的全量，没有放宽阈值。改为FP64 Gram谱并继续对照直接FP64 SVD，
新smoke输出独立保存到`smoke_fp64_spectrum`，新日志含`smoke_fp64_gpu`。

谱的均值/分位数后来改在CPU NumPy上统计，1040组smoke谱与原GPU统计最大
差1.78e-15；矩阵和谱精度未变。原两路screen完成前三模型后，LLaVA保留320图，
剩余180图按原ID奇偶分为95/85图双GPU续跑，互斥覆盖，不改变cohort。
所有提取进程已结束，远程两GPU显存均已释放；最终--stage summarize退出0。
最新运行日志为`outputs/ffn_scalar_subspace_cpu_stats_gpu{0,1}.log`、
`outputs/ffn_scalar_subspace_llava_{even,odd}.log`及
`outputs/ffn_scalar_subspace_summary.log`。计划续跑的SIGINT与前述数值失败分开记录。

## 5. 重现与边界

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_scalar_identity tests.test_ffn_input_geometry
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_scalar_identity.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python scripts/run_ffn_scalar_subspace.py --stage smoke --models qwen2_5_vl_7b llava_1_5_7b --device cuda:0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python scripts/run_ffn_scalar_subspace.py --stage extract --models qwen2_5_vl_7b llava_1_5_7b --device cuda:0
# GPU1同理运行Qwen3/InternVL，全部完成后：
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python scripts/run_ffn_scalar_subspace.py --stage summarize
```

本轮检验当前视觉路径定义下的Jhat，不检验整个decoder的I+Jhat，不据此
删除路径积分或更改检测器。原K4求积尾部、有限精度基底和REAL/HALL的词汇/
生成位置混杂保持披露；描述性差异不作为因果证明。
Jhat包含Norm与FFN的链式导数，尚未把两者的作用单独分离；否定cI也不等价于
证明必须做路径积分，单点Jacobian能否替代积分需要另一项直接比较。
