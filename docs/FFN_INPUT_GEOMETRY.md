# FFN 如何改变视觉 attention WRITE：amplification、rotation 与 cancellation

本实验回答 REAL/HALL 条件下的 FFN 几何变化。结果根目录为
`outputs/ffn_input_geometry_20260912/`。完整 4000 图缓存分析与共享 500 图
rotation 补提取分别报告，不能混称为同一规模的完整三指标实验。

## 1. 定义和研究问题

沿用本项目 v2 的局部 FP32、Gauss–Legendre K4 路径定义。
对一个目标词的预测位置、一层 decoder，m 遍历全部视觉 token，
不是 FFN 神经元。`a_m` 是该视觉 token 在干净 attention 下经 value/output
投影写入残差流的向量；实际 FFN 输入是 `z=h_prev+o_attn`，不是单独的 `a_m`。

令 `G(z)=FFN(Norm(z))`，`A=sum_m a_m`，`z0=z-A`，则

\[
e_m=\int_0^1 J_G(z_0+\alpha A)a_m\,d\alpha,
\qquad \sum_m e_m\approx G(z)-G(z_0).
\]

积分包含 Norm 的导数，FFN 分支不含 residual identity。这里的 `e_m`
也不同于早期 JFFN 实验的单点 `J_G(z)a_m`，更不同于后来的完整来源分解 `c_m`。
实现来源：`features/ffn_visual_path_attribution.py::streaming_vector_path_statistics`。

路径积分和端点差的闭合关系，可对照
[Integrated Gradients 原论文第3节](https://arxiv.org/html/1703.01365v2#S3)。
本项目保留 FFN 的向量响应并按视觉 WRITE 分解；下文的 REAL/HALL 数字来自
本次项目实验，不是该论文中的结论。

逐视觉 token 计算

\[
g_m=\frac{\|e_m\|}{\|a_m\|},\qquad
r_m=\cos(a_m,e_m).
\]

`g_m>1` 为 FFN 分支响应范数放大，`g_m<1` 为缩小。`r_m` 越接近 1
越同向，接近 0 为近似正交，负值表示夹角大于 90 度；cosine 的下降才对应
更大的夹角。由于 G 是一般非线性映射，“rotation”是夹角变化描述，不声称
FFN 是保持长度的旋转矩阵。零分母记为未定义并计数，不加 epsilon 制造值。

每个目标、每层另外计算

\[
\kappa_E=\frac{\|\sum_m e_m\|}{\sum_m\|e_m\|}.
\]

`kappa_E` 越低，视觉来源之间的抵消越强；`1-kappa_E` 才是随抵消增强而增大的量。
分子必须来自实际 `component_sum`。K4 存在求积误差时，不能直接用
`||G(z)-G(z0)||` 代替该分子；本次缓存使用 v2 `N_vec/kappa_vec`。
500 图补提取另计算输入 `kappa_A=||sum a_m||/sum||a_m||` 及
`kappa_E-kappa_A`，判断 FFN 后的抵消是否比进入 FFN 前更强。

`g_m` 不是整个 block 的范数增益。若考察固定其他来源时的 `a_m+e_m`，则

\[
\frac{\|a_m+e_m\|}{\|a_m\|}
=\sqrt{1+g_m^2+2g_m r_m}.
\]

因此只凭 `g_m<1`，不能说整个 block 抑制了该来源。

## 2. 为什么比只比较 P_FFN/P_WRITE 更完整

令 `S_A=sum||a_m||`、`S_E=sum||e_m||`，对非零来源有

\[
\bar g_A=S_E/S_A=\sum_m P_{WRITE,m}g_m,
\qquad
P_{FFN,m}=P_{WRITE,m}\,g_m/\bar g_A.
\]

所以 WRITE/FFN 的归一化空间分布，只保留不同来源之间的**相对增益差异**。
对所有来源施加相同缩放不会改变分布；来源向量的方向与相互抵消也不能从
两张范数分布图确定。原 `R_amp=TV(P_FFN,P_WRITE)` 是分布重分配程度，
不是本次的逐来源 amplification。原 endpoint cosine 是 `cos(e_m,端点差)`，
也不能代替本次的 `cos(a_m,e_m)`。

## 3. 统计口径

- 四模型：Qwen2.5-VL-7B、LLaVA-1.5-7B、Qwen3-VL-8B、InternVL2.5-8B。
- 完整缓存每模型原 4000 图、原始 mentions，保留原 3200/800 图片划分。
  数据取自 `old_validation_fp32_k4.json` 指向的 v2 分片，不混入原生精度 v1。
- 先在每个目标、每层对全部视觉 token 的 `g_m` 求均值、中位数和 `g_m>1`
  比例，再按 REAL/HALL 对 mentions 等权汇总。另列 `S_E/S_A`，它是
  WRITE 范数加权的 gain，不能冒充 token 等权平均。
- 逐层 CSV 保留均值、中位数、四分位数、有效数与未定义数，all/train/test 分开。
  跨层数字均为层统计量的等权平均，不是拿所有来源混在一起重新求比值。
- 同图配对对照：只取同时存在 REAL 与 HALL 的图片，先分别在该图内平均，
  再计算 HALL−REAL 并跨图片等权汇总；标签冲突的相同 target 排除。
  此对照控制图片身份，没有完全控制目标词、生成位置和前缀长度。
- 500 图复用 `outputs/ffn_output_cosine_20260909/cohort500.json`，四模型共享，
  按旧 train/test 分别固定 400/100 图，选择不依赖标签或本次指标。
  新 cosine 保存到逐图分片，避免以后只能从均值反推。

## 4. 完整 4000 图结果

四模型缓存分析完成，共 50,812 mentions。下面都是先按目标计算、再对
mentions 和层取均值，数字按 **REAL / HALL** 排列：

| 模型 | REAL / HALL mentions | token 等权平均 gain | gain>1 比例 | WRITE范数加权 gain | cancellation |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 | 7801 / 916 | .615562 / .615509 | 4.178% / 3.755% | .645471 / .642315 | .512675 / .514489 |
| LLaVA | 11976 / 3487 | .532513 / .528180 | 5.402% / 5.462% | .529319 / .517112 | .503085 / .487463 |
| Qwen3 | 12280 / 2593 | .627794 / .620347 | 5.154% / 5.112% | .660920 / .648388 | .530033 / .533808 |
| InternVL | 9993 / 1766 | .552597 / .555263 | 4.708% / 4.847% | .567292 / .564832 | .500325 / .499776 |

主要观察：

1. **没有跨模型统一的“幻觉增强 FFN 放大”。** Qwen2 的 token 等权平均 gain
   几乎相同，LLaVA/Qwen3 的 HALL 略低，InternVL 的 HALL 略高。
   四模型的 WRITE范数加权 gain 则都是 HALL 略低。InternVL 两种权重给出
   不同方向，不应选择性只报告一种。
2. **没有跨模型统一的“幻觉增强抵消”。** LLaVA 的 HALL cancellation
   较低（差值 −.015622），说明抵消较强；Qwen2/Qwen3 的 HALL 略高，
   InternVL 几乎相同。按 HALL 均值高于 REAL 的层数，分别为
   13/28、6/32、25/36、13/32。
3. **层的差异比类别总体差异更突出。** 图中首层、部分末层的 gain 较高，
   大部分中间层低于 1；整体只有约 4–5% 的来源在 token/mention/layer 等权
   平均口径下 gain>1。此比例不是把所有不同分辨率来源混合后的频率。
   不能用跨层均值代替“每一层、每个来源均衰减”。

同图片配对控制后的跨层平均 HALL−REAL：

| 模型 | token 等权 gain 差值 | 加权 gain 差值 | cancellation 差值 |
|---|---:|---:|---:|
| Qwen2.5 | +.001292 | −.001113 | +.004066 |
| LLaVA | −.003327 | −.010147 | −.011505 |
| Qwen3 | −.008457 | −.013491 | +.003530 |
| InternVL | +.002100 | −.003150 | +.001034 |

LLaVA 的较强抵消在同图比较和原 800 图 test 中方向一致；Qwen3 的 gain
降低也保留。Qwen2 的微小等权 gain 差异、InternVL 的微小 cancellation
差异会随统计口径改变方向，不能描述成稳定规律。

原 test 的 token 等权 gain H−R 依次为 −.000053、−.004241、−.008605、
+.003193；cancellation H−R 为 +.000469、−.015814、+.000848、−.002163。
这是原划分的描述性分组，未重新调参，也不是未接触过的独立确认集。

绘图与数据：

- [4000图全部样本曲线](../outputs/ffn_input_geometry_20260912/all_geometry.png)
- [原800图test曲线](../outputs/ffn_input_geometry_20260912/test_geometry.png)
- [逐层数据](../outputs/ffn_input_geometry_20260912/curves.csv)
- [同图配对数据](../outputs/ffn_input_geometry_20260912/paired_images.csv)
- [汇总差值](../outputs/ffn_input_geometry_20260912/summary.csv)

图中先对一个目标的视觉来源求均值，再对 mentions 汇总；实线是后一层
汇总的均值，虚线是其跨 mentions 中位数，**不是**来源 token 的中位数。
token 内中位数作为独立 `amp_median` 保留在 CSV。

数据检查：只有 LLaVA 存在 174 个 `a_m=e_m=0` 的来源层条目，其他模型为0；
未发现 `a_m=0,e_m>0` 或目标层总 e 范数为0。标签冲突唯一 target 数依次
0/22/2/7，同图控制已排除。原 K4 闭合相对误差最大值依次 .010616/.000639/
.001986/.003653；只有 Qwen2 有2个 mention-layer 大于1%，全部保留并披露，
没有按结果删异常。指标范围、分片完整性、目标唯一性、原划分和 saved
`kappa_vec` 复算检查均通过。

## 5. 共享 500 图的 rotation 与三指标配对

四模型各500图全部完成，共6,496 mentions；原400/100图片划分保持。
以下表内仍为 **REAL / HALL**，使用同一批目标和同一条K4路径：

| 模型 | REAL / HALL mentions | token等权平均 gain | 平均 cos(a_m,e_m) | cancellation |
|---|---:|---:|---:|---:|
| Qwen2.5 | 1006 / 122 | .615301 / .616322 | −.266319 / −.266558 | .513951 / .513551 |
| LLaVA | 1503 / 450 | .533441 / .528215 | −.201412 / −.204931 | .503099 / .485058 |
| Qwen3 | 1560 / 357 | .627856 / .618649 | −.224211 / −.225117 | .530773 / .535874 |
| InternVL | 1278 / 220 | .552282 / .555048 | −.207447 / −.205750 | .501048 / .499227 |

**Rotation 的主体是两个类别共有的层结构。** 平均cosine都为负，说明
FFN响应相对于输入WRITE有反向分量，而非保持输入方向；中间层常更负，
后段很多层接近0。接近0表示近似正交，不能解释为“没有旋转”或“恢复同向”。
四模型平均cosine的H−R依次为 −.000239/−.003518/−.000906/+.001697，
没有跨模型一致的HALL旋转增强。500图中的100张原test图上，Qwen2/Qwen3/
InternVL的这一微小差异方向会翻转；LLaVA仍负。因此不宜把总体cosine
差异包装成稳定的幻觉机制。

### 必须同时看输入中已经存在的抵消

| 模型 | 输入 kappa_A：REAL / HALL | FFN引起的比率变化 kappa_E−kappa_A：REAL / HALL |
|---|---:|---:|
| Qwen2.5 | .462219 / .469034 | +.051731 / +.044518 |
| LLaVA | .460378 / .442700 | +.042721 / +.042359 |
| Qwen3 | .478627 / .487716 | +.052146 / +.048158 |
| InternVL | .441616 / .440924 | +.059432 / +.058303 |

四模型、两个类别的**跨层平均** `kappa_E−kappa_A` 均为正，表示FFN之后
来源之间的抵消平均减弱；并非每一层都如此，例如LLaVA的一些早层为负。
这与 `cos(a_m,e_m)<0` 不矛盾：一个量比较同一来源在变换前后的方向，
另一个量比较不同来源在同一侧的相互抵消。

**LLaVA最重要的解释修正：** HALL输出cancellation低了 `.018041`，但输入
已经低了 `.017679`；FFN前后变化的类别差异仅 `−.000362`。同图配对时，
输入H−R为 `−.012516`，输出为 `−.012067`，FFN变化差为 `+.000449`。
因此，LLaVA的HALL在输出中抵消更强，不能直接归结为FFN额外制造了这些抵消；
数值差距几乎已存在于输入WRITE。

Qwen2/Qwen3的HALL `kappa_E−kappa_A` 则分别比REAL低 `.007213/.003988`，
即FFN在HALL上减少抵消的幅度较小。该方向在同图控制中仍为
`−.007166/−.004030`，在100图test中仍为 `−.006778/−.003683`。
这是本轮较有针对性的条件差异，但仍是小样本描述，不作显著性/因果声明。
InternVL的变化差较小，同图后约 `−.000084`，不支持相同强度的推广。

三项指标合起来，支持的表述是：**FFN的尺度变化、方向变化和来源间抵消
具有清楚的层结构，REAL/HALL大体共享这些结构；类别差异依赖模型和层。
输出几何的差异中，有的已由输入WRITE携带，有的体现为FFN改变抵消程度
的幅度不同。** 这比把所有现象概括成“幻觉被FFN放大”更符合当前结果。

产物：

- [同一500图的三指标逐层图](../outputs/ffn_input_geometry_20260912/cohort500/all_geometry.png)
- [输入/输出抵消及差值图](../outputs/ffn_input_geometry_20260912/cohort500/all_cancellation_change.png)
- [100图test三指标图](../outputs/ffn_input_geometry_20260912/cohort500/test_geometry.png)
- [500图逐层CSV](../outputs/ffn_input_geometry_20260912/cohort500/curves.csv)
- [500图汇总CSV](../outputs/ffn_input_geometry_20260912/cohort500/summary.csv)
- [500图同图片配对CSV](../outputs/ffn_input_geometry_20260912/cohort500/paired_images.csv)

四模型每张图的cosine位于 `cohort500/<model>/extraction/image_*.pt`，
模型级别的 `metrics.npz/mentions.json/audit.json` 保存汇总、身份与检查。
三张500图主图以及4000图all主图均已实际查看。

### 实现与数值复核

三项手算/零向量/非线性积分测试通过，四模型全层真实smoke通过。

优化只改变同一个四节点积分的计算顺序：RMSNorm 在路径点 x 的导数为
`w*r*a - w*x*r^3*mean(x*a)`，其中 `r=(mean(x^2)+eps)^(-1/2)`；
FFN 两条 gated 分支的系数先积分，Norm 导数的第二项作为四个低秩修正保留。
没有把 Norm 的缩放固定在端点，也没有丢弃 gate/up 的任何乘积法则项。
四模型所有层的真实 smoke 与原 direct JVP 向量对照最大相对误差依次
`7.58e-7 / 9.10e-7 / 9.76e-7 / 7.55e-7`；这是实现复现检查，非提高积分阶数。

正式500图全部目标的WRITE范数与旧缓存完全一致；新e范数的逐目标层相对
误差最大值依次为 `9.57e-7 / 1.11e-6 / 2.15e-6 / 7.45e-7`。
所有cosine都在[-1,1]内；LLaVA有18个零向量来源的cosine记为未定义，其他模型
为0，不填0。逐源cosine可重建保存均值；全部500图、mention/目标ID/标签/原划分
与4000图缓存一致。两个远程screen队列正常结束，GPU已释放；汇总命令退出0。

## 6. 解释边界

本次为描述性机制分析，不训练新检测器，不做显著性或因果效应声明。
路径固定干净 attention 下的 WRITE，考察当前 FFN 分支的条件响应；它不等同于
删除图像区域后重新运行 attention 和整个模型。因此类别差异本身不能证明
FFN 导致了幻觉，也未完全排除词汇、生成位置和前缀长度的影响。
原 K4 数值尾部限制保留：即使真实向量和保证 cancellation 在 [0,1] 内，
也不代表积分已在所有样本收敛。新提取需要与原 WRITE/e 范数复现核对；
核对通过只说明同一定义得到复现，不是对 K4 充分收敛的额外证明。

## 7. 运行入口

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_input_geometry
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_input_geometry.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/extract_ffn_input_rotation.py --models qwen2_5_vl_7b --device cuda:0 --smoke
# 每个模型通过 smoke 后，去掉 --smoke 执行固定500图；四模型齐全后：
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_ffn_input_rotation.py --summarize
```
