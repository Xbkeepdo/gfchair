# All-attention WRITE、SS与方向增益：前三项结果

2026-09-15。**三项全部完成，无新VLM提取。** 四模型各4000图，50812 mentions、1622248个唯一target-layer；72个检测头及每模型2000次图片簇bootstrap完成。

结论：视觉和生成来源的REAL/HALL响应强度差，主要随输入WRITE变化；实际来源方向上的gain并不统一，而且S在I之外仍表现出一定检测增量。“大部分强度差来自WRITE”与“响应有额外预测价值”可以同时成立，不能据此将Jacobian近似为恒等映射。

## 固定口径

- 路径z−A_all→z，真实Norm+FFN导数、局部FP32、GL K32；P/V/G全部attention token共享路径，residual保持在背景中，不使用B1/B2响应。
- I=Σ||a_m||，S（本次SS）=Σ||e_m||，G=S/I，先在一个目标层内计算；不是组净向量范数比。
- 机制主表保留全部mentions；同图配对排除标签冲突target、图片等权。all/train/validation/test单独输出。未完全控制词汇、生成位置和前缀长度。
- 检测固定811图片3200/400/400、seeds43/44/45，指标先按seed计算再均值±总体std，不是ensemble。输入统一log1p和train-only逐列Z-score；单隐藏128/ReLU/dropout.3、无BN、Adam lr.001/wd1e-5/batch128、最多150epoch、早停20、最低val BCE checkpoint；所有模型/特征同一配置，无搜索或AE输入。
- PVG保留三来源各层曲线后拼接，ALL先合并全部attention token。两种来源组织均事先固定。I+S输入维数是I/S的两倍，隐藏宽度相同，参数量并不完全相等；未追加重复I的容量对照。

完整[固定协议](ALL_ATTENTION_WRITE_GAIN_811_PROTOCOL.md)及[全部数值/图表](../outputs/all_attention_write_gain_811_v1/summary.md)。

## P0：SS差异主要来自WRITE还是gain？

利用逐样本logS=logI+logG，将HALL−REAL平均log差分开计算。下表先每层跨mentions，再跨层等权；不是平均范数求比，也不是方差解释率。

| 模型 | 视觉Spearman(I,S) | 视觉Δlog I | 视觉Δlog S | 视觉Δlog G |
|---|---:|---:|---:|---:|
| Qwen2.5 | .96051 | −.22926 | −.23002 | −.00076 |
| LLaVA | .93980 | −.19015 | −.20376 | −.01361 |
| Qwen3 | .96694 | −.27871 | −.29383 | −.01512 |
| InternVL | .97540 | −.35646 | −.34799 | +.00847 |

视觉I/S高度相关，HALL响应下降主要对应WRITE下降；gain类别差小，InternVL方向还相反。同图配对及400图test保留“视觉WRITE差绝对值远大于gain差”的观察；Qwen2的微小gain差会变号。相关为逐层跨mentions相关，不是token空间排序相关。

生成来源ΔlogI为+.24450/+.37167/+.15618/+.26700，ΔlogG为−.01773/−.02669/−.02103/−.00323。较高总响应主要伴随较高总WRITE，gain平均抵消一部分差异；前缀长度仍是解释总量时的混杂因素。

不能推广成所有区域gain都可忽略。Prompt四模型ΔlogG为+.04631/+.05540/+.02740/+.08246，抵消部分负的WRITE差；ALL中Qwen3的ΔlogI=+.01141、ΔlogG=−.01470，合成ΔlogS=−.00329，甚至改变符号。全部区域、split及同图结果见[机制汇总CSV](../outputs/all_attention_write_gain_811_v1/mechanism_summary.csv)。

## P0：WRITE-only、SS-only及I+S检测

PVG分别保留时，测试AUROC/HALL-AUPR（%，三seed均值）：

| 模型 | I-only | S-only | I+S |
|---|---:|---:|---:|
| Qwen2.5 | 85.34 / 43.84 | 86.11 / 42.30 | 87.14 / 46.99 |
| LLaVA | 90.12 / 68.69 | 90.25 / 70.75 | 90.53 / 70.55 |
| Qwen3 | 90.54 / 67.90 | 91.35 / 69.39 | 91.89 / 71.02 |
| InternVL | 87.72 / 57.37 | 88.43 / 59.27 | 88.75 / 60.20 |

S-only相对I-only的AUROC四模型点估计均略高，但四个图片配对95%区间均跨零，Qwen2的AP还下降。没有稳健的“仅S全面优于仅I”结论。

I+S相对I-only更直接检查条件增量，下面为百分点和2000次图片簇bootstrap名义95%区间：

| 模型 | PVG ΔAUROC [95% CI] | PVG ΔHALL-AUPR [95% CI] |
|---|---:|---:|
| Qwen2.5 | +1.81 [+.35,+3.36] | +3.15 [−.58,+7.11] |
| LLaVA | +.41 [−.14,+.93] | +1.86 [+.09,+3.65] |
| Qwen3 | +1.34 [+.67,+2.02] | +3.12 [+.64,+5.61] |
| InternVL | +1.03 [+.24,+1.87] | +2.83 [−.19,+5.69] |

PVG的AUROC增量3/4模型区间为正；AP则LL/Qwen3区间为正。I+S对S-only的PVG两项区间四模型全部跨零，没有同样强度的双向互补证据。

ALL版本I+S对I-only的AUROC增量分别+2.16/+1.76/+4.09/+4.26pp，四个区间均为正。但ALL绝对AUROC均低于对应PVG，合并来源会丢掉来源组织信息，不能只挑较大的增量而略去较弱的基线。

这支持当前固定特征编码和分类器下S对I的一定条件增量，不证明与完整WRITE信息统计独立，也不证明路径积分不可替代。I仅保留组范数和，未保留完整WRITE向量/token分布；融合还增加了输入维数和参数量。旧测试集已被查看，48个名义指标区间未多重比较校正，不作为独立确认性结论。完整std/ALL/区间见[结果总表](../outputs/all_attention_write_gain_811_v1/summary.md)及[检测图](../outputs/all_attention_write_gain_811_v1/detection.png)。

## P1：实际source directions上的anisotropic gain

存在可测的方向依赖范数增益：每个目标、每层、每区域的token内lambda IQR都大于0。下面是先在target-layer内统计、再跨mentions和层平均的视觉结果：

| 模型 | lambda中位数 REAL/HALL | token内IQR REAL/HALL | token内CV REAL/HALL |
|---|---:|---:|---:|
| Qwen2.5 | .72644 / .72873 | .11351 / .11377 | .13016 / .12852 |
| LLaVA | .54764 / .54746 | .11392 / .11298 | .13473 / .13140 |
| Qwen3 | .70292 / .69891 | .14899 / .14853 | .14000 / .13706 |
| InternVL | .58785 / .59531 | .08888 / .09171 | .12412 / .12361 |

增益离散度不为零，但REAL/HALL接近。视觉CV在HALL都略低，IQR类别方向依模型不同；Prompt和generation的CV也无统一的“HALL各向异性增强”。第一层及部分后层gain较高，两类共享明显层结构。这里检验相同条件积分算子作用于实际来源方向后的范数比，未重跑完整子空间谱，也未分离Norm与FFN的作用。

[四模型token内分位数图](../outputs/all_attention_write_gain_811_v1/lambda_source_quantiles_all.png)、[400图test](../outputs/all_attention_write_gain_811_v1/lambda_source_quantiles_test.png)：线为各target内部中位数的跨mentions平均，带为内部Q25和Q75各自的平均，不是合池分位数或CI。每模型另有token内IQR/CV曲线及精确CSV。

## 验证与运行记录

- 4000shards/模型、协议/K32、targets/mentions/层数和811划分通过；读取前后原分片size/mtime不变。共50812 mentions，机制与检测使用同一身份表。
- 逐token重算S与保存gross最大相对差低于5.70e-8；All-attention最大闭合误差约4.14e-4。来源sum、G=S/I、log恒等式和IQR恒等式通过；派生统计SHA256见[统计核验](../outputs/all_attention_write_gain_811_v1/statistics_validation.json)。不消除原生捕获精度，也不等于每个方向都经过高阶求积对照。
- 无空区域/未定义区域统计。LLaVA零WRITE仅从对应lambda分位数排除，I/S保留其零贡献。
- 72头全部CPU重载，train-only scaler、最低val-loss epoch、train/validation/test概率与指标复算通过；每模型validation.json保存逐头检查。
- `python -m unittest tests.test_all_attention_write_gain tests.test_single_mlp_search_811`：11项通过；编译、diff通过。图片簇加权bootstrap与含并列分数的显式重复图片样例一致。
- 首次Qwen2运行完成4000图统计后，写cache_audit.json因Torch scalar不能JSON序列化退出；将保存gross转Python float修复，旧内存代码worker停止后重读缓存，未重跑VLM。失败/重启日志保留输出logs/。文档索引首次patch因标题含日期匹配失败，检查原文后修正，无结果数据受影响。

入口：`scripts/analyze_all_attention_write_gain.py --models MODEL --device cuda:0`；汇总：`scripts/summarize_all_attention_write_gain.py`。checkpoint/统计缓存可续用，协议不一致拒绝复用。第四项frozen-RMS直接归因对照本次未执行。
