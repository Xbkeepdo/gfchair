# 生成来源 S_G / position 的检测结果

2026-09-15。四模型36个检测头已完成，无新VLM提取。**当前结果不支持用S_G/position替换原始S_G：四模型AUROC点估计均下降，InternVL两项指标的名义95%配对区间均为负。** 但这个比值仍明显强于position-only，不能说其检测作用完全等同于生成位置。

## 信号和固定协议

S_G是生成前缀token在真实RMS All-attention K32共同路径z−A_all→z上的响应范数和，即Σ_{m∈generation}||e_m||，不是净向量范数；不含prompt、视觉或residual独立归因。position使用目标的response_index，等于其前方已有生成token数量N_G；来源边界实现见features/ffn_all_source_paths.py的partition函数。

逐层比较log1p(S_G)与log1p(S_G/position)，均保留全部层；position-only输入log1p(position)单列。先求比，再log1p，再train-only逐列Z-score，不使用包含prompt/视觉的绝对位置。原4000图全部50812 mentions，固定3200/400/400，seeds43/44/45。统一128/ReLU/dropout.3/noBN单隐藏MLP，Adam lr.001/wd1e-5/batch128，最多150epoch/早停20/LR plateau patience6，最低val BCE checkpoint；无HPO和train+val重训。

position=0预设为raw ratio NaN、检测置0、保留样本。本轮四模型实际均无position=0或未定义比值；position范围依次4–250、4–186、3–484、4–255。position-only不重复成多层输入，因此其输入维数及参数量小于两个逐层强度组；两个强度组的维数、架构和训练协议相同。

[完整协议](GENERATION_PER_POSITION_811_PROTOCOL.md) · [含std/所有配对区间的完整结果](../outputs/generation_per_position_811_v1/summary.md)

## 测试表现

每格为AUROC / HALL-AUPR，单位%，三seed指标均值，不是概率ensemble。

| 模型 | S_G-only | S_G/position-only | position-only |
|---|---:|---:|---:|
| Qwen2.5 | 80.73 / 33.69 | 80.44 / 30.05 | 73.58 / 20.36 |
| LLaVA | 87.16 / 61.52 | 86.61 / 61.63 | 81.43 / 50.77 |
| Qwen3 | 85.19 / 54.17 | 83.99 / 50.43 | 65.91 / 22.46 |
| InternVL | 84.24 / 49.55 | 81.84 / 43.25 | 70.49 / 27.16 |

比值相对原S_G的差值（百分点，2000次测试图片簇bootstrap名义95%区间）：

| 模型 | ΔAUROC [95% CI] | ΔHALL-AUPR [95% CI] |
|---|---:|---:|
| Qwen2.5 | −0.29 [−2.09,+1.56] | −3.65 [−8.48,+0.86] |
| LLaVA | −0.56 [−1.37,+0.28] | +0.11 [−3.04,+3.32] |
| Qwen3 | −1.20 [−2.48,+0.08] | −3.74 [−7.22,−0.56] |
| InternVL | −2.40 [−4.05,−0.75] | −6.29 [−10.78,−1.56] |

Qwen2.5/LLaVA的差异区间均跨零；Qwen3的AUROC区间跨零，AP区间为负；InternVL两项区间均为负。不能把所有模型的点估计下降都称为显著下降，也不能把LLaVA约+0.11pp AP称为有确定增益。

比值相对position-only的AUROC增量依次+6.86/+5.17/+18.08/+11.35pp，两项指标四模型的名义区间都为正。它仍包含比单独位置更有用的预测信号，但这不是与位置统计独立性的证明。

## 如何解释

S_G=N_G×每生成token平均响应。除以N_G改变了保留的信息：总量变成平均量，并不保证消除位置相关性或所有长度混杂。观察到的性能下降与去掉部分有用长度/总量信息相容，但本次没有分离损失来自哪一部分，不能当作因果结论。

本次仅比较生成来源自身的三个信号；这里的S_G-only不是此前拼接P/V/G的S-only，不能直接用此前PVG成绩替代本次对照。已有测试集曾被查看，24个名义指标区间未多重比较校正，结果为探索性结论。

## 核验与产物

缓存统计SHA256、mention/标签/811划分及S_G与旧生成矩阵核对通过；比值复算和ratio×position=S_G通过。36头全部CPU重载，训练集scaler、最低val-loss epoch、train/validation/test概率和指标复算通过。3项新增测试覆盖先除后log、单列position、零分母/合法零比值及非法输入；编译和diff检查通过。原始缓存及已有实验结果不变。

- [检测图](../outputs/generation_per_position_811_v1/detection.png)与[REAL/HALL比值曲线](../outputs/generation_per_position_811_v1/ratio_curves.png)，均有PDF；曲线实线中位数，阴影是跨mentions IQR，不是CI。
- [特征核验](../outputs/generation_per_position_811_v1/feature_validation.json)、[36头核验](../outputs/generation_per_position_811_v1/validation.json)。
- 输出outputs/generation_per_position_811_v1/，含每模型features.pt、协议、9个checkpoint、逐seed/配对CSV及日志。
- 入口scripts/evaluate_generation_per_position_811.py；用--models指定模型，--summarize重新汇总，现有checkpoint按协议指纹复用。
