# SVAR / MetaToken 原生分类器：相同811划分

四模型均为3200训练/400验证/400测试图片，全mentions，seeds43/44/45；均值±std为三seed指标统计，另存概率ensemble。
SVAR：Linear(D,248)-ReLU-Linear(248,2)，Adam .001，batch32，最多50epochs，验证loss最优checkpoint及patience5早停；无BN/dropout/标准化/学习率调度。
MetaToken：StandardScaler+LR(lbfgs,max_iter2000)和StandardScaler+GB100；标准化只拟合训练，固定参数，不搜索、不合并验证集重训。
训练内部1=HALL，所有保存概率转成P(REAL)。阈值取验证REAL-F1并另报0.5，以下AUROC/HALL-AUPR不依赖阈值。
使用项目已有原生分类器实现、controlled同目标特征及811适配，不代表复现论文原始数据集。MetaToken保留完整回答长度/对象span统计。旧800图已有研究使用，本轮是探索性评估。

| 模型 | 原生分类器 | AUROC mean±std (%) | HALL-AUPR mean±std (%) |
|---|---|---:|---:|
| qwen2_5_vl_7b | svar_native | 87.34 ± 0.22 | 47.06 ± 0.66 |
| qwen2_5_vl_7b | metatoken_lr | 83.39 ± 0.00 | 44.08 ± 0.00 |
| qwen2_5_vl_7b | metatoken_gb | 82.73 ± 0.02 | 38.98 ± 1.00 |
| llava_1_5_7b | svar_native | 90.44 ± 0.04 | 70.96 ± 0.53 |
| llava_1_5_7b | metatoken_lr | 88.35 ± 0.00 | 67.36 ± 0.00 |
| llava_1_5_7b | metatoken_gb | 88.61 ± 0.04 | 66.88 ± 0.06 |
| qwen3_vl_8b | svar_native | 88.95 ± 0.23 | 62.66 ± 0.39 |
| qwen3_vl_8b | metatoken_lr | 82.24 ± 0.00 | 45.74 ± 0.00 |
| qwen3_vl_8b | metatoken_gb | 83.66 ± 0.02 | 52.76 ± 0.23 |
| internvl_2_5_8b | svar_native | 86.98 ± 0.61 | 53.73 ± 1.14 |
| internvl_2_5_8b | metatoken_lr | 83.38 ± 0.00 | 46.96 ± 0.00 |
| internvl_2_5_8b | metatoken_gb | 84.89 ± 0.02 | 52.03 ± 0.14 |

## 与统一分类器比较

每格AUROC / HALL-AUPR（%，三seed均值）；只展示已完成三个seed的统一头，不按测试结果选择分类器。

| 模型 | 特征 | 原生 | 统一三层 | 统一单层 | 统一XGB |
|---|---|---:|---:|---:|
| qwen2_5_vl_7b | svar_native | 87.34 / 47.06 | 86.48 / 43.34 | 86.81 / 45.90 | 87.49 / 44.39 |
| qwen2_5_vl_7b | metatoken_lr | 83.39 / 44.08 | 76.85 / 22.20 | 77.92 / 27.12 | 83.74 / 38.66 |
| qwen2_5_vl_7b | metatoken_gb | 82.73 / 38.98 | 76.85 / 22.20 | 77.92 / 27.12 | 83.74 / 38.66 |
| llava_1_5_7b | svar_native | 90.44 / 70.96 | 90.62 / 70.79 | 运行中 | 90.08 / 69.03 |
| llava_1_5_7b | metatoken_lr | 88.35 / 67.36 | 86.22 / 63.77 | 86.33 / 62.05 | 89.14 / 66.33 |
| llava_1_5_7b | metatoken_gb | 88.61 / 66.88 | 86.22 / 63.77 | 86.33 / 62.05 | 89.14 / 66.33 |
| qwen3_vl_8b | svar_native | 88.95 / 62.66 | 运行中 | 运行中 | 运行中 |
| qwen3_vl_8b | metatoken_lr | 82.24 / 45.74 | 运行中 | 运行中 | 运行中 |
| qwen3_vl_8b | metatoken_gb | 83.66 / 52.76 | 运行中 | 运行中 | 运行中 |
| internvl_2_5_8b | svar_native | 86.98 / 53.73 | 87.93 / 54.28 | 运行中 | 88.31 / 57.34 |
| internvl_2_5_8b | metatoken_lr | 83.38 / 46.96 | 78.52 / 38.85 | 78.72 / 37.86 | 85.69 / 50.35 |
| internvl_2_5_8b | metatoken_gb | 84.89 / 52.03 | 78.52 / 38.85 | 78.72 / 37.86 | 85.69 / 50.35 |

## AE拼接参考

同一400测试图；V为视觉AE+log1p(S_V_all)，VP+G为视觉与prompt区域块加generation区域块。列出预先指定的三层和XGB，不按测试分数择优。

| 模型 | V 三层 | VP+G 三层 | V XGB | VP+G XGB |
|---|---:|---:|---:|---:|
| qwen2_5_vl_7b | 86.21 / 39.25 | 87.36 / 46.83 | 86.55 / 40.79 | 87.54 / 48.40 |
| llava_1_5_7b | 90.03 / 69.38 | 90.05 / 72.06 | 90.21 / 69.80 | 90.24 / 70.46 |
| qwen3_vl_8b | 90.12 / 66.51 | 92.29 / 70.04 | 运行中 | 运行中 |
| internvl_2_5_8b | 87.16 / 54.93 | 87.21 / 53.30 | 85.56 / 50.37 | 88.39 / 55.92 |

原生与统一头同时改变了结构/优化或标准化等配置，差值反映整套分类器配置变化，不能单独归因为标准化或层数。尚未计算新的配对bootstrap，不宣称差值显著。

## 结果解读与复核

- 原生SVAR在四个模型的AUROC与HALL-AUPR均高于原生MetaToken LR/GB。相对于统一三层SVAR头：Qwen2提升0.86/3.72个百分点；LLaVA变化−0.17/+0.17；InternVL变化−0.95/−0.55，原生头不是普遍占优。
- MetaToken LR相对于统一三层头：Qwen2提升6.54/21.88、LLaVA提升2.12/3.59、InternVL提升4.86/8.11个百分点。标准化和分类器一同改变，尚不能将增益单独归因于标准化。LR与GB也没有跨模型统一赢家。
- AE的VP+G三层在Qwen3为92.29/70.04，相比原生SVAR88.95/62.66更高；其他模型有AUROC/AP取舍。各配置均完整呈现，不据测试结果事后挑选特征或分类器。
- 四模型36头均从保存分类器重新计算train/validation/test概率，核验与原结果一致；标准化均值/方差仅来自训练，checkpoint对应验证最低loss，划分/mentions/标签与AE逐行对齐。记录：outputs/native_baselines_811_v1/independent_validation.json。首次逐位比较在未统一BLAS线程时出现2.22e-16舍入差，统一单线程后最大概率/指标差均为0。
- 14项本轮相关测试与3项原基线核心回归通过。Qwen2实际断点恢复复用了9个头，result.pt的修改时间保持不变。统一头仍在跑的格子是对照项，四模型原生结果本身已经完整。
