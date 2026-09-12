# attention与attention×gate分组检测说明

2026-09-10启动，2026-09-12完成汇总与文档。六模型训练均已完成，共360个新头。

## 数据与分组

复用 `outputs/prefix_attention_gate/full/region_plots/<model>/regions.npz`，未重新提取模型。每模型原4000图、3200/800划分及全部mentions保持；六模型共78,473条mentions。按mention_id重排到原baseline训练顺序，核对标签、图片和划分。

旧分区顺序为BOS、visual、prompt、generated_text。本轮为了与前面S_g实验一致，三组为：

- prompt：BOS与其余非视觉prompt合并，含system、模板、特殊token；没有独立BOS的模型直接使用原prompt。
- generation：预测目标之前已经生成的前缀，包含当前query位置，不含待预测目标与未来。
- visual：原视觉token位置，MiniGPT-4保持32个原生查询token，无空间映射。

目标t、层ℓ上，记跨头均值注意力为A_m，保存的完整前缀gate为η_m。两种组内总量分别为：

\[
R_g=\sum_{m\in g}A_m,\qquad U_g=\sum_{m\in g}A_m\eta_m.
\]

均按逐位置原值求和，不除token数、不重新归一化。U_g使用原先逐位置乘好的attention_x_gate，不是“区域attention总量×区域gate平均值”。Gate使用hpre直接投影目标unembedding行、完整因果前缀的median/MAD和sigmoid；无final norm或LM-head bias。完整公式见[前缀提取说明](PREFIX_ATTENTION_GATE.md)。此统计支持与原视觉AE不同，且这里保留原始注意力质量，不将U_visual等同于原AE。

## 五组输入与训练

每种信号各有原值与log1p两套，每套五组：prompt、generation、visual单独输入全层向量；concat按[prompt全层,generation全层,visual全层]拼接；visual_prompt_sum为视觉与prompt逐层相加。

log1p在逐mention逐层计算；相加组为log(1+R_V+R_P)或log(1+U_V+U_P)，先相加再变换。单组及相加组L维，拼接3L维。L为32/32/28/32/36/32，两种信号各自拼接，本轮未进行attention与gated之间的混合拼接，也未加入AE或FFN强度。

2种信号×2种尺度×5组×6模型×seeds43/44/45=360头。沿用原MLP隐藏层128/64/32、BN、dropout .3、Adam lr .001/weight_decay 1e-5、batch256、最多100epochs、训练loss调度与早停、最低训练loss checkpoint。未调参或bootstrap；评估只使用原800图holdout，曲线另汇总4000图。

所有主表报告三seed均值±总体标准差（ddof=0），非ensemble。HALL-F1使用各seed训练集REAL-F1阈值；固定0.5与train_f1双阈值保存在CSV。原汇总函数的detection.json兼容保留ensemble字段，主表不使用。

## 三组拼接结果

AUROC，三seed均值±总体标准差（%）：

| 模型 | attention原值 | attention×gate原值 | attention log1p | attention×gate log1p |
|---|---:|---:|---:|---:|
| MiniGPT-4 | 90.605 ± 0.138 | 90.879 ± 0.059 | 90.393 ± 0.244 | 90.878 ± 0.063 |
| Shikra | 85.039 ± 0.321 | 86.081 ± 0.162 | 84.941 ± 0.313 | 86.001 ± 0.226 |
| Qwen2.5-VL | 84.533 ± 0.346 | 87.475 ± 0.489 | 84.567 ± 0.250 | 88.032 ± 0.351 |
| LLaVA-1.5 | 88.954 ± 0.388 | 89.541 ± 0.199 | 89.105 ± 0.355 | 89.611 ± 0.124 |
| Qwen3-VL | 88.091 ± 0.247 | 89.904 ± 0.245 | 88.203 ± 0.483 | 90.165 ± 0.282 |
| InternVL-2.5 | 86.007 ± 0.107 | 87.412 ± 0.171 | 85.503 ± 0.531 | 87.385 ± 0.248 |


HALL_AUPR，三seed均值±总体标准差（%）：

| 模型 | attention原值 | attention×gate原值 | attention log1p | attention×gate log1p |
|---|---:|---:|---:|---:|
| MiniGPT-4 | 66.028 ± 0.459 | 67.161 ± 0.111 | 65.635 ± 1.157 | 67.303 ± 0.217 |
| Shikra | 58.637 ± 0.678 | 60.918 ± 0.046 | 58.666 ± 0.297 | 61.100 ± 0.513 |
| Qwen2.5-VL | 40.933 ± 1.321 | 48.195 ± 1.564 | 40.948 ± 1.373 | 49.342 ± 0.838 |
| LLaVA-1.5 | 66.515 ± 0.702 | 69.361 ± 0.172 | 67.420 ± 0.524 | 69.473 ± 0.151 |
| Qwen3-VL | 62.715 ± 0.934 | 66.928 ± 0.221 | 62.668 ± 0.895 | 67.436 ± 0.673 |
| InternVL-2.5 | 51.775 ± 0.289 | 56.414 ± 0.507 | 50.948 ± 1.327 | 55.987 ± 1.203 |

## 相对单独信号

以下只展示原值。最佳单组按本次三seed平均AUROC作描述性比较，没有据此修改或筛选训练配置。

| 模型 | attention最佳单组 | AUROC → 拼接 | gate最佳单组 | AUROC → 拼接 |
|---|---|---:|---|---:|
| MiniGPT-4 | prompt | 89.209 → 90.605 | visual | 88.771 → 90.879 |
| Shikra | generation | 85.091 → 85.039 | generation | 84.986 → 86.081 |
| Qwen2.5-VL | prompt | 83.437 → 84.533 | prompt | 84.378 → 87.475 |
| LLaVA-1.5 | generation | 88.057 → 88.954 | visual | 87.871 → 89.541 |
| Qwen3-VL | prompt | 85.917 → 88.091 | visual | 85.915 → 89.904 |
| InternVL-2.5 | generation | 84.055 → 86.007 | prompt | 83.872 → 87.412 |

在原值与log1p两套中，attention×gate拼接六模型的平均AUROC和HALL-AUPR都高于对应纯attention拼接，也都高于各自最佳单组。原值gate拼接相对最佳单组的AUROC增量依次约+2.108/+1.095/+3.097/+1.671/+3.990/+3.541个百分点。

纯attention原值拼接在Shikra略低于最佳单组（AUROC -0.052点、AP -0.433点），其余五模型提高；log1p下Shikra也只有很小的变化。log1p对拼接没有跨模型一致收益。未作显著性检验，不把三seed方向或小差异解释成统计显著。

## 相加信号与解释边界

Raw attention的三个互斥组覆盖完整前缀，总量约1，因此R_V+R_P约等于1-R_G。保存矩阵测得六模型最大偏差均小于0.0009，详见raw_total_rounding.json；保留原生FP16/BF16注意力舍入，不调整原值。两者检测器的小差异可能涉及舍入、初始化与优化，不能据此宣称新增信息。

Gate后的总量可变化，U_V+U_P不再是generation的固定互补量。Gate也引入目标词语义得分，检测提升不等于注意力因果解释更正确。Log1p是可逆的逐特征尺度变换，不增加信息。拼接输入从L维变为3L维，未做维度匹配；前缀长度与目标位置的混杂仍在。

## 结果位置与曲线

[完整总表](../outputs/prefix_attention_gate/full/group_detection/summary.md)包含全部单组、相加、拼接和已有F_E参考；[均值/std](../outputs/prefix_attention_gate/full/group_detection/detection.csv)、[逐seed双阈值](../outputs/prefix_attention_gate/full/group_detection/seed_metrics.csv)、[配对差值](../outputs/prefix_attention_gate/full/group_detection/comparisons.csv)。F_E为已有AE+log1p(S_E)，只引用不重训。

[attention原值曲线](../outputs/prefix_attention_gate/full/group_detection/all_attention_raw.png)、[attention log1p](../outputs/prefix_attention_gate/full/group_detection/all_attention_log1p.png)、[attention×gate原值](../outputs/prefix_attention_gate/full/group_detection/all_gated_raw.png)、[attention×gate log1p](../outputs/prefix_attention_gate/full/group_detection/all_gated_log1p.png)。每张图为六模型四列（prompt/generation/visual/V+P），全部4000图按mention等权，实线均值、虚线中位数、阴影IQR（非置信区间）；PNG/PDF与6144条绘图CSV均已保存，两个原值总览已实际查看。

每模型matrices.pt保存与baseline对齐的输入及mentions；protocol.json记录口径；heads保存60个检测头的checkpoint/result及预测概率。最终共120个新特征组、360头；detection.csv含126组（含6旧基线），seed_metrics.csv共756条，comparisons.csv共144条。

## 运行与验证

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/opt/conda/private/envs/vicr/bin/python scripts/train_prefix_attention_groups.py --stage prepare
/opt/conda/private/envs/vicr/bin/python scripts/train_prefix_attention_groups.py --stage train --models minigpt4_7b qwen2_5_vl_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/train_prefix_attention_groups.py --stage train --models llava_1_5_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/train_prefix_attention_groups.py --stage train --models qwen3_vl_8b internvl_2_5_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/train_prefix_attention_groups.py --stage train --models shikra_7b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/train_prefix_attention_groups.py --stage summarize
```

中间四条训练命令分别为独立队列，日志queue_a/b/c/d.log，均已正常退出。prepare、summarize完成。合成BOS合并/质量守恒、拼接/先加再log/零值检查通过；全数据原mention顺序、标签/图片划分、非负/有限性及gated不超过raw检查通过；360头及120组各3seed完整。脚本编译与diff检查通过。没有新增SHA校验、全量checkpoint独立审计或提交上传。
