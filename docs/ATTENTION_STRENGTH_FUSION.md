# 对应attention与完整S_g融合实验

2026-09-12完成。四模型各20组×seeds43/44/45，共240个新检测头，原基线直接复用。

## 范围与特征

用户要求将相应attention和S_g拼接。完整K50来源分解的S_g缓存目前覆盖Qwen2.5、LLaVA、Qwen3、InternVL；本轮只用这四模型。MiniGPT/Shikra未有完整三组S_g，不混用其旧e_m强度。每模型原4000图、3200/800划分及全部mentions保持，共50,812条；attention与S缓存的mention顺序、mention_id、图片、target_key、目标token、response_index和标签逐条匹配。

A_g为对应区域原始attention跨头均值的组内和；U_g为原始attention×gate逐位置乘积的组内和，gate仍使用完整因果前缀MAD统计。S_g=sum_group ||c_j||，视觉分组的S为完整分解S_C。Prompt两路均含BOS/模板/特殊token；generation仅为目标之前已生成前缀。均不除token数或再归一化。

每套五种分组：prompt、generation、visual单独；三组全拼接；视觉＋prompt相加。四种配对方式为：

| 方式 | 输入 |
|---|---|
| attention原值融合 | [A_g, S_g] |
| attention×gate原值融合 | [U_g, S_g] |
| attention双方log1p | [log1p(A_g), log1p(S_g)] |
| attention×gate双方log1p | [log1p(U_g), log1p(S_g)] |

这是同尺度配对，本轮没有额外训练“原始attention＋log1p(S)”的跨尺度组合。

单区域输入先放attention全层向量，再放同区域S全层向量，为2L维。全拼接顺序为[A_P全层,A_G全层,A_V全层,S_P全层,S_G全层,S_V全层]，gated用U替换A，为6L维（168/192/216/192）。相加组为[A_V+A_P,S_V+S_P]或U版本，各自先相加再log1p，不是向量贡献相加后取范数。

## 训练与对照

新增scripts/train_attention_strength_fusion.py，直接复用两套matrices及原base.train。原MLP隐藏层128/64/32、BN、dropout .3、Adam lr .001/weight_decay 1e-5、batch256、最多100epochs、训练loss调度/早停和最低训练loss checkpoint保持；不调参、bootstrap或重提VLM/积分。

每个融合组分别对照同区域、同尺度、同类型的attention（或gated）单独输入，以及对应S单独输入。按三个seed均值±总体std报告；双阈值为固定0.5与训练REAL-F1阈值，主表非ensemble。JSON保留原汇总器兼容的ensemble字段，不用于主结论。

## 三组全拼接结果

AUROC，三seed均值±总体std（%）：

| 模型 | A+S原值 | U+S原值 | log1p(A)+log1p(S) | log1p(U)+log1p(S) |
|---|---:|---:|---:|---:|
| Qwen2.5 | 85.486 ± 0.528 | 86.309 ± 0.398 | 85.405 ± 0.422 | 86.796 ± 0.661 |
| LLaVA | 90.087 ± 0.039 | 90.482 ± 0.221 | 90.125 ± 0.166 | 90.296 ± 0.073 |
| Qwen3 | 90.099 ± 0.516 | 90.525 ± 0.121 | 90.083 ± 0.136 | 90.878 ± 0.366 |
| InternVL | 87.333 ± 0.275 | 87.280 ± 0.172 | 86.939 ± 0.206 | 86.818 ± 0.452 |


HALL_AUPR，三seed均值±总体std（%）：

| 模型 | A+S原值 | U+S原值 | log1p(A)+log1p(S) | log1p(U)+log1p(S) |
|---|---:|---:|---:|---:|
| Qwen2.5 | 43.824 ± 0.954 | 45.278 ± 0.528 | 40.901 ± 1.578 | 44.449 ± 0.928 |
| LLaVA | 70.669 ± 0.822 | 72.194 ± 0.252 | 70.227 ± 0.999 | 71.258 ± 0.727 |
| Qwen3 | 65.673 ± 1.379 | 66.631 ± 0.447 | 66.941 ± 0.753 | 68.972 ± 0.180 |
| InternVL | 58.109 ± 0.562 | 58.627 ± 0.640 | 54.592 ± 1.325 | 56.468 ± 2.340 |

## 单区域结果与本轮最高均值

下表最高组仅为既定20组的描述性比较，未据此修改训练配置，也不作为独立验证的最优结论。不能只看全拼接：InternVL的prompt配对明显更好。

| 模型 | 预设融合组 | AUROC | HALL-AUPR | 对应U单独AUROC | 对应S单独AUROC |
|---|---|---:|---:|---:|---:|
| Qwen2.5 | gated_log1p_concat | 86.796 ± 0.661 | 44.449 ± 0.928 | 88.032 | 85.617 |
| LLaVA | gated_raw_concat | 90.482 ± 0.221 | 72.194 ± 0.252 | 89.541 | 90.052 |
| Qwen3 | gated_log1p_concat | 90.878 ± 0.366 | 68.972 ± 0.180 | 90.165 | 90.141 |
| InternVL | gated_raw_prompt | 88.282 ± 0.184 | 59.672 ± 0.880 | 83.872 | 85.129 |

- LLaVA原值gate全拼接同时超过两个对应基线：ΔAUROC相对U为+.940点、相对S为+.430点；ΔAP为+2.832/+1.295点，三个seed的AUROC差均为正。
- Qwen3双方log1p的gate全拼接同时超过两个基线：ΔAUROC为+.713/+.737点，ΔAP为+1.536/+3.232点，三个seed的AUROC差均为正。
- InternVL原值gate prompt配对为本轮该模型最高AUROC：相对U_prompt/S_prompt的ΔAUROC为+4.411/+3.154点，ΔAP为+10.452/+8.601点，三个seed的AUROC差均为正。其全拼接则有AUROC与AP的取舍，不应以全拼接代表全部融合结果。
- Qwen2最高均值融合组是双方log1p的gate全拼接，虽高于对应S单独输入，但比对应gate单独输入AUROC低1.236点、AP低4.893点，三个seed的AUROC均下降。本轮不存在跨模型统一的融合优势。

以上为当前固定MLP与原holdout上的探索性点估计，不宣称显著性或因果。输入维度从L/3L增加为2L/6L；两套源虽目标一致但来自不同提取缓存，保留原始attention舍入、gate支持及K50来源重建误差限制。注意力部分的V+P约等于1-G，但S_V+S_P不与S_G固定互补，因此相加融合组不能等同于generation融合组。

## 产物与复现

[完整80组融合结果](../outputs/ffn_source_composition_v1/attention_strength_fusion/summary.md)、[均值/std CSV](../outputs/ffn_source_composition_v1/attention_strength_fusion/detection.csv)、[480条逐seed双阈值](../outputs/ffn_source_composition_v1/attention_strength_fusion/seed_metrics.csv)、[160条对两个基线的配对比较](../outputs/ffn_source_composition_v1/attention_strength_fusion/comparisons.csv)。每模型另存matrices.pt、protocol.json、60个头及预测概率。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_strength_fusion.py --stage prepare
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_strength_fusion.py --stage train --models qwen2_5_vl_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_strength_fusion.py --stage train --models llava_1_5_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_strength_fusion.py --stage train --models qwen3_vl_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_strength_fusion.py --stage train --models internvl_2_5_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_strength_fusion.py --stage summarize
```

四条训练命令为独立并行队列，模型名.log保存日志，51744/58384/12064/72109均正常退出；prepare/summarize完成。两源目标/标签/顺序/划分与矩阵形状、有限性和简单拼接检查通过，240头/80组各3seed完整，py_compile与diff检查通过。没有SHA、全量checkpoint独立审计或提交上传。
