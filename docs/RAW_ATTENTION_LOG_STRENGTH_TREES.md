# 原始attention＋log1p(S)八组XGB/RF网格搜索

2026-09-12完成。用户明确要求补跑XGB/RF；本轮与上一轮单层MLP使用完全相同的特征矩阵、内部验证划分和最终holdout。

## 特征与数据

X为原始attention区域和或原始attention×gate区域和，只有完整来源强度S_g取log1p。每种X四组：

| 组 | 输入 |
|---|---|
| visual | [X_V, log1p(S_V)] |
| visual_prompt_sum | [X_V+X_P, log1p(S_V+S_P)] |
| generation | [X_G, log1p(S_G)] |
| vp_generation | [X_V+X_P, log1p(S_V+S_P), X_G, log1p(S_G)] |

前三组2L维，最后4L维；attention不取log，V+P两路先相加。BOS/模板/特殊token并入prompt；generation只含待预测目标前已有文本；gate仍为完整前缀MAD。完整S_g缓存覆盖四模型Qwen2.5/LLaVA/Qwen3/InternVL，保持每模型原4000图、3200/800、全部50,812 mentions和原mention顺序；不混用MiniGPT/Shikra旧S_E。

输入直接复用raw_attention_log_strength_mlp各模型matrices.pt及inner_fit/inner_val，未重新计算或改变划分。详细特征说明见[上一轮MLP说明](RAW_ATTENTION_LOG_STRENGTH_MLP.md)。

## 网格与选参

- XGB：max_depth=[4,6,8]，learning_rate=[.1,.05]，n_estimators=[100,200,500]，18个候选。
- RF：max_depth=[None,10,20]，n_estimators=[200,400,600]，9个候选；None表示不限深度。
- 原3200训练图内部2560/640图片划分与上一轮完全一致，INNER_SEED=20260908。每模型/特征/分类器用seed43遍历网格，按验证AUROC选，精确平局时AP，再候选顺序。
- selection.json先保存所选参数，再用全部3200图重训seeds43/44/45，最终只在原800图上评估；不根据测试结果重选。
- 使用原项目build_classifier：XGBoost2.1.4的原生XGBClassifier，sklearn1.6.1的RandomForestClassifier。保留未搜索参数的原默认值，不增加标准化、类别权重、校准或树模型早停。每估计器2 CPU线程，24个进程并行。
- RF保留bootstrap=True的算法默认训练；本轮不做bootstrap置信区间。XGB原构建器携带的use_label_encoder在该版本已不使用，相关参数警告原样记录，不替换分类器。

四模型×8组×(18+9)=864次候选拟合；两种树分类器各32组×3seed，共192个最终树模型。原三层和单层MLP的192个头仅作已有对照，没有重训。

## 完整AUROC总览

每格为三seed均值（%，原三层MLP / XGB / RF）；完整std、AP、F1及两种阈值见结果文件。

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| attention_visual | 84.624 / 81.881 / 80.004 | 89.379 / 88.425 / 88.320 | 87.516 / 86.811 / 84.002 | 85.244 / 83.435 / 83.184 |
| attention_visual_prompt_sum | 84.078 / 83.752 / 82.153 | 89.720 / 89.416 / 89.068 | 87.725 / 87.865 / 84.767 | 84.880 / 85.704 / 85.021 |
| attention_generation | 82.536 / 81.706 / 81.340 | 88.856 / 88.235 / 88.188 | 86.449 / 86.855 / 85.315 | 84.467 / 85.311 / 83.879 |
| attention_vp_generation | 84.675 / 85.400 / 82.418 | 90.021 / 89.599 / 89.186 | 89.289 / 89.056 / 85.856 | 86.695 / 87.090 / 85.275 |
| gated_visual | 84.973 / 82.763 / 80.439 | 89.501 / 88.880 / 88.452 | 87.744 / 87.138 / 84.405 | 85.419 / 84.483 / 83.944 |
| gated_visual_prompt_sum | 85.528 / 84.844 / 82.502 | 89.699 / 89.275 / 89.315 | 88.350 / 89.257 / 86.359 | 85.402 / 85.665 / 85.374 |
| gated_generation | 83.868 / 84.843 / 81.832 | 88.228 / 88.499 / 88.147 | 86.598 / 87.017 / 84.710 | 84.718 / 84.812 / 83.594 |
| gated_vp_generation | 86.635 / 86.584 / 83.416 | 90.470 / 90.519 / 89.792 | 89.665 / 90.996 / 87.607 | 86.599 / 87.338 / 85.552 |

XGB相对原三层MLP，15/32组平均AUROC提高、8/32组HALL-AUPR提高；RF只有1/32组AUROC和2/32组AP提高。相对上一轮搜参单层MLP，XGB有28/32组AUROC、24/32组AP提高，RF为20/32、22/32。树模型不能跨特征统一替换原三层MLP。

## 完整VP与G拼接的重点结果

AUROC（%），gate完整VP组与G组拼接：

| 模型 | 原三层MLP | 搜参单层MLP | XGB | RF |
|---|---:|---:|---:|---:|
| Qwen2.5 | 86.635 ± 0.660 | 83.111 ± 0.237 | 86.584 ± 0.000 | 83.416 ± 0.164 |
| LLaVA | 90.470 ± 0.306 | 89.601 ± 0.416 | 90.519 ± 0.000 | 89.792 ± 0.035 |
| Qwen3 | 89.665 ± 0.073 | 88.189 ± 0.314 | 90.996 ± 0.000 | 87.607 ± 0.038 |
| InternVL | 86.599 ± 0.558 | 84.825 ± 0.281 | 87.338 ± 0.000 | 85.552 ± 0.064 |


HALL_AUPR（%），gate完整VP组与G组拼接：

| 模型 | 原三层MLP | 搜参单层MLP | XGB | RF |
|---|---:|---:|---:|---:|
| Qwen2.5 | 46.780 ± 2.475 | 39.007 ± 0.630 | 45.783 ± 0.000 | 41.413 ± 0.393 |
| LLaVA | 71.947 ± 0.602 | 71.187 ± 0.711 | 71.134 ± 0.000 | 70.443 ± 0.211 |
| Qwen3 | 66.411 ± 0.973 | 63.007 ± 0.454 | 69.646 ± 0.000 | 63.676 ± 0.118 |
| InternVL | 56.160 ± 0.570 | 52.237 ± 0.690 | 56.765 ± 0.000 | 53.286 ± 0.590 |

| 模型 | XGB depth / lr / trees | RF depth / trees |
|---|---|---|
| Qwen2.5 | 8 / 0.1 / 200 | 不限 / 600 |
| LLaVA | 8 / 0.05 / 500 | 不限 / 600 |
| Qwen3 | 6 / 0.1 / 500 | 不限 / 600 |
| InternVL | 8 / 0.1 / 500 | 20 / 200 |


- Qwen3的gate完整VP/G拼接，XGB相比原三层MLP平均AUROC约+1.33点、AP约+3.24点，是本轮较明显的正向结果。
- InternVL同组XGB两项提高（AUROC约+.74点、AP约+.60点）。LLaVA同组AUROC基本持平但AP下降；Qwen2同组两项略降。
- RF大部分组低于原三层MLP。RF唯一AUROC提高组是InternVL attention_visual_prompt_sum；另外InternVL gated_visual_prompt_sum的AP提高但AUROC略低。
- 本设置下XGB三seed的AUROC/AP相同或只有浮点舍入差，故报告std≈0；这不表示跨图片或未来数据的不确定性为零。RF的seed波动正常保留。

不作显著性或因果声明；固定MLP与树模型的归纳偏置、正则/训练方式不同，结果只对应当前数据与网格。原K50数值、gate支持、前缀长度/位置混杂和重复使用旧holdout的限制继续保留。

## 产物

[四种分类器完整对照](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_summary.md)、[均值/std](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_detection.csv)、[逐seed双阈值](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_seed_metrics.csv)、[相对两种MLP差值](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_comparisons.csv)、[64组选中参数](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_selected_params.csv)。CSV中RF max_depth空白代表None，JSON保留null。

每模型xgb/rf/<group>/search保存全部候选验证指标/耗时/警告；selection.json保存完整候选及选择；seed43/44/45保存最终model.pkl、概率和result.pt。tree_protocol.json记录网格、内部划分、版本和默认参数。主表三seed均值±总体std，不用ensemble；HALL-F1采用训练REAL-F1阈值，另存固定.5阈值。

最终trees_detection.csv包含128组（含两种MLP引用），trees_seed_metrics.csv含768条，trees_comparisons.csv含128条，trees_selected_params.csv/JSON含64组。

## 命令与检查

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/opt/conda/private/envs/vicr/bin/python -u scripts/train_attention_log_strength_trees.py --stage search --workers 24
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_log_strength_trees.py --stage summarize
```

只扩展原共享search_group/fit函数的classifier参数，MLP默认行为保持；新脚本负责树协议和汇总。tiny XGB/RF/MLP fit/predict、18/9网格数量检查通过；576/288候选、各96最终模型完整，64组选参均符合保存的验证排名。搜索会话63332退出0，日志trees_search.log，汇总完成；编译与diff检查通过。

不重提VLM或积分，不计算SHA、bootstrap置信区间或全量checkpoint独立审计。本轮未提交上传，其他scalar-subspace工作保留。
