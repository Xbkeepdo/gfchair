# 原始attention＋log1p(S)分组与单隐藏层MLP搜参

2026-09-12完成。用户确认8组新特征，并要求按给定网格查看单隐藏层MLP。所贴配置同时含XGB/RF，已询问范围但未收到进一步答复；本页记录当时只搜索MLP的阶段；用户随后要求的XGB/RF现已完成，见[XGB/RF说明](RAW_ATTENTION_LOG_STRENGTH_TREES.md)。

## 特征定义

X_g分别取原始attention区域和或原始attention×gate区域和。X不取log，也不额外归一化。S_g为K50完整来源分解中该组c_j贡献范数之和；只有S取log1p。完整S_g缓存覆盖Qwen2.5、LLaVA、Qwen3、InternVL，本轮使用这四模型，不混用MiniGPT/Shikra的旧S_E。

| 组 | 输入 |
|---|---|
| visual | [X_V, log1p(S_V)] |
| visual_prompt_sum | [X_V+X_P, log1p(S_V+S_P)] |
| generation | [X_G, log1p(S_G)] |
| vp_generation | [X_V+X_P, log1p(S_V+S_P), X_G, log1p(S_G)] |

最后一组是完整VP特征与完整G特征的拼接，保留两边S，不是仅拼两条attention向量。每块为全层向量，前三组2L维，最后4L维；L为28/32/36/32。每种X四组，共8组/模型。Prompt包含BOS/模板/特殊token，generation仅含目标前已生成前缀，gate沿用完整前缀MAD统计。

## 数据与选参

原4000图、3200/800划分、全部50,812 mentions及原mention顺序保持，两路缓存身份/标签/顺序/ntrain核对通过。

- 原对照：Torch三隐藏层MLP(128/64/32)、BN/dropout .3、Adam、batch256、原训练loss调度/早停/最低train-loss checkpoint，seeds43/44/45。直接在原3200图上训练。
- 单层网格：项目detection.train.build_classifier的sklearn.MLPClassifier；hidden_layer_sizes=[[64],[128],[256]]，learning_rate_init=[.01,.001]，solver=[adam,sgd]，max_iter=500，共12候选。
- 原3200训练图内用既有inner_image_split、seed20260908，固定划分2560优化图/640验证图。同一图片的mentions不跨划分。
- 每模型/特征组独立以seed43遍历12候选，只用验证AUROC选参；精确平局时HALL-AUPR，再候选顺序。保存selection.json后才在全部3200图重训选中配置的43/44/45，并计算原800图测试指标。
- 没有额外标准化或scaler，符合当前项目MLP构建函数。未用测试集重新选参数，未搜索其他超参数。

sklearn实际版本1.6.1，其他默认值保留：ReLU、alpha=.0001、batch_size=auto（本批为200）、early_stopping=False、tol=.0001、n_iter_no_change=10；SGD为constant学习率、momentum=.9。max_iter=500是上限，不强制跑满500；384次候选拟合实际25–443次迭代，无收敛警告，逐候选loss/n_iter/warnings全部保存。最终每seed的迭代信息保存在result.pt。

这不是只改变隐藏层数量的消融：两套实现还不同于BN/dropout、L2、batch、停止规则。结果只能说明此输入和给定sklearn网格下的表现，不能断言单隐藏层普遍更差。

## 检测结果

以下每格为测试AUROC均值（%，原三隐藏层 → 搜参单隐藏层），完整std、HALL-AUPR/F1在结果总表。

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| attention_visual | 84.624 → 82.914 | 89.379 → 89.009 | 87.516 → 86.883 | 85.244 → 84.534 |
| attention_visual_prompt_sum | 84.078 → 82.240 | 89.720 → 87.816 | 87.725 → 82.591 | 84.880 → 81.879 |
| attention_generation | 82.536 → 78.698 | 88.856 → 87.579 | 86.449 → 83.309 | 84.467 → 83.152 |
| attention_vp_generation | 84.675 → 82.207 | 90.021 → 89.433 | 89.289 → 85.745 | 86.695 → 84.626 |
| gated_visual | 84.973 → 82.719 | 89.501 → 88.828 | 87.744 → 87.126 | 85.419 → 84.475 |
| gated_visual_prompt_sum | 85.528 → 82.765 | 89.699 → 88.415 | 88.350 → 85.916 | 85.402 → 82.425 |
| gated_generation | 83.868 → 79.251 | 88.228 → 87.212 | 86.598 → 83.302 | 84.718 → 81.749 |
| gated_vp_generation | 86.635 → 83.111 | 90.470 → 89.601 | 89.665 → 88.189 | 86.599 → 84.825 |

32组的单层平均AUROC全部低于原对照；只有LLaVA attention_vp_generation的AP提高约.360个百分点、InternVL attention_visual的AP提高约.023个百分点。其余30组AP下降；不作显著性声明。本轮选择的32组参数中28组Adam、4组SGD，宽度64/128/256分别11/11/10组。

用户追加的完整VP与G拼接，以attention×gate版本为例：

| 模型 | 原三隐藏层AUROC | 搜参单隐藏层AUROC | 单层宽度 | 学习率 | 优化器 |
|---|---:|---:|---:|---:|---|
| Qwen2.5 | 86.635 ± 0.660 | 83.111 ± 0.237 | 64 | 0.001 | adam |
| LLaVA | 90.470 ± 0.306 | 89.601 ± 0.416 | 128 | 0.001 | adam |
| Qwen3 | 89.665 ± 0.073 | 88.189 ± 0.314 | 256 | 0.001 | adam |
| InternVL | 86.599 ± 0.558 | 84.825 ± 0.281 | 128 | 0.001 | adam |

该组HALL-AUPR原三层→单层（%）：Qwen2 46.780→39.007，LLaVA71.947→71.187，Qwen3 66.411→63.007，InternVL56.160→52.237。没有因为测试结果较差而增加候选或换选验证排名较低的参数。

## 每组选择的参数

所有max_iter均为500，验证分数来自seed43的2560/640内部划分；不是测试AUROC。

| 模型 | 特征 | 宽度 | 学习率 | solver | 验证AUROC |
|---|---|---:|---:|---|---:|
| qwen2_5_vl_7b | attention_visual | 256 | 0.001 | adam | 81.211 |
| qwen2_5_vl_7b | attention_visual_prompt_sum | 256 | 0.01 | sgd | 81.218 |
| qwen2_5_vl_7b | attention_generation | 64 | 0.001 | adam | 79.395 |
| qwen2_5_vl_7b | attention_vp_generation | 64 | 0.01 | sgd | 82.209 |
| qwen2_5_vl_7b | gated_visual | 64 | 0.001 | adam | 82.184 |
| qwen2_5_vl_7b | gated_visual_prompt_sum | 128 | 0.01 | adam | 82.375 |
| qwen2_5_vl_7b | gated_generation | 64 | 0.001 | adam | 79.307 |
| qwen2_5_vl_7b | gated_vp_generation | 64 | 0.001 | adam | 82.669 |
| llava_1_5_7b | attention_visual | 256 | 0.001 | adam | 90.649 |
| llava_1_5_7b | attention_visual_prompt_sum | 64 | 0.01 | sgd | 88.848 |
| llava_1_5_7b | attention_generation | 128 | 0.001 | adam | 87.926 |
| llava_1_5_7b | attention_vp_generation | 128 | 0.001 | adam | 89.771 |
| llava_1_5_7b | gated_visual | 256 | 0.001 | adam | 90.390 |
| llava_1_5_7b | gated_visual_prompt_sum | 64 | 0.001 | adam | 89.066 |
| llava_1_5_7b | gated_generation | 64 | 0.001 | adam | 88.235 |
| llava_1_5_7b | gated_vp_generation | 128 | 0.001 | adam | 90.144 |
| qwen3_vl_8b | attention_visual | 128 | 0.001 | adam | 86.113 |
| qwen3_vl_8b | attention_visual_prompt_sum | 256 | 0.001 | adam | 84.234 |
| qwen3_vl_8b | attention_generation | 256 | 0.001 | adam | 84.622 |
| qwen3_vl_8b | attention_vp_generation | 64 | 0.001 | adam | 86.012 |
| qwen3_vl_8b | gated_visual | 128 | 0.001 | adam | 86.435 |
| qwen3_vl_8b | gated_visual_prompt_sum | 256 | 0.001 | adam | 86.974 |
| qwen3_vl_8b | gated_generation | 64 | 0.001 | adam | 84.819 |
| qwen3_vl_8b | gated_vp_generation | 256 | 0.001 | adam | 88.855 |
| internvl_2_5_8b | attention_visual | 256 | 0.001 | adam | 84.772 |
| internvl_2_5_8b | attention_visual_prompt_sum | 128 | 0.001 | adam | 82.450 |
| internvl_2_5_8b | attention_generation | 256 | 0.001 | adam | 80.007 |
| internvl_2_5_8b | attention_vp_generation | 128 | 0.001 | adam | 84.816 |
| internvl_2_5_8b | gated_visual | 128 | 0.001 | adam | 84.993 |
| internvl_2_5_8b | gated_visual_prompt_sum | 64 | 0.01 | sgd | 82.734 |
| internvl_2_5_8b | gated_generation | 128 | 0.01 | adam | 79.834 |
| internvl_2_5_8b | gated_vp_generation | 128 | 0.001 | adam | 85.712 |

## 产物与运行

[完整结果](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/summary.md)、[均值/std](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/detection.csv)、[逐seed双阈值](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/seed_metrics.csv)、[逐组差值](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/comparisons.csv)、[最佳参数CSV](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/selected_params.csv)。

每模型matrices.pt和protocol.json保存精确特征与内部图片划分；heads为固定Torch头；one_hidden/<group>/search为12候选JSON；selection.json记录验证排名；seed43/44/45下保存最终sklearn model.pkl、概率与指标result.pt。共384候选拟合、96最终单层头及96新固定对照头。主表报告三seed均值±总体std，不用ensemble；HALL-F1按每seed训练REAL-F1阈值，另有固定.5阈值。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_log_strength_mlp.py --stage prepare
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_log_strength_mlp.py --stage fixed --models <模型列表> --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_log_strength_mlp.py --stage search --workers 12
/opt/conda/private/envs/vicr/bin/python scripts/train_attention_log_strength_mlp.py --stage summarize
```

四模型固定对照在两GPU的四进程并行，CPU搜索12个worker、每worker BLAS单线程。固定35958/91501/69943/62583及搜索7151均退出0，prepare/summarize完成。输入attention不取log、完整VP/G拼接、原图片划分及内层不含测试图片检查通过；384候选/32选择/96+96最终头完整，选择均符合保存的验证排名；编译与diff检查通过。

本轮无需重提VLM或积分，没有SHA、bootstrap、全量checkpoint独立审计、提交或上传。原K50来源数值限制、gate支持、前缀长度/位置混杂以及复用旧holdout的解释边界保留。
