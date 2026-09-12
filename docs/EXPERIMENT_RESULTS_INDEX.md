# 实验结果索引（2026-09-10）

[2026-09-12发布索引及绘图数据](PUBLICATION_20260912.md)：本次图只上传数值数据，PNG/PDF链接用于本地生成产物。

- 原始X＋log1p(S)八组XGB/RF搜索（2026-09-12，完成）：[说明](RAW_ATTENTION_LOG_STRENGTH_TREES.md)、[四种分类器对照](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_summary.md)、[64组参数](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_selected_params.csv)。864候选+192最终树模型；XGB 15/32组AUROC超原三层，RF仅1/32；Qwen3 gate VP/G拼接XGB AUROC90.996%。

- FFN是否近似scalar identity（2026-09-12，完成）：[方法与结果](FFN_SCALAR_IDENTITY.md)、[逐token检验](../outputs/ffn_scalar_identity_20260912/tokens500/all_metrics.png)、[完整子空间检验](../outputs/ffn_scalar_identity_20260912/subspace500/all_subspace.png)。四模型各500图、全部目标/层，206980 target-layers；C/g/R、Q/Y/B谱及leakage均完成。最优signed c的完整子空间残差约.936–.966，子空间外比例.859–.913，Y谱P90/P10约2.15–2.81；两类均不支持cI，类别差异无统一方向。

- 原始attention＋log1p(S)八组及单隐藏层MLP搜索（2026-09-12，完成）：[中文说明](RAW_ATTENTION_LOG_STRENGTH_MLP.md)、[完整结果](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/summary.md)、[选中参数](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/selected_params.csv)。四模型384候选+96单层最终头、96固定对照；32组单层AUROC均低于原三层。仅S取log，含完整VP/G拼接，XGB/RF未运行。

- FFN输入WRITE几何（2026-09-12，完成）：[定义与REAL/HALL结果](FFN_INPUT_GEOMETRY.md)、[4000图曲线](../outputs/ffn_input_geometry_20260912/all_geometry.png)、[同500图三指标](../outputs/ffn_input_geometry_20260912/cohort500/all_geometry.png)、[输入/输出抵消](../outputs/ffn_input_geometry_20260912/cohort500/all_cancellation_change.png)。四模型4000图gain/cancellation、共享500图三指标完成；平均cosine为负但类别差异小，LLaVA输出抵消差异几乎已存在于输入，Qwen的HALL减少抵消幅度较小。无检测器训练/因果声明。

- 对应attention与完整S_g融合（2026-09-12，完成）：[中文说明](ATTENTION_STRENGTH_FUSION.md)、[80组结果](../outputs/ffn_source_composition_v1/attention_strength_fusion/summary.md)、[对应单独基线差值](../outputs/ffn_source_composition_v1/attention_strength_fusion/comparisons.csv)。四模型240新头；LLaVA/Qwen3全拼接、InternVL prompt配对有增益，Qwen2未超对应较强gate基线。

- attention/attention×gate分组单独与拼接检测（2026-09-12，完成）：[中文说明](PREFIX_ATTENTION_GROUP_DETECTION.md)、[完整结果](../outputs/prefix_attention_gate/full/group_detection/summary.md)、[配对差](../outputs/prefix_attention_gate/full/group_detection/comparisons.csv)。六模型360头，原值/log1p各5组；gate拼接六模型均提高平均AUROC/AP，log1p无统一收益；4000图曲线及数据已保存。

- QE/QC softmax温度0.2与对应AE+S融合：[总表](../outputs/ffn_source_composition_v1/q_softmax_tau02/summary.md)、[配对增量](../outputs/ffn_source_composition_v1/q_softmax_tau02/paired.csv)、[说明](FFN_SOURCE_COMPOSITION_EXPERIMENT.md)。四模型24新头完成；QC融合在LLaVA/Qwen3/InternVL相对F_C双指标提高，QE融合仅LLaVA双指标提高。

- 追加AE+视觉S_C（F_C=AE+log1p(S_C)）：[直接对照](../outputs/ffn_source_composition_v1/fc_comparison.csv)、[方法与结果说明](FFN_SOURCE_COMPOSITION_EXPERIMENT.md)。四模型12新头完成，原F复用；AUROC均略低于F，Qwen2的HALL-AUPR提升，InternVL两项下降。当前共10组120头。

- FFN 完整来源分解验证（2026-09-10，完成）：[方法、信号与实际结论](FFN_SOURCE_COMPOSITION_EXPERIMENT.md)、[四模型总表](../outputs/ffn_source_composition_v1/summary.md)、[检测指标](../outputs/ffn_source_composition_v1/detection.csv)、[配对增量](../outputs/ffn_source_composition_v1/paired_detection.csv)。原4000图/K50、108头及400图fixed-QK完成；保存raw attention与少量高维示例。新图没有普遍优于旧路径，Qwen仍有来源重建尾部误差；不做SHA、bootstrap或新图像确认。

- 温度.2 endpoint cosine–JS与F融合：[总表](../outputs/endpoint_tau02_fusion4000/log_s/summary.md)、[完整JSON](../outputs/endpoint_tau02_fusion4000/log_s/summary.json)、[双阈值ensemble指标](../outputs/endpoint_tau02_fusion4000/log_s/ensemble_metrics.csv)。F=AE+log1p(raw S)，同源v1/原3200-800划分，24新头独立重载通过；主报告§5.36。

## 本地新增：原始Q-softmax温度0.07

- [温度1/.07检测对照](../outputs/ffn_q_softmax_js_20260909/temperature007/summary.md)、[逐seed/双阈值完整JSON](../outputs/ffn_q_softmax_js_20260909/temperature007/summary.json)、[四模型曲线对比](../outputs/ffn_q_softmax_js_20260909/temperature007/curves_tau1_vs_007.png)。36个新头及独立重载核验完成；仅Q的softmax温度变化，原3200/800图划分，非cosine/O_FFN实验。主报告§5.35，尚未上传。

- 原4000图差分方向cosine–JS温度对照：[总表](../outputs/endpoint_temperatures4000/summary.md)、[完整JSON](../outputs/endpoint_temperatures4000/summary.json)、[tau=.2均值图](../outputs/endpoint_temperatures4000/mean_curves_tau0.2.png)、[tau=.02均值图](../outputs/endpoint_temperatures4000/mean_curves_tau0.02.png)。原3200/800图片，24个新头及独立重载验证完成；主报告§5.34。

- O_FFN cosine–JS温度0.02：[三温度总表](../outputs/ffn_output_cosine_20260909/cohort500/temperature002/summary.md)、[完整结果](../outputs/ffn_output_cosine_20260909/cohort500/temperature002/summary.json)、[均值曲线](../outputs/ffn_output_cosine_20260909/cohort500/temperature002/mean_curves.png)。共享500图，12个新头完成，独立重载通过；主报告§5.33。

- O_FFN cosine–JS温度0.2：[四模型结果](../outputs/ffn_output_cosine_20260909/cohort500/temperature02/summary.md)、[完整JSON](../outputs/ffn_output_cosine_20260909/cohort500/temperature02/summary.json)、[均值曲线](../outputs/ffn_output_cosine_20260909/cohort500/temperature02/mean_curves.png)。复用共享500图、12个新头完成并独立重载核验；主报告§5.32。

- 实际O_FFN与e_m的cosine–JS，共享500图：[总表](../outputs/ffn_output_cosine_20260909/cohort500/summary.md)、[完整结果](../outputs/ffn_output_cosine_20260909/cohort500/summary.json)、[均值曲线](../outputs/ffn_output_cosine_20260909/cohort500/mean_curves.png)。同子集差分方向对照，四模型2000 image-runs与24头完成，各模型independent_audit.json保存来源及重载核验；主报告§5.31，尚未上传。

- d在e_m上的有符号标量投影softmax–JS：[四模型结果](../outputs/endpoint_projection_js_summary.md)、[完整JSON](../outputs/endpoint_projection_js_summary.json)、[均值曲线](../outputs/endpoint_projection_js_mean_curves.png)。a_m=<d,e_m>/||e_m||，同源v1比较，12头完成；主报告§5.30，尚未上传。

- 直接内积e_m·d的softmax–JS：[四模型结果](../outputs/endpoint_dot_js_summary.md)、[完整JSON](../outputs/endpoint_dot_js_summary.json)、[均值曲线](../outputs/endpoint_dot_js_mean_curves.png)。同源v1与cosine比较，12头完成；主报告§5.29，尚未上传。

## 本地新增：Endpoint-cosine JS分解

- [四模型T集中度/endpoint修正消融总表](../outputs/endpoint_cosine_js_decomposition_summary.md)、[完整结果](../outputs/endpoint_cosine_js_decomposition_summary.json)。将`J_PT=JS(P,T)`精确分为`J_T=JS(U,T)`与有符号`J_E=J_PT-J_T`，训练两项单独及拼接；36个新头完成、checkpoint复算差0，主报告§5.28。本次尚未上传。

## 本地新增：原始Q softmax与T的JS

- [四模型6组检测总表](../outputs/ffn_q_softmax_js_20260909/summary.md)、[全部逐seed/双阈值JSON](../outputs/ffn_q_softmax_js_20260909/summary.json)。原3200/800图，72头完成；温度1、全视觉token、不除以||e_m||，另有同源FP32 cosine-JS对照。主报告§5.27。各模型protocol/curve/audit保存数值检查、层曲线和独立checkpoint重载验证；本次尚未上传。

## 本地后续：停止全量C后的共享500图检测

- [Endpoint-cosine JS四模型总表](../outputs/ffn_endpoint_cosine_js_summary.md)、[完整结果与输入审计](../outputs/ffn_endpoint_cosine_js_summary.json)、[REAL/HALL合并曲线](../outputs/ffn_endpoint_cosine_js_real_hall.png)：`P=softmax(cos(e_m,G(Z)-G(Z0)))`，以全视觉support自然对数`JS(P,T)`作逐层standalone检测特征；主报告§5.26。本地完成，尚未提交/上传。
- [两模型20组总表](../outputs/ffn_target_consequence_cqb_v1/subset500_20260909/summary.md)、[逐seed与双阈值完整结果](../outputs/ffn_target_consequence_cqb_v1/subset500_20260909/summary.json)、[冻结500图清单](../outputs/ffn_target_consequence_cqb_v1/subset500_20260909/cohort.json)。Qwen2/Qwen3各400原train+100原test，120头完成；原全量C及后续模型队列已停止，尚未补提LLaVA/InternVL。
- [Qwen2独立重载/resume核验](../outputs/ffn_target_consequence_cqb_v1/subset500_20260909/qwen2_5_vl_7b_audit.json)、[Qwen3核验](../outputs/ffn_target_consequence_cqb_v1/subset500_20260909/qwen3_vl_8b_audit.json)。两模型概率最大差均0。主报告§5.25说明小样本、完成覆盖偏差和C闭合误差尾部；本次后续尚未上传。

## 2026-09-09 增量发布

本次补充发布下文原标注“本地新增”的§5.20–5.23结果及代码；历史未上传措辞只表示当时状态。C仍在提取，未发布不存在的C检测成绩。

- [C定义、后续层逻辑、优化及待GPT审阅问题](C_TARGET_CONSEQUENCE_DISCUSSION_20260909.md)。
- [注意力JS实验总入口](../outputs/attention_js_studies/README.md)：同图REAL/HALL目标、attention/evidence、跨层JS及检测结果。
- [跨层JS紧凑JSON](../outputs/attention_js_studies/interlayer_js/summary_compact.json)与[完整33,024行统计CSV](../outputs/attention_js_studies/interlayer_js/all_pair_interlayer_js.csv)：替代连接接口无法上传的10 MB重复JSON；逐字段核对无损，原文件在本地保留。
- [联合VJP基准与冻结gate](../outputs/ffn_target_consequence_cqb_v1/optimizations/joint_score_vjp_v1/gate.json)、[2319张原分片保留核验](../outputs/ffn_target_consequence_cqb_v1/optimizations/joint_score_vjp_v1/preservation_check.json)。
- [本次增量发布SHA清单](GITHUB_RESULTS_PUBLICATION_20260909.json)。仅文档/表格/紧凑JSON/代码/测试，不新增图片、权重、原始特征或聊天恢复文件。

本次按用户确认发布截至目前的全部实验结果，包含最新AE+log1p(原始S)专门调参（主报告§5.19）。发布对象为本仓库gfchair；相邻token-detector属于不同实验目录，不混入此仓库。

## 主报告与最新实验

- 本地新增（2026-09-09，未提交/上传）：[Q/B_Q四模型检测总表](../outputs/ffn_target_consequence_cqb_v1/q_bq_summary.md)、[逐seed和双阈值完整JSON](../outputs/ffn_target_consequence_cqb_v1/q_bq_summary.json)、[C同源捕获与FP32缓存实现门控](../outputs/ffn_target_consequence_cqb_v1/implementation_gate.json)，主报告§5.23。Q/B_Q当前96头完成；三分数C的全4000图提取正在双卡运行，C检测未完成，不能纳入v1完成声明。
- 本地新增（未提交/上传）：[三层调度与checkpoint对照](../outputs/ffn_visual_source_consistency_v2/three_layer_scheduler_checkpoints_20260908/summary.md)、[完整结果](../outputs/ffn_visual_source_consistency_v2/three_layer_scheduler_checkpoints_20260908/summary.json)、[352行逐seed/双阈值表](../outputs/ffn_visual_source_consistency_v2/three_layer_scheduler_checkpoints_20260908/groups.csv)、[独立权重/调度/指标/resume核验](../outputs/ffn_visual_source_consistency_v2/three_layer_scheduler_checkpoints_20260908/independent_metrics_audit.json)，主报告§5.22；四模型72条轨迹/96个checkpoint完成，另有24组原AUC checkpoint逐元素一致性核对。
- 本地新增（本轮未提交/上传）：[冻结配置后3200图重训](../outputs/ffn_visual_source_consistency_v2/shallow_refit3200_20260908/summary.md)、[完整结果JSON](../outputs/ffn_visual_source_consistency_v2/shallow_refit3200_20260908/summary.json)、[672行新旧逐seed/双阈值表](../outputs/ffn_visual_source_consistency_v2/shallow_refit3200_20260908/groups.csv)、[独立权重/指标/resume核验](../outputs/ffn_visual_source_consistency_v2/shallow_refit3200_20260908/independent_metrics_audit.json)，主报告§5.21；四模型120个新头完成，原§5.20及全部旧结果保留。
- 本地新增（本轮未提交/上传）：[StandardScaler与浅层probe两阶段搜索](../outputs/ffn_visual_source_consistency_v2/shallow_standardized_search_20260908/summary.md)、[完整指标](../outputs/ffn_visual_source_consistency_v2/shallow_standardized_search_20260908/summary.json)、[逐seed/双阈值表](../outputs/ffn_visual_source_consistency_v2/shallow_standardized_search_20260908/groups.csv)、[全部候选表](../outputs/ffn_visual_source_consistency_v2/shallow_standardized_search_20260908/search_trials.csv)，主报告§5.20。
- [Vector FFN Source Attribution主报告](../ffn_visual_source_attribution_report.md)：v1正式实验及§5.14–5.19增补。
- [第二轮增量验证报告](../jffn_second_round_incremental_validation_report.md)：历史阶段、方法沿革及辅助实验。
- [AE32消融](../outputs/ffn_visual_source_top32_ae_summary.json)与[AE32×cosine消融](../outputs/ffn_visual_source_top32_ae_cosine_summary.json)。
- [WRITE/FFN强度与分布比较](../outputs/ffn_write_ffn_comparison_summary.json)。
- [v2子集数值门控](../outputs/ffn_visual_source_consistency_v2/numerical_subset_20260907/gate.json)与[全量K4验收](../outputs/ffn_visual_source_consistency_v2/old_validation_fp32_k4.json)。全量数值FAIL未改为PASS。
- [单隐藏层MLP/XGBoost：U_SN选参、16组迁移](../outputs/ffn_visual_source_consistency_v2/head_search_20260908/summary.md)、[完整指标](../outputs/ffn_visual_source_consistency_v2/head_search_20260908/summary.json)、[表格](../outputs/ffn_visual_source_consistency_v2/head_search_20260908/groups.csv)。
- [AE+log1p(S)专门扩大调参](../outputs/ffn_visual_source_consistency_v2/ae_direct_log1p_search_20260908/summary.md)、[完整指标](../outputs/ffn_visual_source_consistency_v2/ae_direct_log1p_search_20260908/summary.json)、[独立指标复核](../outputs/ffn_visual_source_consistency_v2/ae_direct_log1p_search_20260908/independent_metrics_audit.json)。

## 四模型原正式cohort的基线对照

以下均来自gfchair的COCO4000-INSLEN-OFFICIAL-TARGET，不是相邻token-detector的ALLMENTION-SWEEP结果。

- [Qwen2.5 baseline native vs shared MLP](../outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/comparison/qwen2_5_vl_7b_baselines_native_vs_shared_mlp_3seed_summary.md)。
- [LLaVA baseline native vs shared MLP](../outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/comparison/llava_1_5_7b_baselines_native_vs_shared_mlp_3seed_summary.md)。
- [Qwen3 baseline native vs shared MLP](../outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/comparison/qwen3_vl_8b_baselines_native_vs_shared_mlp_3seed_summary.md)。
- [InternVL baseline native vs shared MLP](../outputs/internvl_2_5_8b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/comparison/internvl_2_5_8b_baselines_native_vs_shared_mlp_3seed_summary.md)。

各模型的`results/`同时保存历史特征组合、CSV及紧凑JSON。本次新增PNG/PDF图表仅留在本地，旧Git历史内图片不删除。目录名与报告日期区分正式、smoke、失败、中断和探索性阶段；历史快照不应替代最新主报告中的状态说明。

## 发布边界与完整性

- 保留失败、负结果、修复说明及未运行阶段；v2数值FAIL、旧InternVL严格CPU概率复核状态均不改写。近期训练头结果是旧800图上的探索性点估计，独立2000图确认未运行。
- 发布代码、报告、最终/逐seed紧凑指标、候选配置和分片验收；按用户最新确认，不新增发布PNG/PDF图表、原始特征、COCO图片、完整向量、预测张量、checkpoint、逐epoch日志或环境凭据。
- 实验内部checksum清单覆盖本地完整产物，其中指向的`.pt`等文件不代表已放入GitHub。另提供[本次发布文件清单](GITHUB_RESULTS_PUBLICATION_20260908.json)，记录实际加入提交的文件与校验值；此前已在Git历史内的文件不重复列入。
- [中文任务记录](CURRENT_TASK.md)保留命令、耗时、失败和结论。各历史条目中的“不提交/不上传”描述当时执行边界；本次上传已由用户最新指令明确授权。

## MiniGPT-4 / Shikra COCO4000 路径积分（2026-09-09，运行中）

- 入口与协议：[MINIGPT4_SHIKRA_PATH_RUNBOOK.md](MINIGPT4_SHIKRA_PATH_RUNBOOK.md)。
- 已通过两模型K4/K64、真实因果/均值复核、baseline技术训练；正式4000图仍运行，尚无正式检测结果。
- 进度：`outputs/minigpt4_shikra_path_summary/progress.json`；正式输出：`outputs/{minigpt4_7b,shikra_7b}/COCO4000-JACOBIAN-PATH/`。
- `smoke8`和`training_smoke16`为技术验证，不纳入实验性能比较。

## 2026-09-10：MiniGPT/Shikra AE+log1p(S)+Q-JS(τ=.2)

- 已完成同源COCO4000、三seed，共6头；[结果表](../outputs/ae_s_qjs02/summary.md)。
- Q为raw signed path_signed_q；逐层JS(softmax(Q/.2),T)，不是直接拼接概率，也不是cosine-JS。MiniGPT下降，Shikra AUROC近乎持平、HALL-AUPR提高。

## 2026-09-10：MiniGPT/Shikra F+endpoint cosine-JS(.2)

- 同源COCO4000、每模型3seed完成；仅重训6个检测头。[组合对照表](../outputs/ae_s_cosinejs02/summary.md)。
- 均值AUROC分别90.188%和85.644%，相对F下降.718/.621个百分点；此为cos(e_m,G(Z)-G(Z0))后softmax的JS，区别于raw-Q JS。主表均值±总体std，不使用ensemble。

## 2026-09-10：原始视觉注意力质量R + log1p(S)

- 两模型COCO4000、18个新头完成；[检测表](../outputs/raw_attention_strength/summary.md)。
- [MiniGPT曲线](../outputs/raw_attention_strength/minigpt4_7b/curves.png)；[Shikra曲线](../outputs/raw_attention_strength/shikra_7b/curves.png)。均值/中位数/IQR，train/test独立展示。
- R+logS对比F：MiniGPT AUROC较低，Shikra提高约.50个百分点；全部为3seed均值±总体std，非ensemble。

## 2026-09-10：其他四模型原始R + log1p(S)

- Qwen2/LLaVA/Qwen3/InternVL各9头已完成；[六模型结果表](../outputs/raw_attention_strength/summary.md)，[四模型测试曲线](../outputs/raw_attention_strength/four_models_test_curves.png)。
- 四模型R+logS的AUROC都略高于单独logS，但AUROC/HALL-AUPR均低于原AE+logS；按三seed均值±总体std报告。

## 2026-09-10：Cosine-Q vs normalized raw attention JS

- 六模型4000图、J_raw/F+J_raw各3seed，共36个新头完成。[检测结果](../outputs/cosine_raw_attention_js02/summary.md)，[六模型测试曲线](../outputs/cosine_raw_attention_js02/test_curves.png)。
- F+J_raw平均AUROC相对F三升三降，改善均很小且部分模型seed波动增大；不宣称稳定或显著收益。

## 2026-09-10：六模型完整前缀attention / attention×gate（运行中）

- 六模型48图验证和文件审计PASS，已启动每模型4000图提取。[协议和格式](PREFIX_ATTENTION_GATE.md)。
- 进度：`outputs/prefix_attention_gate/full/progress.json`；最终独立验收：`full/audit.json`。本轮只提取信号，无检测训练。

## 2026-09-10：完整前缀四区域曲线（完成）

- 六模型4000图，BOS/visual/prompt/generated_text互斥分区；raw attention与attention×gate分别求和，train/test、HALL/REAL分开，均值+IQR。
- [全部图与CSV索引](../outputs/prefix_attention_gate/full/region_plots/README.md)。无独立BOS不伪造；各区域相加守恒检查通过，不除以区域token数，不重归一化。

## 2026-09-10：视觉/generation注意力比值曲线

- 六模型逐mention逐层先作比，再汇总REAL/HALL；raw和attention×gate分别画all/train/test，原始比值统计、对数纵轴显示。
- [图及数据索引](../outputs/prefix_attention_gate/full/region_plots/visual_generation_ratio/README.md)，共2304条CSV统计，实际无零分母；不含检测训练。

## 2026-09-12：视觉/generation比值检测汇总

- 六模型两种原值比值、各三seed共36头完成；[结果表](../outputs/prefix_attention_gate/full/region_plots/visual_generation_ratio/detection/summary.md)。
- attention×gate比值相对raw比值四模型AUROC提高、两模型下降；两种比值均低于原F，未额外取log或拼接其他特征。
