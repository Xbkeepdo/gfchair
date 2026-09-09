# 实验结果索引（2026-09-09）

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
