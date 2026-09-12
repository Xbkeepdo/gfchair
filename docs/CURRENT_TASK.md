# 当前任务摘要

整理：2026-09-12；scalar identity条目已按实际完整结果核验更新，其余历史状态由原记录汇总。

## 2026-09-12：代码、结果与绘图数据发布

- 用户明确授权上传GitHub，图只上传绘图数据。本次纳入当前实验代码/测试/报告及六个结果目录的CSV、结果JSON、协议；不上传PNG/PDF、PT/权重/训练缓存或旧对话交接文件。
- 发布索引 `docs/PUBLICATION_20260912.md`；基于 `2a96b3d`，目标origin/main，普通提交/推送。保留原任务归档、数值限制及不同cohort/统计口径。
- 10项定向测试、分组及树网格检查通过；数据JSON解析和凭据特征检查通过。仅整理发布，不重跑VLM或训练实验。

## 已完成：FFN scalar identity / 完整子空间

- 四模型各500图、全部目标和层完成：2000个image-runs，6371唯一目标、206980 target-layers、6496 mentions。没有缩为20图或Top-K；不要重复提取。报告 `docs/FFN_SCALAR_IDENTITY.md`，输出 `outputs/ffn_scalar_identity_20260912/{tokens500,subspace500}`。
- 逐token正交比例均值约.944–.966；即使各token独立拟合g，仍有大正交残差。共享c允许负值并逐目标逐层最优拟合，原writes残差约.957–.975。
- 完整子空间cI残差REAL/HALL：Qwen2 .936482/.936819，LLaVA .959864/.958857，Qwen3 .951351/.950788，InternVL .965951/.966461；子空间外比例约.859–.913，Y谱P90/P10约2.15–2.81。均不支持scalar identity，但REAL/HALL差异小且无统一方向。
- Q探测保留原writes积分路径；FP64 Gram谱经直接FP64 SVD smoke验证（最大差7.56e-12）。七项测试PASS，完整图片/mention/标签/层、谱维度排序、谱能量与残差恒等式检查通过；恒等式误差≤1.78e-15，旧e范数重建误差≤2.62e-6。主秩阈值1e-5与1e-6全部一致；更严1e-4会改变LLaVA3619个case，正式未采用。
- 最后LLaVA保留320图，剩余95/85图奇偶双GPU续跑已完成，分片互斥覆盖。所有远程screen结束、两GPU显存0；`run_ffn_scalar_subspace.py --stage summarize`会话95307退出0。日志 `ffn_scalar_subspace_cpu_stats_gpu{0,1}.log`、`ffn_scalar_subspace_llava_{even,odd}.log`及`ffn_scalar_subspace_summary.log`均在outputs；失败/计划中断历史见报告。
- 中文报告、两张子空间图、逐token图、CSV/NPZ/mention表/逐图完整谱已齐全；三张主图已查看。未删路径积分、未训练检测器、未做显著/因果声明；代码和表格纳入本次发布，图片和大型张量留在本地。用户偏好完成后统一总结，不持续播报进度。

## 最近已完成及入口

- 原始 attention/gate X + log1p(S) 八组：三层 MLP、搜参单层 MLP、XGB/RF 全部完成；说明为 `docs/RAW_ATTENTION_LOG_STRENGTH_MLP.md`、`docs/RAW_ATTENTION_LOG_STRENGTH_TREES.md`。
- 输出 `outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/`（含 `trees_summary.md`）。四模型原 4000 图、3200/800 划分；搜参仅用训练内 2560/640，冻结后 seeds 43–45，测试集不参与选择。
- attention 与对应 S_g 融合：240 头完成；`docs/ATTENTION_STRENGTH_FUSION.md`、`outputs/ffn_source_composition_v1/attention_strength_fusion/summary.md`。不能宣称跨模型统一融合优势。
- amplification / rotation / cancellation：4000 图缓存分析和共享 500 图 cosine 提取完成；`docs/FFN_INPUT_GEOMETRY.md`、`outputs/ffn_input_geometry_20260912/`。
- 完整结果入口：`docs/EXPERIMENT_RESULTS_INDEX.md`；实验协议、指标和数值局限以对应报告为准。复用已完成结果，只有任务需要时才重跑。

## 历史与维护

- 本地有其他未提交工作；交接文档整理本身不改实验代码，后续执行以对应任务条目为准。历史提交授权和旧工具会话不是新任务的自动执行指令。
- 历史还标记过 MiniGPT/Shikra COCO4000 积分、K4 探索训练、LLaVA 数值子集和“双净量”为执行中；标记可能过时，接手对应任务时检索后续记录及输出核实。
- [完整原始记录](archive/CURRENT_TASK.before-context-optimization-20260912T111400Z.md)逐字节保留；按任务词或日期 `rg -n` 定位后只读附近段落，不全文加载。
- 此处只更新当前状态；完成细节进入专门报告或 `docs/archive/`，保持摘要简短。
