# 当前任务摘要

## 当前：All-attention / B1 / B2结果分析；B2精度待修正

- 用户已批准完整实现、50-case审计、共享500图机制分析、4000图固定检测；B1和B2都进入正式实验且分别对照，不做B1+B2融合。强度保留raw/log1p两版，共15特征组×4模型×seeds43/44/45。
- 本机GPU0 Qwen2.5、GPU1 LLaVA；SSH别名32678服务器GPU0 Qwen3、GPU1 InternVL。两端共享userdata和vicr环境，禁止重复同步或并发修改环境。远程由server_4090子代理负责，主代理负责实现/汇总。
- 运行于9月12日全部结束：四模型各4000分片、180 seed级检测结果（12个F0复用）、每模型10000次bootstrap齐全；49939唯一目标、50812 mentions、1622248 target-layers。进度文件completed仅表示流程结束，不代表数值验收。
- 9月13日逐图全量复核发现B2 K4纯积分>1%：Qwen2 43269/242312、LLaVA15386/478432、Qwen3 58826/529344、InternVL3595/372160，总计7.46%。原50-case只选5层，漏掉Qwen2早层2–6、Qwen3层2–5、InternVL层20等异常层。不得把旧B2表面提升视为可靠优势。
- All-attention K32、B1稳定sinh自适应积分全量无>1%闭合；B1最大约1.54e-5。Visual K4仅Qwen2有2个case略超1%。原生端点重构误差另外报告。
- 正式分析 `docs/ALL_SOURCE_PATHS_RESULTS_20260913.md`；数值汇总每模型full_numerical_summary.json；对比图outputs/ffn_all_source_paths_v1/analysis_20260913/。原分数summary.md已加数值限制，数值未改。本次只分析已有数据，未重跑模型/训练。
- 其他信号完整报告 `docs/ALL_SOURCE_SIGNALS_20260913.md`：75字段、10组all/train/test PNG/PDF及同图配对数据在outputs/ffn_all_source_paths_v1/signals_20260913/。补算routing/token标量，6488 mentions对齐；生成每token响应复算误差≤7.54e-7。HALL生成总量↑而每token写入/响应↓、组内抵消↑；B1相对残差的G−V余弦差↓；非因果结论。B2图保留并标记精度未通过。
- 检测逐组定义与历史对照 `docs/ALL_SOURCE_DETECTION_COMPARISON_20260913.md`，数据在detection_comparison_20260913/：60本轮组、384历史组，统一seed均值；旧composition总表为ensemble不能直接混比。F6=F0+F2+F4，不含F1/F3范数。全部维度及五套历史mentions/标签/划分精确对齐；本次未重训。
- B1-F6对F0两指标四模型CI均跨零，未得稳健提升；可保留机制观察为HALL视觉上下文调制增强、视觉–生成输出余弦通常更高。B2曲线/相对B1优劣需待精度修复。
- 待修：审计覆盖全部层并加强异常层、提高/自适应B2求积、正式提取/训练增加数值门禁；仅补算B2和重训相关60头及bootstrap，复用其余分支。尚未启动修复任务。
- 入口scripts/run_ffn_all_source_paths.py；v3日志outputs/ffn_all_source_{qwen2,llava,qwen3,internvl}_pipeline_v3.log。旧两次失败已归档，协议/恢复入口见docs/ALL_SOURCE_PATHS.md；30项定向测试通过记录保留。
- 固定原cohort/split/mentions；机制主曲线及paired均排除冲突目标；训练保留所有mentions，NaN固定置0。分离积分与来源重建误差，不制造补齐来源。
- 完成后交付中文报告/绘图及数据/检测及10000次图片级paired bootstrap。旧未跟踪方案与交接文档保留，不提交凭据。

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
- [完整原始记录](CURRENT_TASK.before-context-optimization-20260912T111400Z.md)逐字节保留；按任务词或日期 `rg -n` 定位后只读附近段落，不全文加载。
- 此处只更新当前状态；完成细节进入专门报告或 `docs/archive/`，保持摘要简短。
