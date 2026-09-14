# 当前任务摘要

## 当前：区域 AE＋All-attention gross，三层/单层MLP与XGB

- 用户确认四组：[AE_V,log1p(SV_all)]、[AE_VP,log1p(SV_all+SP_all)]、[AE_G,log1p(SG_all)]、[完整VP组,完整G组]；AE原值，只有gross取log，所有层拼接。新增对应AE-only对照，三种分类器各自比较。
- AE沿用旧visual定义：各区域内部attention归一化，hpre目标logit在同域Gaussian MAD后sigmoid。不是full-prefix U或U/mass；旧缓存缺VP/G的该定义，须补forward，复用真实RMS All-attention K32 gross，不重算积分，不用B2。
- 原4000图3200/800、全部mentions、seeds43/44/45。固定MLP128/64/32；单层12候选(64/128/256,.01/.001,adam/sgd,max500)，XGB18候选(depth4/6/8,lr.1/.05,100/200/500树)。沿用训练内2560/640、seed20260908；seed43验证AUROC/AP选参后全3200重训，不用test选参、不加scaler。
- 四fusion＋四AE-only，每模型72正式头/240候选；总288正式头/960候选。主入口scripts/train_all_attention_ae_strength.py --stage pipeline；提取scripts/extract_region_ae.py；输出outputs/all_attention_ae_log_strength_v1/。空G AE保存NaN，检测置0，不删mentions。
- 子代理server_4090负责32678(Qwen3/InternVL)，主代理本机Qwen2/LLaVA。复用共享userdata/vicr；每卡一模型，CPU搜索每模型8worker。进度watcher新增--progress-root，原默认兼容，仍原子覆写两根目录进度文件。
- 四模型8图visual AE核验现均PASS，max_abs≤2.98e-7、零失败、未放宽容差。旧失败来自分块attention及forward形状；现Qwen逐目标原生eager，LLaVA/InternVL完整回答原生eager，复现旧执行形状。四模型正式4000图提取已启动，提取代码签名冻结。新特征拼接/空G/独立区域及进度回归共8项PASS。
- 2026-09-13 13:55 UTC：Qwen2.5/LLaVA/InternVL各4000图、72头完成；Qwen3约3500/4000仍提取。216头概率独立复算AUROC/AP与表一致，报告docs/ALL_ATTENTION_AE_STRENGTH_PARTIAL_20260913.md；partial_20260913/保存72组、216seed与验证。
- 自动汇总screen aeall_report（scripts/report_all_attention_ae_strength.py --wait）等待Qwen3完成，后核验全部288头/960候选并生成最终报告及图。状态outputs/all_attention_ae_log_strength_v1/report_progress.json，日志outputs/aeall_report.log；两进度文件保留。
- 已出36组拼接均提高AE-only AUROC，33组提高AP；单层AUROC全部低于三层；VP+G的XGB在三模型均优于同输入三层。新视觉All-attention对旧Visual-only三层增益不一致；未做新bootstrap、不宣称显著性，无gross-only对照。尚未发布/提交。

## 前序 All-attention / B1 / B2 已有结果与未修事项

- 四模型4000分片、15组×3seed共180结果，10000次图片级配对bootstrap齐全；49939目标/50812 mentions/1622248 target-layers。运行结束不等于数值通过。
- B2 K4全量7.46% case积分误差>1%；原50-case只查5层漏掉早层等异常。B1/All-attention全量无>1%纯闭合，Visual仅Qwen2有2case略超。B2提高/自适应求积、全层审计及相关60头重训仍待做，本轮不混用其特征。
- 结果docs/ALL_SOURCE_PATHS_RESULTS_20260913.md；75信号与图docs/ALL_SOURCE_SIGNALS_20260913.md；15组定义及历史比较docs/ALL_SOURCE_DETECTION_COMPARISON_20260913.md。F6=F0+F2+F4，不含F1/F3范数；B1-F6对F0的8个CI均跨零。旧composition总表用ensemble，比较须取seed_mean_std。
- 前序详细任务状态归档docs/archive/CURRENT_TASK.before-region-ae-20260913.md；保留原方案与未跟踪交接文件，不提交凭据。

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
