# Current Task

## 2026-09-08：截至目前全部实验结果发布快照

- 用户随后缩减为不上传图片，并再次明确“代码也传”：最终提交仅包含代码/测试、结果文档、CSV/JSON表格和说明，不新增PNG/PDF图表；本地文件与旧Git历史不删除。传输中的图片队列已停止，尚未更新main；已传输但未引用的临时Git对象不加入最终提交。
- 用户明确确认“包含最新§5.19，上传截至目前全部结果”，覆盖此前不提交/上传的执行边界。目标为现有`Xbkeepdo/gfchair`的main，远端当前`9b00af17e7f5094203069527ab969095e4ce8673`；相邻token-detector不混入。
- 按Ponytail复用Git与已连接GitHub应用，不引入发布框架或依赖。终端HTTPS无可用凭据，SSH亦无认证agent；原生dry-run返回缺少Password。GitHub应用已确认账号与仓库push权限，将通过Git数据API创建单个快照并以非force方式更新main，不读取auth.json或用户私钥。
- 补齐§5.14–5.19代码/测试、图表/紧凑指标/协议/状态/门控，以及原正式cohort历史结果和四模型baseline对比；排除原始特征、完整向量、预测张量、checkpoint、逐epoch日志和凭据。保留本地全部artifact、旧数值FAIL/CPU复核状态及无关未提交文件。新增`docs/EXPERIMENT_RESULTS_INDEX.md`与本次发布清单，主报告更新发布范围。
- 发布前回归命令 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_ae_log1p_search tests.test_ffn_consistency_alternative_heads tests.test_ffn_visual_source_consistency tests.test_ffn_visual_source_study tests.test_ffn_visual_path_attribution tests.test_ffn_write_comparison`：69/69 PASS，19.539秒（不含导入），日志`/tmp/gfchair_publish_regression_20260908.log`。发布清单、敏感信息检查、staged diff与远端commit/tree一致性需在发布时核对；远端是否成功以实际commit及分支复查为准。

## 2026-09-08：AE + log1p(原始 S) 单独扩大调参（完成）

- 用户要求S直接用log(1+S)、针对AE+S单独调参、多试参数并参考论文。严格定义新输入为concat(全AE,log1p(raw S))，不除tau、不使用AE32、不额外标准化；现有local-FP32 K4紧凑特征和全部旧结果保留。按Ponytail新增薄入口 `scripts/train_ffn_ae_log1p_search.py`，复用旧数据校验、图片split、DGSTStyleProbe/训练器、checkpoint推理及指标函数，不修改已封存旧脚本。
- 四模型/三分类器(one_hidden、three_hidden、XGBoost)各48组候选：12固定锚点+36确定性随机候选，随机seed20260908。原3200/800划分不变；3200内按同一seed分2560/640，同图mention不跨分区。先seed43筛48组，内部验证前三补seed44/45，再按三seed平均内部AUROC、HALL-AUPR及固定index选参。每模型三类均封存selection后才运行原800图最终评估。共576初筛+72复核=648搜索头；最终3类×2变换×3seed×4模型=72头，另12个原固定三隐藏层参数direct-log控制，总计732个新头。
- 单隐藏宽度32/64/128/256/512；三隐藏为[w,w/2,w/4]，与单隐藏相同48候选预算。扩展lr 1e-5至1e-2、dropout 0至.5、weight_decay 0或1e-6至1e-3。XGBoost扩展深度2/3/4/5/6/8、树数150/300/600/1000（锚点含100）、lr .001至.3（锚点含.03/.1）、min_child_weight1至100、lambda .01至100、alpha0或1e-5至10、行列采样.5至1。MLP仍Adam/batch256/最多100epochs/minimum-train-loss checkpoint；树模型hist/CPU4线程/固定候选树数，不加类别权重。
- 配对对照是用direct-log选定的同一参数，另训旧log1p(S/tau)输入；tau仅来自完整3200训练mentions，不独立优化scaled。另用原[128,64,32]/dropout.3/lr.001/wd1e-5训练direct-log，与旧固定scaled头对照，以分开变换与调参因素。全部仅探索性旧800图结果，不做bootstrap、不启动独立2000图、保留旧数值FAIL及InternVL旧CPU复核状态。
- 文献已查主文和作者配置：Gorishniy等NeurIPS2021，验证集选参、100次MLP配置搜索、lr/宽度/dropout/weight_decay、多seed评价；Grinsztajn等NeurIPS2022约400次随机搜索并纳入默认配置；Bergstra/Bengio JMLR2012随机搜索基准。URL封存到protocol/新汇总报告。本轮48而非100/400；保留原Adam、BatchNorm、训练损失checkpoint，不照搬论文AdamW、quantile预处理或验证早停，不声称完整复现论文。
- 新产物单独写各模型v2 `production_k4/ae_direct_log1p_search_20260908/`；总汇总为 `outputs/ffn_visual_source_consistency_v2/ae_direct_log1p_search_20260908/`。新单测检查直接log无tau、图片隔离、48候选/排名、三类真实小训练/weight_decay生效/同设备恢复及错误指纹拒绝、648搜索和84最终任务顺序。实际测试和启动状态随后记录；不把脚本完成写成实验完成。不提交、不上传。
- 实际新增4/4测试PASS，8.354秒（不含环境导入），命令 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_ae_log1p_search`，日志 `/tmp/gfchair_direct_log_tests_20260908.log`。py_compile与git diff --check通过；另起含旧相关套件的完整回归，日志 `/tmp/gfchair_direct_log_regression_20260908.log`，状态待检查。
- 09:27:52 UTC（服务器CST显示17:27:52）启动两路screen：`420993.gfchair-log1p-qwen2-llava-20260908`（worker420994）及 `421004.gfchair-log1p-qwen3-intern-20260908`（worker421005）。命令前缀 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/train_ffn_ae_log1p_search.py`，分别传 `--models qwen2_5_vl_7b llava_1_5_7b --device cuda:0`、`--models qwen3_vl_8b internvl_2_5_8b --device cuda:1`；日志 `outputs/ae_direct_log1p_qwen2_llava_20260908.log` 与 `outputs/ae_direct_log1p_qwen3_intern_20260908.log`。启动时两卡空闲，无其他VLM任务；先导入环境和验证已有特征，后训练，尚不能报告新检测成绩。
- 首次启动Qwen2/Qwen3均在首头优化前失败：`torch.cuda.reset_peak_memory_stats(device)`不会自动初始化CUDA allocator，报`RuntimeError: Invalid device argument / did you call init?`。两路screen已退出，检查均0个result.pt、1个failure记录；不是训练数值失败。修复为统计前创建该设备的零元素tensor触发初始化，并新增双GPU实际小训练/reload测试。两个首次协议/失败目录完整移到同级 `ae_direct_log1p_search_20260908_startup_failed_cuda_init/`；未删除或改写旧结果。首轮日志原样保留，重启使用新日志名，修复后新协议绑定新源码SHA。
- 首次完整回归65/65 PASS，19.102秒（不含导入），但当时没有GPU路径测试，因此未覆盖上述初始化错误；不把这次CPU测试通过当作GPU已验证。修复后的新增测试/完整回归另行记录。
- 修复后5/5新增测试PASS（8.680秒），明确在cuda:0和cuda:1各做真实训练、峰值显存统计及同设备reload，概率最大差均0。完整相关回归66/66 PASS（19.597秒），日志 `/tmp/gfchair_direct_log_regression_cuda_20260908.log`。09:30:27 UTC重启screen `421533.gfchair-log1p-qwen2-llava-r1-20260908`（worker421535）与 `421536.gfchair-log1p-qwen3-intern-r1-20260908`（worker421538）；训练命令不变，新日志为 `outputs/ae_direct_log1p_qwen2_llava_r1_20260908.log` 和 `outputs/ae_direct_log1p_qwen3_intern_r1_20260908.log`。09:31已观察到Qwen2首个真实搜索结果，修复后训练开始；尚未完成全部搜索。
- 09:59 UTC进度：Qwen2/Qwen3各162搜索+21最终头完成，均有summary/verification且无修复后failure；GPU0已转LLaVA（92/162搜索），GPU1已转InternVL（44/162搜索）。仅两个模型完成，不把四模型阶段写成完成。
- 已完成的Qwen2直接log单隐藏/三隐藏/XGBoost ensemble AUROC分别87.867/87.347/85.613%，HALL-AUPR44.951/45.566/43.320%；原固定三隐藏scaled基线88.198/47.011%。固定旧三隐藏参数只改direct-log为88.214/45.774%，AUROC几乎不变但HALL-AUPR下降。Qwen3直接log单隐藏/三隐藏/XGBoost为88.927/89.755/89.025% AUROC、65.089/64.433/62.975% HALL-AUPR；其同参数新三隐藏scaled为90.214/66.214%。均不反向修改已选参数。
- 独立Python进程已从原输入重建Qwen2/Qwen3直接log矩阵指纹并验证图片隔离；训练集raw S范围分别[.0590083,305.8301]/[.00616821,168.4061]，log1p后[.0573329,5.726294]/[.00614926,5.132299]。另从原标签+保存概率独立复算两个模型共1464份双阈值指标、366个头的候选配置/前三/多seed选型/全部ensemble与mean/std、1224个artifact SHA，均PASS，94.094秒；核对文件时序为所有搜索先于selection、selection先于最终result。训练保存时同设备checkpoint复算记录最大误差均0；独立指标进程没有冒充又做全量权重推理或独立新图确认。
- 四模型全部完成：576初筛+72追加seed复核=648搜索头；84最终头；总732记录齐全。两路screen正常退出，10:12:43.790 UTC生成总报告，修复后启动至汇总2536.790秒（42分17秒）。Qwen2/LLaVA/Qwen3/InternVL主流程分别832.574/1430.475/1406.675/1124.566秒，含输入读取与逐头校验，不含最初模块导入；两路并行不能把四者相加作墙钟时间。相应纯fit累计777.382/1346.864/1326.085/1062.305秒。PyTorch峰值allocated分别22.121/22.175/22.230/22.175 MiB，不含CUDA上下文和缓存；运行期间nvidia-smi约495 MiB/卡，退出后两卡0MiB/0%。
- LLaVA的旧固定scaled/固定direct/调参单隐藏direct/调参三隐藏direct/XGBoost direct分别为AUROC 90.646/90.496/89.813/90.258/89.583%，HALL-AUPR72.298/71.928/70.325/71.644/69.409%。InternVL对应AUROC86.388/86.381/85.230/86.381/85.267%，HALL-AUPR55.221/54.615/52.926/54.615/51.026%；InternVL三隐藏选回原配置，因此固定direct与tuned direct重合。所有精确配置与逐seed指标保留，不按这些测试成绩改选型。
- 全量独立验收完成：2448个artifact SHA、732个头、2928份双阈值报告、全部候选/多seed选型、seed均值std/ensemble均PASS，最大同设备checkpoint复算记录误差0。Qwen2+Qwen3/LLaVA/InternVL独立指标进程耗时94.094/62.142/42.388秒；总summary与四模型来源逐项一致，SHA256为 `a09f8dfa588ad30ee5ae4e67ddda22adc9b791dc8725baec8151fb985e405244`。总目录新增 `independent_metrics_audit.json`，明确独立指标核验与训练保存时checkpoint推理复算的区别。
- 总目录 `outputs/ffn_visual_source_consistency_v2/ae_direct_log1p_search_20260908/` 已有summary.md/summary.json/independent_metrics_audit.json；各模型有protocol、三类shortlist、selection、162搜索+21最终result、checkpoint和verification。主报告§5.19补齐四模型结果、同参数scaled对照、参数选择及论文实践。结论：原参数只改direct-log时四模型HALL-AUPR均下降；48组专门调参亦没有使direct-log的三类头超过原固定三隐藏scaled基线的两项ensemble指标。本轮更支持保留原log1p(S/tau)，但只报告探索性点估计，不宣称显著性或全局最优。
- 旧实验、数值gate和CPU失败状态均未改写；首次启动两份失败与协议已完整保留，修复后0失败。66项相关单测、最终py_compile、git diff --check通过；不提交、不上传，不做bootstrap或独立2000图确认，不自动替换生产特征。

## 2026-09-08：单隐藏层 MLP 与 XGBoost 小规模调参（完成）

- 用户明确“bootstrap”指XGBoost/boosting，且“单层MLP”指一个隐藏层。按Ponytail新增薄脚本 `scripts/train_ffn_consistency_alternative_heads.py`，复用现有v2紧凑特征、共同尺度、16组拼接、DGSTStyleProbe/训练器、阈值与汇总函数。环境已有XGBoost2.1.4/sklearn1.6.1/torch2.8，无新依赖、VLM forward、bootstrap CI或独立2000图实验。全部旧代码和旧实验不覆盖、不提交、不上传。
- 固定U_SN调参基准，四模型各进行12个单隐藏层MLP候选及12个XGBoost候选，搜索seed43。原3200训练图片按seed20260908固定划分2560/640内部训练/验证，同图所有mentions只在一侧；内部S/I尺度仅拟合2560图对应mentions。内部验证AUROC选参，精确平局按HALL-AUPR再按候选顺序。两类搜索均封存到selection.json后，才开始原800测试图评估；测试指标不进入选择函数。
- MLP候选为hidden[32,64,128]×dropout[0,.3]×lr[3e-4,1e-3]，保留BatchNorm和既有Adam/lr调度、weight_decay1e-5、batch256、最多100epochs/minimum-train-loss checkpoint。XGBoost为max_depth[2,3,5]×lr[.03,.1]×trees[100,300]，hist/CPU4线程、subsample=.8、colsample_bytree=.8、min_child_weight5、lambda1、alpha0，无类别加权。参数选择后按每模型/分类器一套参数迁移全部16组，并在原3200图上按seeds43/44/45重新训练；最终S/N尺度仍为原完整训练集尺度。
- 本轮新增96个搜索头、384个最终头，复用192个旧三隐藏层固定头对照。新头调过参、旧头固定参数，不能把成绩差全部解释为隐藏层数量影响；U_SN上选参也不等于每个特征组都单独最优。新头在原计算设备/FP32/固定batch256复核训练和测试概率（绝对误差≤1e-7），保留旧InternVL CPU复核FAIL和原K4数值FAIL例外，不擅自改变旧gate。
- 新增5个测试覆盖12候选、图片隔离、内层尺度无验证/测试泄漏、只按内部验证选参、两类真实小训练的checkpoint/指标复核及恢复不重训/指纹拒绝、两类搜索先封存再评估和16组调度。命令 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_consistency_alternative_heads`，日志 `/tmp/gfchair_alternative_heads_tests_20260908.log`，5/5 PASS（2.201秒，不含环境导入）；py_compile和git diff --check通过。
- 独立产物根为各模型v2 `production_k4/head_search_20260908/`，总体表/JSON/Markdown位于 `outputs/ffn_visual_source_consistency_v2/head_search_20260908/`。自动resume验证内容/指纹/checksum，保留不完整attempt；已完成head和汇总不重写。正式启动与完成状态在后续条目追加，不把测试通过当作真实实验完成。
- 08:36:59 UTC（服务器CST显示16:36:59）实际启动两路screen，PPID均为1：`412240.gfchair-heads-qwen2-llava-20260908`（worker412241，cuda:0）和 `412249.gfchair-heads-qwen3-intern-20260908`（worker412250，cuda:1）。命令前缀均为 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/train_ffn_consistency_alternative_heads.py`，分别加 `--models qwen2_5_vl_7b llava_1_5_7b --device cuda:0` 与 `--models qwen3_vl_8b internvl_2_5_8b --device cuda:1`。日志为 `outputs/alternative_heads_qwen2_llava_20260908.log` 和 `outputs/alternative_heads_qwen3_intern_20260908.log`。启动后首先读取旧特征并检查输入指纹，不能将环境/特征加载等待写成已完成训练。
- 全部96搜索头及384最终头完成，两路screen正常退出，无failure文件。四模型内部图片划分完全相同且2560/640/800互斥。总报告于09:04:34 UTC生成，启动至汇总约27分35秒；逐模型主流程Qwen2/LLaVA/Qwen3/InternVL耗时568.095/945.886/913.789/715.783秒（含读取、训练及逐头复核，不含首次环境导入；两路并行，不能相加当总墙钟）。
- 选定单隐藏层MLP为：Qwen2 hidden64/dropout0/lr3e-4；LLaVA hidden128/dropout.3/lr3e-4；Qwen3 hidden128/dropout.3/lr1e-3；InternVL hidden64/dropout.3/lr1e-3。XGBoost四模型均depth5/trees300；Qwen2/LLaVA lr.03，Qwen3/InternVL lr.1。所有参数在内部验证上选定，未因最终测试成绩重新修改；候选范围和转移到其他组的局限完整记录。
- U_SN的ensemble AUROC，旧三隐藏层/新单隐藏层/XGBoost分别为：Qwen2 .891296/.890580/.871018；LLaVA .904150/.904249/.899342；Qwen3 .906117/.901468/.895458；InternVL .882044/.875344/.860684。单隐藏层在Qwen2 H_SN提高到.896602（旧.880095），Qwen3 H_SK提高到.912141（旧.905339），但在64个同特征组对照中仅16个AUROC高于旧头；XGBoost仅1/64个提高，部分HALL-AUPR仍有正增量。不据此宣称浅层或树模型普遍更好，也不把调参收益完全归因于深度。
- 新增+相关回归61/61 PASS，11.375秒（不含导入），日志 `/tmp/gfchair_alternative_heads_regression_20260908.log`。所有384个最终头在保存后同设备/同batch重新加载复算，训练/测试最大概率差为0；真实小模型的resume不重训/不改SHA及mtime已由新增单测覆盖。
- 另起独立Python进程复核1440个产物SHA、96条搜索配置/选型、384个最终头的1536份训练/测试×双阈值报告、seed/ensemble汇总及384行总CSV，全部PASS。核验记录为总目录 `independent_metrics_audit.json`；该独立进程从保存概率重算指标并核对已有checkpoint复算记录，不冒充又做了一次全量特征/权重推理或新图独立确认。诊断耗时289.542秒包含等待总汇总生成。
- 总目录保存 `summary.md`、`summary.json`、`groups.csv`，各模型保存protocol、selection、全部search/final checkpoint及verification。主报告新增§5.18，旧K4数值FAIL、旧InternVL严格CPU复核状态和全部旧产物保持不变；未提交或上传。

## 2026-09-08：InternVL checkpoint 复算超差只读诊断（完成，未修改验收规则）

- 当前192/192头均训练完成；Qwen2.5/LLaVA/Qwen3各48头独立验收PASS。InternVL原验收因CPU复算概率超过绝对阈值1e-5而退出，跨模型联合汇总未执行。本次用户仅要求查明“差多少、为什么”，不重训、不覆盖预测/checkpoint、不放宽门槛、不改原失败状态，也不启动独立2000图确认。
- 按Ponytail复用现有函数，在 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u -i` 内调用 `load_training_data('internvl_2_5_8b','fp32_k4',True)` 与 `checkpoint_probabilities()`，扫描全部48头的训练/测试预测；数据加载56.02秒，加载加CPU概率扫描58.64秒（不含初次环境导入）。当前特征、mention顺序、标签及协议重建的训练fingerprint与原manifest完全相同，全部48头artifact SHA256通过。
- 共复算564432个概率（48×[9378训练mentions+2381测试mentions]），仅1个超过1e-5：`H_SK/seed44/train`，行2847，image183364，mention `183364:2`，target `183364:101`，词keyboard，label0。保存的REAL概率为 `.6065546870231628`，CPU复算为 `.6065648198127747`，绝对差 `1.0132789611816406e-5`，比门槛高约1.328%；该头训练集平均绝对差 `1.0486591763645061e-7`。所有头测试集最大概率差 `6.318092346191406e-6`，没有测试概率超差。
- 同一故障checkpoint、同一特征的隔离对照：CPU FP32改回原固定batch256仍出现完全相同的最大差；原cuda:1 FP32+固定batch256复算训练/测试均与保存概率逐元素相同（最大差0）；GPU改为CPU复核用的均分batch，最大差仅 `1.7881393432617188e-7`，故障样本仍完全复现。说明这次超差主要来自跨CPU/GPU FP32运算，不是批次划分或权重/特征改变。所有模型均eval，未启用AMP；matmul TF32为False。
- 同权重/输入临时CPU FP64参考（仅内存转换，无文件改动）：该样本概率 `.6065573237977084`；相对保存GPU值差 `2.6367745455946334e-6`。逐层追踪该样本，首Linear的CPU/GPU最大分量差 `6.556510925292969e-7`，首BatchNorm后 `5.170702934265137e-6`，末logit差 `4.26173210144043e-5`；经过sigmoid得到上述概率差。FP64仅为诊断参考，不替换原预测。
- 48头测试集按原阈值复算的分类翻转均为0，REAL/HALL P/R/F1保持不变；跨全部头测试AUROC最大绝对变化 `1.3351420057317043e-6`，REAL-AUPR `7.974826073953167e-7`，HALL-AUPR `3.916974185669275e-6`。故障头测试AUROC仍为 `.8624363137263279`、HALL-AUPR仍为 `.5432015915187369`，REAL-AUPR只差 `1.2324496201365776e-8`，不声称全部指标逐位相同。
- 另外21个头在固定保存的train-F1阈值下各有1个训练样本翻转，均不是测试翻转；故障头对应训练行5705恰等于原阈值 `.7825173735618591`，CPU概率为 `.7825140953063965`，不是行2847的超差样本。重新用CPU训练概率选阈值时该头得到后一个值，测试分类仍不变。诊断未更新任何阈值。下一步如需继续验收，应显式区分同设备重现与跨设备数值一致性，不能直接把原CPU FAIL标成PASS。

## 2026-09-08：保留原 K4，按用户授权继续探索性训练（执行中）

- 用户在获知两个K4闭合误差尾部案例及K64诊断结果后要求“就先这样吧，先训练把”。本次明确按保留现有K4特征执行：不替换两个case、不删除case、不做自适应K64、不改变原数值阈值，不把 `FAIL_NUMERICAL_OLD_COHORT` 改成PASS；先运行旧cohort训练及验证，暂不执行独立2000图确认。
- 按Ponytail复用现有16组拼接、train-only共同尺度、MLP及恢复实现，仅新增显式 `--train-despite-known-failure` 和 `train-only` 队列。例外只允许当前已诊断的Qwen2.5第23层最大误差 `.010615984949452763`；绑定原失败gate的SHA256，其他模型/层失败、缺失cohort或指纹改变均拒绝。没有取消源shard checksum、mention对齐、finite、split、checkpoint复算或训练阈值检查。
- 探索性例外记录单独保存到 `outputs/ffn_visual_source_consistency_v2/exploratory_training_20260908/numerical_exception.json`；四模型产物分别写入原v2 `production_k4/exploratory_training_20260908/training/fp32_k4/`，与正式训练目录分开。manifest、训练指纹和summary显式携带原数值FAIL及用户例外来源；跨模型汇总也注明探索性质。正常入口仍要求原数值PASS，例外入口禁止freeze/generate/evaluate独立确认。
- 原4000图、3200/800划分、16组、seeds43/44/45、MLP[128,64,32]、dropout.3、batch256、最多100epochs、minimum-train-loss checkpoint和train-REAL-F1阈值不变；S和N_vec共用训练集逐层tau的log1p，I单独训练中位数；不新增bootstrap、标准化、加权或重采样。
- 新增例外保留FAIL/拒绝指纹变化测试与仅训练队列禁止提取/封存/独立生成测试；相关完整56/56 PASS（9.934秒，不含导入），日志 `/tmp/gfchair_exploratory_training_tests_final_20260908.log`。合成48头训练/CPU复核/resume测试同时覆盖例外元数据；`py_compile`、`git diff --check` PASS，四模型原 `extract_old()` 源码指纹逐一匹配旧manifest。另对新目录自动创建的训练队列做定向检查，日志 `/tmp/gfchair_training_queue_check_20260908.log`。
- 已启动独立screen：`screen -dmS gfchair-train-k4-exploratory-20260908 bash -lc 'cd /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair && exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_consistency.py train-only --production-after-subset --train-despite-known-failure --resume > outputs/consistency_v2_exploratory_training_20260908.log 2>&1'`。会话394084，协调进程394085；GPU0按Qwen2→LLaVA，GPU1按Qwen3→InternVL，各模型48头后独立CPU复核，全部完成再汇总，不自动进入独立确认。当前启动/运行不代表192头已完成。
- 实际worker为394200（Qwen2，cuda:0）和394201（Qwen3，cuda:1），workflow为 `outputs/ffn_visual_source_consistency_v2/workflow_runs/1788850053990256501/`。两个模型均已通过特征读取/校验，写入带数值例外的训练manifest并开始epoch优化；首次检查Qwen2已有2个完成头、Qwen3正在第一个头的后期epoch。MLP每卡显存约495MiB，低于VLM提取时是正常现象。新目录自动创建的队列定向测试也PASS（1项，0.012秒）；原gate仍为FAIL，未启动任何新VLM任务。

## 2026-09-08：全量 K4 验收停止，定点复核两个 Qwen2.5 案例

- 四模型旧4000图K4-only提取全部完成，均有4000个shard和对应sidecar。Qwen3/LLaVA/InternVL全量数值验收PASS；Qwen2.5第23层失败，四模型总门控为 `FAIL_NUMERICAL_OLD_COHORT`。队列于2026-09-07 20:39:41 UTC按预设上限自动停止，screen已退出，训练0/192、独立确认未开始，禁止把子集PASS或提取完成写成全量PASS。
- 逐条只读复核Qwen2.5全部242312个target-layer，发现且仅发现两个闭合相对误差超过1%的case：`248069:17:23 = .010615984949452763`，`546325:30:23 = .010395158690002427`。它们的κ_vec、finite和差异界检查均正常，没有OOM/缺失shard导致的失败。原门控文件为 `outputs/ffn_visual_source_consistency_v2/old_validation_fp32_k4.json`。
- 用户同意只对上述两case做局部FP32 K4/K64对照。按Ponytail新增薄诊断入口 `scripts/diagnose_ffn_visual_source_tail.py`，复用未修改的生产 `_extract_one_image()`、完整caption因果query-row捕获、local_fp32、streaming及原OOM分块回退；只在第23层额外计算K64，不切换为截断prefix审计，不重跑4000图、不调整门槛、不开始训练。
- 诊断先检查生产核心函数/依赖/输入checksum；保持原目标batch计算，在同次捕获的同一z/writes上比较K4/K64。保存两case的完整分量、component_sum、端点、S/N_vec/κ_vec、分布差异及原K4重现检查。独立输出为模型v2根下 `diagnostics/tail_k4_k64_20260908/`，原shard和失败gate的checksum/mtime不得改变。
- 新增snapshot/重现判断的合成检查；运行相关完整单测命令 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_visual_source_consistency tests.test_ffn_visual_source_study tests.test_ffn_visual_path_attribution`，日志 `/tmp/gfchair_tail_diagnostic_tests_20260908.log`。诊断代码的 `py_compile` 与当前 `git diff --check` 已通过，实际运行结果待后续记录。
- 实际运行已完成：`screen -dmS gfchair-tail-20260908 bash -lc 'cd /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair && exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/diagnose_ffn_visual_source_tail.py --device cuda:0 > outputs/consistency_v2_tail_diagnostic_20260908.log 2>&1'`。只加载一个Qwen2.5模型，顺序复核两图；screen已随正常完成退出。上述单测最终54/54 PASS，11.976秒（不含导入）。
- 两个case的S、N_end、N_vec、κ_vec、P_FFN及闭合相对误差均完全复现原K4，重现差异为0；同一case的K4/K64端点向量逐元素相同。K64闭合相对误差分别降至 `5.314491046890357e-7 / 7.064114137168922e-7`，绝对误差由 `.7252963/.6163259` 降至 `3.630921e-5/4.188293e-5`。支持这两个异常主要来自K4求积不足，不是输入改变或局部FP32未启用；K64仍是数值参考，不称真值。
- 相对K64，K4的S差异为 `.3515%/.3818%`，N_vec为 `.5274%/.5656%`，κ_vec绝对差为 `.00117472/.00114873`；P_FFN Spearman为 `.999999708/.999999599`，Top32 overlap均1。单图复核耗时9.412/5.376秒，不含模型加载；最大allocated显存16.83 GiB，chunk256，无OOM重试或诊断失败。
- 额外CPU验收直接从保存的完整分量重算component_sum/S/N_vec，component_sum相对误差≤1.14e-7、S相对差≤1.61e-7（FP32归约顺序），全部通过。禁止模型加载后调用同一main做resume，完成产物checksum/mtime均未改写；验收记录为诊断目录 `verification.json`。原生产shard、manifest及失败gate的checksum/mtime亦保持不变。
- 主报告§5.16已新增日期化复核结果，诊断 `summary.json` 状态为 `COMPLETE_TARGETED_DIAGNOSTIC`。这仅代表两case诊断完成，正式总gate仍FAIL，未替换生产特征、未开始训练、未擅自选择自适应K或全量K64；继续实验需要用户确认数值复算协议。未提交、未上传，保留全部无关改动。

## 2026-09-07 15:45 UTC：v1 P_FFN/P_WRITE、S_FFN/S_WRITE 与 R_amp 对照（完成）

- 按用户确认的方案只读取四模型 v1 正式 COCO4000 compact shard，不占 GPU、不重跑 VLM、不训练检测器或做 bootstrap；正在执行的 v2 screen/GPU 流程未修改。新增独立 `scripts/analyze_ffn_write_comparison.py`，逐 shard 释放原始张量；`S_WRITE=sum_m||a_m||`（旧 I，不是 `||sum_m a_m||`），`S_FFN=sum_m||e_m||`（gross_strength），`R_amp=0.5*sum_m|P_FFN-P_WRITE|`（TV）。
- 四模型按 mention 等权、REAL/HALL 分开的跨层中位数平均 `R_amp` 为：Qwen2 `.04835/.04609`、LLaVA `.05082/.04886`、Qwen3 `.05435/.05324`、InternVL `.04712/.04589`。HALL 中位数更高仅 `8/28、10/32、14/36、10/32` 层，不支持幻觉有更强 FFN 空间重分配；但末层 REAL/HALL 达 `.1056/.0985、.2402/.2479、.3970/.4008、.1187/.1378`，不能外推成所有层近似相同。
- P 的 Spearman 跨层中位数平均为 `.9836–.9950`、Top32 overlap `.9375–.9590`，确认大部分空间排序继承 WRITE。Top1 agreement 的跨层类内 rate 为 REAL `.8657/.8750/.8720/.8781`、HALL `.8858/.8947/.8838/.8935`（模型顺序同上）；HALL 的最高点反而略稳定，但不是更高全排序相似度。
- `S_WRITE–S_FFN` 每层全 mention Spearman 跨层平均为 Qwen2/LLaVA/Qwen3/InternVL `.9614/.9452/.9633/.9753`，log-Pearson为 `.9707/.9536/.9683/.9813`。gross gain 的 REAL/HALL 跨层中位数平均为 `.63525/.63177、.52094/.51418、.65421/.63776、.56228/.55701`；FFN response 主要随 WRITE magnitude 变化，但层间比例明显变化，不能解释成 Jacobian≈I。
- 总产物为 `outputs/ffn_write_ffn_comparison_summary.json` 及两张四模型总图；各模型 v1 根下保存完整逐层 CSV、相关表、分布/强度/固定差值图和 JSON。覆盖每模型4000图、合计50812 mentions；归一化最大误差2.45e-7，signed差值和最大2.02e-16，TV正负质量恒等误差2.23e-16，无越界或零WRITE强度。新JS/S_FFN与既有曲线最大差2.20e-7/2.71e-6。
- 新手算测试覆盖TV、正负迁移质量、Spearman、Top-K、非法输入及二元Top1使用均值汇总；新增3/3、相关完整23/23 PASS，`py_compile`通过。产物已目视检查；结果写入主报告§5.17。遵循Ponytail复用既有正式shard和依赖，未改旧分析脚本或增加依赖；未提交、未上传。

## 2026-09-07 13:01 UTC：续跑双卡 LLaVA，改用 screen 托管

- 用户再次要求“继续2卡提取”。实时检查发现上一组交互长任务的协调进程698397及worker698430/698431均已退出，两卡显存为0；实际已完成372/500张。此前没有持续运行到本次检查时刻，不能沿用旧RUNNING状态报告进度。现有日志没有数值失败或Python traceback，退出原因未从日志确定；原workflow另存 `INTERRUPTED_NO_LIVE_PROCESSES_RESUME_IN_SCREEN`，不伪造正常完成。
- 使用机器上已有的 `screen` 托管，未安装依赖：`screen -dmS gfchair-llava-dual-20260907 bash -lc 'cd /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair && exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_consistency.py llava-dual-subset --resume > outputs/consistency_v2_llava_dual_screen_20260907.log 2>&1'`。
- 已确认screen会话 `215712.gfchair-llava-dual-20260907` 为Detached、父进程为PID1，协调进程215713。继续原500图清单的rank0/1两份分片，所有完成shard保留；当前数值提取核心 `extract_old()` 的SHA256与四模型旧manifest逐一核对，全部一致。仅任务托管方式改变，没有更换计算方法。
- 13:06:47 UTC复核：cuda:0/1 worker为216031/216032，GPU利用率95%/89%，显存15913/15555 MiB；两条rank分别新增完整图片50811/50746，总计374/500，确认实际恢复产出。新workflow为 `outputs/ffn_visual_source_consistency_v2/workflow_runs/1788786186057847579/`。本次环境/权重冷读取较慢，不能把启动等待误记为GPU提取时间。后续检查先核对这个screen及PID，禁止再与它并行启动重叠任务。
- 数值验收未完成前不宣称LLaVA PASS；其余三模型已验收结果保持。双卡500图完成后汇总四模型gate，通过后才按既定授权继续4000图K4-only，未授权全量K64重启。

## 2026-09-07 12:33 UTC：LLaVA 剩余数值子集改为双 GPU（执行中）

- 用户要求两卡同时完成当前 LLaVA 提取并回报验证结果。先停止旧子集协调进程 644148，再计划中断单卡 LLaVA 进程 644360；停止后核实两卡空闲。已完成 367/500 张全部保留，剩余133张按固定清单 `rank/world_size=0/2、1/2` 分成66/67张，无结果筛选、重复分配或第501张计算。
- 按 Ponytail 只扩展既有分片边界与队列，未改冻结的 `extract_old()` 数值实现及依赖指纹。双rank分别写完成状态，之后汇总核验完整500张；新增双rank恢复/无重复/边界单测，相关完整测试53/53 PASS（9.739秒，不含导入），`py_compile` 和 `git diff --check` PASS，日志 `/tmp/gfchair_consistency_dual_tests_20260907.log`。
- 启动命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_consistency.py llava-dual-subset --resume > outputs/consistency_v2_llava_dual_subset_20260907.log 2>&1`；长任务session57629，协调进程698397，cuda:0/1 worker为698430/698431，workflow为 `outputs/ffn_visual_source_consistency_v2/workflow_runs/1788784409103190401/`。原workflow保留过程记录并新增 `CANCELLED_FOR_USER_DUAL_GPU_RESHARD` 说明，不把计划中断算作数值失败。
- Qwen2.5 1007图、Qwen3 873图以及InternVL 500图已经分别完成子集数值验收并PASS。InternVL共有1469个唯一目标、47008个target-layer，闭合相对误差P90 `4.062403e-5`、最大 `.003571673`；S/N_vec相对FP32 K64差异P90低于1.15e-7，所有层及κ边界通过。LLaVA尚待补齐并验收，不能提前宣称四模型子集gate已通过。
- 当前双卡仍计算局部FP32 K4特征及额外FP32 K64参考；后续K4-only仅取消额外K64，不改当前K4特征公式/精度。它与v1的差异是局部FP32数值路径、真实 `component_sum` 的N_vec/κ_vec及并列端点诊断，并非整模型FP32。4000图和3200/800划分不变，所有子集PASS后才接续获准的K4-only队列。
- 尚未完成完整4000图K4-only、192个训练头或独立2000图确认。未提交、未上传、未运行bootstrap。

## 2026-09-07 范围修订：子集数值验证后，4000图仅计算K4（执行中）

- 用户明确要求Qwen系列先用已完成图片验证，LLaVA/InternVL各500图；随后确认“仅缩减数值验证，验证通过后4000图只算K4，仍用原3200/800训练划分”。不缩减训练cohort，不拿子集训练代替原实验；原16组×3seeds×4模型、train-only共同尺度、独立2000图确认和不做bootstrap均保持。
- 先按精确PID停止旧全量队列560825，再向两条Qwen提取任务556303/556313发送SIGINT；两卡显存已释放。日志中的KeyboardInterrupt是用户范围变更导致的计划中断，不是数值失败。已完成shard和sidecar全部保留，无孤立半成品或删除；旧workflow新增 `CANCELLED_BY_USER_SCOPE_AMENDMENT` 状态说明，原过程记录不改写。
- 子集清单已封存到 `outputs/ffn_visual_source_consistency_v2/numerical_subset_20260907/selection.json`：Qwen2.5为1007图，Qwen3为873图，均为停止时完整落盘的全部图；LLaVA/InternVL为原定image-ID处理顺序前500图（含已有smoke），不依据标签或结果筛选。不宣称这个处理顺序前缀是额外随机抽样。
- 按Ponytail复用原数值计算、逐图保存、`gate_group`及训练代码，不新增依赖；保持被冻结的 `extract_old()` 数值函数源码与原指纹不变。新增薄子集边界适配，在请求集合结束后停止，不计算第501图；Qwen仅读取已有结果，不再新增JVP。
- 所有子集通过后，完整4000图仍逐条检查闭合误差、κ边界、finite及轨迹对齐；取消尚未处理图片逐条K64参考。不以缩减后的子集报告冒充原定4000图K64逐条验收。
- 新正式产物放在各模型v2根下的 `production_k4/`，保留原 `old/fp32_k4/` K4+K64参考产物。已有K4特征复用到新namespace时逐文件保存父路径、原checksum/fingerprint与 `original_included_k64_reference=true`，不冒充原先没计算过K64；源文件不改写。仅剩余图片运行新K4提取。
- 已启动修订队列命令 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_consistency.py subset-pipeline --resume > outputs/consistency_v2_subset_pipeline_20260907.log 2>&1`（长任务session43631）。先完成子集数值验收，再进入带 `--production-after-subset` 的K4-only全量队列；原全量K64 CLI在检测到修订清单后拒绝意外重启。任何子集或完整K4数值失败均停止后续训练，不擅自升级为全量K64。
- Qwen子集数值验收已完成且通过：Qwen2.5为1007 processed images、2135唯一目标、2148 mentions（59780个target-layer）；Qwen3为873图、3138唯一目标、3171 mentions（112968个target-layer）。全部层及整体 `gate_group` 均PASS，κ_vec范围检查全部通过，无case遗漏或源文件checksum/mtime改变。
- Qwen2.5/Qwen3闭合相对误差P90为 `1.061384e-5 / 3.139963e-5`，最大为 `.003739372 / .001384320`；相对FP32 K64的median Spearman与Top32 overlap均1，median JS约 `3.99e-16 / 4.89e-16`，S/N_vec相对差P90均低于1.24e-7，κ_vec绝对差P90低于6.20e-8。完整逐层及最差case结果见子目录 `{model}_validation.json`，这是数值子集验证，不是检测成绩。
- 修订队列已运行到LLaVA cuda:0、InternVL cuda:1的500图任务，真实VLM PID644360/644391；父队列PID644148，workflow为 `workflow_runs/1788763847833252110/`。两模型各最多一个VLM；Qwen没有继续增加图片。
- 新增集合边界停止/恢复测试和K4复用父指纹测试均通过；相关完整测试52/52 PASS，耗时10.317秒（不含环境导入），日志 `outputs/consistency_v2_subset_tests_final_20260907.log`，`py_compile`、`git diff --check` PASS。临时复用测试仅复制synthetic/指定smoke产物，不据此写正式全cohort完成标记。
- Qwen两模型各用真实已完成smoke文件做production复用预检，在临时目录创建带父指纹的副本，再禁止模型加载调用原 `extract_old()`；K4-only manifest指纹与原计算函数完全匹配，完成副本checksum/mtime不变。预检PASS记录为 `numerical_subset_20260907/qwen_production_reuse_preflight.json`，未提前启动正式生产或训练。
- 外层500图边界复用原提取函数，因此内层日志的 `pending=3999` 仍描述原4000图manifest，而不是新执行上限；真实上限来自不可变子集清单，边界测试验证不会计算第501图，子集完成后写独立500图状态。当前最后一次检查LLaVA/InternVL为6/13个完整shard，两卡正常忙碌，Qwen固定1007/873未增加。
- 当前不是整轮实验完成：LLaVA/InternVL子集、完整4000图K4-only提取、192个探针和独立2000图仍待执行/验收。本节修订优先于下面保留的旧全量K64执行过程记录；未提交或上传。

## 2026-09-07 第七点：双净量、联合尺度与独立确认（执行中）

- 起点为 `9b00af17`，保留此前 AE32/cosine 未提交代码、报告、测试与全部 v1 artifacts；不提交、不上传。按 Ponytail 复用 capture、streaming JVP、OT 和 trainer，不新增依赖，不改旧分析脚本的 checksum。
- 严格先数值门控：固定 seed20260907 的异常/匹配正常及20图全层样本，同次捕获比较 native K4/K64 与局部 FP32 K4/K64。从真实 `component_sum` 保存 N_vec，保留 N_end、两种 kappa 及闭合误差；不裁剪、不从旧误差反解。
- 四模型共同候选顺序 native K4 → 局部 FP32 K4 → 局部 FP32 K64，按用户阈值冻结；最高候选不通过则停止训练和独立确认，明确报告数值阶段未完成，不放宽阈值。
- 通过后才补算旧4000图、按同一训练尺度构造16组×3seeds×4模型，再封存后运行共享新2000图的独立评估。不做 bootstrap、AE32/cosine、JS 扩展、Shapley 或因果重跑；融合容差固定 AUROC .005 / HALL-AUPR .01，不作显著性或统计非劣声称。
- 初始只读检查：两张 RTX4090 各24GB且空闲；每卡最多一个 VLM。正式 Python 为 `/opt/conda/private/envs/vicr/bin/python`。代码和结果统一使用独立 `ffn_visual_source_consistency_v2` 命名，随后记录实际命令、耗时、显存及失败。
- 已新增双净量审计 CLI、数值门控/共同尺度/16组薄适配代码，以及独立测试。最初8项新测试通过，与既有路径和研究测试合计41项通过；随后增加 cohort ID/checksum 隔离、融合全模型双容差和 checkpoint 复算测试，新测试11项通过。`py_compile` 和 `git diff --check` 通过。
- 冻结审计清单：Qwen2/Qwen3旧唯一异常数28/2421，分别选28/100个异常及等量正常，加20图全层，最终616/920个去重case；LLaVA旧唯一异常0，InternVL为76（清单均已保存）。原50-case完整向量四模型均复核了两种净量，不能把该子集当作异常尾部覆盖。
- 已完成 Qwen2/Qwen3 各1图全层smoke以及全部冻结审计（包含smoke），两进程退出0，无跳过；smoke主流程约180.9/176.2秒（含首次权重加载），随后恢复主流程805.5/803.8秒。实际全量审计命令为 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/run_ffn_visual_source_consistency.py audit --model MODEL --device CUDA`；Qwen2 cuda:0、Qwen3 cuda:1；日志 `outputs/consistency_v2_qwen{2,3}_audit_20260907.log`。LLaVA/InternVL随后各占一张卡，先 `--smoke` 再恢复全部audit。
- 独立 cohort 准备已完成：`analyze_ffn_visual_source_consistency.py prepare-independent` 扫描本仓库及相邻仓库448份可识别COCO/POPE使用记录，排除4451个已用ID（含额外451个POPE图），从36053张本地候选固定选2000张并校验ID/JPEG checksum隔离。清单为 `outputs/ffn_visual_source_consistency_v2/independent_cohort.json`；此时没有生成新caption、查看新标签或评估检测器。
- 四模型数值审计全部完成：Qwen2/LLaVA/Qwen3/InternVL分别616/640/920/792个case，共2968×4条件=11872条。LLaVA/InternVL首次smoke主流程164.07/164.83秒，随后恢复全部audit898.82/512.53秒；四进程均退出0，无跳过和未解释失败。`audit_gate.json` 正式状态为 `PASS_AUDIT`，共同选定局部FP32 K4；native K4与native K64均未通过，局部FP32 K4的闭合P90/max分别为 `1.624e-5/1.123e-4、1.856e-5/1.114e-4、1.093e-4/8.361e-4、1.524e-4/1.116e-3`，所有要求的分组及全层检查通过。K64 FP32只作参考，不称真值。
- 四模型旧cohort各1图全层提取smoke已完成，候选FP32 K4，同次捕获额外逐case FP32 K64参考，P_FFN相关OT重新计算。Qwen2/LLaVA/Qwen3/InternVL单图约32.36/67.79/37.41/36.04秒，均退出0。Qwen2/Qwen3审计及全量smoke恢复时禁止模型加载，完成文件checksum与mtime不变；审计恢复证明保存于 `outputs/ffn_visual_source_consistency_v2/qwen_audit_resume_verification.json`。
- 已按严格全量K64参考口径启动Qwen2 cuda:0和Qwen3 cuda:1旧4000图补算，每卡一个VLM。命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_consistency.py extract-old --model MODEL --candidate fp32_k4 --device CUDA`；日志为 `outputs/consistency_v2_qwen{2,3}_full_fp32k4_ref64_20260907.log`。已向用户说明逐case额外K64导致工作量约17倍、完整任务可能多日；在没有范围调整回复时不省略参考验收。旧cohort必须通过后才能开始192个检测器；目前未训练、未生成新2000图。
- 审计脚本全文件checksum已冻结，不改其代码；后续全量适配和训练/独立确认入口放在新的分析CLI，以免破坏已完成审计的可恢复性。全量适配自身函数及实际数值依赖单独指纹化；原v1源文件均未修改。
- 开发期间第一次全量smoke发现provenance缺少数值实现指纹，停止该开发版本并保留到精确的 `old/fp32_k4_pre_code_fingerprint_smoke/`（Qwen2已完成1图、Qwen3收到SIGINT退出130）。Qwen2停止请求因其已退出返回NoSuchProcess，不是实验失败。随后加入代码指纹，在独立canonical目录重跑四模型smoke全部退出0；没有删除或覆盖这些开发产物。
- 最新联合验证命令 `python -m unittest tests.test_ffn_visual_source_consistency tests.test_ffn_visual_source_study tests.test_ffn_visual_path_attribution` 为46/46 PASS（13项新测试）；`git diff --check`通过。一次只读检索包含不存在的 `evaluation/` 目录报错，随后直接定位 `coco-labeling/coco_chair.py`，没有影响实验。
- 后续实现与验收：增加双GPU普通进程队列 `analyze_ffn_visual_source_consistency.py pipeline --resume`，严格按“全量数值验收→16×3×4训练→CPU复算→封存→新2000图生成/标注→AE/R_cos与v2提取→独立checkpoint复算→本地报告”顺序衔接；每卡至多一个VLM。预设候选用尽仍不通过时硬停止，无跳过、放宽阈值或提前训练。每次命令的退出码、耗时、日志checksum写入 `outputs/ffn_visual_source_consistency_v2/workflow_runs/`；只有完整验收通过才追加主报告与中文终末记录，不把进程启动当完成。
- 合成小数据集成测试实际训练48个两epoch CPU探针，核验train/test checkpoint预测、两阈值指标、ensemble和train-only阈值，再禁止trainer调用验证恢复不改checksum/mtime；这些全部位于临时测试目录，已清理，不属于正式检测结果。加上零目标图片、紧凑轨迹与双净量一致性、独立阶段封存前拒绝、最高候选失败停止等测试，终末相关测试为50/50 PASS；`py_compile`与`git diff --check` PASS。
- 四模型全审计恢复验收全部PASS，Qwen两模型记录见前述文件；LLaVA/InternVL保存为 `llava_internvl_audit_resume_verification.json`，检查1280/1584个数据及sidecar文件。四模型全层提取smoke的实际记录和恢复验收为 `full_smoke_resume_verification.json`；数据shard内Qwen2/LLaVA/Qwen3/InternVL单图耗时32.359/67.697/37.411/35.903秒，峰值显存16.524/14.433/17.614/17.492GiB，OOM重试均0。前面的67.79/36.04秒为控制台整体值，与shard实际单图计时口径略有差别。
- 队列首次尝试 `nohup ... &` 返回PID560587，但复查发现该进程没有持续运行、没有产生workflow记录（运行器回收短命shell派生任务）；不能据此宣称后台启动成功。随后改为与正在运行的全量worker相同的长任务会话机制，命令 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_consistency.py pipeline --resume >> outputs/consistency_v2_pipeline_20260907.log 2>&1`，必须以真实进程及workflow状态文件验收启动。
- 队列先等正在执行的Qwen2/Qwen3全量worker锁释放，校验恢复后依次跑LLaVA/InternVL，不会另起VLM抢占同一GPU。当前数值计算函数和审计脚本指纹保持不变；修改的是后续验收/队列逻辑。服务器或运行器中断后，可用同一命令从已完成的逐图shard恢复，不依赖未完成进程的内存状态。
- 队列启动已再次真实核验：Python PID560825持续运行，长任务session43396；`workflow_runs/1788734321233811355/` 已产生Qwen2/Qwen3两个 `WAITING_FOR_EXISTING_WORKER` 记录，正常等待现有PID556303/556313（session39590/31396）。GPU上仍只有这两个全量VLM进程；旧4000图完成数以各模型 `old/fp32_k4/shards/image_*.pt` 和sidecar为准，不以进程存在当作完成。
- 当前交接状态仍为**执行中，不是实验全部完成**：Qwen2/Qwen3旧4000图只完成前几十张，LLaVA/InternVL全量待队列调度；正式192头与新2000图尚未开始。逐case K64参考使旧cohort补算本身预计需要数日；若后续失败，队列保留失败记录并停止后续依赖阶段。没有提交或上传任何v2结果。
- 开发期间一次只读 `rg` 把公式 `S_{net}` 当正则报错 `repetition quantifier expects a valid decimal`；改用多个字面检索项后继续，不影响数据或实验。

## 2026-09-07 AE32×平均 cosine 检测消融（完成）

- 按新请求定义 `X=AE32*C32`，`AE32` 为原始视觉 token AE 最大32项的总和；`C32` 为相同 Top32(T)/Top32(AE) 集合内、预测位置与视觉 token hpre 状态余弦的无权算术平均，精确复用旧 `dgst_t_hpre_raw_logit_gauss_target_cosine_topk32_hpre_per_layer`。保留 cosine 正负号，不取绝对值、不截断负值，不改为逐 token AE×cos 的加权和。
- 沿用 Ponytail 最小增量，在既有未提交 Top32 AE 成果上新增独立 `scripts/analyze_ffn_visual_source_top32_ae_cosine.py`；复用 Top32、20组矩阵、AE替换、trainer、resume与CSV实现，不改旧脚本或原产物checksum。无新依赖、VLM forward或bootstrap，不提交/上传。
- 四模型各20组×seeds43/44/45，共240个新训练头；直接对照上一轮AE32，并补充原全AE及13正式组的全AE×cosine旧结果。固定原3200/800图片split、MLP[128,64,32]、dropout.3、batch256、最多100epochs、minimum-train-loss checkpoint、train-REAL-F1及固定0.5，不标准化/加权/重采样，S/I/N/kappa/R_cos/JS/OT全部保持原值。
- 输入先核验当前AE32矩阵与旧baseline fingerprint完全一致，再比较原始attention×gate直接Top32求和，以及 `AE32*C32 = full_AE*old_EV`（old_EV=M32*C32）的实际数值误差。恢复校验cohort/feature SHA256，完成后不重训或重写。输出独立命名为 `metrics/top32_ae_cosine_*`、`tables/top32_ae_cosine_*`。
- 运行前两张RTX4090空闲，主机可用内存约459GiB；采用四个互不重叠的单模型进程，两卡各运行两个小MLP任务，保留每个训练头原协议，不修改trainer来并行同一产物。
- 启动前定向单测20/20 PASS、py_compile、git diff --check通过。命令为 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_top32_ae_cosine.py --models MODEL --training-device CUDA`；Qwen2/Qwen3使用cuda:0，LLaVA/InternVL使用cuda:1。逐模型日志 `outputs/top32_ae_cosine_MODEL_20260907.log`，不复用上轮日志。
- 四模型全部完成20组×3seed，共240个新checkpoint；单独X的ensemble AUROC（顺序Qwen2/LLaVA/Qwen3/InternVL）为 `.80173/.87244/.85497/.84770`，相对AE32为 `-.00134/+.00394/-.00347/+.00030`。80个组×模型点差仅20个为正，各模型为6/5/4/5；X+S、F、U、H_JS、H_OT的四模型集成分数均下降，不支持普遍乘cosine。逐seed方向并非全部一致，不做显著性或机制断言；全部20组及另13组全AE×C32对照写入主报告5.15。
- 原始attention×gate直接Top32总和与上轮AE32最大绝对误差分别 `1.79e-7/2.98e-7/2.98e-7/2.38e-7`；乘积恒等式最大误差均小于1.8e-7。X实际范围为 `[-.016727,.593118]/[-.002277,.417609]/[-.008076,.628776]/[-.008606,.570178]`；负值条目773/1625/1376/4270来自有符号cosine，AE32本身仍非负。
- 独立重读真实compact和features.pkl，再从CPU复算每模型159个新旧checkpoint（60新+60AE32+39全AE×C32），共636个，预测最大误差 `1.67e-6/3.04e-6/3.28e-6/2.38e-6`。每模型resume禁止trainer调用并验证完成产物hash/mtime不变；全局独立验收 `outputs/ffn_visual_source_top32_ae_cosine_audit.json` 为PASS。80行组汇总、960行seed×threshold双版本指标完整有限；7个非正式组合无全AE×C32旧对照，CSV相应字段预期留空。
- 四训练进程均退出0，单模型主流程耗时506.2/913.8/891.5/699.6秒（含读取、校验、训练，不含初始导入），四进程并行不能累加当墙钟耗时。两卡显存已释放；终末单测20/20 PASS、py_compile、git diff --check通过。跨模型汇总为 `outputs/ffn_visual_source_top32_ae_cosine_summary.json`，仍未提交/上传，之前Top32脚本及其checksum保持不变。
- 失败记录：训练和独立验收无失败；首次 `scripts/analyze_ffn_visual_source_top32_ae_cosine.py --summarize-only` 提前于LLaVA最终JSON落盘，报 `FileNotFoundError: .../llava_1_5_7b/.../metrics/top32_ae_cosine_feature_results.json`，未输出不完整总表。确认四模型JSON及训练退出0后，原命令重跑退出0，没有修改或重训旧/新实验。只读rg无匹配返回1属于正常检索。
- 终末只读核验将主报告80个主对照与52个次要对照单元自动对照JSON，全部通过；跨模型汇总与逐模型JSON完全一致，764条结果内artifact checksum再次逐字节通过。同期修正本记录两个手工抄录的5位小数末位，未改任何实验产物。

## 2026-09-07 全量 AE / Top-32 AE 曲线与检测消融（完成）

- 当前基线为已上传的 `9b00af1`；除无关交接文件外工作树干净。本轮只做用户新增的 AE 曲线与替换消融，不提交/上传新成果、不更改旧实验。
- `AE32=AE*sum(Top32(T))=sum_{j in Top32(T)} attention_j*gate_j`，按 T/AE 贡献排序，不按原始 attention 排序；视觉 token 不足 32 时全取，不重新归一化选中区域，不乘 cosine。直接读取保存的 AE/T；零 AE 返回零，保留 FP32 T 的数值归一化误差。
- 遵循 Ponytail，新增独立 `scripts/analyze_ffn_visual_source_top32_ae.py`，复用现有矩阵、core/net 特征拼接、AE 替换、median/IQR 绘图和 MLP trainer；不修改旧分析脚本、无新增依赖或 VLM forward、不做 bootstrap。四模型各训练 20 组×3 seeds（13 正式组 + AE+I、AE+S、AE+I+S、AE+N、AE+S+N、U、D_OT+strength），只替换 AE，所有其余轨迹不变。
- 原始 S/I/N、不标准化/重采样/加权，固定原图片 3200/800 split、seeds 43/44/45、原 MLP `[128,64,32]`、dropout .3、batch 256、最多100 epochs、minimum-train-loss checkpoint、train-REAL-F1 和固定0.5双阈值。JS 沿用全 token support，OT 沿用双方 Top-32 union；Top32AE 不改变距离定义。
- 输出使用独立 `top32_ae_feature_*` 和 `ae/ae_top32_real_hall_curve`，恢复先校验 cohort/features SHA256，完成后核验产物 checksum 并直接返回，不重训/覆盖。训练前两张 RTX 4090 均空闲；新增单测检查 TopK 原始总量、退化值、20组仅 AE block 被替换和错误 cohort 恢复拒绝。
- 启动前 `19/19` 定向单测、`py_compile`、`git diff --check` 通过。命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_top32_ae.py --models qwen2_5_vl_7b,qwen3_vl_8b --training-device cuda:0`；另一路 `--models llava_1_5_7b,internvl_2_5_8b --training-device cuda:1`。日志为 `outputs/top32_ae_{qwen,llava_intern}_20260907.log`；完成后使用同脚本 `--summarize-only` 汇总。
- 四模型曲线已生成并逐图检查：全 AE 的 REAL 中位数更高层数为 Qwen2/LLaVA/Qwen3/InternVL `19/28、32/32、36/36、29/32`，Top32 AE 为 `10/28、27/32、26/36、28/32`。全部正式 train+test mentions、mention 等权，中位数/IQR（不是CI），不是检测器输出概率。
- 四模型最终均完成60个新训练头，共240个；复用240个全AE旧对照。独立对全部480个新旧 checkpoint 在当前特征上复算CPU预测，最大概率差分别 `2.03e-6/2.50e-6/3.28e-6/2.98e-6`。逐token直接加权求Top32与 `AE*mass32` 的FP32结果完全一致；恢复时trainer被禁止调用，完成产物hash/mtime均未变化。独立验收为 `outputs/ffn_visual_source_top32_ae_audit.json`，状态PASS。
- 全部80行组汇总、960行双版本×seed×threshold指标、512行曲线统计完整有限；两张总图与八张单图已目视检查，报告80组数字及delta自动对照JSON全部通过。本轮无训练/验收失败；终末19/19单测、py_compile、git diff --check通过。两训练进程退出0，GPU已释放。
- Top32 AE单独ensemble AUROC为Qwen2/LLaVA/Qwen3/InternVL `.80307/.86850/.85844/.84740`，相对全AE `+.01537/−.00798/+.01000/+.01278`；AE32+S为 `.87292/.89704/.87911/.85981`，相对原AE+S `+.00588/−.00195/+.00019/+.00551`。F32在Qwen2/InternVL提高、LLaVA/Qwen3下降；U32为 `.88483/.90610/.88836/.87582`。完整H_OT32为 `.89117/.90923/.89506/.88198`，仅LLaVA的ensemble提高，Qwen2近乎不变、另两模型下降。因此Top32乘法不是单纯归一化，也不能普遍替换全AE；A32/E_JS32虽四模型点差均正，仍不能在未做bootstrap时称显著。
- 共80个点差中50个为正（各模型17/7/11/15）。主报告5.14给出公式、完整20组对照、两图及边界；跨模型汇总为 `outputs/ffn_visual_source_top32_ae_summary.json`，总图为 `outputs/ffn_visual_source_{ae,ae_top32}_real_hall.{png,pdf}`，逐模型结果使用前述独立命名，不覆盖旧研究。原代码文件checksum不变；本轮新代码/报告尚未提交或上传。
- 单模型主流程耗时 `548.7/927.2/919.8/707.6` 秒，含读取、验证、绘图和60个训练头，不含初始模块导入；两条GPU队列并行，不能相加当墙钟总时长。恢复不改写首次完成耗时。运行/测试无失败；只读 `rg` 查询无匹配返回1属于正常检索结果。
- 补充图片元信息只读检查时系统未安装 `file` 命令，改用已有 Pillow 确认两张总图均为2160×1350且内容未裁切。报告四舍五入数字与JSON逐项对照，并修正交接记录中少数手工抄录的末位；未改实验产物。

## 2026-09-07 GitHub 增量发布

- 用户要求上传到 GitHub；本次在 `main` 的 `d6f0728` 上发布后续代码及结果，远端为 `Xbkeepdo/gfchair`。提交前 `git ls-remote origin refs/heads/main` 确认远端与本地基线一致，不需要强制推送或改写历史。
- 发布范围：6 个实验代码/测试/报告文件，以及 195 个紧凑 JSON/CSV/PNG/PDF 结果文件（10,861,512 bytes）；包括 Union-Top32 JS、net/bounded/relative/log1p strength、OT/JS REAL/HALL 曲线和四模型核心信号消融。沿用 Ponytail 复用现有实现，没有重训、VLM forward、新 bootstrap 或新增依赖。
- 原始 shard、完整向量、预测 `.pt`、模型 checkpoint 留在本地；无关的 `docs/CODEX_CONVERSATION_HANDOFF_20260830.md` 不加入提交。主报告明确区分 GitHub 紧凑发布文件与 checksum 清单中的本地完整验收文件。
- 提交前核验：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -m unittest tests.test_ffn_visual_source_study` 为 `16/16 PASS`；4 个相关 Python 文件 `py_compile`、源代码/文档的 `git diff --check` 通过。全部新增 JSON 可解析，报告 15 个本地链接均纳入发布；敏感凭据模式扫描无命中，382 个既有实验文件的 SHA256 逐字节复核全部通过。
- 发布检查记录：首次完整 `git diff --cached --check` 将 Python CSV writer 的标准 CRLF 行尾报告为 trailing whitespace；没有修改实验 CSV 字节或破坏 checksum。使用命令级 `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff HEAD^ HEAD --check` 明确接受 CRLF、保留其他空白检查后通过，未更改全局/仓库 Git 配置。
- 发布命令：`git commit -m "Publish core signal ablations and JS/OT analyses"`、`git push origin main`；以推送返回值及 `git ls-remote origin refs/heads/main` 与本地 HEAD 一致作为上传验收，不把仅本地提交当作上传成功。

## 2026-09-06 核心强度与距离增量消融（完成）

- 新任务只复用紧凑特征，不做 VLM forward、bootstrap 或新的数值变换。`I=sum_m||a_m||`（gross WRITE）、`S=sum_m||e_m||`、`N=||G(z)-G(z0)||`，逐样本逐层核验 `N=S*kappa` 后使用原始 I/S/N。新增 `--study core_signals`，复用现有矩阵加载、三种子 trainer 和结果格式。
- 六组新增训练：`AE+I、AE+S、AE+I+S、AE+N、AE+S+N、U`。复用正式 `B/F/H_OT/H_JS` 作为 `AE/AE+S+kappa/H_OT/H_JS`；U 拼接顺序为 `[R_cos,AE,S,kappa]`，恰好是 H 删除三条距离后的顺序。H_JS 使用正式全视觉 support。四模型已完成 72 个新 checkpoint、48 个复用 checkpoint，不覆盖已有对照。
- 命令 1：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python -u scripts/analyze_ffn_visual_source_signal_ablation.py --study core_signals --models qwen2_5_vl_7b,qwen3_vl_8b --training-device cuda:0 --bootstrap-resamples 0 --resume`。命令 2：同脚本使用 `--models llava_1_5_7b,internvl_2_5_8b --training-device cuda:1`，其余参数相同。
- 运行前检查：两张 RTX 4090 空闲；新增拼接/零强度/错误 N/非有限 WRITE 检查与相关测试 `16/16 PASS`，`py_compile`、`git diff --check` 通过。新增输出使用 `core_signal_feature_*`，恢复时核验固定 image split 和所有特征矩阵的 SHA256，防止复用不同 cohort/特征的预测。
- 主要结果（顺序 Qwen2/LLaVA/Qwen3/InternVL，均为三 seed 概率集成 AUROC）：`AE+I=.84437/.89275/.86551/.84637`；`AE+S=.86704/.89900/.87892/.85430`；`AE+I+S=.86164/.90035/.89687/.86443`；`AE+N=.86790/.89408/.87996/.85451`；`AE+S+kappa=.86938/.90100/.88551/.86974`；`AE+S+N=.87996/.89724/.88856/.86539`；`U=.88118/.90548/.89065/.87684`。
- 结论：`AE+I+S−AE+I=+.01727/+.00760/+.03137/+.01806`，四模型全部 12 个配对 seed 均为正，WRITE 强度未解释全部 S 收益；净 effect 单独总体接近 S，但不及 S+kappa。`(AE+S+kappa)−(AE+S+N)=-.01059/+.00376/-.00306/+.00435`，两种参数化各赢两模型；S>0 时 N=S*kappa，可互相确定，不能把 κ 增益当作独立语义冲突机制的证据。
- `H_OT−U=+.01006/+.00133/+.00680/+.00740`，`H_JS−U=+.00123/-.00015/+.00390/+.00275`。特别说明：LLaVA 的 H_OT 三个单 seed 均低于 U（seed-mean `.900563` 对 `.901817`），只有集成后反超；不能称为稳定或显著的距离增益。本轮不做 bootstrap、未基于 test 选 checkpoint/threshold，但同一 test cohort 的反复事后选择边界不变。
- 数值边界：1,650,608 个 mention-layer 条目全部 S>0，N−S*kappa 最大绝对/相对偏差为 `8.56e-6/<5.96e-8`。原始 κ>1 条目为 `28/0/2427/76`，最大为 `1.5420/.9630/3.2319/1.5140`；分子来自有限端点、分母来自数值积分 gross，代数一致性不代表 vector completeness，也不能把全部已存 κ 当作严格 `[0,1]` 的精确抵消率。本轮保留原值，没有裁剪或新增积分前向。
- 验收完成：四模型各 4000 processed images，40 行组汇总、240 行 seed×threshold 指标、52 行比较全部 finite；独立在 CPU 从 120 个新/复用 MLP checkpoint 重算当前 test 预测，最大绝对差 `<3e-6`。恢复验收禁止 trainer 调用，完成产物 hash/mtime 全部不变；382 文件 checksum 保存于 `outputs/ffn_visual_source_core_signal_ablation_audit.json`，跨模型总结果为 `outputs/ffn_visual_source_core_signal_ablation_summary.json`，主报告新增 5.13。原定向测试再次 `16/16 PASS`，`py_compile`、`git diff --check` 通过；无训练/验收失败，GPU 已释放，未提交或上传。
- 首次模型主流程耗时 `144.1/280.2/261.7/228.7` 秒，不含初始模块导入，含读取、校验与 18 个新 MLP 训练；两 GPU 并发、各串行两模型，不能把这四项之和当作墙钟总时长。resume 返回已完成结果而不覆写耗时或产物。
- 终末只读 checksum 复核脚本首次使用当前 Python 环境没有的 `hashlib.file_digest`，报 `AttributeError`；改用标准 `hashlib.sha256` 分块读取后，382 个文件逐字节复核全部通过，没有改动训练或任何实验产物。

## 2026-09-06 JS(P_FFN,T) 的幻觉/非幻觉曲线

- 用户要求查看 FFN/目标证据这一对分布。已核对 `T=ATTENTION_EVIDENCE=normalize(attention_support×hpre_raw_logit_gauss_gate)`，所以复用 `D_EF^JS=JS(T,P_FFN)`；JS 对称，等于用户所说的 `JS(P_FFN,T)`。此前 `E_JS` detector 输入还含 AE，本图不含 AE/检测概率。
- 按 Ponytail 给已有绘图脚本增加 `--pair ef`（默认 `wf` 不变），复用正式矩阵、union JS 与原曲线函数；无新依赖、无训练、无 VLM 重跑、无 bootstrap。全视觉 token 与 `Top32(T)∪Top32(P_FFN)` 分别重归一化的两版均已完成，旧图及所有既有未提交成果保留。
- 样本仍是四模型各 4000 processed images 的全部正式 train+test mentions，中位数实线/IQR 阴影、mention 等权。全 token 版 HALL 中位数较高层数为 Qwen2/LLaVA/Qwen3/InternVL `14/28、28/32、30/36、23/32`，union 版为 `16/28、27/32、31/36、21/32`。相比 WRITE/FFN JS，后三模型多数层更常 HALL 较高，LLaVA 的可见间距较大；Qwen2 方向混合、两类区间仍重叠，不据此声称显著性或分类准确率。
- 命令：`CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py --pair ef --distance js`；末尾替换为 `--distance js_union_topk` 运行并集版。两进程退出 0，主流程耗时 105.7/254.5 秒（不含初始模块导入）。总图/统计为 `outputs/ffn_visual_source_d_ef_js_real_hall.*` 和 `outputs/ffn_visual_source_d_ef_js_union_topk_real_hall.*`，另存 `_full_ln2.png/pdf`；逐模型产物在原 `tables/figures/metrics`，主报告新增 5.12 节。
- 验收：两版合计 512 行；cohort counts 与已有图完全一致，finite/范围、标签数/层数/去重、分位数顺序、总/逐模型 JSON 一致性和图表存在性全部通过，四张总图已目视检查。运行时验证 E_JS/G_JS 中的 D_EF block 精确一致；原 union-feature 测试新增 E_JS 断言，运行 `python -m unittest tests.test_ffn_visual_source_study.FFNVisualSourceStudyTest.test_union_topk_js_uses_pairwise_union_and_rebuilds_only_js_groups tests.test_ffn_visual_source_study.FFNVisualSourceStudyTest.test_js_is_symmetric_and_zero_on_identity`（正式 vicr Python）为 `2/2 PASS`；`py_compile`、`git diff --check` 通过。无失败，未提交或上传。

## 2026-09-06 JS 的幻觉/非幻觉曲线

- 按用户要求对照上一轮 OT，画纯 `D_WF^JS=JS(P_WRITE,P_FFN)` 的四模型逐层曲线；明确此前 `D_JS` 训练组仍包含 AE，曲线不包含 AE/检测概率。正式全视觉 token 与后续双方 Top-32 并集两个版本均已完成，JS 使用自然对数、未开根号/除以 ln(2)，理论范围 `[0,ln(2)]`。
- 遵循 Ponytail 复用路径：只给现有 `scripts/plot_ffn_visual_source_d_ot.py` 增加 `--distance js/js_union_topk`，默认 OT 不变；复用正式矩阵读取、`union_topk_js_matrices()` 与 `write_label_curve()`。无新增依赖、无训练、无 VLM 重跑、无 bootstrap，旧 OT 图及其他未提交成果保留。
- 样本仍是每模型 4000 processed images 的全部正式 train+test mentions，每 mention 等权；线为中位数、阴影为 IQR，不是 CI。全 support 的 HALL 中位数更高层数为 Qwen2/LLaVA/Qwen3/InternVL `7/28、9/32、15/36、12/32`；union 为 `10/28、11/32、14/36、12/32`。两版都多次交叉且大量重叠，没有跨模型一致的“JS 越大越幻觉”方向。全 support 多数层为 `10^-3` 量级，Qwen3 末层 REAL/HALL 为 `.100795/.103195`，两类同时升高；不能由曲线数值或边际重叠直接推断检测器优劣。
- 命令：`CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py --distance js`；将末尾替换为 `--distance js_union_topk` 运行并集版。两进程均退出 0，CPU 主流程分别 83.7/228.2 秒（不含初始模块导入）。总图/JSON 为 `outputs/ffn_visual_source_d_js_real_hall.*` 和 `outputs/ffn_visual_source_d_js_union_topk_real_hall.*`，另有 `_full_ln2.png/pdf`；各模型保存 `tables/figures/metrics`，主报告补充 5.11 节。
- 验收：两版共 512 行统计；cohort counts 与已有 OT 完全一致，标签数/层数/去重、finite、范围、分位数顺序、JSON 与逐模型统计一致性、PNG/PDF/CSV 存在性全部通过，四张总图已目视检查。运行 `python -m unittest tests.test_ffn_visual_source_study.FFNVisualSourceStudyTest.test_union_topk_js_uses_pairwise_union_and_rebuilds_only_js_groups tests.test_ffn_visual_source_study.FFNVisualSourceStudyTest.test_js_is_symmetric_and_zero_on_identity`（正式 vicr Python）为 `2/2 PASS`，`py_compile`、`git diff --check` 通过。
- 失败记录：绘图、单测、产物核验均无失败；最初定位 AGENTS/skill 时广域 `rg --files` 在无权限的 `hpc-home` 和两个 NLTK 目录报告 permission-denied，随后只使用实际项目路径，不影响任务。检索旧 JS 曲线无匹配返回 1 属正常的“未找到”，已新增独立命名产物；未提交或上传。

## 2026-09-06 D_OT 的幻觉/非幻觉曲线

- 本次仅分析已有正式特征：画纯 `D_WF^OT=OT(P_WRITE,P_FFN)`，不是此前含 AE 的 `D_OT` detector 组预测。复用正式矩阵读取和现有曲线函数，新增独立脚本 `scripts/plot_ffn_visual_source_d_ot.py`，无新增依赖、无训练/特征重提取、无 bootstrap。
- 四模型各 4000 processed images，Qwen2/LLaVA/Qwen3/InternVL 的全部 train+test mentions 为 `8717/15463/14873/11759`。每个 mention 等权；线为 REAL/HALL 中位数，阴影为 IQR（不是 CI）。两类多次交叉，HALL 中位数更高为 `12/28、12/32、13/36、12/32` 层；多数层约 `.01–.04`。LLaVA/Qwen3 末层两类同时升高，不能据峰值判定幻觉；也不能把含 AE 的旧 D_OT AUROC 当作纯距离训练结果。
- 已保存四模型单图/CSV/JSON，以及 `outputs/ffn_visual_source_d_ot_real_hall.{png,pdf,json}` 和统一 `[0,1]` 纵轴的 `_full_01.{png,pdf}`；主报告新增 5.10 节。
- 命令：`CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_visual_source_d_ot.py`。CPU 主流程 89.6 秒（不含初始模块导入），退出码 0。合计 256 行统计，所有输入距离 finite/范围、AE/距离精确拆分、标签样本数、分位数顺序及图表存在性检查通过；两张总图已目视检查，`py_compile` 和 `git diff --check` 通过。
- 失败记录：正式绘图和产物核验无失败。只读定位时曾访问不存在的 `features/ffn_visual_source_study.py` 和 `outputs/ffn_visual_source_attribution_v1`，随后定位到实际脚本及各模型结果根目录；未修改旧实验，未提交或上传。

## 2026-09-04 Vector FFN Source Attribution 全实验完成

- 主定义已改为 `e_m=∫_0^1 J_G(z0+αA)a_m dα`；实现 `streaming_vector_path_statistics()`、vector sampled Shapley、四个独立 CLI，并复用现有 capture/JVP/OT/trainer/manifest/bootstrap。没有新增依赖、没有提交、没有覆盖旧实验，也没有修旧 scalar-path Qwen3、跑 VQA/neuron intervention 或优化 bbox。
- 数值 audit：四模型各 200 图、50 FP32/scaling/Shapley cases、8/16 regions、128 permutations、0 remaining failure。K 只由 Qwen2/LLaVA 冻结为 Gauss–Legendre K4；扩展模型不能重选 K。`linearize` 未同时满足误差/显存/速度 gate，正式 backend 为 `vmap(jvp)`。
- 正式全层提取：Qwen2/LLaVA/Qwen3/InternVL 均为 4000 processed images；unique targets=`8654/14951/14704/11630`，mentions=`8717/15463/14873/11759`。无目标图 `321/29/57/73` 也保留在 manifest。所有层 shape/finite/归一化、跨 shard 无重复、target/mention、固定 8:2 split 均通过。
- Detector：四模型均完成 13 groups×seeds 43/44/45，固定 0.5 与 train-REAL-F1 threshold 指标齐全，每模型 10 个图片级 paired-bootstrap 比较×10,000 resamples。主 gate 四项全部通过：Qwen2 `G_JS−C_JS=.07192 CI [.04113,.10490]`、`G_OT−C_OT=.06685 [.03775,.09814]`；LLaVA `.01139 [.00228,.02090]`、`.01456 [.00505,.02374]`。扩展结果不得反向参与 gate。
- 追加探索性 `strength+kappa` 三 seed：Qwen2/LLaVA/Qwen3/InternVL 的 seed-ensemble AUROC 为 `.86368/.89124/.87209/.85943`，相对 strength 的图片级 10,000-bootstrap 增益为 `+.00683/+.00523/+.00170/+.01511`；除 Qwen3 CI 跨 0 外，其余下界均大于 0。四模型 REAL/HALL strength 的逐层 median+IQR 曲线及 CSV 已写入各自 source-attribution `figures/`、`tables/`；REAL median 较高的层数为 `24/28、23/32、33/36、29/32`。
- A 组来源已澄清：它是新方案内部 `R_cos+AE strength` reference，不是 baseline comparison 中的 SVAR，也不是旧 selected-feature 的 `R_cos+EV`。正式表报三 seed 概率 ensemble AUROC 且 `drop_last=False`；旧 summary 报 per-seed AUROC mean±std 且 `drop_last=True`。两边固定图片 split、mention 数和标签 cohort 一致。
- 完成 `AE strength→旧 top32 mass×mean-cosine EV` 的 13 组严格替换实验：四模型各 39 checkpoints，每组与原 AE 版本做 10,000 次图片级 paired bootstrap。52 个比较无一显著支持旧 EV；LLaVA 的 F/C_OT/D_OT 与 Qwen3 的 F/H_JS/G_OT/H_OT 显著下降，其余 CI 跨 0。结果为各模型 `metrics/old_ev_feature_group_results.json`、`tables/old_ev_feature_group_metrics.csv`；主报告 5.3 给出公式和完整 AUROC 表。
- 完成 `AE strength×旧 Top-32 mean hpre cosine` 的 13 组严格替换实验：四模型各 39 checkpoints，共 156 个；按用户新要求不做 bootstrap，只报告三 seed ensemble 点估计。52 个组×模型差值中 23 个为正，平均 `+0.00007`；Qwen2/LLaVA/Qwen3/InternVL 的跨组平均差为 `+.00459/-.00127/-.00034/-.00269`，没有跨模型一致提升。结果为各模型 `metrics/ae_cosine_feature_group_results.json`、`tables/ae_cosine_feature_group_metrics.csv`；主报告 5.4 给出 AE/M 区别与完整表。
- 完成 JS 的 pairwise Union-Top32 对照：每个 `E/W/F` 分布对独立取双方 Top-32 并集、在并集内分别重归一化后计算 JS；只重训 `C_JS/D_JS/E_JS/G_JS/H_JS`，其余 block 不变。四模型共 60 个新 checkpoint；按用户要求不做 bootstrap。20 个 AUROC 点差中 6 个为正，平均 `-.00097`；`E_JS` 四模型全降，G/H 的小幅增益不跨模型一致，因此保留正式全视觉 support JS。结果为各模型 `metrics/union_topk_js_feature_group_results.json`、`tables/union_topk_js_feature_group_metrics.csv`；主报告 5.5 给出完整对照表。
- 使用边界复核：`C_m` 是需要 downstream target gradient 的 Riesz target consequence，按正式全层 stage“不跑 downstream gradient”的预注册约束，没有进入本轮 compact shard、detector、gate 或因果 ranking；`R_cos` 不是 `C_m`。`Q_m=hat(delta)^T e_m` 已在四模型所有 target-layer 保存为 `PATH_SIGNED_Q` 并用于验收/固定示例图，但没有作为 13 组 detector 输入，因果 ranking 也只用 `P_FFN/P_WRITE/random`。
- 完成 `S_net` 与 `D_OT+strength` 追加实验：`S_net=||sum_m e_m||` 单独 AUROC 为 Qwen2/LLaVA/Qwen3/InternVL `.8586/.8799/.8670/.8393`，相对 gross strength 为 `+.0017/-.0061/-.0034/-.0050`，没有稳定提升。`D_OT+strength` 明确定义为 `AE+D_WF^OT+gross_strength`，AUROC `.8697/.8987/.8811/.8606`，相对 D_OT 为 `+.0445/+.0017/+.0141/+.0071`，四模型均提高但仍低于正式 G_OT。按用户要求不做 bootstrap。
- 新增四模型 `S_net` 与 kappa 的 REAL/HALL median+IQR 曲线及逐层 CSV。Snet 的 REAL median 高于 HALL 层数为 `26/28、26/32、36/36、32/32`；kappa 为 `14/28、25/32、10/36、14/32`，且多次跨层反向，不能解释成“越高越真实”的单调标量。结果写入各模型 `metrics/net_strength_feature_results.json`、`tables/net_strength_feature_metrics.csv`、`tables/{net_strength,kappa}_real_hall_curve.csv` 与对应 PNG；主报告 5.6 给出公式、完整表和使用边界。
- 完成 `bounded_strength=S_raw/(S_raw+tau_layer)` 追加对照；`tau_layer` 仅由每模型训练 mentions 的逐层中位数确定，测试集复用，不读取 label/test 统计，且不改 `e_m/P_FFN/kappa`。Bounded strength 单独相对 raw strength 的 AUROC 差为 Qwen2/LLaVA/Qwen3/InternVL `-.00930/+.00732/+.01084/+.01063`；`D_OT+bounded_strength` AUROC 为 `.88807/.90642/.90274/.86813`，相对 `D_OT+raw_strength` 四模型一致提高 `+.01835/+.00776/+.02165/+.00757`。组合 HALL-F1 也全部提高；HALL-AUPR 在 Qwen2/Qwen3 提高、LLaVA/InternVL 小降。实际 bounded 范围均严格位于 `[0,1)`，分别为 `[.0427,.9651]/[.0292,.9605]/[.0145,.9763]/[.0153,.9420]`。按用户要求不做 bootstrap，暂不替换正式预注册结果；主报告 5.7、各模型 `metrics/tables/figures` 已更新。
- 完成无训练标定的提取时相对强度 `relative_strength=R/(R+W)`，其中 `R=sum_m||e_m||`、`W=sum_m||a_m||`，退化分母返回 0，构造时即限制在 `[0,1)`；不读取 label 或 train/test 统计。单独相对 raw strength 的 AUROC 差为 Qwen2/LLaVA/Qwen3/InternVL `-.03313/-.00303/+.01672/+.02274`；`D_OT+relative_strength` AUROC 为 `.85745/.90640/.89285/.86966`，相对 `D_OT+raw_strength` 为 `-.01227/+.00773/+.01176/+.00910`，相对 D_OT 单独仍全部提高。实际 train/test 合并范围为 `[.1930,.6893]/[.1355,.7487]/[.0977,.8834]/[.1502,.8466]`。Qwen2 明显退化，故不替换 raw/bounded/formal G/H；按用户要求不做 bootstrap。代码层没有统计泄漏，但由于同一 test cohort 已被反复用于后验特征选择，该结果只属探索性，确认需冻结公式后使用独立 held-out cohort。主报告 5.8 与各模型 relative-strength JSON/CSV/PNG 已更新。
- 完成 plain `log1p_strength=ln(1+gross_strength)` 对照；不使用 `tau`、标准化、裁剪或任何 train/test 统计。单独 AUROC 为 Qwen2/LLaVA/Qwen3/InternVL `.84841/.89211/.87744/.85477`，相对 raw 为 `-.00844/+.00610/+.00705/+.01045`；`D_OT+log1p_strength` 为 `.88312/.90609/.89369/.86936`，相对 `D_OT+raw_strength` 四模型一致提高 `+.01339/+.00743/+.01260/+.00880`，但相对 bounded 组合仅 InternVL 略高。变换后实际范围为 `[.0573,5.7246]/[.1250,4.2965]/[.00615,5.1331]/[.0155,4.1988]`，明显压缩但不在 `[0,1]`。由于严格单调，逐层排序和单层 AUROC 不变；差异来自固定 MLP 的数值条件。按用户要求不做 bootstrap，且同一 test cohort 的后验选择边界不变；主报告 5.9 与各模型 log1p JSON/CSV/PNG 已更新。
- 因果阶段：四模型各 100 张 label-balanced 图、400 target-layer cases，regular 8-region；fixed-QK、activation patching、pixel counterfactual 全部正式执行，重复 region 只跑一次并保留三策略映射。每模型 27 个 score/ranking bootstrap 比较均为 10,000 resamples。`P_FFN−random` 的 log-probability 绝对效应在 12 个模型×intervention 项上 CI 下界都大于 0；`P_FFN−P_WRITE` 只有 LLaVA activation patch 明确为正，InternVL pixel CF 明确为负，其余跨 0。因此两种非随机 ranking 均获因果支持，但 FFN ranking 没有稳定取代 WRITE。

正式命令模板：

```bash
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_visual_source_audit.py --model MODEL --device CUDA --num-images 200 --audit-cases 50 --permutations 128 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_study.py --models qwen2_5_vl_7b,llava_1_5_7b,qwen3_vl_8b,internvl_2_5_8b --stage freeze
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_visual_source_attribution.py --model MODEL --device CUDA --k 0 --jvp-backend auto --num-images 0 --shard-images 10 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_study.py --models MODEL --stage detectors --training-device CUDA --bootstrap-resamples 10000 --resume
/opt/conda/private/envs/vicr/bin/python scripts/run_ffn_visual_source_counterfactuals.py --model MODEL --device CUDA --num-images 100 --bootstrap-resamples 10000 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --models MODEL --training-device CUDA --bootstrap-resamples 10000 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study old_ev --models MODEL --training-device CUDA --bootstrap-resamples 10000 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study ae_cosine --models MODEL --training-device CUDA --bootstrap-resamples 0 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study union_topk_js --models MODEL --training-device CUDA --bootstrap-resamples 0 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study net_strength --models MODEL --training-device CUDA --bootstrap-resamples 0 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study bounded_strength --models MODEL --training-device CUDA --bootstrap-resamples 0 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study relative_strength --models MODEL --training-device CUDA --bootstrap-resamples 0 --resume
/opt/conda/private/envs/vicr/bin/python scripts/analyze_ffn_visual_source_signal_ablation.py --study log1p_strength --models MODEL --training-device CUDA --bootstrap-resamples 0 --resume
```

- `elapsed_seconds` 是每份最终 manifest 对应的最近一次/resume command，不是被覆盖前各段的累计时间：Qwen2 full repair/counterfactual=`13.97/372.92s`；LLaVA 双 rank full=`1876.82/1871.47s`、counterfactual=`246.23s`；Qwen3 audit recovery=`1091.25s`、双 rank full recovery=`1705.12/1720.20s`、counterfactual=`273.84s`；InternVL audit/full/counterfactual resume=`861.31/5343.81/131.57s`。Qwen2/LLaVA 最终 audit no-op verification 为 `6.24/4.99s`，不能冒充此前完整 audit 总耗时。Detector trainer 没有独立 elapsed 字段；本轮双 GPU 扩展训练加 bootstrap 观察约十余分钟，以 39 checkpoints/模型和 10,000 resamples 验收，不伪造更精确时间。
- 可恢复失败全部保留：Qwen2 image `571678` 的整层 FP32 copy OOM、`213525` 的 K64 chunk256 OOM；Qwen3 `150779` 因保留非代表层 capture 导致全部 chunk OOM；InternVL counterfactual `120777` 因缓存多个完整 32-layer pixel captures OOM。对应根修复分别是当前 norm+FFN 原位 FP32、统一 `256/128/64/32` fallback、立即释放非代表层 capture、只缓存四个所需层的 visual `h_prev`/target row/logits。所有失败 case 都由 resume 原位重试，无跳过。LLaVA 与 Qwen3 全量从单 rank 安全转双 rank 的操作性中断也保留 resume 命令。
- 最终验收：相关定向测试 `64/64 PASS`，`py_compile`、`git diff --check` PASS；26 份 checksum manifest、2823 个唯一文件、32,562,887,917 bytes 已独立逐字节复算 PASS。主报告为 `ffn_visual_source_attribution_report.md`；旧 `jffn_second_round_incremental_validation_report.md` 已加 superseding addendum，旧 bbox 明确降为 auxiliary sanity check。最终状态 `FORMAL COMPLETE / PASS`。
- Union-Top32 JS 追加验收：四模型分别为 5 行结果、15 个 seed 预测和 15 个 checkpoint，JSON 全部 finite，均明确记录 `bootstrap=NOT_RUN_BY_USER_REQUEST`；新增及相关定向测试 `11/11 PASS`，`py_compile`、`git diff --check` PASS，两张 GPU 均释放。一次只读进度查询误用系统 Python，因该环境没有 torch 报 `ModuleNotFoundError`；立即改用正式 vicr Python 后查询通过，四个实验进程均未受影响。
- Snet/D_OT+strength 追加验收：四模型各 2 行结果、6 个 seed 预测和 6 个 checkpoint，结果与两类曲线 CSV 全部 finite，8 张 PNG 已逐图检查，均明确记录不做 bootstrap；相关定向测试最终 `12/12 PASS`，`py_compile`、`git diff --check` PASS，GPU 释放。新增矩阵测试首次用 `assert_array_equal` 比较 float32 与 Python float，最大 `1.19e-8` 的表示误差导致 1 项失败；改成 `assert_allclose` 后通过，正式训练在修正前尚未启动。最终复核误用默认 `/opt/conda/bin/python -m pytest` 时因环境未安装 pytest 在收集前退出；随后改用正式 `vicr` 环境的 `unittest`，12 项全部通过，未修改依赖。
- Bounded-strength 追加验收：四模型各 2 行结果、6 个 seed 预测、6 个 `model.pt` checkpoint，JSON/CSV 数值全部 finite，逐层 `tau>0`，所有 bounded 值均满足 `0<=x<1`，四张曲线已目视检查，GPU 释放；新增及相关定向测试 `13/13 PASS`，`py_compile`、`git diff --check` PASS。第一次产物审计错误假设 checkpoint 名为 `best_model.pt` 而触发断言，实际训练器固定保存 `model.pt`；修正只读审计路径后四模型全部 PASS。另一次临时汇总误读取一个旧 detector JSON 的不存在 `summaries` 键而报 `KeyError`，随后直接从已核验 CSV 汇总，不影响任何训练或 artifact。首次合并补丁也因补丁工具禁止同文件多次声明而未应用，拆分后正常完成。
- Relative-strength 追加验收：四模型各 2 行结果、6 个 seed 预测、6 个 `model.pt` checkpoint；JSON/CSV 全部 finite，所有 train/test 值满足 `0<=x<1`，四张 REAL/HALL 曲线已逐图检查，四模型 image split 均为 `3200/800` 且交集为 0，GPU 释放。新增及相关定向测试 `14/14 PASS`，两个脚本 `py_compile` 与 `git diff --check` PASS。只读产物汇总首次调用 `jq` 时发现环境未安装该命令；改用标准库读取。随后第一版 CSV finite 审计误把字符串 `model` 列转成浮点而报 `ValueError`，限定到五个数值列后四模型全部 PASS；两次均未修改实验产物或训练状态。
- Log1p-strength 追加验收：四模型各 2 行结果、6 个 seed 预测、6 个 `model.pt` checkpoint，JSON/CSV 和全部预测 finite，四张曲线已逐图检查，GPU 释放；新增及相关定向测试 `15/15 PASS`，两个脚本 `py_compile` 与 `git diff --check` PASS。定位项目 `AGENTS.md` 时首次对整个 `/home/apulis-dev` 执行 `rg --files`，因无权读取 `hpc-home` 和两个 NLTK 目录而返回 permission-denied；命令仍找到目标文件，随后直接读取仓库内 `AGENTS.md`，没有访问或修改这些目录。一次等待测试时误把 nested exec 的 chunk id 传给 cell-wait 接口而得到 `exec cell not found`，随即使用真实 session id 续取，测试进程正常完成且未重跑训练。

## 2026-09-03 risk+EV 的单输出 BCE 与双输出 CE 头对照

- 按用户要求只将三隐藏层探针末端由 `Linear(32,1)+BCEWithLogitsLoss` 换为 `Linear(32,2)+CrossEntropyLoss`；输入仍是 original `old_hpre_cos risk + hpre_raw_logit_gauss mass_x_cosine EV`，并完整复用 DHCP 类别反频率 `WeightedRandomSampler`、现有图片级 split、`[128,64,32]`、batch 256、100 epochs、seeds 43/44/45、无标准化、minimum-sampled-train-loss checkpoint 和 train-REAL-F1 阈值。双输出正类概率固定取 `softmax(logits)[:,1]`，其中 label 1=REAL；固定阈值结果使用 0.5，与双输出 argmax 等价。不做 bootstrap。
- `scripts/train_torch_probe_feature_sets.py` 以默认不变的 `output_mode` 小开关支持单输出 BCE/双输出 CE；新增 `scripts/train_dhcp_two_output_risk_ev.py` 和 `tests/test_two_output_torch_probe.py`。两个模型均完成 3 seeds，所有双输出 checkpoint 的末层权重形状核验为 `(2,32)`，配置确认 CE、DHCP replacement sampler、无 loss sample-weight。
- LLaVA：train-REAL-F1 阈值下，单输出/双输出 AUROC=`0.889179/0.887270`，Hall-AUPR=`0.657014/0.661106`，Hall-P/R/F1=`0.652399/0.616831/0.633924` 与 `0.636341/0.631406/0.633671`。双输出相对变化为 AUROC `-0.001908`、AUPR `+0.004092`、Hall-F1 `-0.000254`，没有稳定净提升。固定 0.5/argmax 时 Hall-P/R/F1 从 `0.513757/0.879643/0.648489` 变为 `0.556350/0.812882/0.660075`，表现为 precision 上升、recall 下降、F1 `+0.011586`。
- InternVL：train-REAL-F1 阈值下，单输出/双输出 AUROC=`0.846005/0.846307`，Hall-AUPR=`0.506057/0.500854`，Hall-P/R/F1=`0.528139/0.470063/0.497310` 与 `0.527285/0.443253/0.481276`。双输出 AUROC仅 `+0.000302`，AUPR `-0.005202`、Hall-F1 `-0.016034`。固定 0.5/argmax Hall-P/R/F1 从 `0.448831/0.674710/0.535904` 变为 `0.453098/0.688114/0.544742`，F1 `+0.008838`。
- 结论：两个参数化在二分类表达能力上近似等价，双输出 CE 多了一个不影响 softmax 概率的共同平移自由度，实际差异主要来自优化和校准。当前三 seed 中，双输出没有改善排序能力；它只在固定 0.5/argmax 下令 Hall-F1 小幅提高，而重新选择训练阈值后 LLaVA持平、InternVL反而下降。因此没有证据把主方法从更简洁的单输出 BCE 改成双输出 CE。
- 跨模型汇总为 `outputs/dhcp_sampler_one_vs_two_output_risk_ev_2model.{md,json}` 与 `_metrics.csv`；各模型详情、6行 seed 对照、3个新 checkpoint 和图位于 `results/jffn_second_round/dhcp_sampler_two_output_risk_ev/`。JSON全部 finite、三 seeds与图文件完整，GPU已释放。
- 验证：相关 `py_compile`、最终 `git diff --check` 和 sampler/weight/双输出定向单测通过。双输出 tiny test 首次使用13行、batch 4，末 batch 仅1行触发 BatchNorm `Expected more than 1 value per channel`；fixture改为12行后 `13/13` PASS，正式实验 batch/样本数不受影响。一次只读 artifact 展示使用系统未安装的 `jq`，命令以127退出；随后用正式 vicr Python 完成相同 JSON/checkpoint 核验，不影响任何实验 artifact。

## 2026-09-02 original risk+EV 的 DHCP 反频率 sampler 消融

- 按用户要求将上一轮平方根 HALL loss-weight 改为 DHCP 官方训练代码使用的类别反频率重采样：训练集中每行权重为 `1/N_class`，`WeightedRandomSampler(num_samples=N_train,replacement=True)`，因此每个 epoch 的期望 REAL/HALL 抽样比例为 1:1；loss 保持普通未加权 BCE。为确保单变量对照，特征仍为 original `old_hpre_cos risk + hpre_raw_logit_gauss mass_x_cosine EV`，并严格复用现有图片级 split、`[128,64,32]` 三隐藏层头、batch 256、最多100 epochs、seeds 43/44/45、无标准化、minimum-train-loss checkpoint。该实验只复现 DHCP 的 sampler，不把 head 同时改成 DHCP 的单隐藏层128维头。
- `scripts/train_torch_probe_feature_sets.py` 新增互斥的 `train_sampler_weights` 入口：使用确定性 seed 的有放回 `WeightedRandomSampler`；明确禁止与 `train_sample_weights` 同时启用；训练完成后在 config/metrics 中记录 sampler 类型、replacement 和每 epoch 抽样数。新增 `scripts/train_dhcp_sampler_risk_ev.py` 与 `tests/test_dhcp_sampler_risk_ev.py`。报告同时复用无权重与平方根权重结果，并给出当前 full-train REAL-F1 阈值及固定0.5（二分类 argmax 等价）结果；按用户先前要求不做 bootstrap。
- LLaVA 训练 REAL/HALL=`9539/2778`，DHCP HALL/REAL 单行抽样权重比=`3.433765`。train-REAL-F1 阈值下，无权重/平方根/DHCP 的 AUROC=`0.888141/0.889209/0.889179`，Hall-AUPR=`0.656175/0.656811/0.657014`，Hall-P/R/F1 中 DHCP=`0.652399/0.616831/0.633924`；相对无权重 AUROC/AUPR仅 `+0.001037/+0.000839`，Hall-F1 `-0.010408`。固定0.5时 DHCP Hall-P/R/F1=`0.513757/0.879643/0.648489`，相对无权重主要把 recall 从 `0.690644` 推至 `0.879643`，但 precision 从 `0.620203` 降至 `0.513757`，F1没有提升。
- InternVL 训练 REAL/HALL=`7985/1393`，HALL/REAL 单行抽样权重比=`5.732233`。train-REAL-F1 阈值下，无权重/平方根/DHCP AUROC=`0.848825/0.848040/0.846005`，Hall-AUPR=`0.502827/0.509767/0.506057`，DHCP Hall-P/R/F1=`0.528139/0.470063/0.497310`；相对无权重 AUROC `-0.002819`、AUPR `+0.003230`、Hall-F1 `+0.031000`。固定0.5时 DHCP=`0.448831/0.674710/0.535904`，比无权重 Hall recall/F1 增加 `+0.183200/+0.051395`，也略高于平方根权重的 Hall-F1 `0.532949`。
- 结论：DHCP sampler 没有改善 risk+EV 的排序质量（两模型 AUROC约不变或下降），主要改变输出校准并大幅提高固定0.5下的 HALL recall；代价是 precision下降。full-train REAL-F1 重新选阈值后，这种变化在 LLaVA 被抵消并使F1下降，在更不平衡的 InternVL仍保留约3.1点 Hall-F1收益。因此类别平衡对 InternVL更有用，但不能解释为特征本身变得更可分。
- 结果位于两模型各自 `results/jffn_second_round/dhcp_sampler_risk_ev/`；跨模型汇总为 `outputs/dhcp_sampler_risk_ev_2model.{md,json}` 与 `_metrics.csv`。两模型各3个新 checkpoint、各9行三策略逐seed指标、JSON全部 finite，sampler config 均确认 loss weighting=`none`、sampling=`weighted_random_replacement`，GPU已释放。
- 验证：相关 `py_compile`、`git diff --check`、sampler/loss-weight/平方根消融定向单测 `10/10` PASS，两个模型的样本数、3个 checkpoint、9行 CSV、等类别总采样质量及图文件均通过核验。一次额外只读 artifact 审计最初用字典浮点精确相等断言，LLaVA 的 REAL 总采样质量因浮点求和得到 `1.0000000000000002` 而触发 `AssertionError`；改为 `1e-12` 容差后全部通过，训练与 artifact 未受影响。另一次展示 CSV 使用了系统未安装的 `column` 命令，表格展示子命令报 `command not found`，原始 CSV 与实验结果均正常。

## 2026-09-02 original risk+EV 平方根 HALL sample-weight 消融

- 按用户要求保持 `label 1=REAL, label 0=HALL` 不变，仅在训练损失中对每条 HALL 样本使用 `sqrt(N_REAL/N_HALL)` sample weight，REAL 权重为1；权重只从训练 rows 计算，测试集完全不加权。首轮只验证当前 original `old_hpre_cos risk + hpre_raw_logit_gauss mass_x_cosine EV`，不加入 S、不运行 P_JFFN risk；复用相同图片级 split、`[128,64,32]` MLP、batch 256、100 epochs、seeds 43/44/45、无特征标准化和 train-REAL-F1 阈值，不做 bootstrap。
- `scripts/train_torch_probe_feature_sets.py` 新增可选 `train_sample_weights`：无权重调用保持原 `BCEWithLogitsLoss` 路径；加权路径使用逐样本 BCE 并按 `Σw_i loss_i/Σw_i` 归一化，checkpoint依据加权 train loss，训练后阈值仍在未加权训练 rows 上选择。新增 `WeightedMatrixDataset` 的形状、有限性和正权重校验；保存的 metrics/config 明确记录 sample weighting。新增实验脚本 `scripts/train_sqrt_hall_weight_risk_ev.py`，以及 `tests/test_weighted_torch_probe.py`、`tests/test_sqrt_hall_weight_risk_ev.py`。
- LLaVA 训练 REAL/HALL=`9539/2778`，HALL weight=`1.853042`。默认 train-REAL-F1 阈值下，无权重 AUROC/Hall-AUPR/Hall-P/Hall-R/Hall-F1=`0.888141/0.656175/0.632111/0.657734/0.644332`；加权=`0.889209/0.656811/0.641582/0.635637/0.637918`。AUROC/AUPR仅 `+0.001068/+0.000636`，Hall recall/F1反而 `-0.022097/-0.006414`，因此该默认协议下 LLaVA 没有实质收益。
- InternVL 训练 REAL/HALL=`7985/1393`，HALL weight=`2.394208`。默认阈值下，无权重=`0.848825/0.502827/0.540956/0.412869/0.466311`；加权=`0.848040/0.509767/0.527547/0.436997/0.477875`。AUROC `-0.000785` 基本不变，Hall AUPR/recall/F1 分别 `+0.006940/+0.024129/+0.011564`，以 precision `-0.013409` 为代价，符合类别权重主要改善少数类召回的预期。
- 加权后的 train-REAL-F1 决策阈值整体下降（LLaVA三 seed 从 `[0.532159,0.480259,0.411247]` 到 `[0.308665,0.366833,0.247987]`；InternVL从 `[0.528221,0.347256,0.539726]` 到 `[0.323023,0.230509,0.350705]`），抵消了一部分 HALL 加权带来的召回变化。固定0.5阈值可分离这个效应：LLaVA Hall P/R/F1 从 `0.620203/0.690644/0.651382` 变为 `0.548077/0.827927/0.658482`，InternVL从 `0.506833/0.491510/0.484509` 变为 `0.438029/0.697051/0.532949`。因此权重确实让模型更倾向检出 HALL，但当前 train-REAL-F1 阈值重新校准后，最终收益主要只保留在更不平衡的 InternVL。
- 输出位于两模型各自 `results/jffn_second_round/sqrt_hall_weight_risk_ev/`；跨模型汇总为 `outputs/sqrt_hall_weight_risk_ev_2model.{md,json}` 与 `_metrics.csv`。每模型新增3个 weighted checkpoint，并精确复用3个无权重 seed 结果；结果 JSON 全部有限、6行 seed CSV 和两张图片已核验，GPU/训练进程均已释放。
- 验证：weighted BCE 数值、非法权重、训练集权重公式、聚合与原无权重路径回归均通过；相关 `py_compile` 和 `git diff --check` PASS。第一次附加回归命令误写不存在的测试类名 `ExtractionModeTest`，导致 unittest loader `AttributeError`；改用真实类名 `ExtractionModeTests` 后通过，属于测试定位命令错误，未进入实验或影响 artifact。无关的 `docs/CODEX_CONVERSATION_HANDOFF_20260830.md` 保持未触碰。

## 2026-09-02 LLaVA/InternVL risk+Union-S 无 EV 消融

- 按用户要求在上一轮完全相同的 Union support、split 和 MLP 协议下去掉 EV，比较 original `old_hpre_cos` risk-only / risk+Union-S 与 P_JFFN risk-only / risk+Union-S。`S_U=Σ_{j∈U}||J_f(z)a_j||₂/(Σ_{j∈U}||a_j||₂+epsilon)`，`U=Top32(P_JFFN)∪Top32(Q_hpre_raw_logit_gauss)`；特征维度为 risk-only 32、risk+S 64，缓存中的 EV 明确不进入矩阵。三 seed 43/44/45、`[128,64,32]`、batch 256、100 epochs、无标准化、minimum-train-loss checkpoint、train-F1 threshold；不做 bootstrap。
- 新增 `scripts/train_union_topk_risk_s_no_ev.py` 与 `tests/test_union_topk_risk_s_no_ev.py`，直接读取上一轮 134.04/102.86 MB compact cache，不再扫描约 6.3/2.2 GB paired shards。两模型在 GPU 0/1 并行完成，每模型 4 feature sets×3 seeds=12 个新训练头。
- LLaVA original risk-only 的 AUROC/Hall-AUPR/Hall-F1=`0.870446/0.630539/0.587302`，+S=`0.886082/0.655453/0.631205`，增量=`+0.015636/+0.024915/+0.043903`。P_JFFN risk-only=`0.847202/0.594926/0.524126`，+S=`0.884921/0.666505/0.622577`，增量更大，为 `+0.037719/+0.071579/+0.098451`。
- InternVL original risk-only=`0.762005/0.371828/0.280208`，+S=`0.847181/0.523135/0.475387`，增量=`+0.085176/+0.151307/+0.195180`。P_JFFN risk-only=`0.737744/0.323021/0.275036`，+S=`0.848088/0.508339/0.485323`，增量=`+0.110344/+0.185317/+0.210287`。InternVL 的 S-only AUROC=`0.848491`，与两种 risk+S 持平甚至略高，说明该模型中排序能力几乎由 S 主导；risk 主要改善 train-F1 阈值下的 Hall-F1。
- 与含 EV 版本对照：在 original risk+S 上再加 EV，LLaVA/InternVL AUROC 分别再增 `+0.005712/+0.019424`，说明 EV 对 original risk 尤其 InternVL 仍有独立信息；在 P_JFFN risk+S 上再加 EV仅增 `+0.003358/+0.004359`，InternVL Hall-AUPR只变 `+0.000224` 且 Hall-F1变 `-0.003101`，EV 对该组合的边际贡献很小。若追求紧凑方案，P_JFFN risk+S 已接近 risk+EV+S；若追求最高 AUROC，保留 EV 仍略优。
- 输出位于两模型各自 `results/jffn_second_round/union_topk_risk_s_no_ev/`；跨模型汇总为 `outputs/union_topk_risk_s_no_ev_2model.{md,json}` 与 `_metrics.csv`。两模型结果 JSON 全部有限，各自4组维度、12行 seed CSV、12个 checkpoint完整，两张 metrics PNG 已目视检查，GPU和训练进程均已释放。相关 `py_compile`、`git diff --check` 与定向单测 9/9 PASS；无关的 `docs/CODEX_CONVERSATION_HANDOFF_20260830.md` 保持未触碰。

## 2026-09-02 LLaVA/InternVL 全量 Union-TopK I/R/S 检测消融

- 按用户要求复用已完成的两轮 JFFN 分片，不重新运行 VLM/JVP。每层固定区域 `U=Top32(P_JFFN)∪Top32(Q_hpre_raw_logit_gauss)`；定义 `I_U=Σ_{j∈U}||a_j||₂`、`R_U=Σ_{j∈U}||J_f(z)a_j||₂`、`S_U=R_U/(I_U+epsilon)`。I/R/S 联合项均为 block 拼接，不是先做分类器级归一化或数值相加。两种 risk 分别是原始 `old_hpre_cos` P 与 `P_JFFN`，二者共享相同 target Q、Union support、`sqrt_matched_state` cost 和现有 `mass_x_cosine` EV。
- 新增 `scripts/train_union_topk_irs_ablation.py` 与 `tests/test_union_topk_irs_ablation.py`，并扩展 `scripts/train_union_topk_region_s_mlp.py::collect_matrices()` 返回对齐的 old risk、JFFN risk 与 EV。每模型比较 I-only、R-only、S-only、I+R+S，以及 original/P_JFFN `[risk,EV]` 分别追加 I、R、S、I+R+S，共 14 feature sets × seeds 43/44/45=42 组结果。严格复用 `[128,64,32]` MLP、batch 256、100 epochs、minimum-train-loss checkpoint、train-F1 threshold、现有图片级 split、无特征标准化；按用户要求不做 bootstrap。
- LLaVA 全量 cohort 为 14951 unique positions、15463 mentions，train/test=`12317/3146`。I/R/S/I+R+S 单独 AUROC/Hall-AUPR/Hall-F1 分别为 `0.873471/0.641057/0.581208`、`0.878886/0.651995/0.604126`、`0.874894/0.625940/0.607913`、`0.884852/0.650374/0.627091`。I+R+S 的 AUROC 与 Hall-F1 最好，说明绝对 write、FFN response 与相对增益含有互补信息；单信号中 R 的 AUROC/AUPR 最好，S 的 Hall-F1 最好。
- LLaVA original `[risk,EV]`=`0.888141/0.656175/0.644332`；+I=`0.888888/0.667713/0.658052`，+R=`0.893297/0.663003/0.638730`，+S=`0.891793/0.672379/0.640199`，+I+R+S=`0.894870/0.670014/0.649803`。因此联合项 AUROC 增量最大 `+0.006729`，+S 的 Hall-AUPR 增量最大 `+0.016204`，+I 的 Hall-F1 最好。P_JFFN `[risk,EV]`=`0.876717/0.659444/0.611740`；+I/+R/+S/+IRS AUROC 分别为 `0.883542/0.887541/0.888278/0.890868`，+IRS 的 AUROC/Hall-AUPR/Hall-F1 增量为 `+0.014150/+0.014641/+0.021773`。
- InternVL 全量 cohort 为 11630 unique positions、11759 mentions，train/test=`9378/2381`。I/R/S/I+R+S 单独=`0.827443/0.476635/0.360788`、`0.832901/0.484103/0.287482`、`0.848491/0.521512/0.374066`、`0.854411/0.520276/0.431676`。单信号中 S 明显最强，联合项进一步取得最高 AUROC/Hall-F1，但 Hall-AUPR 与 S 基本持平。
- InternVL original `[risk,EV]`=`0.848825/0.502827/0.466311`；+I=`0.849898/0.520156/0.433004`，+R=`0.859690/0.533008/0.459017`，+S=`0.866605/0.552289/0.500021`，+IRS=`0.867020/0.545717/0.490279`。+IRS 的 AUROC 最高，但 +S 的 Hall-AUPR/Hall-F1 更高，是跨指标最稳定的单一追加项。P_JFFN `[risk,EV]`=`0.840525/0.483769/0.478527`；+I/+R/+S/+IRS AUROC=`0.846019/0.856753/0.852447/0.862542`，+IRS 的 AUROC/Hall-AUPR 增量=`+0.022017/+0.048513`，Hall-F1 则 +S 略高于 +IRS。
- 曲线解释：LLaVA 的 I/R/S 分别有 25/25/28 层 REAL>HALL，最强层分别为 L1/L1/L19；InternVL 的 I、R 全部 32 层 REAL>HALL，S 为 19/32 层。I/R 是区域内视觉写入量和经过 FFN Jacobian 后的响应量，受总能量与 Union 大小影响；S 消除了输入能量尺度，更接近 FFN 局部放大效率。I 与 R 高相关，所以全拼通常提高排序型 AUROC，却不保证阈值相关 Hall-F1 最大；综合两模型，S 是最稳定的单独新增信号，I+R+S 则更适合追求 AUROC。
- 输出位于两模型各自 `results/jffn_second_round/union_topk_irs_ablation/`，跨模型汇总为 `outputs/union_topk_irs_ablation_2model.{md,json}` 与 `_metrics.csv`。每模型结果含 14×3=42 个 seed rows；精确复用了 3 组既有头，因此实际新训练 33 个 checkpoint。为避免之后每次读取约 6.3 GB/2.2 GB paired shards，各生成 134.04 MB/102.86 MB compact feature cache。完整 JSON 均有限、14 组维度和 42 行 CSV 通过审计，六张 PNG 已目视检查，GPU/训练进程均已释放。
- 耗时说明：若把前一轮看到的约 24 分钟理解为纯 MLP GPU 训练会高估训练成本；前一轮两模型 66 个独立头共最多 6600 epoch passes，且还包含脚本实现、依赖导入、数 GB 分片的 CPU 反序列化/对齐、绘图和报告。小 MLP 仅占约 486 MiB 显存，主要受大量小 batch 的 Python/调度与 I/O 限制，4090 不会持续满载。本轮通过两卡并行、复用 9 个既有 seed-head/模型和 compact cache，LLaVA 从大分片加载至完成约 10 分钟，InternVL 约 7 分钟；后续从 cache 做新拼接会更快。
- 验证：相关 `py_compile`、`git diff --check` 与 4 个定向模块共 17/17 单测 PASS。一次图片检查将正确文件名误输入为 `realelassen_hall`，只读查看报 file-not-found；随后用正确 `real_hall` 路径检查全部图片，无 artifact 受影响。无关的 `docs/CODEX_CONVERSATION_HANDOFF_20260830.md` 保持未触碰。

## 2026-09-02 LLaVA/InternVL normalized I/R sum 与 aggregate S 消融

- 按用户要求定义 `N=I/||I||₂+R/||R||₂`：每个 mention 分别沿 32 个 decoder 层对 aggregate visual-write norm I 与 aggregate FFN-response norm R 做逐样本 L2 归一化，再逐层相加为 32 维；不做第二次归一化，也不使用跨样本 StandardScaler。S 严格复用分片中已有 aggregate gain `R/(I+epsilon)`。表中的 `N+S` 是两个 32 维 block 拼接为 64 维，不是再次做数值相加。
- 新增 `scripts/train_jffn_normalized_ir_s_ablation.py` 和 `tests/test_jffn_normalized_ir_s_ablation.py`。每模型训练 N-only、S-only、N+S，以及 original old-hpre/P_JFFN 两种 `[risk,EV]` 分别追加 N、S、N+S，共 11 feature sets × seeds 43/44/45 = 33 个 `[128,64,32]` 三隐藏层 MLP；两模型合计 66 个头。split、batch 256、100 epochs、minimum-train-loss checkpoint、train-F1 threshold、无分类器级标准化均保持不变，按用户上一条要求不做 bootstrap。
- LLaVA 全量 cohort 为 80 shards、14951 positions、15463 mentions，train/test=`12317/3146` mentions。N/S/N+S 单独 AUROC/Hall-AUPR/Hall-F1=`0.871648/0.632786/0.562015`、`0.864967/0.622850/0.589011`、`0.878449/0.637600/0.626564`；N+S 对 N 的增量=`+0.006802/+0.004814/+0.064549`。N 曲线 23/32 层 REAL>HALL，最强 L11 `d=-0.9696`、单层最佳方向 AUROC=`0.7665`；S 为 25/32 层、最强 L19 `d=-0.6829/AUROC=0.7036`。
- LLaVA original `[risk,EV]` AUROC/Hall-AUPR/Hall-F1=`0.888141/0.656175/0.644332`；+N=`0.892321/0.664836/0.644156`，+S=`0.893087/0.675664/0.658950`，+N+S=`0.895169/0.678783/0.653397`。N+S 的 AUROC/Hall-AUPR 最好，分别比基线 `+0.007027/+0.022608`；Hall-F1 则 +S 最高。P_JFFN `[risk,EV]`=`0.876717/0.659444/0.611740`；+N=`0.886031/0.666376/0.631153`，+S=`0.883950/0.672963/0.624144`，+N+S=`0.892915/0.680232/0.647254`。这里 N+S 三指标全部最好，对基线增量=`+0.016197/+0.020788/+0.035514`。
- InternVL 全量 cohort 为 80 shards、11630 positions、11759 mentions，train/test=`9378/2381` mentions。N/S/N+S 单独=`0.817755/0.471264/0.356491`、`0.847344/0.508109/0.332601`、`0.864059/0.555407/0.495597`；与 LLaVA 相反，S 单独 AUROC 明显强于 N，但 N+S 对 S 仍提升 `+0.016715/+0.047298/+0.162996`。N 曲线 22/32 层 REAL>HALL、最强 L26 `d=-0.3892/AUROC=0.6105`；S 为 19/32 层、最强 L13 `d=-0.4780/AUROC=0.6434`。
- InternVL original `[risk,EV]`=`0.848825/0.502827/0.466311`；+N=`0.851431/0.510616/0.495720`，+S=`0.866745/0.546172/0.510817`，+N+S=`0.861279/0.554411/0.506169`。+S 的 AUROC/Hall-F1最高，+N+S 的 Hall-AUPR最高；N 在 S 已存在时使 AUROC下降 `0.005466`。P_JFFN `[risk,EV]`=`0.840525/0.483769/0.478527`；+N=`0.846111/0.502062/0.477121`，+S=`0.855042/0.524322/0.485575`，+N+S=`0.864111/0.553556/0.509602`。与 LLaVA 一致，P_JFFN 下 N+S 三指标最佳，对基线增量=`+0.023585/+0.069788/+0.031075`。
- 结论：N 和 S 有互补性，但 N 的独立可迁移性弱于 S；N 在 LLaVA 有较强信号，在 InternVL 明显偏弱。若保留 original risk，`risk+EV+S` 是更稳定的跨模型方案，N+S 只在 LLaVA 提高 AUROC、在 InternVL 更偏向提高 Hall-AUPR。若使用 P_JFFN risk，N+S 在两模型上均为最强组合，说明归一化后的跨层 I/R 形状能补足 P_JFFN risk 丢失的总响应信息。因未做 bootstrap，这些增量只作三 seed 描述性结论。
- 单模型报告/曲线/metrics、33 个 checkpoint 与 progress 位于各自 `results/jffn_second_round/normalized_ir_s_ablation/`；跨模型汇总为 `outputs/normalized_ir_s_ablation_2model.{md,json}` 与 `_metrics.csv`。两模型 stored S 与重算 R/I 最大误差均为 0，I/R 单位范数最大误差均为 `1.192e-7`；所有 66 个 checkpoint、11 组维度、JSON/CSV 有限性和图片均通过审计，GPU 已释放。相关 `py_compile`、`git diff --check` 和回归测试 15/15 PASS。一次只读字段探查误用系统默认 `python` 导致 `ModuleNotFoundError: torch`，改用正式 vicr 解释器后通过；汇总生成后第一次 `sed` 多写了 `_summary` 后缀而报文件不存在，随后读取正确的 `outputs/normalized_ir_s_ablation_2model.md` 通过，均未影响实验 artifact。无关的 `docs/CODEX_CONVERSATION_HANDOFF_20260830.md` 保持未触碰。

## 2026-09-02 LLaVA 全量 aggregate JFFN I/R 曲线与两种 risk 拼接消融

- 按用户要求仅复用已完成的 LLaVA JFFN 分片，不重新运行 VLM/JVP。`I_l=||sum_j a_j^{visual}||_2`、`R_l=||J_f(z_l)sum_j a_j^{visual}||_2`；两种 risk 分别固定为 `old_hpre_cos` 原始 P 与 `new_jffn` P，二者共享 `hpre_raw_logit_gauss` target Q、Union Top-K、`sqrt_matched_state` cost 和同一 `mass_x_cosine` EV。正式 cohort 为 3971 images、15463 mentions；train/test=`3174/797` images、`12317/3146` mentions，test REAL/HALL=`2437/709`。
- 新增 `scripts/train_jffn_i_r_feature_ablation.py` 和 `tests/test_jffn_i_r_feature_ablation.py`，训练 I-only、R-only、I+R，以及两种 `[risk,EV]` 分别追加 I、R、I+R，共 11 feature sets × seeds 43/44/45 = 33 个三隐藏层 `[128,64,32]` MLP。协议保持 batch 256、100 epochs、无标准化、minimum-train-loss checkpoint、train-F1 threshold 和现有图片级 split。用户在全部 33 个头完成后明确要求不做 bootstrap，因此中止正在进行的 bootstrap，并以 `--bootstrap-replicates 0 --resume` 复用全部训练头生成最终结果；报告只给三 seed 均值±标准差，不给显著性 CI。
- I/R 曲线均以 REAL 为主：I 有 26/32 层 REAL>HALL，最强 L1 的 Hall−Real=`-0.098628`、Cohen's d=`-1.0566`、最佳方向单层 AUROC=`0.7796`；R 有 27/32 层 REAL>HALL，最强 L11 为 `-0.958545/-1.0034/0.7732`。这说明真实物体通常得到更强的 aggregate visual write 与其 FFN 局部响应，但两类分布仍有重叠。
- 单独训练的 AUROC/Hall-AUPR/Hall-F1：I=`0.877711±0.001584/0.642977/0.591980`，R=`0.874441±0.000459/0.645850/0.559944`，I+R=`0.881668±0.001390/0.648914/0.604165`。I 单独的 AUROC 和 Hall-F1 高于 R；I+R 对 I 的均值增量为 AUROC `+0.003957`、Hall-AUPR `+0.005937`、Hall-F1 `+0.012185`，说明两者有少量互补但信息高度相关。
- 原始 `[risk,EV]` 基线 AUROC/Hall-AUPR/Hall-F1=`0.888141±0.001422/0.656175/0.644332`；追加 I=`0.890281/0.668572/0.643421`，追加 R=`0.893901/0.666282/0.641042`，同时追加 I+R=`0.892193/0.660835/0.635218`。因此原始 risk 下 R 带来最大 AUROC 增量 `+0.005759`，I 带来最大 Hall-AUPR 增量 `+0.012397`；三种追加均未提高 Hall-F1，同时加入 I/R 不如只加 R。
- P_JFFN `[risk,EV]` 基线为 `0.876717±0.004153/0.659444/0.611740`；追加 I=`0.887550/0.679365/0.634203`，追加 R=`0.888169/0.667310/0.631459`，同时追加 I+R=`0.885945/0.661473/0.631002`。P_JFFN risk 自身比原始 risk 的 AUROC低 `0.011424`，但追加 I/R 的补益更大：I 的 AUROC/Hall-AUPR/Hall-F1 增量=`+0.010832/+0.019921/+0.022462`，R 为 `+0.011452/+0.007866/+0.019719`。只加 R 的 AUROC最高，只加 I 的 Hall-AUPR与 Hall-F1最高；同时加入仍不如单独追加，支持 I/R 冗余解释。
- 最终结果、三张 PNG/PDF、逐层 CSV、33 个 checkpoint/progress 与报告位于 `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/i_r_feature_ablation/`。`results.json` 核验为 80 shards、14951 unique positions、15463 mentions、所有输入有限、11 个维度均符合预期且 `paired_bootstrap={}`。第一次临时只读 finite 审计误把包含逐 seed list 的 JSON 节点当成纯标量字典而触发 `TypeError`；改用递归遍历后完整审计 PASS，artifact 未损坏。相关 `py_compile`、`git diff --check` 和 JFFN/I-R/S 回归单测 12/12 通过；GPU 已释放。未触碰无关的 `docs/CODEX_CONVERSATION_HANDOFF_20260830.md`。

## 2026-09-02 TC-FVPA 当前结果汇总与 GitHub 可审阅快照

- 按用户要求新增 `docs/TC_FVPA_CURRENT_RESULTS_20260902.md`，统一汇总 Qwen2.5 与 LLaVA 已封存正式结果、Qwen3 partial/OOM 失败和 InternVL 未启动状态。Qwen2/LLaVA 的 Local/Path 规模、Path completeness、frozen-write 相关、真 FP32、Shapley 和描述性空间指标均从 compact artifact 重新计算；未运行的 detector/bootstrap/VQA/fixed-QK/activation/pixel/neuron 项继续明确标为 `NOT RUN/BLOCKED`。
- Qwen3 不能升级为正式结果：Local shard 实际为 `3743 success + 1 OOM`，虽然旧 resume 状态误写 PASS；Path 两 rank 均扫描 100 images 后以 `743+703=1446 success`、`5+5=10 OOM` 结束为 FAIL。控制器因 Qwen3 未生成 checksum-verified formal root 于 2026-08-30 23:03 停止，InternVL 没有启动或生成 formal root。
- 将待提交的 report builder 从 LLaVA 专用硬编码泛化为按 `FORMAL_LAYERS_BY_MODEL` 核验冻结层位；Qwen2 `7/14/21/28` 与 LLaVA/InternVL `8/16/24/32` 均可正确判断。FP32 报告现仅在有 measured rows、0 blocked layers、0 failures 时标 PASS；formal scope、handoff 与 verdict 不再错误写死 LLaVA/Qwen。
- 用冻结 LLaVA reports 命令只重建报告、handoff 和 checksum，没有重算或改写数值 shard。重建后 checksum `78/78`、handoff `SHA256SUMS 50/50` 全部匹配；checksum-enabled loader 成功读取 WRITE `3656` rows。定向 6 模块 `20/20` PASS，相关 `py_compile` 与 `git diff --check` PASS。首次测试因报告措辞已泛化但旧断言仍匹配旧句子而 `1/20` 失败，更新断言后完整重跑通过；这是测试预期同步问题，不是实验失败。发布前一次临时 checksum 审计又误把 manifest 的字典 entry 当成纯 digest 字符串而产生 78 个假 mismatch，按 `entry['sha256']` 修正后 78/78 通过，artifact 未损坏。
- GitHub 快照沿用精简策略：纳入汇总、代码/测试、LLaVA reports/manifests/metrics/schemas、约 10 MiB compact CSV.GZ 与小型 Shapley case 表，以及 Qwen3 小型状态/失败日志；不纳入约 486 MiB LLaVA 与 199 MiB Qwen3 根中的 token maps、rank shards、audit vectors、Shapley running estimates 或 tarball。公开快照足以复核汇总统计，但不冒充完整张量归档。

## 2026-08-30 双 RTX 4090 LLaVA TC-FVPA 正式实验完成与终验

- 正式根 `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260830` 已由 detached coordinator PID `499771`（PPID 1、SID/PGID 499771）按冻结命令完成 local、path、真 FP32、Shapley、analyze、counterfactuals 和 reports，顶层日志 `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/tc_fvpa_comprehensive_v1_formal_repaired_20260830.resume01.coordinator.log` 逐阶段记录 complete，最终所有 worker 退出、两张 RTX 4090 均回到 0 MiB/0%。没有运行 VQA benchmark、Qwen3 或 InternVL。
- Local 两 shard 为 `1728+1928=3656` 个唯一 target-layer cases、500 images、四层各 914、REAL/HALL=`2528/1128`，全部 MEASURED、failure=0、stage PASS。首次 rank1 重复 causal position 失败的 traceback、失败 manifest 和旧日志继续保留；修复后只安全复用 rank0 PASS，rank1 重跑通过，没有覆盖失败证据。
- Path 两 shard为 `692+728=1420` 个唯一 target-layer cases、200 images、四层各 355、REAL/HALL=`884/536`，全部 MEASURED、failure=0、stage PASS。所有行均保持 requested/effective `path_batch_size=1`、OOM fallback=0，并完整包含 `3 scalars × K(1/4/8/16/32) × 2 quadratures=30` 项，共 42600 convergence rows。
- 真 FP32 两 shard 各 50 images，共 60144 个 layer-32 MEASURED intervention rows，全部 `final_block_true_fp32`/`torch.float32`、failure=0；layers 8/16/24 各 shard各有一条带 error 的 BLOCKED 记录，因此两个 FP32 stage 诚实为 BLOCKED。Shapley 完成 50 images、300 个 layer-32 cases、38400 running estimates、8/16 regions、128 permutations、failure=0；layers 8/16/24 明确为 `not_in_scope_layers`。
- Analyze stage PASS，生成 3656 case rows、42600 path convergence rows、76032 token-score sample rows和 57464 spatial rows；Parquet 因正式环境无 pandas/Parquet 依赖明确 BLOCKED，权威 PT/CSV.GZ 均存在。Counterfactual 表共 46830 行，其中 frozen-write 46815 MEASURED，另外 15 行诚实 NOT_RUN；fixed-QK、activation patching、pixel counterfactual 均 NOT_RUN，所以 stage 为 BLOCKED。Reports stage PASS，但最终科学 verdict 保持 PARTIAL。
- 终验发现旧 report builder 会把中央 smoke 版 `19_FORMAL_SCOPE_AND_BLOCKER_AUDIT.md` 静态带入并误报本次 formal 计数为 0/NOT RUN；已改为从本根 summary/run_status 动态生成，并修复最终 report-log/checksum 时序。修复后 report 05/06 分别把已达标的正式 Local/Path 标 PASS，report 18 保持 PARTIAL，report 19 逐项保留 lower-layer FP32、非最终层 Shapley、未实现 counterfactual、Parquet、detector/bootstrap、cross-model 与 VQA blocker。数值 shard 未因报告修复改变。
- 最终 output checksum 77/77、handoff 49/49 及内部 `SHA256SUMS` 全部匹配；正式 loader 在 checksum 验证开启时读取 WRITE 3656 rows 成功。PT 递归 finite 检查为 0 个 NaN/Inf；六张 CSV.GZ 表行数、状态和有限值审计通过。两次临时 CSV 审计脚本先后因误把 CSV 当 JSONL、再因漏计 6 条合法 BLOCKED intervention rows 退出，均是只读审计假设错误；第三次使用 `csv.DictReader` 和正确分母通过，未改 artifact。
- 定向 `unittest` 37/37 PASS；`vicr` 环境没有 pytest，附加 pytest 命令在收集前报 `No module named pytest`，不属于实验失败。共享 Qwen3 目录在本次期间确有旧队列写入，但其 hardware manifest 明确是双 RTX 3090、独立 qwen3 exact command；本机 4090 进程树只含 LLaVA，未向 Qwen3/InternVL/VQA 根发出写命令。Git HEAD 仍为 `6e92f8f3d8e44e43f53569ad35561efc2b0e295e`，未 commit、未 push。

## 2026-08-30 双 4090 LLaVA TC-FVPA local 冲突位置去重修复

- LLaVA 正式根 `tc_fvpa_comprehensive_v1_formal_repaired_20260830` 的双卡 local 首轮完成两边各 250 图计算；rank0 以 1728 measured cases、0 failures 落盘并 PASS，rank1 在提交原子 shard 前因 `Duplicate case inside shard: llava_1_5_7b:449731:9:8` 诚实 FAIL。协调器按零失败门禁停止，完整 traceback 保留在 `logs/local_rank01.log` 和 `manifests/run_status.json`；没有进入 Path，也没有覆盖 rank0 产物。
- 根因是 InsLen 官方首 subtoken/首次出现协议允许不同 detected word 指向同一因果位置。`image=449731` 的 REAL `toilet` 与 HALL `toothbrush` 都解析到 `response_index=9,target_token_id=304`；TC-FVPA 的 `_choose_target_indices()` 先按 HALL、REAL 取候选时把同一索引 append 两次，违反仓库既有“同一 `(image_id,response_index)` 只计算一次”的唯一位置原则。不能靠扩展 case key 把同一个预测事件伪装成两个独立 case。
- 最小修复仅在选择器 append 前检查该位置尚未选中；随后原有补位循环用下一个唯一位置补足上限。固定 seed `20260829` 的 500 图 cohort 审计显示旧版只有 `449731:[9,9]`、`325736:[10,10]` 两张发生重复，均在 rank1；修复后重复选择为 0，rank0 选择完全不变，因此已有 rank0 PASS shard 可安全复用。200 图 Path cohort 也包含这两张，修复同时防止后续 Path 落盘失败。
- 新增回归 `test_conflicting_label_position_is_selected_only_once`，覆盖 REAL/HALL 冲突位置只选一次并由下一唯一位置补位。相关 `py_compile`、`git diff --check` 和 7 个 TC-FVPA 定向模块共 30/30 测试通过。恢复运行前仍需保留首轮失败日志，并用新的 coordinator 日志/PID 文件启动 `--resume`，不覆盖失败证据。

## 2026-08-30 gfchair：当前 3090 队列改为 Qwen3 → InternVL，LLaVA 留给 4090

- 按用户要求，将当前双 RTX 3090 机器的后续顺序改为 Qwen3 完成后只运行 InternVL；LLaVA 不再由本机自动启动，留给稍后的 4090 新对话。原队列父 shell `558757` 已先 `SIGSTOP` 并确认 Qwen3 coordinator/path 子进程继续满载，再精确终止该父 shell；Qwen3 PID `558763` 及两个 Path worker `559546/559547` 没有重启或丢失当前进度。
- 新的独立控制器 PID=`593002`，日志为 `outputs/tc_fvpa_qwen3_then_intern_20260830.log`。它等待 Qwen3 coordinator 退出后，先用 checksum-verified loader 检查 Qwen3 L36 `PATH_LOG_PROBABILITY_GAUSS_LEGENDRE_K32`；只有完整封存验证通过才启动 InternVL formal resume。LLaVA 明确不在控制器命令中。
- 当前 Qwen3 Path 两 rank 约完成 `66/100` 与 `67/100` images，但日志已累计 `4+1=5` 个失败 case。正式零失败门禁保持不变：若 Qwen3 因这些 case 结束为 FAIL，控制器会记录 `BLOCKED` 并且不启动 InternVL，等待本对话诊断/修复；不会跳过失败伪造“Qwen3 完成”。
- 4090 新对话应从 GitHub `main` 拉取当前代码，在确认目标机器具备同一模型、COCO4000/CHAIR/InsLen 输入与环境后，仅启动 `llava_1_5_7b` 的 formal root。不要复用当前 3090 的 InternVL root，也不要同时在共享输出根写同一模型。

## 2026-08-30 gfchair：GitHub 阶段性审阅快照

- 按用户要求准备将当前 TC-FVPA 代码与已完成结果发布到 GitHub，供 ChatGPT 先行审阅。新增 `docs/TC_FVPA_CURRENT_RESULTS_20260830.md`，明确分开 Qwen2 已完成正式测量、Qwen3 正在运行、LLaVA/InternVL 排队以及 fixed-QK/pixel/detection/VQA 等 `BLOCKED/NOT RUN` 项；同时修正 runbook 中已经过期的“Qwen 尚未通过门禁”说明。
- Qwen2 完整结果根约 `438 MiB`，其中大型 PT token maps/FP32 shards/audit vectors 不适合普通 GitHub。发布快照只纳入可读 reports、权威 `run_status.json`、metrics、schemas，以及约 `19 MiB` 的 compact CSV.GZ/小型 Shapley case 表；不纳入重复 handoff 副本、环境变量清单和大型张量分片。摘要明确记录该裁剪，不把 GitHub 子集冒充完整本地归档。
- 发布前验证：TC-FVPA 定向单元测试 `42/42` PASS；相关 `py_compile`、`bash -n run_tc_fvpa_comprehensive.sh`、`git diff --check` PASS；敏感信息扫描未发现凭据值。后台 Qwen3 正式 Path 双卡进程保持运行，未因发布操作中断。

## 2026-08-30 gfchair：四模型安全 Path 加速实测与 Qwen3 长前缀门禁修复

- 对四模型统一执行同样本 A/B：2 images、双 shard、四个预注册层、三 scalar、K=`1/4/8/16/32`、两种 quadrature，正式安全 `path_batch_size=1`。按 12 个 case 的 `runtime_seconds` 合计，Qwen3=`313.393→293.604 s`（`1.067x`，下降 `6.31%`），Qwen2=`288.991→274.670 s`（`1.052x`，下降 `4.96%`），InternVL=`422.307→405.685 s`（`1.041x`，下降 `3.94%`），LLaVA=`579.272→563.347 s`（`1.028x`，下降 `2.75%`）。四模型均 12/12 MEASURED、0 failures；各自 528/528 方法 token maps 与旧结果逐元素完全一致，全部 frozen-write LOO observed effects 也逐值一致。
- Qwen3 正式 local 两 rank 首轮完成 `1947+1796=3743` measured cases，但 rank0 在唯一一个超长前缀 `image=259342,response_index=445,L9` 的重复 full-model gradient parity 中 OOM；rank1 PASS、rank0 因该 1 个诚实失败使队列按门禁停止。该 full backward 是每个 case 重复验证正式前已经通过的 Qwen3 L9/L18/L27 多层真实梯度 gate，不是 attribution estimand 本身。
- Qwen3 L9/L18/L27 现固定使用正式前已冻结并通过的 full-row causal suffix gradient 路由；每个 case 仍与 clean capture 检查全词表 logits、target/competitor、三 scalar 和 argmax，超阈值即回退 full replay。这样移除了稀有长前缀的冗余 full backward OOM，同时不改变已通过路由的 Path/LOCAL 数值；Qwen3 安全 A/B 的 528/528 完全一致也覆盖该改动。
- Qwen2 不能采用同样的固定层路由：真实 A/B 发现其 suffix 接受会随 prefix 改变。一次探索性固定路由得到 `-4.80%` 退化且 43/528 方法图变化，已立即撤回并保留独立结果根供审计；恢复原逐 case logit+gradient gate 后重测为 `+4.96%`、route 12/12 与旧正式 case 一致、528/528 图完全一致。该探索结果没有写入正式根。
- 安全 A/B 根分别为 Qwen3 `tc_fvpa_fullk_path_optimized_safe_prevalidated_20260830`、Qwen2 `tc_fvpa_fullk_path_optimized_safe_dynamic_20260830`、LLaVA/InternVL `tc_fvpa_fullk_path_optimized_safe_20260830`。定向 Qwen/path 回归 `5/5` PASS，相关 `py_compile` 通过。
- Qwen3 修复后正式队列已恢复，PID=`558757`、PPID=`1`、SID=`558757`，日志为 `outputs/tc_fvpa_primary4_path_optimized_queue_resume2_20260830.log`；先重跑 Qwen3 失败的 local rank0（rank1 复用 PASS shard），再继续 Qwen3 剩余阶段与 LLaVA/InternVL。Qwen2 已在上一队列完成 FP32/Shapley/analyze/counterfactuals/reports，不重复执行。

## 2026-08-30 gfchair：Path 计算优化、低精度稳定性 A/B 与正式队列恢复

- Path 的真实瓶颈确认在每个 case 的 96 个唯一积分内点（K=`1/4/8/16/32`、trapezoid/Gauss-Legendre 共享端点与重复节点）逐点执行下游 decoder；另有 11 个 frozen-write LOO 原本只需要 scalar score，却仍构造并反传三套随后丢弃的梯度。现已新增 exact-suffix 路径节点 microbatch、LOO score-only 前向、三 scalar 梯度统一入口及 OOM 后释放失败图并自动降为 batch 1 的保护。FP32/FP64 可使用 batched VJP；原生 BF16/FP16 保持原三 scalar 串行 VJP，避免三宽 vmap 增加显存和改变低精度累加顺序。
- Qwen3 同一 2 图、双 shard、12 case、四层、三 scalar、完整 K 网格 A/B：旧版 case runtime 合计 `313.393 s`；稳定优化版、显式 `--path-batch-size 2` 为 `281.250 s`，即 `1.114x`、下降 `10.26%`，12/12 MEASURED、0 failures。短序列早层约 `1.48x`，末层约 `1.50--1.60x`；最长序列早层会自适应回退到 batch 1，约 `1.04--1.05x`。稳定版 rank elapsed=`186.41/110.95 s`，峰值 allocated=`24.25/24.35 GB`（十进制）。结果根为 `.../qwen3_vl_8b/.../tc_fvpa_fullk_path_optimized_stable_20260830/`。
- 数值门禁没有只看运行成功：稳定版 36/36 LOCAL token maps 与旧版逐元素完全一致；BF16 路径内点因 batch=2 改变 GEMM 的 M 维，全部 PATH 图的最大绝对差 `0.0040283`、最大相对差 `0.12177`、最低 cosine `0.994753`。正式主 `Gauss-Legendre K32` 的最低 Spearman=`0.998967`、Top-1 全部一致，但个别 Top-4 overlap=`0.75`。考虑已完成 Qwen2 正式 Path 使用旧 batch-1 形状，四模型正式默认冻结为保守 `--path-batch-size 1`，保证跨模型/既有结果一致；batch 2 只作为显式可选吞吐优化，不进入当前正式队列。
- 修复双 rank 共享结果竞争：FP32 不再让两个 rank 写同一个 `.tmp`/最后一写覆盖半份 CSV，而是以 rank-local PT shard 为权威、文件锁内重建合并后的 `tables/interventions.csv.gz` 与 metrics；现有 Qwen2 两 shard无需重算已汇总为 `216384` measured rows、0 blocked/0 failures、`all_ranks_present=true`，CSV 行数逐行核验一致。Shapley 的共享 case/running-estimate 表与 summary 同样改为加锁从全部 rank shard 合并，避免正式报告只看到最后一个 rank。
- 验证：相关 `py_compile` 与 `git diff --check` 通过；Path/Qwen adapter 定向测试 `16/16` PASS；双 rank 汇总临时目录烟测同时覆盖 FP32 CSV/metrics 和 Shapley PT/metrics，均 PASS。最初 batch-2 + 低精度 batched-VJP 真机 A/B 曾在长序列早层触发 3 个 OOM case，已作为失败证据保留在 `tc_fvpa_fullk_path_optimized_b2_20260830/`；修复后的 adaptive/stable 两轮均 12/12、0 failures，未删除失败记录。
- 已恢复四模型正式断点队列，PID=`539400`、PPID=`1`、SID=`539400`；日志为 `outputs/tc_fvpa_primary4_path_optimized_queue_20260830.log`。顺序为 Qwen2 FP32 resume/完整汇总→Shapley/analyze/counterfactuals/reports，再依次 Qwen3、LLaVA、InternVL 全阶段；`set -euo pipefail` 保持任一真实失败即停。正式 Path 使用默认 batch 1；本轮优化的无损收益主要来自 LOO score-only，未用 batch-2 A/B 的 `10.26%` 直接缩短正式 ETA。

## 2026-08-30 gfchair：Qwen 真 FP32 干预切批修复与四模型剩余工期校准

- Qwen2.5 正式 FP32 在两个 50-image shard 跑到结尾后分别保留 `178/147` 个失败 case，根因不是 102552 行落盘，而是每个 layer-case 将 `1 + 14~15 directions × 8 eta × 2 signs = 225~241` 个真 FP32 query 变体一次性扩展；GQA `_repeat_kv` 额外申请约 `1.23 GiB` 时，24 GiB RTX 3090 只剩约 `675 MiB`，因此 CUDA OOM。旧 shard 已诚实保留为 FAIL，两个正式队列均已按门禁停止。
- `features/qwen_fp32_suffix.py` 已新增 `query_batch_size`：共享 causal-prefix K/V 只做一次 head expansion，query 默认按 32 个 microbatch 执行；前向和 autograd 图仍通过 `torch.cat` 合并，未减少 15 strategies、8 eta、正负扰动或 3 scalars。`scripts/run_tc_fvpa_fp32_causal.py` 复用 `--logit-batch-size` 控制 Qwen query microbatch，并将其写入 payload/metrics/run status；协议升级为 `tc_fvpa_true_fp32_causal_suffix_v3_microbatched`。
- 同时修正 FP32 resume 门禁：不再因 shard 文件存在就无条件写成 PASS。只有协议、microbatch 配置匹配且 failures 为空才复用；旧协议或包含失败的 shard 会重算，成功结果提交前旧 shard 仍保留。
- 真机双卡回归：Qwen2.5 4/4 图、8256 measured rows、0 blocked/0 failures，单 shard 计算 elapsed=`23.66/17.37 s`，峰值=`19.97/20.25 GiB`；Qwen3 2/2 图、4032 rows、0 blocked/0 failures，elapsed=`15.52/11.84 s`，峰值=`21.31/21.22 GiB`。两模型 batch=32 均在 RTX 3090 上通过，运行后 GPU 已释放。
- 为避免用旧末层 smoke 低估正式 Full-K Path，补做三模型各 2 图、双 shard、四层、三 scalar、K=`1/4/8/16/32` 真机校准：Qwen3 两 shard 为 `210.25 s/8 cases`、`119.65 s/4 cases`；LLaVA 为 `394.02/8`、`207.97/4`；InternVL 为 `276.00/8`、`178.57/4`，全部 PASS/0 failures。结合 Qwen2 正式 Path 已完成的 `16558/17917 s`，从当前进度算，可运行阶段顺序完成的安全墙钟估计约 `30--36 h`；其中 Qwen3/LLaVA/InternVL Path 约 `5.5/9.5/8.2 h`，其余 local/FP32/Shapley/analyze/report 与共享盘冷启动合计约 `4--8 h`。
- 该 `30--36 h` 仅表示当前已实现阶段跑完并生成诚实报告，不表示综合协议全部 PASS。LLaVA/InternVL 低层 true-FP32、fixed-QK、activation patching、pixel counterfactual 等仍未实现，相关项会保持 BLOCKED；在补实现和新增真机门禁前，无法给“所有预注册实验均 PASS”承诺完成时间。
- 已按用户先前“修复后执行正式实验”的授权启动新断点队列，PID=`522887`、session/PPID1 均确认脱离当前终端，日志为 `outputs/tc_fvpa_primary4_microbatched_queue_20260830.log`。顺序为 Qwen2.5 剩余阶段（先重算两个旧 OOM FP32 shard）→ Qwen3 全阶段 → LLaVA 全阶段 → InternVL 全阶段；`set -euo pipefail` 保证任一阶段非零退出时停止，不会越过失败伪造后续完成。
- 验证：相关 Python `py_compile` 通过；`tests.test_qwen_fp32_suffix + tests.test_tc_fvpa_qwen_adapter + tests.test_tc_fvpa_path_runner` 共 `5/5` PASS，新增测试覆盖 microbatch/未切批前向与 VJP 等价；成功 v3 shard 的 `--resume` 真实复用校验为 PASS。

## 2026-08-29 gfchair：TC-FVPA/Riesz/path 综合框架与四架构真实烟测

- 按用户给定综合协议先冻结 `outputs/tc_fvpa_comprehensive_v1/reports/00_PREREGISTERED_HYPOTHESES.md`，并新增问题/符号、旧管线审计和正式范围阻塞审计；没有把旧 JFFN 结果或本轮烟测升级为新正式结论。核心实现位于 `features/ffn_visual_path_attribution.py`：支持三种 target scalar、fixed clean competitor、批量 JVP、VJP/Riesz pullback、local duality、current-block zero-write vector/scalar path、trapezoid/Gauss-Legendre K 网格、frozen-write LOO、direct-unembedding、对称差分和保留正负质量。另新增 Shapley、SwiGLU product-rule、geometry/whitening 与可审计 artifact/loader 模块。
- 新增全部要求的启动器：`scripts/run_tc_fvpa_{comprehensive,path_attribution,fp32_causal,counterfactuals,shapley}.py`、`scripts/analyze_tc_fvpa_comprehensive.py`、`scripts/build_tc_fvpa_reports.py` 和根 `run_tc_fvpa_comprehensive.sh`。统一支持 model/device/devices/resume/shard/output/seed/layers/scalars/K/dry-run/smoke/formal；原子 PT shard、锁保护状态、输入/输出 SHA256、完整 token map、CSV.GZ/可选 Parquet、reports 00--19、handoff bundle 及 tar.gz 均已接通。缺失值不会用 0 代替，负贡献不裁剪。
- LLaVA 真实开发烟测：旧版 1 图、1 HALL target、L32、三标量、两种积分法、K=1/4 path stage/case 分别为 `12.725/4.888 s`；修复后 exact final-block suffix 的 stage/case 为 `7.902/0.361 s`，全词表 clean-logit parity 最大误差 `0.0078125`，12 条 convergence 与 11 个 frozen-write intervention 完整、failures=0。另有真 FP32 336 rows 和 8-region×8-permutation Shapley 3 scalar rows。旧结果根为 `.../tc_fvpa_comprehensive_v1_smoke_20260829/`，修复烟测根为 `.../tc_fvpa_comprehensive_v1_smoke_splitfix_20260829/`。
- InternVL packed-QKV 旧烟测的 `248.028 s` 是含初始化/共享盘 I/O 的 stage elapsed，不是单 case 路径耗时；旧持久化 case runtime 实为 `3.828 s`。修复后 BF16 exact final-block suffix 的 stage/case 为 `11.196/1.041 s`；全词表/target-competitor/三 scalar parity 最大误差为 `0.125/0/0.001483`，argmax 一致，均在持久化的 BF16 `0.25` 阈值内，12 条 convergence、11 个 intervention 完整且 failures=0。修复烟测根为 `.../tc_fvpa_comprehensive_v1_smoke_splitfix_bf16_20260829/`。
- Qwen causal-prefix 修复已完成：Qwen2.5/Qwen3 均复用已有精确 token-ID 追加路径，目标 token 不进入前缀，并按 processor 的 image-pad span 与 `image_grid_thw` 动态解析视觉范围；path/FP32/Shapley 的四模型门槛已打开。正式层位在任何 outcome 产生前冻结为深度四分位：Qwen2.5=`7/14/21/28`、Qwen3=`9/18/27/36`。双模型 1 图末层 parity smoke 与 3 图四层 path smoke 均 exit 0；四层 smoke 各 16 个 target-layer cases、REAL/HALL 各 8、0 failures，12 个低层 case 走 full multimodal replay、4 个末层 case 走 exact suffix。末层全词表 parity 最大误差均不超过 `0.125`、argmax 一致，动态网格分别出现 `18×13=234` 与 `13×20=260` 等真实非固定视觉 token 数。
- Qwen 真 FP32/末层 Shapley smoke 也已执行：每模型 FP32 `432` measured rows、Shapley `6` case rows、failures=0，Shapley completeness absolute error=0；但旧 FP32 实现仍只对末层做真 FP32，三个低层及缺失 path-ranked 方向被诚实持久化为 BLOCKED，Shapley 也因把末层限定误写成低层 blocker 而处于 BLOCKED。不能把该 smoke 冒充完整正式结果；当前正补低层 downstream-FP32 route，并将末层 Shapley scope 明确化。fixed-QK、activation/pixel counterfactual、真实 neuron intervention、train-only geometry calibration、正式 spatial/detection/bootstrap 和 POPE/CLEVR/AMBER 仍为 `BLOCKED/NOT RUN`。
- Qwen 低层真 FP32 suffix 已补齐并通过真实门槛：Qwen2.5 L21 单 target 产生 `336` 行（14 directions×8 eta×3 scalars），全部 route=`qwen_causal_query_suffix_true_fp32`、7 个下游层/上下文均真 FP32、0 blocked/0 failures、stage PASS；微型真实 Qwen2/Qwen3 decoder layer 的 full-sequence 最末行与 compact causal-query forward/VJP parity 最大误差约 `1.19e-7`，相关 5 项定向测试通过。末层 Shapley 已明确冻结为 preregistered final-layer subset，其他层写入 `not_in_scope_layers` 而非伪 BLOCKED。
- Qwen2.5 repaired formal local 已实际完成：新根 `.../tc_fvpa_comprehensive_v1_formal_repaired_20260829/` 下 2 shards 共 500 images、3496 个唯一 target-layer cases、四层各 874、REAL/HALL=`2348/1148`、全部 MEASURED、0 failures，双 stage PASS；尚未进入 analyze，故 `output_checksums.json` 尚未生成，严格 loader 对未封存 partial 根按设计拒绝。首次 formal path 的 full K 网格实测 rank1 单图约 6.5 分钟，且尚无 shard 时主动停止；之后加入 Qwen2 clean multimodal embedding/mRoPE 缓存与 suffix parity gate。最终冻结 route：Qwen2 L7/L21 保留 cached-embedding full-text replay，L14 使用通过 target/scalar/gradient gate 的 full-row suffix，L28 exact suffix；Qwen3 L9/L18/L27 full-row suffix 的 gradient relative error 均不超过约 2%、cosine 不低于约 `0.99986`，L36 exact suffix。Qwen2 formal path 已从同一冻结样本恢复运行，不会把近似 query-only route用于未过 gate 的层。
- 原 `800 × 248.028 = 55.1 GPU-hours` 估算建立在错误的 stage/case 混淆上，现已撤回；也没有用末层烟测替代四层 K=1/4/8/16/32 正式耗时。协调器已把 500 图 local-only 与 200 图 path/LOO 拆为独立 cohort/原子分片，path 同键记录在分析/loader 中优先于 local 副本；alpha=0/1 downstream gradient/pullback 会跨 quadrature/K 复用。Qwen formal dry-run 各展开 2 GPU/2 shard 的 11 个命令且传递正确等价层位；新的 repaired formal 结果根不会覆盖旧 BLOCKED 根。正式 local 阶段可先行，FP32/后续报告必须等待低层 blocker 修复后再进入。恢复入口仍为 `PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python bash run_tc_fvpa_comprehensive.sh`。
- 真实 local-only 烟测在 LLaVA L32 为 case `0.294 s`：`path_convergence=[]`、`frozen_write_interventions=[]`、token map 无 `PATH_*`，`available_experiments.path_riesz/finite_loo=false`；与 path 分片同根分析后只得到 1 个去重 case、12 条 path convergence 和 33 条 counterfactual rows，证明 path 富记录正确覆盖 local 同键副本。
- 验证：旧 JFFN/Jacobian 定向回归 28/28 PASS；全部新增 TC-FVPA 定向集 37/37 PASS；相关 `py_compile` 与 11-command formal dry-run PASS。全仓 discovery 现为 336 tests 中 312 pass、12 fail、12 error；24 个失败/错误仍全部来自既有 active-YAML/manifest 预期、缺失 NLTK `punkt` 等非 TC-FVPA 项，没有新增模块回归。另修正 analysis 在更新 `run_status.json` 前过早写 output checksum 的顺序，重跑后 LLaVA/InternVL 修复烟测根分别核验 `20/18` 个 manifest 文件、mismatch=0。Parquet 因两环境均无 pandas/pyarrow/fastparquet 明确 BLOCKED，PT/CSV.GZ 不受影响。

## 2026-08-29 gfchair：LLaVA 完整 3971 图四种 Jacobian S 单独训练

- 在用户确认 500 图筛选后，续跑 LLaVA Union-TopK aggregate S 的完整正式 cohort。上次会话中断时已有 72 个分片、3600 张图且无重复；本轮通过两 rank resume 分别补齐 186/185 张。最终为 80 个原子分片、3971 张正式图片，两个 extraction audit 均为 `worker_complete=true`、`failures=[]`，峰值显存约 15.65/15.67 GB。
- 将 `scripts/train_four_s_only_mlp_500.py` 泛化为可显式指定 cohort label、artifact stem 与 `--no-allow-subset`；完整实验要求 aggregate mention 与正式 JFFN cohort 精确一致，防止把部分分片误报成全量结果。四个分类头仍分别只输入自己的 32 层 S，不包含 risk 或 EV；训练使用原图片 split（train/test images=`3174/797`、mentions=`12317/3146`）、三层 `[128,64,32]`、batch 256、100 epochs、seeds `43/44/45`、无标准化、minimum-train-loss checkpoint 和 train-F1 threshold。
- 完整三 seed AUROC/Hall-F1/Hall-AUPR：all-token aggregate S=`0.864417±0.000909/0.582569/0.623901`；all-token tokenwise S=`0.879812±0.001614/0.591302/0.651029`；Union-TopK tokenwise S=`0.874654±0.001957/0.596298/0.625348`；Union-TopK aggregate S=`0.862309±0.001108/0.576299/0.602870`。最佳 S-only 是 all-token tokenwise；它相对 all-token aggregate 的 seed-ensemble AUROC/Hall-AUPR 增量为 `+0.013941/+0.023964`，10,000 次图片级 paired bootstrap 95% CI 分别为 `[+0.006120,+0.021859]` 与 `[+0.000116,+0.047159]`。
- all-token tokenwise 相对 Union tokenwise 的 ensemble AUROC 增量 `+0.004837`，CI=`[-0.000063,+0.009667]`，几乎触及 0；Hall-AUPR 增量 `+0.025323`，CI=`[+0.011462,+0.038318]`。因此全视觉 tokenwise 在 Hall 排序质量上更稳定，但 AUROC 不能按传统 95% 门槛宣称显著优于 Union tokenwise。
- 500 图上的“tokenwise S-only 优于 risk+EV”没有在完整 cohort 复现。完整 all-token tokenwise S-only 的 AUROC/Hall-AUPR/Hall-F1=`0.879812/0.651029/0.591302`，低于同 cohort risk+EV 的 `0.888141/0.656175/0.644332`；而 `risk+EV+all-token tokenwise S` 为 `0.894717/0.668844`（AUROC/Hall-AUPR）。最终结论是 S 单独具有强信号，但尚不能替代 risk+EV；S 作为补充块仍是当前更好的用法。
- 完整 JSON、逐 seed CSV、PNG/PDF 和中文报告位于 `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/four_s_only_mlp_4000_fair/`。四种方法×三 seeds、六组 10,000 次图片级 bootstrap 均完成，全部结果有限；相关 `py_compile` 通过，S-only/Union/JFFN 定向回归 14/14 通过，最终 artifact 审计确认 3971 images、15463 mentions、12×32-D heads、6×10000 bootstraps 且无 NaN/Inf，`git diff --check` 通过，两张 GPU 已释放。

## 2026-08-28 gfchair：LLaVA 500 图四种 Jacobian S 单独训练

- 为回答“四种 S 本身是否具有幻觉检测信号”，新增 `scripts/train_four_s_only_mlp_500.py`，严格复用既有 Union aggregate 公平实验的同一批 500 张 LLaVA 图片和现成 JFFN/aggregate 分片，不重新执行 VLM forward 或 JVP。四个分类头分别只输入 32 层 `all-token aggregate S`、`all-token tokenwise S`、`Union-TopK tokenwise S`、`Union-TopK aggregate S`；每个输入都恰好 32 维，不含 risk、EV 或其他特征。
- 训练协议保持一致：原图片 split 在该 500 图 cohort 中的交集为 train/test images=`395/105`、mentions=`1513/404`；使用三隐藏层 `[128,64,32]` MLP、batch 256、最多 100 epochs、seeds `43/44/45`、无标准化、minimum-train-loss checkpoint 和仅由 train F1 选择阈值。
- 三 seed 平均 AUROC/Hall-F1/Hall-AUPR 分别为：all-token aggregate S=`0.826055±0.004179/0.558968/0.559724`；all-token tokenwise S=`0.851866±0.010224/0.624512/0.633475`；Union-TopK tokenwise S=`0.849838±0.006390/0.577861/0.620931`；Union-TopK aggregate S=`0.830376±0.003443/0.562651/0.549222`。因此这批 500 图上 tokenwise 两种定义明显强于 aggregate 两种定义；all-token tokenwise 的均值最好，但与 Union tokenwise 的 seed-ensemble AUROC 差仅 `+0.004204`，10,000 次图片级 paired bootstrap 95% CI=`[-0.011050,+0.018550]`，不能认为二者有可靠差异。
- all-token tokenwise 相对 all-token aggregate 的 ensemble AUROC/Hall-AUPR 差为 `+0.027457/+0.077759`，95% CI 分别为 `[+0.001905,+0.054067]` 与 `[+0.015498,+0.146401]`；说明在这批数据上，“先对每个视觉 token 求模长再相加”的 sensitivity gain 比“先把方向向量相加再求模长”的 cancellation-sensitive aggregate gain 更稳定。Union tokenwise 相对 Union aggregate 的 Hall-AUPR 增量 `+0.072868`、CI=`[+0.013322,+0.133260]`，AUROC 增量 `+0.019016` 的 CI=`[-0.002263,+0.040435]` 仍跨 0。
- 结论仅限 500 图小样本筛选：四种 S-only 的 AUROC 都显著高于随机，证明 Jacobian S 自身携带标签信息；当前证据支持优先保留 tokenwise S，但 105 张测试图不足以在 all-token 与 Union-TopK tokenwise 之间定胜负，也不能替代完整 4000 图结果。
- 完整 JSON、逐 seed CSV、PNG/PDF 和中文报告位于 `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/four_s_only_mlp_500_fair/`。新增 `tests/test_four_s_only_mlp_500.py` 强制四个头只能读取自己的 32 维 S 块并拒绝错误宽度；相关 `py_compile` 通过，S-only/Union/JFFN 回归测试共 14/14 通过，所有 12 个训练结果与 6 组 10,000 次 bootstrap 均完成。

## 2026-08-28 gfchair：JFFN 综合讨论文档与 GitHub 代码发布

- 新增 `docs/JFFN_EXPERIMENT_SUMMARY_FOR_DISCUSSION.md`，自包含整合 `jacobian_visual_ffn_validation_report.md`、`jffn_second_round_incremental_validation_report.md` 及后续 `risk+EV+S`、`S×risk+EV`、JFFN–target Union-TopK JS、all/Union tokenwise S、Union aggregate S 和原生分类头消融。文档明确区分：JFFN-P 替换旧 P 的负结果、WRITE-only 对空间收益的解释，以及 aggregate/tokenwise S 作为旧 risk+EV 附加块的正结果；另列出可声称/不可声称、推荐实验和可直接交给 ChatGPT 的讨论问题。
- README 增加 JFFN 入口；`.gitignore` 明确排除 `outputs/`、日志、checkpoint/分片和根目录生成的 JFFN CSV/JSON/PNG。用户随后明确要求一并发布 `docs/CURRENT_TASK.md`，因此该中文实验交接记录纳入第二次提交；仍不包含本地大规模特征、预测、权重、图片或运行数据。
- 在原本没有 `.git` 的 gfchair 目录初始化 `main`。远端 `https://github.com/Xbkeepdo/gfchair.git` 发布前 `ls-remote` 为空。staged 审计为 250 个文件、无软链接、最大文件约 361 KB；高置信私钥/API token 和通用 credential assignment 扫描均无命中，`git diff --cached --check` 通过。
- 验证：正式解释器下 JFFN/second-round/Union-S/JS/S×risk/native-head 定向单测 `29/29` 通过；`compileall -q` 覆盖代码与测试目录通过；全部根目录及 `scripts/*.sh` 的 `bash -n` 通过。
- GitHub 发布已完成：初始代码/报告提交为 `5f594b8`，纳入完整 `CURRENT_TASK.md` 的提交为 `df4d3d0`。服务器最初无 HTTPS/SSH credential，随后在用户明确要求网页弹窗认证后，下载并校验官方 GitHub CLI 2.97.0，通过 device-login 网页授权 `Xbkeepdo`，最终远端 `main` 与本地同步。认证过程中没有把 PAT 写入仓库或聊天；GitHub CLI 报告其 credential 保存在用户目录的配置中，可在不再需要推送时用 `gh auth logout --hostname github.com` 主动清除。

## 2026-08-28 gfchair：LLaVA 500 图 Union-TopK aggregate S

- 新增精确 Union aggregate 定义：每层先取 `U=Top32(P_JFFN) ∪ Top32(Q_hpre_raw_logit_gauss)`，再计算 `S_union_agg=||sum_{j∈U} J_f(z)a_j||_2/(||sum_{j∈U}a_j||_2+eps)`。它与上一轮 `S_union_tokenwise=sum_{j∈U}||J_f(z)a_j||/sum_{j∈U}||a_j||` 不同，保留视觉 token 向量之间的相长/抵消。实现位于 `features/visual_ffn_jacobian.py`，在逐 token `a_j/J a_j` 尚在显存时直接聚合，只保存每层 gain、输入/响应合向量模长、方向余弦、两种 cancellation ratio 和 Union 大小，不保存 hidden-size 方向向量。
- `features/dgst_t.py`、LLaVA/InternVL wrapper 和 `scripts/run_jffn_second_round_extraction.py` 已接入可选 `--union-aggregate` 路径；目标 Q 从现有 feature parts 严格按 `normalize(attention_support*hpre_raw_logit_gauss_gate)` 重建。LLaVA 5 图 smoke 为 21 positions/22 mentions，全部有限，gain 恒等式最大误差 0，Union min/mean/max=`33/40.69/62`，峰值显存约 15.22 GB，分片约 9.5 MB。
- 用户要求先看 500 图后，已停止正在进行的全量抽取；停止时根 `results/jffn_union_aggregate/shards/` 已有 50 个完整 50-image 分片（约 2500 图），均保留以便后续 resume。500 图实验通过双 rank 各前 5 个分片硬链接到 `results/jffn_union_aggregate_500/shards/`，精确核验为全局前 500 张图，无缺失/额外；共 1861 positions、1917 mentions，实际 Union min/mean/max=`32/41.54/64`。
- 新增 `scripts/train_union_topk_aggregate_s_mlp.py`。公平 500 图实验严格在相同 cohort 上重训四组：`risk+EV`、`risk+EV+all-token aggregate S`、`risk+EV+Union tokenwise S`、`risk+EV+Union aggregate S`；均使用原图片 split 在 500 图中的交集（train/test images=`395/105`，mentions=`1513/404`）、三层 `[128,64,32]`、seeds `43/44/45`、batch 256、100 epochs、无标准化、train-loss checkpoint 与 train-F1 threshold。第一次只训练新头却复用全量训练旧头的非公平结果保留在 `_500` 目录，仅作审计；正式结论只使用 `_500_fair`。
- 公平三 seed 结果：risk+EV 的 AUROC/Hall-F1/Hall-AUPR=`0.821245±0.004099/0.610406/0.549276`；全 token aggregate S=`0.829718±0.005879/0.603090/0.570312`；Union tokenwise S=`0.845628±0.004829/0.627679/0.603987`；Union aggregate S=`0.839762±0.007118/0.612775/0.576353`。Union aggregate 相对 risk+EV 的 ensemble AUROC `+0.018379`，95% CI `[-0.004855,+0.043193]`；相对全 token aggregate `+0.008982`，CI `[-0.006139,+0.025114]`；相对 Union tokenwise `-0.004045`，CI `[-0.017100,+0.011561]`，均因 500 图测试集较小而跨 0。
- 为回答500图与4000图趋势是否一致，在相同500图 cohort 上补训了全视觉 tokenwise `S_all=sum_jR_j/sum_jI_j` 三 seeds；AUROC/Hall-F1/Hall-AUPR=`0.840155±0.005531/0.596042/0.615408`。此前三种已有全量 S 加 risk+EV 的 AUROC 排序：500图为 `Union tokenwise (0.845628) > all-token tokenwise (0.840155) > all-token aggregate (0.829718) > risk+EV (0.821245)`；4000图为 `all-token tokenwise (0.894717) > all-token aggregate (0.893087) > Union tokenwise (0.891793) > risk+EV (0.888141)`，名次不一致但三种 S 的 AUROC 均高于 baseline。Hall-AUPR 也都保持三种 S 高于 baseline，但 S 内部排序从500图的 `all-token tokenwise > Union tokenwise > aggregate` 变为4000图的 `aggregate > Union tokenwise > all-token tokenwise`；Hall-F1 排序及部分增益方向直接翻转。500图只有105 test images/404 mentions/105 HALL，且 S 之间 AUROC差仅约0.005–0.016，所以可验证“有 S 信号”，不能用于选择最佳 S。新 Union aggregate 只有500图结果，没有伪造4000图名次。
- 曲线中 all-token aggregate / Union tokenwise / Union aggregate 分别有 `25/28/25` 层 REAL>HALL；Union aggregate 最强 L7 的 Hall−Real=`-0.078661`、Cohen's d=`-0.7645`，强于 all-token aggregate 同层的 `-0.064217/-0.7074`，但略弱于 Union tokenwise 最强 L19 的 `d=-0.7750`。组件图显示 HALL 在大量中后层的输入和响应 cancellation ratio 都更低，即幻觉目标的 Union token 方向内部抵消更强；aggregate 比值同时受响应与输入两侧抵消调制，因此没有像 tokenwise S 那样稳定提升下游 Hall-AUPR。
- 为避免原始均值曲线因共同尺度而视觉重叠，新增 `scripts/plot_s_clear_visualizations_500.py`，在同一公平 500 图 cohort 上生成四类诚实可视化：全层原尺度并列 L4--L30 明示缩放、逐层 REAL−HALL 差值加 10000 次图片级 bootstrap 95% CI/Cohen's d/单层 AUROC、train-only 最大绝对 Cohen's d 选层后的 held-out test violin+box（四种 S 均选到 L19，另固定展示 L16/L24）、以及五个三层 MLP 头的 AUROC/Hall-AUPR/Hall-F1 对比。最清晰的证据来自差值/效应量图和 L19 测试分布：REAL 整体右移但仍有明显重叠，说明 S 有统计分离而非近乎完美的单层判别；分类器图进一步显示多层拼接后增益。完整 PNG/PDF、逐层 CSV 和审计 JSON 位于正式目录的 `clear_visualizations/`。
- 正式报告、原曲线、新增四张 PNG/PDF、CSV、JSON、15 个模型 checkpoint 位于 `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/jffn_second_round/union_topk_aggregate_s_mlp_500_fair/`。验证：合成 cancellation 单测证明 aggregate gain `4.0` 与 tokenwise gain `4/3` 不同；相关 `py_compile` 通过，Union/JFFN 回归测试 12/12 通过，500 图 cohort、3 seeds×5 specs、数值有限与 gain 恒等式均通过；可视化审计确认 395/105 train/test images、1513/404 mentions、10000 次 image-level bootstrap、10 个精确 50-image shard，所有输出均可读取；GPU 已释放。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-28 gfchair：Union-TopK 区域逐 token Jacobian S

- 新增 `scripts/train_union_topk_region_s_mlp.py`，只连接已完成的 JFFN round-one 与 second-round 分片，不重新执行 VLM forward。每层区域固定为 `Top32(P_JFFN) ∪ Top32(Q_hpre_raw_logit_gauss)`；新特征定义为 `S_union = sum_{j∈U} ||J_f(z)a_j||_2 / (sum_{j∈U} ||a_j||_2 + eps)`。它是 Union 区域逐 token gain 的 `I_j` 加权平均，不是旧的 cancellation-sensitive `||sum_j J a_j||/||sum_j a_j||`。为分离“标量聚合方式”与“区域筛选”的作用，同时计算全视觉控制 `S_all=sum_j R_j/sum_j I_j`。
- 两个新输入均为 96 维 `[old-hpre raw-logit sqrt OT risk_32, mass×cosine EV_32, S_32]`，使用现有 `[128,64,32]` 三隐藏层 MLP、seeds `43/44/45`、图片级 8:2 split、batch 256、100 epochs、minimum-train-loss checkpoint、train-F1 threshold、无标准化。既有 `risk+EV` 与 `risk+EV+old aggregate S` 的逐 seed 预测直接复用；正式比较包括 `S_all` 与 `S_union`，因此能单独判断 Union 区域筛选是否有效。
- LLaVA 曲线中 `S_union` 有 28/32 层 REAL>HALL，最强 L19 的 `Hall−Real=-0.05043, d=-0.815`，比 `S_all` 的 27/32 层及最强 `d=-0.790` 略增强单层分离；但下游 `risk+EV+S_union` 三 seed AUROC/Hall-AUPR=`0.891793±0.001312 / 0.672379±0.001749`，低于 `S_all` 的 `0.894717±0.001654 / 0.668844±0.008230`（AUROC更低、Hall-AUPR略高）。seed-ensemble 的 Union−all AUROC=`-0.004153`，图片级 95% CI `[-0.008058,-0.000272]`，说明 LLaVA 上区域裁剪稳定损失排序信息；Union 对 `risk+EV` 的 ensemble AUROC 增量 `+0.006808`，CI `[-0.001083,+0.014666]`，不跨越可靠显著门槛。旧 aggregate S 的均值 AUROC/Hall-F1=`0.893087/0.658950`，仍优于 Union 的 `0.891793/0.640199`。
- InternVL 曲线中 `S_union` 有 19/32 层 REAL>HALL，最强 L11 为 `Hall−Real=-0.03658, d=-0.528`；`S_all` 只有 17/32 层 REAL>HALL，且最强 L21 反而为 HALL>REAL (`+0.01874, d=+0.564`)。下游 `risk+EV+S_union` AUROC/Hall-AUPR=`0.866605±0.001525 / 0.552289±0.002817`，优于 `S_all` 的 `0.863213±0.001334 / 0.545624±0.009251`；ensemble Union−all AUROC=`+0.008302`，95% CI `[+0.001906,+0.014983]`。相对 `risk+EV`，Union ensemble AUROC/Hall-AUPR 分别提升 `+0.021988/+0.045065`，两项 CI 均不跨 0；但相对旧 aggregate S 的 AUROC `+0.003927`、CI `[-0.004651,+0.012282]`，只能视为打平。Union 的 Hall-F1=`0.500021` 仍低于旧 aggregate S 的 `0.510817`。
- 结论是 Union 区域本身具有模型依赖性：InternVL 从排除区外视觉 token 中获益，LLaVA 则会丢失有用的分布式 FFN gain。当前证据不支持把 `S_union` 作为跨模型统一替代；LLaVA 更适合 `S_all`/旧 aggregate S，InternVL 可保留 `S_union` 作为模型特定变体。实际 Union token 数为 LLaVA min/mean/max=`32/41.49/64`、InternVL=`32/39.71/59`。
- 目标 Q 对两模型均从现有 feature parts 按 `normalize(attention_support×raw-logit gate)` 重建，LLaVA 512 个重复位置和 InternVL 129 个重复位置最大误差均为 0；LLaVA 后期分片内嵌 Q 与重建 Q 最大误差 `3.81e-8`。round-one/second-round 的正式 positions 与 mentions 全部一一对应；两次 BF16 JFFN 抽取的 P 最大差为 LLaVA/InternVL `2.63e-4/1.72e-3`，新 `S_all/S_union` 均统一使用 second-round 内部一致的 `R/I/P_JFFN`，所以主比较 Union−all 不受跨轮选择差异影响。
- 单模型结果、曲线、CSV、各 6 个 checkpoint 与报告位于各自 `results/jffn_second_round/union_topk_region_s_mlp/`（两模型共 12 个新 checkpoint）；两模型汇总为 `outputs/union_topk_region_s_mlp_2model_summary.{md,json}` 与 `outputs/union_topk_region_s_mlp_2model_metrics.csv`。验证：正式 train/test mentions 为 LLaVA `12317/3146`、InternVL `9378/2381`；相关 `py_compile` 通过，新增手算与 JFFN 回归测试 11/11 通过，全部 JSON/CSV/曲线和指标有限。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-20 gfchair：Ours 特征使用 S-VAR/MetaToken 原生分类头

- 新增 `scripts/train_ours_with_native_baseline_heads.py`，只做分类头替换，不重新执行模型 forward 或特征提取。对已完成完整 JFFN cohort 的 LLaVA-1.5-7B 与 InternVL2.5-8B，分别使用 64 维 `[old-hpre raw-logit sqrt OT risk, mass×cosine EV]` 和 96 维 `[risk,EV,Jacobian S]`；样本、标签、InsLen 目标、图片级严格 8:2 split 与 seeds `43/44/45` 全部不变。
- 三种原生头为：S-VAR 的单隐藏层 `input→248→2`、ReLU、Adam `lr=1e-3`、batch 32、50 epochs、CE、无标准化；MetaToken 的 `StandardScaler→LogisticRegression(lbfgs,max_iter=2000)` 与 `StandardScaler→GradientBoosting(100 estimators)`。输入宽度按 Ours 的 64/96 维适配，其余头协议保持当前 native-paper baseline 实现；固定 0.5 与 train Real-F1 阈值均保存。当前 `[128,64,32]` 三层 Torch MLP的既有逐 seed 结果和预测作为严格对齐参考，不重新训练。
- LLaVA 64维 `risk+EV`：当前三层 MLP / S-VAR / Meta-LR / Meta-GB 的 AUROC 为 `0.888141/0.882879/0.874598/0.873417`，Hall-AUPR 为 `0.656175/0.637729/0.618606/0.626063`。96维加入 S 后为 `0.893087/0.887273/0.880830/0.888901`，Hall-AUPR 为 `0.675664/0.664128/0.646143/0.675866`；Meta-GB 的 Hall-AUPR 与三层 MLP基本相同，但 AUROC、Hall-F1仍较低。
- InternVL 64维 AUROC 为三层 MLP / S-VAR / Meta-LR / Meta-GB=`0.848825/0.822749/0.818551/0.812016`，Hall-AUPR=`0.502827/0.466272/0.436445/0.468151`。96维为 `0.866745/0.851424/0.839470/0.840995`，Hall-AUPR=`0.546172/0.505964/0.472539/0.518275`；三层 MLP仍全面最好。
- S 在两模型×三原生头的六个组合中全部提升 AUROC。LLaVA 的 S 增量为 S-VAR/LR/GB `+0.004393/+0.006232/+0.015484`，InternVL 为 `+0.028674/+0.020919/+0.028979`；对应 Hall-AUPR 也全部提升。这证明 Jacobian S 的信息不是三层 MLP专属，但当前三层 MLP对 risk、EV、S 的联合利用更充分。原生头没有带来总体性能提升，因此不建议替换主实验头，可作为“分类器头敏感性”消融。
- 追加 S-VAR 隐藏宽度 128 的单变量消融，输入仍为 64/96 维。LLaVA `risk+EV / risk+EV+S` 的 h128 AUROC 为 `0.881584/0.885765`，均低于 h248 的 `0.882879/0.887273`；InternVL 为 `0.817035/0.845775`，低于 h248 的 `0.822749/0.851424`。Hall-F1、Hall-AUPR和 ensemble AUROC 同样四组全部下降，且 InternVL 96维 h128 的 seed 标准差由 `0.001563` 增至 `0.004036`。因此缩窄到128没有改善泛化，保留原生248更合理。
- 每模型 JSON、报告、24 个 checkpoint 和可恢复 progress 位于各自 `results/jffn_second_round/ours_with_native_baseline_heads/`；两模型汇总为 `outputs/ours_native_baseline_heads_2model_summary.{md,json}` 与 `outputs/ours_native_baseline_heads_2model_metrics.csv`。验证：LLaVA/InternVL train/test mention=`12317/3146`、`9378/2381`，每模型 2 feature sets×4 heads×3 seeds=24 项，全部结果有限，GPU 已释放；新增测试及 baseline/S-VAR 回归共 24/24 通过，`py_compile` 通过。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-20 gfchair：P_JFFN–target Union-TopK JS + EV 三层 MLP

- 新增 `scripts/train_jffn_union_topk_js_ev_mlp.py`：只复用已完成全量 JFFN 分片的 LLaVA-1.5-7B 与 InternVL2.5-8B，不重新做模型 forward，也不续跑 Qwen。每层取 source `P_JFFN` Top-32 与 target Top-32 的并集（最多 64），在并集内分别归一化后计算自然对数 Jensen–Shannon 散度；将 32 维 JS 与同一 target gate 的 32 维原有 `mass×cosine` EV 拼接为 64 维。两个 target gate（hpre raw-logit Gaussian、hpre softmax-prob Gaussian）均训练三隐藏层 `[128,64,32]` MLP，严格复用现有图片级 split、seeds `43/44/45`、batch 256、100 epochs、minimum-train-loss checkpoint、train-F1 threshold，且不做特征归一化。
- 同时训练严格同构的 `old_hpre_cos P + target Union-Top32 JS + EV` 控制。LLaVA raw/softmax gate 的 `P_JFFN JS+EV` 三 seed AUROC 分别为 `0.876057±0.002547 / 0.877191±0.002651`，低于 old-hpre 控制的 `0.889554±0.001292 / 0.890203±0.001922`；seed-ensemble AUROC 差为 `-0.014881/-0.012102`，10,000 次图片级 paired bootstrap 的 95% CI 分别为 `[-0.024194,-0.005572] / [-0.021909,-0.002441]`，均不跨 0。
- InternVL raw/softmax gate 的 `P_JFFN JS+EV` AUROC 为 `0.838730±0.007828 / 0.842990±0.005356`，old-hpre 控制为 `0.848808±0.002423 / 0.844465±0.001018`。raw gate 的 ensemble AUROC 差 `-0.010174`、CI `[-0.023899,+0.003084]`，但 Hall-AUPR 差 `-0.050323`、CI `[-0.086775,-0.010059]`；softmax gate 基本打平，ensemble AUROC 差 `-0.000855`、CI `[-0.015399,+0.013373]`。
- 与同一 `P_JFFN + sqrt_matched_state OT-risk + EV` 对照相比，JS+EV 的 ensemble AUROC 差异在两模型、两 gate 上都没有显著不跨 0：LLaVA raw/softmax 为 `-0.001980/+0.005109`，InternVL 为 `-0.002952/+0.002405`。因此本轮不支持“用无几何信息、上界为 ln2 的 JS 替代 OT risk”会稳定提高检测性能；`P_JFFN` 的局部 FFN 响应能量与 target gate 的分布差异仍有信号，但在 Union Top-K 后只比较概率错位会丢掉 OT cost 中的状态几何。
- 曲线方面，LLaVA raw gate 有 27/32 层 `JS(Hall)>JS(Real)`，全层均值为 `0.097283 vs 0.085722`；softmax gate 为 25/32 层，但平均差仅 `0.002803`。InternVL raw gate 为 23/32 层且平均差仅 `0.001518`，softmax gate 只有 14/32 层且总体方向轻微反转。可分性主要来自多层联合和 EV，而不是每层 JS 都有强、稳定的单调间隔。
- InternVL/LLaVA 的紧凑 JFFN 分片没有重复保存完整 target Q，因此从既有 `features.part*.pkl` 严格按 `normalize(attention_support × gate)` 重建；所需唯一位置分别为 `11630/14951`，缺失为 0，重复行最大误差为 0，且重建后的 EV 与 JFFN 分片已存 EV 最大绝对误差为 0。正式 train/test mention 分别为 `9378/2381`、`12317/3146`，全部 64 维输入和 JSON/CSV 为有限值。
- 每模型结果、曲线、checkpoint 和报告位于 `results/jffn_second_round/jffn_union_topk_js_ev_mlp/`；两模型汇总为 `outputs/jffn_union_topk_js_ev_2model_summary.{md,json}` 与 `outputs/jffn_union_topk_js_ev_2model_metrics.csv`。验证：相关 `py_compile` 通过，Union-TopK JS 手算/既有实现 parity 与 JFFN 回归测试 12/12 通过；系统默认 Python 因没有 PyTorch 导入失败，改用正式实验解释器 `/opt/conda/private/envs/vicr/bin/python` 后全部通过。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-20 gfchair：S、S×risk 曲线与 S×risk+EV 三层 MLP

- 新增 `scripts/train_s_times_risk_ev_mlp.py`：从已完成的 LLaVA/InternVL JFFN 分片读取逐层 Jacobian sensitivity `S=R/(I+epsilon)`，与同层 `old_hpre_cos + hpre_raw_logit_gauss + sqrt_matched_state` Union Top-K OT risk 做严格逐元素乘积；分别输出 S、S×risk 的 REAL/HALL 均值及 mention-level 95% CI、Hall−Real、Cohen's d 和逐层 AUROC。另训练 `[S×risk_32, EV_32]` 64 维三隐藏层 `[128,64,32]` MLP，split、seeds `43/44/45`、batch 256、100 epochs、train-loss checkpoint、train-F1 threshold、无归一化均与原实验一致。
- 曲线方向发生系统翻转。LLaVA 的 S 有 25/32 层 REAL>HALL，最强 L19 为 `Hall−Real=-0.05209, d=-0.683`；S×risk 则有 20/32 层 HALL>REAL，最强 L21 为 `+0.01910, d=+0.631`。InternVL 的 S 有 19/32 层 REAL>HALL，最强 L13 为 `-0.05343, d=-0.478`；S×risk 有 24/32 层 HALL>REAL，且 L10–L29 连续 HALL>REAL，最强 L24 为 `+0.01173, d=+0.512`。这表明 risk 会把主要偏向 REAL 的 FFN sensitivity 调制为更接近幻觉风险方向的量。
- LLaVA 的 S×risk+EV 三 seed AUROC/Hall-AUPR 为 `0.890893±0.000247 / 0.672932±0.001593`，对 old risk+EV 的均值增量为 `+0.002752/+0.016757`；seed-ensemble AUROC 增量 `+0.002295`，图片级 95% CI `[-0.004802,+0.009246]`。Hall-F1 为 `0.640065`，低于 old risk+EV 的 `0.644332`。相对 old risk+EV+S 的 ensemble AUROC 低 `0.007185`，95% CI `[-0.014471,-0.000020]`。
- InternVL 的 S×risk+EV 三 seed AUROC/Hall-AUPR 为 `0.851277±0.004707 / 0.511333±0.013386`，对 old risk+EV 的均值增量为 `+0.002453/+0.008506`；seed-ensemble AUROC 增量 `+0.002989`，95% CI `[-0.008205,+0.014293]`。Hall-F1 为 `0.463574`，低于 old risk+EV 的 `0.466311`。相对 old risk+EV+S 的 ensemble AUROC/Hall-AUPR 分别低 `0.015071/0.037707`，两项 95% CI 均不跨 0。
- 结论：S×risk 是有清晰方向翻转的机制交互量，但 `[S×risk,EV]` 对 `[risk,EV]` 的小幅排序增益不可靠，而且明显弱于让 MLP 分别接收 `[risk,EV,S]`。硬乘法压缩了 risk 与 S 的独立幅度及反向信息；它适合解释性消融，不应替代三块独立输入。
- 每模型 CSV、原始曲线、Hall−Real/effect 图、checkpoint、JSON 与报告位于各自 `results/jffn_second_round/s_times_risk_ev_mlp/`；两模型汇总为 `outputs/s_times_risk_ev_mlp_2model_summary.{md,json}` 和 `outputs/s_times_risk_ev_mlp_2model_metrics.csv`。
- 验证：两个模型正式 train/test mention 为 `12317/3146`、`9378/2381`，64 维输入、3 seeds 和 split 全部一致；S×risk 逐元素等式、128 条 label-curve 行、64 条 difference 行、所有 JSON/CSV 有限值及 4 张 PNG 尺寸均通过；相关 `py_compile` 与 `tests.test_s_times_risk_ev_mlp + tests.test_jffn_experiment` 为 8/8 通过。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-20 gfchair：old risk + EV + Jacobian S 三层 MLP

- 新增 `scripts/train_old_risk_ev_s_mlp.py`，只复用已完成全量 JFFN 分片的 LLaVA-1.5-7B 与 InternVL2.5-8B，不续跑尚未完成的 Qwen。输入严格按层拼接：32 维 `old_hpre_cos + hpre_raw_logit_gauss + sqrt_matched_state` 的 Union Top-K OT risk、32 维原有 `mass×cosine` EV、32 维 Jacobian sensitivity `S=R/(I+epsilon)`，合计 96 维；没有做特征归一化。
- 训练头与原对照一致：三隐藏层 `[128,64,32]`、batch 256、最多 100 epochs、seeds `43/44/45`、minimum-train-loss checkpoint、train-F1 threshold、严格复用现有图片级 8:2 split。协议完全相同的 `old risk+EV` 训练结果及逐样本预测直接从既有 JFFN comparison 产物读取，未重新随机训练 baseline。
- LLaVA 的 `old risk+EV → old risk+EV+S` 三 seed 均值：AUROC `0.888141→0.893087`（`+0.004946`），Hall-AUPR `0.656175→0.675664`（`+0.019489`），Hall-F1 `0.644332→0.658950`（`+0.014618`）；seed-ensemble AUROC `0.893401→0.902881`，图片级 10,000 次 paired bootstrap 的 AUROC 增量 95% CI 为 `[+0.001897,+0.017336]`。
- InternVL 的三 seed 均值：AUROC `0.848825→0.866745`（`+0.017920`），Hall-AUPR `0.502827→0.546172`（`+0.043345`），Hall-F1 `0.466311→0.510817`（`+0.044506`）；seed-ensemble AUROC `0.856685→0.874746`，paired bootstrap AUROC 增量 95% CI 为 `[+0.004740,+0.031558]`。两模型 AUROC 的置信区间均不跨 0；LLaVA 的 Hall-AUPR 区间轻微跨 0，InternVL 不跨 0。
- 单模型 checkpoint、progress、JSON 和报告位于各自 `results/jffn_second_round/old_risk_ev_s_mlp/`；两模型汇总为 `outputs/old_risk_ev_s_mlp_2model_summary.{md,json}` 与 `outputs/old_risk_ev_s_mlp_2model_metrics.csv`。
- 验证：LLaVA/InternVL train/test mention 分别为 `12317/3146`、`9378/2381`，与现有正式 cohort 一致；两者均为 96 维、3 seeds，全部输入/结果无 NaN/Inf，输出文件完整；脚本 `py_compile` 通过，`tests.test_jffn_experiment` 为 7/7 通过。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-19 gfchair：LLaVA 576-token exact FFN-JVP 五图显存验证

- 新增 `features/visual_ffn_jacobian.py` 和 `scripts/validate_llava_visual_ffn_jacobian.py`：从每层真实 attention row/value/output projection 重建 576 个视觉 token 对目标预测位置的 residual 方向 `a_j`，并用同一个 `J_f(z)` 在 `torch.vmap(torch.func.jvp)` 中一次并行计算全部 `J_f(z)a_j`。新 P 按每个方向的 FFN 响应模长归一化；定位仍使用 InsLen 首 subtoken 的 `toke_idx-1`。
- 复用固定 500 图 selection 的前 5 张 LLaVA 图（image IDs `408805/459408/163155/246746/521804`），覆盖 14 个唯一目标位置、32 层、160 个 image-layer JVP batch，失败为 0。模型加载后 allocated/reserved 为 `13.157/13.176 GiB`；完整 forward 峰值 `13.813 GiB`；全 576 并行 JVP 峰值 `14.037 GiB`，相对 JVP 调用前最大增量 `0.319 GiB`。纯 JVP 总时间 `0.922 s`；预热后约 `7.6 ms/层`。
- 在第 1/16/32 层与 64-token 分块做对照：全并行额外工作显存约 `326 MiB`，分块约 `120 MiB`；预热后全并行约 `7.6 ms`，分块约 `15.6 ms`，relative-L2 差异 `5.7e-4–2.5e-3`（FP16 batch/kernel 舍入差异）。
- 数值自检：attention 重建 cosine 最低 `0.99999887`；`sum_j J a_j = J sum_j a_j` 最大 relative error `4.36e-3`；中心有限差分与 JVP 的 cosine 中位数 `0.9884`。新 P 的归一化熵均值 `0.787`、Top-32 mass 均值 `0.514`、最大 token/均匀值中位数 `41.5`，这 5 图上并非“576 个 token 都差不多”；Real/Hall 曲线仍高度重合，小样本不用于推断分类效果。报告与 CSV/JSON/PNG 位于 `outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/visual_ffn_jacobian_exact_5images/`。
- 验证：相关 Python `py_compile` 通过；`tests.test_visual_ffn_jacobian` 1/1 通过。本轮仅做 5 图显存/数值 smoke，未修改正式特征提取或训练配置。

## 2026-08-18 gfchair：同图不同目标的 P 相似度

- 新增 `scripts/analyze_inslen_same_image_p_similarity.py`，从现有 feature parts 按图片分组，只比较 `token_str` 不同的正式 InsLen 目标。主统计采用图片等权，同时在 CSV 保存 pair 等权；分别分析 hmid/hpre P 的 raw cosine、去均匀分量 cosine、归一化 JS、TV、Top-32 重合率及 `pair JS / uniform JS`。
- 四模型至少有两个不同目标词的图片数为 LLaVA/InternVL/Qwen2.5/Qwen3 `3457/3231/2584/3466`，对应不同目标 pair `26782/16860/9286/28952`。原始 cosine 为 `0.918–0.955`，但去均匀后为 `0.40–0.65`；Top-32 重合率为 `0.260–0.537`，仍是随机基线的约 `2.8–5.1` 倍。结论是 P 并非完全与目标无关，但由显著的同图公共模板和中等强度的目标特异变化共同组成。
- LLaVA/Qwen2.5 的 pair-JS/uniform-JS 约 `1.0–1.2`，不同目标带来的变化与 P 偏离均匀的幅度相当；InternVL/Qwen3 约 `0.69–0.76`，共享图像结构更强。四模型 Hall–Hall 的 JS 均低于 Real–Real，与幻觉目标缺少各自视觉锚点、更多退化到共享模板的解释一致，但该统计不验证高 P token 是否落入正确对象区域。
- 每模型完整报告、CSV 和 PNG/PDF 位于对应 `results/feature_curves/*same_image_different_target_p_similarity*`，跨模型汇总为 `outputs/same_image_different_target_p_similarity_4model_summary.md`。验证：四模型 feature parts 均无跨 part 图片，P 形状及归一化检查通过；脚本 `py_compile` 和合成三目标手算通过；产物无 NaN/Inf。LLaVA 因官方 raw-surface/归一化词关系排除同图同 `token_str` 重复行 1,088 条，其余三模型为 0。

## 2026-08-18 gfchair：四模型 P 视觉-token 分布统计

- 新增 `scripts/plot_inslen_p_token_distribution.py`，直接分析既有 `features.part*.pkl` 中的两套完整 P 矩阵：`dgst_t_source_dist_per_layer`（hmid）和 `dgst_t_source_hpre_cos_dist_per_layer`（hpre）。未重新生成、labeling、抽特征或训练；raw/softmax target gate 与 sqrt/cosine cost 共用 P，因此不重复统计。
- 按正式对象样本和 CHAIR Hall/Real 标签统计每层视觉-token P50/P90/P99/max、`N_vP_i`、Top-1/5/10/32 mass、归一化熵、有效 token 比例、均匀阈值以上的质量与 token 比例；另保存 token 等权的 `log10(N_vP_i)` 直方图。每模型输出指标曲线和分布热图 PNG/PDF、两个 CSV 与中文报告，四模型汇总为 `outputs/p_visual_token_distribution_4model_summary.md`。
- 为降低峰值内存，脚本优先逐个读取两个 feature part，处理后立即释放，不加载合并后的 root `features.pkl`；分位数和 Top-K 共用一次排序，直方图按样本向量化累计。全量覆盖 LLaVA/InternVL/Qwen2.5/Qwen3 的 15,463/11,759/8,717/14,873 个正式目标，共 50,812 条。
- 主要观察：P 的归一化熵约 `0.95–0.99`，整体较分散但非均匀；最大 token 约为均匀值的 `2.4–7.1` 倍。八组 P-source×模型的全层平均均为 Hall 略更分散，但差异强烈依赖层；InternVL 31/32 层方向一致，LLaVA/Qwen2.5 存在明显反转。hpre 对 Qwen2.5 的尖锐化最明显，而 LLaVA 的 hpre 略变平，说明 hpre 并非跨模型统一地让 P 更尖。
- 验证：两套 P 的每层归一化最大误差为 `3.47e-7`；四模型指标 CSV 分别为 128/128/112/144 行，直方图为 46,080/46,080/40,320/51,840 行，均与层数×2 source×2 labels×360 bins 严格一致；全部 CSV 无 NaN/Inf，PNG 已抽样目视检查，脚本 `py_compile` 和合成分布手算通过。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-17 gfchair：Union-topK JS + mass×cosine 四模型三种子结果

- 直接复用四模型 `COCO4000-INSLEN-OFFICIAL-TARGET/features.pkl`，未重新生成 caption、labeling 或模型特征。使用现有 alias 分别训练 raw-logit/softmax-prob 的 union-topK JS、mass×cosine 和二者拼接，共 4 模型 × 3 seeds × 6 feature sets = 72 个 Torch MLP；训练设置与当前主实验完全相同（严格图片级 8:2、seeds 43/44/45、Real 正类、train-F1 阈值、`[128,64,32]`、100 epochs）。
- Union-topK JS 使用 source/target 各 Top-32 的并集（并集最多 64）；mass×cosine 为既有 target Top-32。拼接在 8/8 个模型×gate 分支上均高于对应两个单特征，AUROC 增益为 `+0.0136` 到 `+0.0380`。
- 每模型最佳拼接（train-F1 报告，AUROC/Real-F1/Hall-F1）：LLaVA softmax `0.8891/0.8938/0.6378`；InternVL raw `0.8524/0.9157/0.4706`；Qwen2.5 raw `0.8299/0.9341/0.3648`；Qwen3 raw `0.8649/0.9153/0.5486`。相对各模型现有最佳 risk+mass AUROC 分别为约 `+0.0000/+0.0009/-0.0069/+0.0016`。
- 单模型 seed 产物写入各自 `results/union_topk_js_mass_cosine_seed{43,44,45}/`，单模型三 seed 汇总位于各自 results 根目录；四模型完整双阈值汇总为 `outputs/union_topk_js_mass_cosine_4model_3seed_summary.{md,csv,json}`，简明解释为 `outputs/union_topk_js_mass_cosine_4model_comparison.md`。
- 验证：四模型每个 seed 均严格包含 6 项，seed provenance 为 43/44/45；输入维度依次为 LLaVA/InternVL `32→64`、Qwen2.5 `28→56`、Qwen3 `36→72`；四模型日志均无 Traceback/Error/NaN/OOM，最终两张 GPU 均恢复 0 MiB。样本严格覆盖现有正式对象：15,463/11,759/8,717/14,873。

## 2026-08-17 gfchair：删除过时 feature-manifest adoption 断言

- 删除 `test_legacy_feature_file_cannot_be_adopted_into_v2_resume`：该测试仍期待 `_validate_or_write_feature_manifest(..., adopt_legacy=True)` 抛出 `adoption is intentionally disabled`，但按当前实验配置 feature manifest 已是可选项，写入函数也早已不执行这条拒绝逻辑。
- 同步删除 feature manifest 写入函数未使用的 `adopt_legacy` 形参和 5 个无效传参；将另一项测试更名为 feature provenance 内容/locator 跟踪，并移除与其断言无关的 manifest 文件准备。generation artifact 复用、labeling resume、特征完整覆盖和 split/token/label 一致性检查均未修改。
- 验证：相关 Python `py_compile`、`bash -n run.sh` 通过；`tests.test_stage_resume + tests.test_inslen_official_targeting + tests.test_training_provenance` 共 33/33 通过，原先 stage-resume 的 16/17 现为 16/16。gfchair 没有 `.git`，无法运行有效的 `git status`/`git diff --check`。

## 2026-08-17 gfchair：修复 InternVL InsLen 候选词误取 BOS

- InternVL2.5 的 `InternLM2Tokenizer` 实测默认 `add_bos_token=True`：例如 `" officer"` 默认编码为 `[1, 9581]`，其中 `1=<s>`，而回答的 `response_token_ids` 不包含该 BOS。此前逐字复现发布版 `find_word()` 的 `input_ids[0,0]` 因而总是搜索 ID 1，现有 4,000 张 InternVL 标签中的 36,480 个 CHAIR mention 全部 unresolved，抽取阶段得到 0 个样本并因缺少 `features.pkl` 退出。
- `internvl_find_word` 现仅在候选字符串重新 tokenize 时设置 `add_special_tokens=False`，使 `[0,0]` 指向真实首 lexical subtoken；候选前后缀顺序、归一化词、首次回答 ID、跳过和组内去重规则均不变。LLaVA 原本已使用该参数；Qwen2.5/Qwen3 继续走 `tokenizer.encode(surface)`，本地两者均无 BOS，未受修改影响。
- 新标签的 `resolver_compatibility_note` 明确记录 InternVL BOS compatibility fix。labeling resume 同时比较 resolver model 与非空 compatibility note，因此旧的 `note=null` InternVL 标签不会继续被复用；下一次 `run.sh` 会复用原 4,000 条 generations，只重新做一次 labeling，之后的新标签仍可正常跳过。
- 回归测试新增带自动 BOS 的 tokenizer，确认 InternVL 每个候选都收到 `add_special_tokens=False`、命中真实 ID 9581；另确认旧 BOS 标签被 resume 判定为不兼容，新标签可复用。`py_compile`、`bash -n run.sh` 与 InsLen 定向测试 12/12 通过；真实本地 tokenizer 对图 230008 的 `officer` 解析为候选 `" officer"`、ID 9581。
- 失败记录：第一次未设置 `NLTK_DATA` 的定向测试有 4 个既有 NLTK 用例因找不到 `punkt` 报错；按正式环境设置 `NLTK_DATA=/home/apulis-dev/userdata/nltk_data` 后 12/12 通过。`tests.test_stage_resume` 为 16/17，唯一失败仍是既有 `test_legacy_feature_file_cannot_be_adopted_into_v2_resume`，它要求 manifest adoption 报错，而当前项目按用户要求关闭/移除了该强制逻辑，与本次 tokenizer 修改无关。gfchair 没有 `.git`，无法运行有效的 `git status`/`git diff --check`。

## 2026-08-17 gfchair：修复关闭 manifest 时 labeling 无法 resume

- `run.sh` 每次都会以 `--resume` 调用 generation/labeling 入口，完整缓存应由 `label_coco.py` 内部快速判定并跳过。实际重复 labeling 的根因不在现有 artifact：InternVL 的 generations/labeling 均为 4,000 张、ID/caption/InsLen 协议一致，ground truth 也完整。
- 根因位于 `_reuse_complete_labeling` 的 ground-truth hash 分支：活动配置 `run.validate_manifests=false` 时 `actual_manifest=None`，因此 `saved_ground_truth_sha256=None`；旧 `elif saved_hash != actual_hash` 没有受开关保护，恒为 true 并返回 false，导致完整 labeling 永远不能复用。现已将 hash 对比限制为 `validate_manifest=true`；关闭 sidecar 校验时仍执行 ground-truth 文件的结构、cohort 和内容 hash 计算，只是不再要求不存在的 manifest 保存 hash。
- 新增回归覆盖：关闭 manifest、无 labeling manifest、ground truth 完整时必须 resume；ground-truth cohort 损坏时即使关闭 manifest 仍必须拒绝。新旧两个 ground-truth resume 用例 2/2 通过，相关 `py_compile` 通过。真实 InternVL 与 LLaVA 各 4,000 条 artifact 的只读判定现均返回 `would_skip_labeling=True`，并打印正式 skip 信息。
- 完整 `tests.test_stage_resume` 为 16/17；唯一失败仍是既有 `test_legacy_feature_file_cannot_be_adopted_into_v2_resume`，它要求关闭/缺失 manifest 时抛错，而当前项目按用户要求允许不使用 feature manifest，与本次 labeling 分支无关。第一次新增测试时误把原测试后半段放入新测试，定向运行 1/2 失败；恢复两个测试边界后重跑为 2/2，通过结果如上。gfchair 没有 `.git`，无法运行有效的 `git diff --check`。

## 2026-08-17 LLaVA InsLen：risk 与 mass×cosine 的 Real/Hall 逐层曲线

- 对 `COCO4000-INSLEN-OFFICIAL-TARGET` 的全部 15,463 个正式对象样本（Hall 3,487、Real 11,976）聚合当前活动的 8 条 risk 曲线：raw/softmax gate × P=hmid/hpre × sqrt/cosine cost；另聚合 raw/softmax 两条 target-side `mass × cosine`。后者不依赖 P-source 或 transport cost，因此没有把相同 EV 曲线重复画四遍。
- 新增 `scripts/plot_inslen_risk_mass_cosine_by_label.py`。默认优先逐个读取 `features.part*.pkl`、累计 sum/sum-square 后释放分片，避免同时保留完整 5.8 GB root artifact 与所有隐藏状态；输出类别均值及 95% CI、Hall−Real 及独立均值差近似 95% CI、逐层 Cohen's d、显著方向与 CSV。三组 PNG/PDF、完整 CSV 和中文汇总位于 `results/feature_curves/llava_1_5_7b_inslen_risk_mass_x_cosine_by_label*`。
- 曲线显示 risk 整体为 Hall>Real，主要强区间为 L19–L27，最大标准化效应集中 L20/L21、`|d|≈0.59–0.72`；raw risk 的正向方向比 softmax 连续。`mass × cosine` 在 L1–L5 有轻微 Hall>Real，L6 后翻为 Real>Hall，L18–L30 最强，L28 的均值差约 `-0.05`；raw/softmax 最大 `|d|` 分别为 `0.899/0.978`。两类特征的方向和层区间互补，与 risk+EV 的下游增益相符。
- 验证：两个 worker part 分别 7,796/7,667 行，合计 15,463；10 个字段均覆盖 Hall/Real，曲线均为 32 层且全部有限；CSV 为 10×32 条数据，脚本 `py_compile` 与实际全量运行通过，三张 PNG 已做可视检查。gfchair 没有 `.git`，`git status` 按预期报 not a repository。

## 2026-08-17 LLaVA seed 43：方法与 shared-MLP baseline 的 200 epoch 收敛对照

- 在 `COCO4000-INSLEN-OFFICIAL-TARGET` 上，以完全相同的严格图片级 8:2 split、seed 43、三层 Torch MLP 和 train Real-F1 阈值协议，隔离运行当前 16 个 `risk/risk+EV` 方法分支与 SVAR、MetaToken shared-MLP baseline，最大 epoch 从 100 改为 200；结果分别写入 `results/seed43_ep200` 与 `baseline/results/shared_torch_mlp/seed43_ep200`，没有覆盖已有 seed43/44/45 结果。native-paper baseline head 的训练器不同，未混入“相同 MLP 的 200 epoch”比较。
- 为允许 baseline 做同协议对照，`scripts/train_baselines.py` 新增可选 `--num-epochs`，仅覆盖 shared Torch MLP 的 `training.torch_probe.max_epochs`，默认不传时仍保持原 YAML 配置；非正值会明确拒绝。实际结果 provenance 确认 SVAR/MetaToken 的 trainer config 均记录 `num_epochs=200`。
- 16 个方法分支相对 100 epoch 的平均变化为：train loss `-0.02719`、train AUROC `+0.01662`，但 test AUROC `-0.00165`、Real-F1 `-0.00235`；risk-only test AUROC 平均变化 `-0.00040`，risk+EV 为 `-0.00291`。200-epoch 最高 test AUROC 仍是 raw-logit/P=hmid/cosine risk+EV 的 `0.8894`，该分支其实在 epoch 83 已最优、epoch 93 early-stop，与 100-epoch 结果完全一致。
- SVAR 从 100 到 200 epoch 的 train loss `0.1379→0.0484`、train AUROC `0.9949→0.9999`，但 test AUROC `0.8853→0.8799`、Hall-F1 `0.6510→0.6351`，表现为明显过拟合；MetaToken 在 epoch 47 最优、epoch 57 early-stop，两个最大 epoch 设置得到完全相同结果。结论是训练损失在 100 epoch 后仍能下降，但测试泛化并非未收敛，盲目延长训练没有收益。
- 完整 16 项与两个 baseline 的 100→200 数值、best/run epoch 和横向比较已写入 `results/seed43_ep200/llava_1_5_7b_seed43_100_vs_200_epoch_comparison.md`。验证：两个结果 JSON 可解析且分别完整包含 16/2 项；主方法 12/16、MetaToken 均按 train-loss patience 提前停止；训练进程全部退出，两张 GPU 恢复 0 MiB/0%；`scripts/train_baselines.py` 的 `py_compile` 和 `--help` 检查通过。gfchair 按隔离设计没有 `.git`，无法执行 `git diff --check`。

## 2026-08-17 gfchair：修复 InsLen 合并阶段误删 346 个正式样本

- LLaVA 训练在关闭 manifest 校验后暴露正式样本完整性差异：labeling 期望 15,463，合并后的根特征只有 15,117。核验两个 worker part 后确认二者合计恰好 15,463、逐项与 labeling 完全一致，遗漏发生在最终 merge 而非模型抽取。
- 根因是 InsLen 对 detected word 去重；LLaVA raw-surface 分支允许同图中的单复数（如 `skateboard`/`skateboards`）分别入选，但二者可能共享同一 normalized `word`、首次 token 位置和标签。旧 merge 使用集合键 `(image_id, response_token_idx, token_str)`，把这类合法的多重样本压成一条，共误删 346 条。
- `_merge_feature_parts` 现改为按 artifact 做 Counter 多重集合并集：root 与 part 的重复副本不会重复加入，但同一 artifact 内的合法 multiplicity 会完整保留；key 同时加入 `label`。resume 的 controlled key 比较也从 set 改为 Counter，能发现同键少一条的情况。
- 已使用保留的两个 worker parts 原子重建 LLaVA `features.pkl` 与 `baseline/features.pkl`，分片未删除、可恢复。最终两者均为 15,463 条；相对 labeling 的 missing=0、extra=0。主文件约 5.79 GB，baseline 约 158 MB，无需重新做 GPU 特征抽取。
- 验证：新增 merge 多重集合幂等与 resume multiplicity 回归；新用例加 training provenance 共 8/8 通过，相关 `py_compile` 通过。组合 `tests.test_stage_resume + tests.test_training_provenance` 为 20 pass/1 fail；唯一失败是已有 `test_legacy_feature_file_cannot_be_adopted_into_v2_resume` 仍期待 manifest adoption 报错，而当前 manifest 写入函数早已不实现该拒绝逻辑，与本轮 merge 修复无关。

## 2026-08-17 gfchair：关闭 feature manifest 强制校验

- 按用户要求，训练阶段现在与特征阶段共同服从 `run.validate_manifests`。活动配置为 `false` 时，不再要求或读取 `features_manifest.json`，可直接使用现有 `features.pkl` 和 `baseline/features.pkl`；因此 LLaVA 已完成的特征无需重提取或补写 manifest。配置为 `true` 时，原有 provenance 内容一致性校验仍可选启用。
- 仍保留与论文实验数值正确性直接相关的结构检查：特征必须完整覆盖 labeling 中的正式对象样本、图片 split 必须严格覆盖 cohort、token key 与二元标签必须一致。这些检查不依赖 manifest，也不会改变任何特征值或训练公式。
- 验证：`scripts/training_provenance.py` 与对应测试 `py_compile` 通过；`tests.test_training_provenance` 5/5 通过，其中新增用例确认关闭开关时没有 `features_manifest.json` 也能成功读取完整特征；活动 InsLen 配置解析确认 `validate_manifests=False`。gfchair 没有 `.git`，无法执行有效的 `git status`/`git diff --check`。

## 2026-08-17 gfchair：P-source 的 hmid/hpre cosine 对照与 GLSim softmax 核对

- compact four-gate 的 P-source 现可由 `feature_extraction.dgst_t.source_modes` 配置。活动配置设为 `[hmid_cos, hpre_cos]`：原基线为 `softmax(cos(o_ffn(q_t), hmid(v_i))/tau)`，新增对照为 `softmax(cos(o_ffn(q_t), hpre(v_i))/tau)`；二者是并行 marginal，不相加、不共享归一化，`o_ffn(q_t)` 与既有因果预测行均不变。默认未配置时仍只计算 `hmid_cos`，保持旧调用兼容。
- 新分支保存 `dgst_t_source_hpre_cos_dist_per_layer`，risk 字段在 method 后增加 `source_hpre_cos_`，同时保存 `dgst_t_source_modes` 与逐分支公式。VV/VP/VPend、普通/capped、source-tau/Top-K sweep、COCO compact、QA compact 和训练 alias 均已接通；活动 sqrt/cosine 两种 cost 与 raw-logit/softmax-prob 两种 gate 都会使用两套 P。活动训练列表从 8 项增至 16 项：每个 gate、source、cost 都训练 risk-only 与 risk+原 target EV。
- 核对 GLSim NeurIPS 2025 原论文：VLL 定义为 `h_l(v_i) W_U`，随后公式直接读取 `softmax(VLL_l(v_i))[o]` 并据此选 Top-K，未出现 temperature；论文中的 `tau` 是最终 real/hall 判定阈值。官方 `deeplearning-wisc/glsim` 代码同样对 LM-head logits 做普通词表 softmax；CLI 的 `inference_temp=0.1` 只传给 `model.generate(..., temperature=temp)`，不参与 VLL softmax。因此本项目当前 `hpre_softmax_prob_gauss` 的 T=1 词表 softmax与 GLSim 的该计算一致；InsLen 复现的 `0.1*logits`/T=10 是另一套实现，不能归到 GLSim 论文。
- 验证：新增手算测试同时核验两套 cosine-softmax P、两套 source 确实不同、新 risk 字段、COCO/QA 序列化与训练 alias；`test_four_gate_dgst + test_extraction_modes + test_cost_variants` 共 50/50 通过，wrapper sweep/prompt-CAFE 2/2 通过，相关 Python `py_compile` 与 `bash -n run.sh` 通过，继承后的活动 YAML 解析为 source modes 2、cost modes 2、feature sets 16，全部 alias 可解析。未启动完整 COCO4000 GPU 重提取。
- 失败记录：`test_pipeline_config + test_stateupd_alpha_sweep` 仍为 7 failures/4 errors，原因是活动配置已关闭 manifest 校验且保留 VV-only，而旧测试要求 manifest 防护与 `[vv,vpend]`；`test_raw_attention_integration + test_train_and_eval_stage` 仍有 2 failures，原因同为旧测试强制 VPend feature。均与本轮 P-source 数值/路由无关，和本文件上一条既有失败记录一致。gfchair 本身按隔离要求没有 `.git`，因此无法执行 `git status`/`git diff --check`。

## 2026-08-17 gfchair：InsLen 官方目标协议隔离对照版

- 从当前 dirty `token-detector` 工作区完整复制代码与未跟踪配置到 `/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair`；没有复制 `.git`、`outputs`、`logs`、`__pycache__`、`.pyc` 等生成物，也没有改写原项目。环境没有 `rsync`，首次复制命令因此未执行；随后用带精确 exclude 的 tar stream 完成复制并核验目录差异。
- 正式目标协议固定为 `inslen_official_first_token_first_occurrence`，sample unit 为 `inslen_unique_detected_word`。实现逐行对照 `Fraserlairh/Instruction-Lens-Score@d36f6f76023e7625ef86df7b60acf7a99ae3707d` 的 `util/chair.py`、`lvlm/utils.py` 和 `detector/detector.py`：CHAIR 使用 `nltk.word_tokenize(caption.lower()) -> inflect 单数化 -> 双词合并 -> synonym/canonical 映射`；REAL/HALL 各自维护 detected-word 去重表；找不到候选就跳过；成功后取候选首 subtoken ID 在回答中的第一次出现。
- 按用户后续更正，gfchair 正式流程完全移除了 response exact-offset 与 shadow exact：不计算、不保存、也不在特征记录中携带真实 surface token span。`all_object_token_spans` 只保留 InsLen resolved/skipped 决策审计，`object_token_spans` 只保留成功进入官方评测的对象。`word` 明确表示 CHAIR 归一化后的生成词（如 `woman`），`canonical_object` 单独保存 node word（如 `person`）；特殊模型只搜索前者。
- 模型映射为：LLaVA-1.5 使用原始 caption NLTK token 单独 tokenize 后的首 ID；InternVL 使用官方 `internVL` `find_word`；Qwen2.5/Qwen3 使用发布代码保留的 `QwenVL3` 分支，并在标签 provenance 说明官方没有发布 Qwen2.5 wrapper；LLaVA-OneVision 使用 `llava-one-version1.5`；LLaVA-NeXT 在启动 labeling 前直接报 unsupported。`find_word` 候选严格按 prefix `("", " ")`、suffix `("", "s", "es")` 顺序，只查看每个候选的第一个 ID。
- 特征仍是本项目的 DGST/spatial-cost/ADS-CGC 定义。正式 span 同时保存官方 resolved ID 与 first-occurrence index；抽取时强校验该 ID 等于 `generations.json` 对应位置的真实 ID。wrapper 使用该目标的因果预测行（位置 `i>0` 时为回答 token `i-1` 的状态），所有 `W h` target-logit/softmax 分支索引同一 resolved ID。feature extraction 与 training 都会验证 locator、sample unit、官方上游 commit 和 feature manifest，拒绝旧 exact-offset 标签或陈旧特征。
- 黄金样例使用原项目 Qwen2.5 图 `414289` 的真实 caption/response IDs 与本地 tokenizer：`handbag` 的候选为 `" handbag"`，首 subtoken ID=`1424`；该 ID 在回答位置 `15/24/43` 出现，正式 gfchair 目标固定为第一次位置 `15`，输出确认不存在 shadow exact 字段。没有启动完整 COCO4000 GPU 实验。
- 默认入口使用 `configs/model_configs_inslen_official_target.yaml`，`run.sh` 默认绝对输出为 gfchair 自身的 `outputs/<model>/COCO4000-INSLEN-OFFICIAL-TARGET`，CHAIR cache 也写入 gfchair；QA output root 同样是 gfchair 绝对路径。初始代码副本没有携带 outputs；随后按用户新要求，在 gfchair 下为 LLaVA-1.5、InternVL2.5、Qwen2.5-VL、Qwen3-VL 建立默认实验目录，并分别从原项目 `COCO4000-COST-spatial` 只复制 `generations.json`。四份均为 4000 行、caption/response IDs 无空值，且与来源 SHA-256 逐字节一致；没有复制 labeling、features、results、manifest 或日志。
- gfchair 的活动 cost 对照随后改为两个非空间模式：主 `cost_mode=sqrt_matched_state`，`cost_modes=[sqrt_matched_state, cosine_matched_state]`；不再提取 `sqrt_matched_state_plus_spatial`。训练列表同步改成 raw-logit Gaussian 与 softmax-prob Gaussian 各自的 sqrt/cosine risk，并分别训练 risk-only 与 risk+EV，共 8 个 feature set。运行时规范化结果为 `('sqrt_cosine_matched_state', 'cosine_matched_state')`，8 个训练 alias 全部可解析，定向 cost/risk/alias 测试 3/3 通过。
- 定向测试覆盖 raw-surface/normalized-word 分支、空格与单复数候选顺序、person/people、缺失跳过、官方去重时机、首次 ID、多次 ID、因果 prefix、resolved-ID 一致性、`W h` 列索引、resume/feature/training provenance，最终相关组合全部通过。全仓 `unittest discover -s tests -p 'test*.py'` 为 259 项、12 failures、5 errors；剩余项均来自复制前活动 YAML 与旧断言/关闭 manifest 保护的既有不一致（VPend、capped sweep、pipeline manifest 等），InsLen 定向项与 training provenance 均通过。第一次未指定 `-s tests` 的 discover 运行了 0 项，已用正确命令重跑。

## 2026-08-16 COCO4000 spatial-cost 过夜运行预检

- 对四个 `COCO4000-COST-spatial/generations.json` 做了正式运行前只读核验：四份均为 4000 图、无空 caption/response token，且 image cohort 与共享 strict 8:2 split 完全一致；模型、COCO、CHAIR cache、NLTK 资源路径均存在，两张 RTX 4090 空闲，存储空间充足。
- 在 `/tmp/token-detector-label-preflight.hKyaEK` 中按当前 unified YAML 对四份 caption 完整预演 schema-v2 labeling，四模型 4000/4000 全部完成且没有 exact-response-offset 对齐失败；当前 unified 实际使用 `first_canonical_mention`，训练 mention 数依次为 LLaVA 12700、InternVL 9960、Qwen2.5-VL 7879、Qwen3-VL 11751。临时预演没有写入正式输出目录。
- 四个真实模型分别在 COCO 图 230008 上完成一 token 的 method-only spatial-cost 前向：视觉网格依次为 `24x24`、`16x16`、`13x23`、`11x20`，两种活动 gate 均输出完整的 `plus_spatial` risk（层数 32/32/28/36），语义/空间权重均核验为 1.0。四模型训练 dry-run 均正确展开 4 个 feature set × seeds 43/44/45；禁用的 source-tau/Top-K sweep CLI 对四模型都正常 no-op。
- 运行环境风险：当前非交互默认 `/opt/conda/bin/python` 缺少 PyYAML，直接依赖 `run.sh` 的默认 `PYTHON_BIN=python` 会立即失败；过夜命令必须显式设置 `PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python`，并保持 `NLTK_DATA=/home/apulis-dev/userdata/nltk_data`。
- 验证命令 `CUDA_VISIBLE_DEVICES='' /opt/conda/private/envs/vicr/bin/python -m unittest discover -s tests -p 'test_*.py'` 共 248 项，结果为 15 failures、5 errors。失败集中在活动 YAML 与旧测试预期不一致：`validate_manifests=false` 导致 manifest/provenance 保护测试不成立，当前 VV-only/spatial-only 与旧 VPend 断言不符，unified/fj01 的 sample unit、support scope 等运行语义不再镜像，以及已关闭 capped sweep 但旧测试直接校验其 risk modes；这些失败不出现在本次 unified method-only 正式命令的已验证路径。spatial cost 数值/序列化 22 项与五模型/直接 four-gate/prompt-target plumbing 6 项通过，相关 Python `py_compile`、`bash -n run.sh`、`git diff --check` 通过。

## 2026-08-16 新增语义 + 视觉物理距离 cost

- 为四模型 spatial-cost 正式实验新建 `outputs/{llava_1_5_7b,internvl_2_5_8b,qwen2_5_vl_7b,qwen3_vl_8b}/COCO4000-COST-spatial/`，每个目录只复制该模型 `COCO-4000-ALLMENTION-SWEEP/generations.json`，没有复制旧 labeling、features、split 或 results。四份 generation 均为 4000 行，目标文件与各自来源 SHA-256 完全一致；前三模型的来源还与同模型 `COCO4000-ALLMENTION/generations.json` 逐字节一致。
- 后续按用户要求把 `configs/model_configs_unified.yaml` 与 `configs/model_configs_server_fj01.yaml` 都改成只计算 `sqrt_matched_state_plus_spatial`，语义/空间系数均为 `1.0`。普通 MLP 训练项只保留 raw-logit、softmax-prob 各自的 spatial-risk 与 spatial-risk+原 EV，共 4 项；source-tau/transport-TopK 与 capped 提取、旧的自动 sweep 训练入口均关闭，避免新 cost 单独运行时继续计算无关变体或让旧 sweep trainer 查找不存在的旧 risk 字段。两个 YAML 均通过运行时配置校验，四个 feature-set alias 全部可解析，`git diff --check` 通过。
- 开始修改前先保存全部已跟踪/未跟踪本地改动，随后将 `hope-best` 从 `6fac6eb` 快进到远端最新 `5b64fae`；恢复本地内容时仅 `docs/CURRENT_TASK.md` 冲突，已同时保留远端 7 月 27 日记录与本地 7 月 28 日记录。保险 stash `codex-preserve-before-spatial-cost-pull-20260816` 仍保留，用户原有 `utils/config_utils.py`、`tests/test_pipeline_config.py` 和 7 份未跟踪 VQA YAML 未被本轮覆盖。
- compact four-gate 新增独立 cost mode `sqrt_matched_state_plus_spatial`（规范名 `sqrt_cosine_matched_state_plus_spatial`），旧 `sqrt_matched_state` 不变。新定义为 `C = 1.0 * Csem + 1.0 * Cspatial`：`Csem=sqrt((1-cosine)/2)`；`Cspatial` 为视觉 token 在二维 patch 网格上的欧氏距离除以网格对角线，范围 `[0,1]`。VP/VPend 中只对视觉—视觉 token 对叠加空间项，只要任一端是 prompt token，空间项即为 0。
- 两个系数可通过 `feature_extraction.dgst_t.spatial_cost_semantic_weight` 与 `spatial_cost_distance_weight` 调节，默认均为 `1.0`。Qwen2.5、Qwen3、LLaVA、LLaVA-OneVision、InternVL 和 prompt-target 路径均透传真实视觉网格；启用新 cost 但模型无法给出矩形网格时会明确报错，不会猜测 token 布局。
- 新 risk 字段为 `dgst_t_<method>_risk_sqrt_<hpre|hmid>_plus_spatial_per_layer`，VV/VP/VPend、capped 和 source-tau/transport-TopK sweep 均沿用现有 exact EMD 与新提交的前处理缓存/去重路径。COCO 训练 alias 为 `<method>_risk_sqrt_matched_state_plus_spatial`（VP/VPend 使用原 scope 前缀），QA compact/probe 同步支持同名 component；特征中保存系数、视觉网格、距离归一化及 prompt 策略供审计。
- 本轮没有修改活动 YAML、`run.sh`，也没有启动正式 GPU 特征提取或训练。做 cost 对比时可令 `cost_mode: sqrt_matched_state`，并把 `cost_modes` 设为 `[sqrt_matched_state, sqrt_matched_state_plus_spatial]`；训练列表分别选择旧 risk alias 与新的 `..._plus_spatial` alias。
- 验证：空间矩阵手算、`Csem+Cspatial` exact EMD、COCO/QA 序列化、训练 alias、缓存开关逐值一致均通过；four-gate/prompt-target/prompt-CAFE/视觉 support 组合 `37/37` 通过，state-update 相关定向 `3/3`、五个 wrapper/直接 four-gate/prompt-target plumbing `6/6` 通过；相关文件 `py_compile` 与 `git diff --check` 通过。

## 2026-07-27 four-gate exact-EMD 前处理缓存与 COCO500 capped 子集

- compact four-gate 新增默认开启的 `DGST_FOUR_GATE_PREP_CACHE=1`。缓存严格限制在单层内部，只复用 source/target transport Top-K、capped Top-Mass indices/union、target region、union-support 上重新归一化后的 marginal，以及同一 `(cost_mode, matched_state, exact_union_support_indices)` 的 cost matrix；每个不同 source/target distribution 仍分别调用 POT exact EMD，不复用 transport plan，也不从大 support 的解推导小 support。
- 为避免完整矩阵切片可能引入不同 GEMM kernel 的末位浮点差异，cost 采用“第一次按历史 subset-first 路径计算、后续相同 support 原样复用”，而不是预计算整个视觉区域的矩阵。因此 feature 定义、support、renormalization 和 exact solver 均未改变；可用 `DGST_FOUR_GATE_PREP_CACHE=0` 随时切回旧路径做 A/B。
- 新增缓存开关前后端到端嵌套输出逐值比较，覆盖两个 target 方法、三个 cost、两个 source tau 和三个 capped alpha，所有 tensor、risk、cosine、EV、support 与 sweep payload 完全相等。four-gate/cost/sweep 组合回归 44 项通过；唯一额外失败仍是旧测试要求活动 YAML 启用 VPEND，而用户当前 unified 配置明确为 VV-only，与缓存无关。`py_compile`、两份 shell `bash -n`、新 YAML 解析及 `git diff --check` 通过。
- 等价合成负载（576 visual support、VV+VPEND、两方法、4 tau、5 alpha、单 cost）中，每层 cost 构造由 120 次降为 20 次；预热后的 exact-EMD 输出完全相同，中位耗时由 `0.3671s` 降为 `0.1310s`，该单层准备/求解负载约 `2.80x`。正式模型端到端仍需用 COCO500 实测，不能把该合成倍率直接外推到整条 pipeline。
- 从 `COCO4000-512-CAPPEDSWEEP` 按 seed 42、原 train/test 和是否含幻觉 mention 分层抽取 500 张物理图，保存为 `outputs/qwen2_5_vl_7b/COCO500-capped`，严格 400/0/100 split；500 份 generation/label/ground truth 完全对齐，未复制旧 feature。新增 `configs/model_configs_server_fj01_coco500_capped.yaml` 和 `run_coco500_capped.sh`，专用入口跳过 generation/labeling，先验证既有 500 captions 再执行 extraction、training 与 sweep，避免普通 `run.sh` 按 4000 图配置扩充或拒绝该子集。

## 2026-07-27 capped Top-Mass alpha 超参验证

- 将 compact four-gate 路径中原先固定的 `capped_topmass_085` 扩展为 YAML 可配置的多 alpha 实验。两份活动配置均新增 `feature_extraction.dgst_t.capped_topmass_alphas: [0.7, 0.75, 0.8, 0.85, 0.9]`；字段名分别使用 `capped_topmass_070/075/080/085/090`。旧开关 `compute_capped_topmass_085`、旧标量 `capped_topmass_085_alpha` 和所有 0.85 字段继续保留，因此已有 0.85 缓存与旧训练入口仍可读取。
- VV、VP、VPEND 的各模型 wrapper 和 prompt-target 共用入口现在都会透传 alpha 列表。特征提取对每个 alpha 分别构造 source/target capped union support、exact EMD risk、target-only cosine 与 EV，并在根记录和 `dgst_t_hparam_sweep` 中序列化动态字段；同时保存 `dgst_t_capped_topmass_alphas` 与 slug 到数值的映射供审计。
- sweep 训练新增 `capped_topmass_alpha_sweep` 模式。固定 Top-K 仍训练 `source tau × transport Top-K`，新 capped 模式训练 `source tau × capped alpha`，并验证同一 source tau 下不同固定 Top-K 变体中的 capped risk 完全一致，避免重复训练。两份 YAML 的默认 `risk_modes` 已改为 `[fixed_topk, capped_topmass_alpha_sweep]`；0.85 只由新 sweep 训练一次，不再与旧单值模式重复。
- 验证：相关 Python 文件 `py_compile` 通过；four-gate 数值、动态字段和序列化回归 19/19 通过；训练矩阵、YAML 选择和去重回归 7/7 通过；五个模型 wrapper、两个直接 four-gate 路径与 prompt-target plumbing 3/3 通过；两份 YAML 运行时解析、fj01 禁用开关 CLI no-op 与 `git diff --check` 通过。首次训练测试因动态 EV 尚未登记在静态 alias 表而失败，已改为读取带完整 provenance 的动态字段后通过；首次 plumbing 检查发现 LLaVA/InternVL 的直接 four-gate快路未透传 alpha 列表，补齐后通过。本轮未启动正式 GPU 特征重提取或训练。
- 扩展回归中 LLaVA/InternVL 完整 caption batch-forward 2/2 通过。`test_stateupd_alpha_sweep.py` 为 3/4：唯一失败仍是旧断言强制 unified YAML 必须启用 `support_modes=[vv,vpend]`，而用户当前 unified 配置明确保留为 `[vv]`；该失败与 capped alpha 改动无关，本轮未覆盖用户的 scope 选择。
## 2026-07-28 Qwen3/Qwen2 三个 VQA benchmark 的 Ours 完成

- Qwen3-VL-8B 与 Qwen2.5-VL-7B 的 `VPEND-ALL` 现均已完成 POPE、CLEVR-Exist-9K、AMBER 三个 benchmark 的 Ours 三种子训练。Ours 严格为全层 `hpre_raw_logit_gauss_risk_sqrt_matched_state + hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine`，source tau=`0.05`；Qwen3 target Top-K=`8`，Qwen2.5 target Top-K=`32`。本轮新增 Qwen2 的 `configs/model_configs_vqa_ours_only_target_topk32.yaml` 及 AMBER prompt-only 子配置；Qwen3 使用对应 target-TopK=8 的 Ours-only 配置。新提取的 Qwen2 CLEVR/AMBER 未抽取或训练 ADS、CGC、baseline。
- 两个模型的 POPE/CLEVR 均有 9000 条 prompt-last 与 9000 条 question-object 特征；AMBER 均有 14216 条 prompt-last、0 条可靠 question-object 特征，因此 AMBER 只训练 `prompt_last_token`。六个目录的 `extraction_failures.jsonl` 均为 0 字节，正式汇总均位于 `outputs/qa_benchmarks/{qwen2_5_vl_7b,qwen3_vl_8b}/VPEND-ALL/{pope,clevr_exist_9k,amber_discriminative}/results/summary_mean_std.json`，训练结束后 GPU 0/1 已释放。
- train-Real-F1 阈值下，Qwen2.5 的 `answer_correctness_all` prompt-last AUC/Hall-F1：POPE `0.8257+-0.0016 / 0.3487`、CLEVR `0.8852+-0.0142 / 0.3574`、AMBER `0.8356+-0.0071 / 0.3963`；Qwen3 对应为 POPE `0.7535+-0.0197 / 0.2861`、CLEVR `0.8733+-0.0132 / 0.2643`、AMBER `0.8563+-0.0038 / 0.4034`。`object_hallucination_yes_only` prompt-last AUC/Hall-F1：Qwen2.5 为 `0.7747/0.1306`、`0.7989/0.0128`、`0.7666/0.3250`；Qwen3 为 `0.8541/0.3237`、`0.8083/0.0897`、`0.8275/0.5118`。
- Qwen2 CLEVR 首次续跑被旧的、不兼容 extraction fingerprint 分片拒绝，未生成新行。为保留可恢复性，只把 CLEVR 约 1.1 GB 与 AMBER 约 14 MB 的旧根方法缓存分别移动到其 `.qa_parallel/features.stale-before-ours-topk32-20260728`；根 generation、labels、split、baseline 均未移动或删除。清理精确缓存后两项均复用 generation、双卡完成新特征并正常退出。

## 2026-07-28 InternVL 三个 VQA benchmark 的 Ours-only 续跑

- 新增 `configs/model_configs_vqa_ours_only_target_topk8.yaml`：继承既有 target-TopK=8 配置，只把 QA extraction mode 改为 `method_only`；实际参数为 source tau `0.05`、target Top-K `8`，唯一 probe 输入为全层 `hpre_raw_logit_gauss risk_sqrt_matched_state + target-dist mass x cosine`，未抽取/训练 ADS、CGC 或 baseline。另增 `configs/model_configs_vqa_ours_only_target_topk8_prompt_last.yaml`，仅用于没有可靠问题物体 span 的 AMBER。
- InternVL `VPEND-ALL` 的 POPE、CLEVR-Exist-9K、AMBER 已全部续跑完成；三者均复用已有 generation，双卡完成 Ours 特征抽取，随后完成 seeds `43/44/45` 与两个 label protocol 的 MLP。POPE/CLEVR 各有 9000 条 prompt-last 和 9000 条 question-object 特征；AMBER 有 14216 条 prompt-last、0 条 question-object。三个 `extraction_failures.jsonl` 均为 0 字节，训练结束后两张 GPU 均已释放。
- train-Real-F1 阈值下三种子 `AUC / Hall-F1`：POPE `answer_correctness_all` prompt=`0.7279+-0.0060 / 0.2369+-0.0210`、object=`0.5056+-0.0127 / 0.0665+-0.0253`，`object_hallucination_yes_only` prompt=`0.8367+-0.0101 / 0.3159+-0.0437`、object=`0.6333+-0.0371 / 0.1317+-0.0938`；CLEVR correctness prompt=`0.9543+-0.0022 / 0.4437+-0.0667`、object=`0.6957+-0.0389 / 0`，yes-only prompt=`0.8721+-0.0520 / 0.2340+-0.0272`、object=`0.5416+-0.0640 / 0.1333+-0.1886`；AMBER correctness prompt=`0.8547+-0.0036 / 0.4442+-0.0262`，yes-only prompt=`0.8483+-0.0090 / 0.5790+-0.0127`。
- 正式汇总分别位于 `outputs/qa_benchmarks/internvl_2_5_8b/VPEND-ALL/{pope,clevr_exist_9k,amber_discriminative}/results/summary_mean_std.json`。`run_qa.sh` 的 baseline 与 cross-family comparison 阶段按 `method_only` 正常跳过，没有改写已有 baseline 结果。

## 2026-07-28 Qwen3 TC tau=0.05 / transport Top-K=16 risk-only 核验

- 对 `outputs/qwen3_vl_8b/COCO4000-512-TC/features.pkl` 的 VV raw-logit Gaussian sweep 变体 `tau0p05_topk16` 完成全 36 层 risk-only 三种子训练；严格复用原 sweep 的 3200/800 图片 split、11,751 条 first-canonical token、三层 MLP、seeds 43/44/45 和 train Real-F1 阈值。AUROC/Real-F1/Hall-F1/Hall-AUPR/Accuracy=`0.8240+-0.0041 / 0.8839+-0.0026 / 0.4731+-0.0561 / 0.5783+-0.0055 / 0.8100+-0.0009`（样本标准差）。产物位于 `results/vv_tau0p05_topk16_hpre_risk_only/`。
- 同一 risk 与 TC 根缓存 legacy EV（target Top-K=32 mass x cosine）拼接的既有三种子结果为 AUROC/Real-F1/Hall-F1/Hall-AUPR/Accuracy=`0.8723+-0.0035 / 0.8960+-0.0010 / 0.6148+-0.0006 / 0.6781+-0.0045 / 0.8362+-0.0013`。
- 审计 `COCO4000-512-TC/new/features.pkl` 全部 11,751 行：schema 声明并实际只含 target-cosine Top-K `16/32/64/128`，固定 target mass 仅有 recovered Top-K=32；没有任何 target Top-K=8 字段，因此未用 K=16 静默替代用户原先指定的 K=8。
- 后续按用户明确要求计算 `new` target-K=16：从 TC 根缓存按 `normalize(attention_support * hpre_raw_logit_gauss_gate)` 逐层重建 target distribution，其重建 mass@32 与 `new` 已存 mass@32 在 11,751 行上的最大误差仅 `2.90e-7`；据此取真正的 mass@16，乘以 `new` cosine@16，再与同一 `tau=0.05 / transport Top-K=16` risk 拼接。三种子 AUROC/Real-F1/Hall-F1/Hall-AUPR/Accuracy=`0.8659+-0.0046 / 0.8944+-0.0017 / 0.6015+-0.0090 / 0.6620+-0.0062 / 0.8331+-0.0019`。产物位于 `results/vv_tau0p05_transport_topk16_risk_plus_new_target_topk16_mass_x_cosine/`。
- 补充交叉组合（同一 `tau=0.05`、split/MLP/seeds/阈值）：risk transport Top-K=64 + TC-root target mass x cosine K=32 的 AUROC/Real-F1/Hall-F1/Hall-AUPR/Accuracy=`0.8675+-0.0041 / 0.8983+-0.0055 / 0.6069+-0.0216 / 0.6706+-0.0053 / 0.8385+-0.0085`；risk transport Top-K=32 + `new` target mass x cosine K=16 为 `0.8629+-0.0028 / 0.8915+-0.0016 / 0.5977+-0.0151 / 0.6598+-0.0094 / 0.8292+-0.0024`。后者产物位于 `results/vv_tau0p05_transport_topk32_risk_plus_new_target_topk16_mass_x_cosine/`。
- 将上述两组的 source tau 换为 `0.06`：risk transport K=64 + target K=32 为 `0.8704+-0.0080 / 0.8978+-0.0029 / 0.6204+-0.0246 / 0.6727+-0.0162 / 0.8390+-0.0058`；risk transport K=32 + `new` target K=16 为 `0.8680+-0.0043 / 0.8895+-0.0022 / 0.6006+-0.0126 / 0.6710+-0.0170 / 0.8269+-0.0028`。后者产物位于 `results/vv_tau0p06_transport_topk32_risk_plus_new_target_topk16_mass_x_cosine/`。
- target mass x cosine K=32 不使用 source tau，sweep 的所有 tau/transport 变体均复用同一条 VV raw-logit Gaussian EV 曲线；因此 `tau=0.06` 下单独训练等同于已有 `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` standalone 三种子结果。train Real-F1 阈值下 AUROC/Real-F1/Hall-F1/Hall-AUPR/Accuracy=`0.8445+-0.0045 / 0.8906+-0.0051 / 0.5348+-0.0064 / 0.6049+-0.0105 / 0.8228+-0.0069`（样本标准差）。

## 2026-07-28 AMBER attribute 的 risk + mass*cosine 改进审计

- 严格限定 Ours 只包含 `risk + target-distribution mass x cosine`，不拼接 ADS/CGC。对当前 LLaVA `VPEND-ALL` 的 VV raw-logit 特征进行了 outer-train 内部按图像隔离的层窗口与 MLP 超参筛选；全层仍最好，但 attribute 内层 AUC 约 0.78，未显示能稳定达到 SVAR 0.85 的证据。attribute-only head 内层 AUC 也仅约 0.80，且会改变统一训练协议，因此未作为正式结果。
- 找到旧 LLaVA AMBER 缓存中的纯 VP softmax-Gaussian `risk + mass*cosine` 结果。严格三种子均值的 attribute AUC/Hall-F1 为 `0.8438+-0.0455 / 0.5331+-0.0797`，即一位小数 `84.4/53.3`；相对当前 native SVAR `85.0/47.6`，AUC 低 0.6 点、Hall-F1 高 5.7 点。
- 对同三个既有 checkpoint 做等权 probability soft-voting，阈值仍只在 outer-train 的 ensemble prediction 上按 Real-F1 选择。attribute AUC/Hall-P/Hall-R/Hall-F1=`0.8800/0.6371/0.5663/0.5996`，整体 AMBER AUC/Hall-F1=`0.9297/0.6893`。该结果超过当前 native SVAR，但属于 ensemble 口径，不能直接替换原表的三种子均值行，除非 baseline 也按相同 ensemble 协议评估。
- 完整说明位于 `outputs/qa_benchmarks/llava_1_5_7b/amber_discriminative/results/attribute_improvement/vp_softmax_risk_ev_report.md`。探索产物全部隔离在 `results/attribute_improvement/`，未覆盖正式 summary/checkpoint；此前误试的 Ours+ADS+CGC 已明确排除。

## 2026-07-26 sweep 全层默认与 run.sh 自动训练开关

- Sweep 训练新增 `risk_modes` 选择，支持 `fixed_topk` 与 `capped_topmass_085`；两份活动 YAML 当前都配置为同时训练两类（总开关仍为 `false`）。fixed 模式沿用 source tau × transport Top-K 笛卡尔网格；capped 模式以自适应 source/target top-mass union 取代固定 Top-K，因此只按 source tau 训练一次，不会把同一 tau 下不同 Top-K 的等价 capped 曲线重复训练。
- capped sweep 不是复用主参数 `tau=0.07` 的 capped risk：特征提取现在会对每个 sweep source tau 重新构建 source distribution、重新选 capped union support 并求 exact EMD，同时复用与 source tau/transport Top-K 无关的 branch-specific capped EV。嵌套 sweep payload 同时保存 fixed risk 与对应 tau 的 capped risk；训练读取 capped risk + capped EV，仍默认全层拼接。旧 `features.pkl` 的 `dgst_t_hparam_sweep` 只有 fixed risk，选择 capped 前必须用新代码重提特征。
- `scripts/train_source_tau_transport_topk_sweep.py` 的默认 risk slice 从固定层 `[20,29)` 改为 `[0, 实际曲线长度)`；未传 `--layer-end` 时会按模型自动使用全部 decoder 层，并检查同一次训练中的 risk 曲线层数严格一致。EV 仍使用对应方法的全层曲线，因此 Qwen3 的默认组合为 36 层 risk + 36 层 EV，32 层模型为 32+32；仍可显式传 `--layer-start/--layer-end` 做旧式子区间实验。
- 两份活动 YAML 在 `training.source_tau_transport_topk_sweep` 新增训练开关和方法列表，当前均为 `enabled: false`，方法为 raw-logit Gaussian 与 softmax-prob Gaussian。打开开关时，`run.sh` 会在普通 `train_and_eval.py` 完成后调用 sweep 脚本；关闭时该入口只打印 skip 并立即退出，不加载大 `features.pkl`。自动训练按 YAML 当前 `support_modes` 选择 VV/VPEND scope，并把两类方法分别写入既有的 `...full_hpre_risk_plus_full_ev` 与 `...full_hpre_softmax_prob_gauss_risk_plus_full_ev` 目录。
- 要实际启用，除了把训练开关设为 `true`，还必须给 `feature_extraction.dgst_t.source_tau_values` 与 `transport_top_k_values` 配置非空网格并重新提取含 `dgst_t_hparam_sweep` 的特征；缺少网格时会在加载大特征前明确报错。本轮保持两份 YAML 的网格为 `null`、开关为 `false`，没有启动正式 GPU 提取或 sweep 训练。
- 验证：新增 full-layer 拼接、fixed/capped 选择、capped Top-K 去重、CLI 默认值、YAML 开关/方法、scope 推导、缺失网格前置报错和 `run.sh` 阶段顺序回归 6/6；capped sweep 的 source-tau 重算、手算 EMD、VV/VPEND 序列化定向测试通过，既有 sweep wrapper plumbing 3/3。禁用开关的真实 CLI no-op、相关 Python `py_compile`、`bash -n run.sh`、两份 YAML 运行时解析及 `git diff --check` 均通过。第一次尝试 pytest 因项目环境未安装 pytest 而无法启动；改用标准库 unittest 后通过。扩展合跑旧 `test_train_and_eval_stage + test_stateupd_alpha_sweep` 为 12/14，其中 2 个失败仍是旧断言要求 unified 必须启用 VPEND，而用户当前 unified 明确为 `support_modes=["vv"]`，与本轮开关/全层逻辑无关；另一次定向 unittest 首次因输入了不存在的测试方法名而报 loader error，改用实际方法名后通过。

## 2026-07-26 LLaVA/InternVL 恢复完整 caption batch-forward

- 经典 `LLaVAWrapper` 与 `InternVLWrapper` 的公开 `extract_token_features_batch()` 已从“每个目标对象 token 单独做一次 causal-prefix forward”恢复为“一张图、完整 teacher-forced caption 只做一次 forward”，随后从同一次 decoder 输出中读取所有请求的 `j-1` prediction rows，并把全部 target IDs/positions 一次交给 DGST batch 计算。Qwen、Qwen3、LLaVA-OneVision 与独立 prompt-target API 未改。
- 因果语义仍成立：完整 caption 虽一次送入模型，但 decoder causal mask 保证第 `j-1` 行不能看到目标 token `j` 或未来 token；变化只是共享此前重复计算的图像、prompt 和历史 response prefix。MetaToken/HalLoc 需要的完整 response hidden/logit statistics 也直接复用这一次 forward，不再额外执行共享 baseline forward。
- 原来保留但不调用的 full-response reference implementation已成为正式 batch 实现；删除两个 wrapper 不再需要的 `dataclasses.replace`。测试改为断言公共 batch API 只委托一次 full-response 实现、绝不调用 single-target forward。
- 验证：LLaVA/InternVL batch 路由、causal positions、prompt-target、extraction modes、four-gate/capped、sweep plumbing、prompt CAFE 与视觉 support 相关组合 53/53 通过；两个 wrapper 与测试 `py_compile`、`git diff --check` 通过。本轮未启动正式 GPU 特征抽取。

## 2026-07-26 compact four-gate 接入 capped-topmass 0.85

- 仓库原有 `capped_topmass_085` 只在 legacy DGST 路径生效；当前活动的 `target_gate_mode=four_gate` 会提前进入 compact 快路径，旧的 capped 参数此前不会被消费，当前 `COCO4000-512-TC/features.pkl` 因而没有任何 capped 字段。本轮把该逻辑正式接入 compact VV/VP/VPEND，并保持未显式开启时不新增计算。
- 当前定义为：source 与 branch-specific target distribution 分别按概率降序取达到累计质量 `0.85` 的最小 K，再把每侧 K 截断到 `[32,64]`；capped risk 在两侧索引并集上做同一 exact EMD，capped target-cosine 与 capped EV 在 target-only capped 区域上计算。输出同时记录 alpha/min-K/max-K 和定义 provenance。
- Qwen/Qwen3/LLaVA/InternVL/OneVision 以及 prompt-target 共用路径均已透传开关；compact serializer 和训练 alias 支持 capped risk、target-cosine、EV，包括所有已有 cost 与 VV/VP/VPEND 前缀。
- 两份活动 YAML 均启用 `compute_capped_topmass_085=true`、alpha=`0.85`、min-K=`32`、max-K=`64`，并各新增四个 `capped risk + capped target-cosine` 训练项：VV/VPEND × raw-logit/softmax-prob。训练 dry-run 为 3 seeds × 15 个当前 method feature sets，四个 capped 组合均被保留。
- 新增合成回归精确核验 capped region、union support、EMD risk、cosine、VV/VPEND 序列化和两块 36 层向量的 72 维训练拼接；同步补齐 extraction-mode 旧 fixture 缺失的 VV scope 元数据。four-gate/state-update/sweep plumbing/prompt-target/QA/extraction/raw-attention 相关组合 46/46 通过；单独的 config-mirror 测试仍因用户保留的 unified/fj01 extraction/baseline/QA 选择差异失败。
- 现有 TC 的 11,751 条 first-canonical 特征不能直接训练 capped，因为它们没有对应字段；需要重新抽取。为避免在关闭 manifest 后把旧/新 schema 混入同一缓存，正式尝试应使用新的输出目录并只复用 generation，再生成 all-mentions labeling/features。

## 2026-07-26 Qwen3 risk + raw target-cosine 三 seed 训练

- 复用 `outputs/qwen3_vl_8b/COCO4000-512-TC/features.pkl`，新增训练 VV/VPEND × raw-logit/softmax-prob 四组 `risk + raw target_cosine`；每组输入为 36 层 risk 与 36 层 cosine 拼接，共 72 维。协议沿用严格图片级 8:2、seeds `43/44/45` 和当前三层 Torch MLP，同时报告固定 0.5 与 train Real-F1 阈值。
- train Real-F1 阈值下，四组的 `AUROC / Hall-F1 / Hall-AUPR` 分别为：VV-softmax `0.8482±0.0037 / 0.5609±0.0223 / 0.6220±0.0132`；VPEND-raw `0.8474±0.0106 / 0.5615±0.0246 / 0.6232±0.0112`；VV-raw `0.8434±0.0022 / 0.5554±0.0174 / 0.6290±0.0053`；VPEND-softmax `0.8417±0.0082 / 0.5320±0.0104 / 0.5942±0.0127`。
- 与同一缓存、split、head 的 risk-only 相比，加入 raw cosine 后四组 AUROC 均提升 `+0.0186` 至 `+0.0242`，Hall-F1 提升 `+0.0428` 至 `+0.0735`，Hall-AUPR 提升 `+0.0296` 至 `+0.0477`；但仍全面弱于已有 `risk + EV`，后者把 cosine 乘上 target-distribution mass，四组 AUROC 还高 `+0.0219` 至 `+0.0313`。说明 cosine 有独立信号，但 target mass 加权是组合效果的重要组成。
- 汇总位于 `results/qwen3_vl_8b_risk_plus_target_cosine_3seed_summary.{md,csv,json}`；3 seeds × 4 feature sets 的 12 个 checkpoint/config/history 及逐 seed JSON/Markdown 均完整，聚合项全部为有限值。
- 重要口径：当前 TC artifact 仍是 11,751 条旧 `first-canonical` labeling/features；活动 YAML 虽已改成 `all_mentions`，但尚未重新标注和补提特征。因此本轮结果只与同一旧缓存上的 risk-only/risk+EV 公平可比，不能当作未来 32,631 条 all-mentions 重提后的最终结果。

## 2026-07-26 all-mentions、关闭 manifest 与停止 sweep

- 两份活动 YAML 的 COCO `labeling.sample_unit` 改为 `all_mentions`；schema-v2 labeling 现在会把 caption 中每次有效 COCO 对象出现都写入 `object_token_spans`，不再只保留每个 canonical object 的第一次出现。训练报告中的 label protocol 同步改为动态记录实际 sample unit。
- 新增 `run.validate_manifests` 总开关并在两份活动 YAML 设为 `false`。关闭时 COCO generation/labeling/features、协调器复用路径及 QA baseline 提取/训练均不读取、不校验、也不写 provenance manifest；generation/label/features 的字段完整性、实际 response token ID、一致 caption、样本 cohort 与严格 split 检查仍保留。
- `source_tau_values` 与 `transport_top_k_values` 均设为 `null`，普通特征抽取不再生成 tau×Top-K sweep 变体；主参数 `tau=0.07`、`transport_top_k=64` 不变。
- 已删除 `outputs/qwen3_vl_8b/COCO4000-512-TC` 内 4 个现有 sidecar：根 generation/labeling/features manifest 和 baseline/features manifest；generation、labeling、features、split、模型结果均未删除。
- TC 当前 `labeling.json` 仍是旧 first-canonical 数据，根与 baseline 特征各 11,751 条。只读审计其已保存的 `all_object_token_spans` 得到 32,631 个可提取且无重复的 all-mentions 目标，3,713 张图片需要补充特征。下一次 `run.sh` resume 会复用完整 generation，检测 sample-unit 不同后重建 all-mentions labeling；特征 resume 已从“按图片是否出现”加强为“按目标 token 键是否完整”，会重处理这 3,713 张图片并在合并时去重旧记录，而不会错误跳过整张图。
- 验证：新增/直接相关回归 18/18 通过，相关 Python 文件 `py_compile`、两份 YAML 解析、`bash -n run.sh run_qa.sh` 与 `git diff --check` 通过。扩展组合的 3 个 stage-resume 失败仍是仓库此前已有的旧 manifest 拒绝契约断言；活动 QA/config 扩展组合另有 2 个失败，来自用户已保留的 unified/fj01 非路径配置差异及当前 unified baseline-only 使旧测试期待的 22 个 method feature sets 为空，未覆盖这些配置。

## 2026-07-26 VQA 问题物体 token 与 prompt 末 token 对比

- 再次以 `model_configs_server_fj01.yaml` 的当前实验选择为基准同步 `model_configs_unified.yaml`，只保留两台机器各自的数据、模型与输出绝对路径。同步后的 COCO extraction mode 为 `all`，baseline 提取/训练只启用 MetaToken，source tau 网格为 `[0.02, 0.03, 0.04, 0.05, 0.06]`，transport Top-K 网格为 `[16, 32, 64, 128]`。
- 两份活动 YAML 的 VQA `position_protocols` 现同时启用 `prompt_last_token` 和 `question_object_pre_token`。前者取完整 prompt 最后一行并预测第一个回答 token；后者在完整多模态上下文中定位问题物体的第一个真实 contextual sub-token `j`，严格截断到 `j` 之前并用预测行 `j-1` 提取同一套 DGST 特征。训练脚本按 `@position` 分别训练与汇总，因此形成同方法、同数据划分下的直接位置消融。
- POPE 使用官方问题中的精确物体 surface/span；CLEVR 使用末端 `exist` program 消费的 entity head 并显式统计覆盖率。AMBER 的 existence/attribute/relation 问题没有统一的唯一物体 span，本轮不引入可能污染对比的启发式定位：这些行继续保留 `prompt_last_token`，对象位置状态为 `unavailable`。
- QA feature schema 升级为 `qa-position-comparison-v7`，防止旧的 prompt-last-only shard 被静默续跑。新增回归测试核验活动 YAML 会为每个方法生成两个位置的 feature set，并核验 object 分支实际传入 `cat` 的原始字符 span、保存 contextual target token ID 和 `j-1` prediction position；显式只选 prompt-last 的旧路径仍保证不会额外执行 object forward。
- 本轮只修改配置、说明与测试，不启动 VQA 特征提取或训练。
- 验证：QA/config/object-position/report 相关测试为 `51 passed, 6 subtests passed`；sweep/state-update/raw-attention/wrapper 相关测试为 `12 passed, 4 subtests passed`；`py_compile` 与 `git diff --check` 通过。提交后尝试推送 `hope-best` 失败，原因是 fj01 的 HTTPS GitHub remote 没有可用用户名/凭据；本地分支仍比 `origin/hope-best` 超前 4 个提交。

## 2026-07-25 unified/fj01 配置语义同步

- 以 `model_configs_server_fj01.yaml` 的当前实验选择为基准同步 `model_configs_unified.yaml`：QA extraction mode=`method_only`、两个 QA label protocol、VV+VPend support、source tau=`[0.01,0.03,0.06]`、transport Top-K=`[32,64,128]`、活动 method/ADS/CGC feature sets 及全部训练参数现已一致。unified 继续保留 apulis 环境路径，fj01 继续保留 `/root/rivermind-*` 路径，不跨机器覆盖模型、数据和输出绝对路径。
- 新增 `tests/test_config_mirror.py`，显式移除 18 个环境路径字段后比较两份 YAML 的完整解析对象；今后任何非路径配置漂移都会失败。同步更新 state-update、raw-attention 和 QA active-config 旧断言。
- 当时 sweep 提取与 sweep 训练仍刻意分离：`extract_features.py` 只写含 `dgst_t_hparam_sweep` 的 `features.pkl`；该行为现已被文档顶部 2026-07-26 的 YAML 自动训练开关取代，开关关闭时仍保持分离。
- 验证：配置镜像、state-update、QA config、raw-attention、train/eval stage 和 sweep wrapper plumbing 共 `24 passed`，`py_compile` 与 `git diff --check` 通过。首次把完整 `test_pipeline_config.py` 一并加入时为 `11 failed, 45 passed`：其中 1 项是本次已修正的旧 cost-mode 断言；其余 10 项均为仓库已有 pipeline manifest/resume 契约失败（未写 manifest 或未抛出旧预期异常），与本次 YAML 同步无调用关系。

## 2026-07-25 InternVL/LLaVA 改为逐目标 token 的 causal-prefix forward

- 将 InternVL 与经典 LLaVA 的公开 `extract_token_features_batch()` 对齐 Qwen 协议：接口仍一次接收同一回答中的多个目标位置，但内部按目标逐个截取 `response_ids[:response_index]`，每个目标分别调用一次 `extract_token_features()`；不再用一次完整回答 forward 同时计算多个目标的 DGST。
- 单目标路径会使用该目标之前的真实 causal prefix，目标 token 本身只作为待预测 ID，不放入输入。启用 MetaToken/HalLoc 所需的 `response_hidden_states` 时，另做一次不安装 DGST hooks 的轻量整句共享 forward，行为与 Qwen/Qwen3/OneVision 一致；该共享 forward 只提供整句 hidden states 和 compact logit statistics，不参与目标 DGST。
- 新增 `tests/test_sequential_token_forward_wrappers.py`，同时覆盖 InternVL/LLaVA 的三个非连续目标位置、精确 prefix、目标 ID/response index 映射，以及整句 baseline capture 只执行一次并共享给全部目标。
- 验证：两个 wrapper 与新测试 `py_compile` 通过；`test_sequential_token_forward_wrappers + test_dgst_sweep_wrapper_plumbing + test_causal_token_positions + test_four_gate_dgst + test_prompt_cafe + test_visual_support_ranges` 为 `35 passed, 10 subtests passed`；未启动训练或正式特征提取。
## 2026-07-25 Qwen3 旧 all-mentions 特征复用当前共享 MLP

- 复用 `outputs/qwen3_vl_8b/COCO4000-512/baseline/features.pkl` 的 32,628 条历史 token 特征，不重新生成 caption、label 或特征。该缓存属于旧 `all mentions + prefix token count locator` 协议，不满足当前 `schema_version=2 / first canonical + exact offsets` 的受控训练入口；因此没有关闭正式入口的 schema 防护，而是把本次兼容实验明确隔离到 `baseline/legacy_all_mentions_current_mlp/`，结果不能冒充新协议受控实验。
- image split 复用当前 `COCO4000-512-ENDAC/image_splits.json` 的严格 3200/0/800；其 train 与历史 train 完全相同，test 为历史 val+test 并集。旧缓存中实际有有效 token 特征的图片为 3158/0/790，对应 token 样本 26098/0/6530，train/test 标签计数分别为 real/hall=`23476/2622` 和 `5863/667`。
- 使用当前 YAML 共享 MLP 完成 MetaToken、SVAR、ProjectAway × seeds `43/44/45`：输入维度 `42/448/37`，网络 `128→64→32`、BatchNorm-ReLU-Dropout 0.3、`drop_last=true`、Adam `lr=1e-3`、weight decay `1e-5`、batch 256、最多 100 epochs、train-loss scheduler/early stopping、minimum-train-loss checkpoint；固定 0.5 与 train Real-F1 阈值均有报告。
- train Real-F1 阈值下的 Test mean±population-std：MetaToken Accuracy/Real-F1/Hall-F1/AUROC=`0.9002±0.0002/0.9469±0.0001/0.1705±0.0073/0.8627±0.0011`；SVAR=`0.9325±0.0011/0.9626±0.0005/0.6537±0.0144/0.9265±0.0022`；ProjectAway=`0.8980±0.0000/0.9463±0.0000/0.0040±0.0014/0.6010±0.0060`。固定 0.5 下 Hall-F1 分别为 `0.0933±0.0570/0.6585±0.0077/0.0000±0.0000`；类别不平衡下 Accuracy/Real-F1 会掩盖 MetaToken/ProjectAway 几乎不召回幻觉样本的问题，当前最强仍是 SVAR。
- 三 seed 汇总为 `baseline/legacy_all_mentions_current_mlp/results/shared_torch_mlp/qwen3_vl_8b_legacy_all_mentions_baselines_shared_torch_mlp_3seed_summary.md`；逐 seed JSON、9 个 checkpoint/config/history 均已保存。核验 3 个结果 seed/方法/协议/样本数一致，9/9 checkpoint 配置与当前 YAML 相符，聚合结果全部为有限值；本轮未修改用户已有的 dirty `configs/model_configs_unified.yaml`。
- 为与上述 baseline 做真正同模型、同样本、同 split、同 head 的比较，又复用根 `features.pkl` 将历史报告的全部 23 组自有特征按相同严格 8:2、seeds `43/44/45` 和当前共享 MLP 重训，共 69 个 checkpoint。三 seed 均完整覆盖 23/23 feature sets；汇总位于 `results/legacy_all_mentions_current_mlp/qwen3_vl_8b_legacy_own_features_current_mlp_3seed_summary.{md,csv,json}`，直接比较报告为 `qwen3_vl_8b_legacy_own_vs_baselines_current_mlp_3seed_comparison.md`。
- 当前 parser 对 `hmid_*` 要求真正的 `*_hmid_per_layer`，但该历史缓存的 `hmid_*` risk/cosine/EV 实际以 `*_hpre_per_layer` 保存；第一次三路运行因此在完成前 8 组后同时明确报缺字段。兼容续跑只在内存中把 6 个历史 hpre key 映射到 parser 所需 key，数组数值不变，并把完整映射写入各 hmid 结果 provenance；没有修改正式 alias 或放宽当前 schema 保护。
- 23 组自有特征中 CGC 全面最强。train Real-F1 阈值下，CGC Accuracy/Real-F1/Hall-F1/AUROC/Hall-AUPR=`0.9294±0.0013/0.9611±0.0007/0.6123±0.0191/0.9276±0.0008/0.6817±0.0057`；旧 hmid raw risk+cosine+EV 排第二，为 `0.9233±0.0010/0.9575±0.0006/0.6033±0.0027/0.9258±0.0019/0.6586±0.0031`。同协议 SVAR 的 Hall-F1/Hall-AUPR 高 CGC `0.0415/0.0251`，Accuracy 高 `0.0032`，CGC 的 AUROC 仅高 `0.0011`；因此旧缓存上 SVAR 的少数类识别更强，两者排序能力近似。

## 2026-07-24 VP softmax-Gaussian R / AE 的 STD、SEM、95% CI 曲线

- 使用 `COCO4000-512-ENDAC-SOFT/features.pkl` 全部 12,686 个有效 token（Real 9,540，Hall 3,146；拒绝 0）绘制完整 32 层 label 曲线。R 精确字段为 `dgst_t_vp_hpre_softmax_prob_gauss_risk_sqrt_hpre_per_layer`；AE 精确字段为 `dgst_t_vp_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine_topk32_hpre_per_layer`。STD 使用样本标准差 `ddof=1`，SEM=`STD/sqrt(n)`，95% CI=`mean±1.96*SEM`。
- 分别生成 R/AE × STD/SEM/CI95 的 6 张 PNG/PDF和一张 2×3 汇总图。95% difference CI 显示：R 在 29/32 层 Hall>Real、L2 Hall<Real、L4/L23 跨零；AE 在 27/32 层 Hall<Real、L4–L7 Hall>Real、L8 跨零。重点层 18/21/24/27/30 均为 R Hall>Real、AE Hall<Real。
- 结果位于 `analysis/vp_softmax_prob_gauss_risk_ev_uncertainty_by_label/`，另含 128 行逐层 label 统计、64 行 Hall−Real 差异及 JSON。服务器端验证全部 n、均值、STD、SEM、CI 关系和有限值；7 张 PNG 的尺寸与像素方差均通过，未下载分析图或数据到本地。

## 2026-07-24 500-token `Describe this image.` 质量分布

- 全部计算与绘图在 fj01 服务器完成。按总计 500 token 抽样：Real/Hall 各 250，seed=42；每类每图最多一个 token，且两类 image ID 完全不重叠。分析 VP 第 18/21/24/27/30 层的 `Des / cribe / ▁this / ▁image / .`，信号为 raw、`normalize(raw * hpre_softmax_prob_gauss_gate)` target 和 source。
- 五层平均 instruction 总质量：raw Real=`0.00236498`、Hall=`0.00163387`（Hall/Real=`0.6909`，rank AUC=`0.2279`，BH p=`1.91e-25`）；target Real=`0.00376442`、Hall=`0.00297925`（ratio=`0.7914`，AUC=`0.3750`，BH p=`1.33e-6`）；source Real=`0.00758666`、Hall=`0.00817812`（ratio=`1.0780`，AUC=`0.6983`，BH p=`2.57e-14`）。raw/target 的 Hall 质量显著更低，而 source 的 Hall 质量总体更高。
- 分层方向：raw 与 target 在五层全部为 Hall<Real；source 在 18/24/27/30 层 Hall>Real，但第 21 层反向（Hall/Real=`0.8799`）。条件 token 构成的类别差异很小：raw/target 均主要落在 `▁image`，source 主要落在 `▁this`，说明主要差异来自 instruction 总质量而不是五个词片内部重排。
- 结果位于 `outputs/llava_1_5_7b/COCO4000-512-ENDAC-SOFT/analysis/describe_this_image_mass_500tokens_vp_softmax_prob_gauss_seed42/`，包含 4 张 PNG/PDF、500 行 manifest、7500 行 token-layer-signal、37500 行逐 piece、1500 行 token 聚合、总体/逐层/逐 piece 统计、JSON 和 Markdown。服务器验证 500 个 image ID 全部唯一、组合行数完整、数值非负有限、每个条件分布和为 1；4 张 PNG 尺寸和像素方差检查通过。

## 2026-07-24 image 831 在 `Describe this image.` 上的逐 token 质量

- 从既有 prompt-token CSV 中严格截取 VP support `583:588` 的五个 instruction token：`Des / cribe / ▁this / ▁image / .`，比较 Real `skateboard` 与 Hall `handbag` 在第 18/21/24/27/30 层的 raw、softmax-Gaussian target、source。输出绝对质量热力图、instruction 内条件占比热力图和 instruction 总质量逐层曲线。
- 五层平均 instruction 总质量：skateboard raw=`0.001315`、target=`0.001713`、source=`0.007123`；handbag raw=`0.001286`、target=`0.002109`、source=`0.009023`。Hall 的 target/source instruction 质量分别比 Real 高约 `23.2%/26.7%`，说明这组样本里“对指令关注更多”也不能直接作为真实性解释。
- 条件分布上，target 的 skateboard 主要落在句末 `.`（36.45%），handbag 主要落在 `▁image`（42.75%）；source 对五个 instruction token 更均匀。7 个 PNG/PDF/CSV/JSON 产物追加保存到 `analysis/paper_heatmap_pairs_top32_softmax_prob_gauss_vpfix_auto/831/vp/prompt_mass_pair_skateboard_handbag_softmax_prob_gauss/`。验证 150 行完整组合、全部非负有限，且每个条件分布之和为 1；三张图已人工检查。

## 2026-07-24 image 831 的 prompt 质量与 R / AE 逐层曲线

- 对同图 Real `skateboard`（response idx 37）与 Hall `handbag`（idx 123）完整复现 image 252573 的 VP `hpre_softmax_prob_gauss` 分析。prompt 统计使用第 18/21/24/27/30 层、视觉列 `5:581`、user-instruction `583:588`；当前 wrapper 无 SYSTEM。五层平均（visual/full prompt/user）：skateboard raw=`0.139808/0.860192/0.001315`、target=`0.324627/0.675373/0.001713`、source=`0.973531/0.026469/0.007123`；handbag raw=`0.213101/0.786899/0.001286`、target=`0.396467/0.603533/0.002109`、source=`0.969860/0.030140/0.009023`。
- R=`risk_sqrt_hpre`，AE=`target_dist_mass_x_cosine_topk32_hpre`；Real skateboard 用实线、Hall handbag 用虚线绘制全部 32 层。R 均值 skateboard=`0.526933`、handbag=`0.540952`；AE 均值 skateboard=`0.189805`、handbag=`0.169795`。
- prompt 的 PNG/PDF、30 行 scope CSV、510 行 prompt-token CSV、JSON/Markdown 位于 `analysis/paper_heatmap_pairs_top32_softmax_prob_gauss_vpfix_auto/831/vp/prompt_mass_pair_skateboard_handbag_softmax_prob_gauss/`；R/AE 的 PNG/PDF、32 行 CSV 和 JSON 位于相邻 `risk_ae_skateboard_handbag_softmax_prob_gauss/`。所有质量守恒、连续层、有限值和线型断言均通过，主图已人工检查。

## 2026-07-24 image 252573 的 R / AE 逐层曲线

- 为同图 Real `cat`（response idx 6，实线）和 Hall `mouse`（idx 26，虚线）分别绘制 VP `hpre_softmax_prob_gauss` 的 32 层曲线。R=`dgst_t_vp_hpre_softmax_prob_gauss_risk_sqrt_hpre_per_layer`；AE=`dgst_t_vp_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine_topk32_hpre_per_layer`。
- R 的 32 层均值：cat=`0.484504`、mouse=`0.513284`，两条曲线频繁交叉；AE 均值：cat=`0.240716`、mouse=`0.193143`，cat 在大部分中后层更高。单样本曲线仅用于说明这组 pair 的层间行为，不代表总体类别统计。
- 两张 PNG/PDF、32 行原始 CSV 和 JSON 位于 `analysis/paper_heatmap_pairs_top32_softmax_prob_gauss_vpfix_auto/252573/vp/risk_ae_cat_mouse_softmax_prob_gauss/`。验证 32 层连续、全部值有限、线型要求正确，且两张图已人工检查。

## 2026-07-24 image 252573 Real/Hall pair 的 VP prompt 质量分布

- 对同图 Real `cat`（response idx 6）和 Hall `mouse`（idx 26）按同一口径分析第 18/21/24/27/30 层：raw=`dgst_t_vp_attention_support_per_layer`，target=`normalize(raw * dgst_t_vp_hpre_softmax_prob_gauss_gate_per_layer)`，source=`dgst_t_vp_source_dist_per_layer`。精确 VP support 为 593，视觉列 `5:581`，其余 17 个位置为完整模板 prompt；当前 wrapper 无 SYSTEM，user-instruction-only 为 `583:588` 的 `Describe this image.`。
- 五层平均质量（visual/full prompt/user instruction）：Real cat 的 raw=`0.157680/0.842320/0.002463`、target=`0.337912/0.662088/0.004543`、source=`0.975497/0.024503/0.007066`；Hall mouse 的 raw=`0.118314/0.881686/0.002056`、target=`0.393972/0.606028/0.003324`、source=`0.976348/0.023652/0.006195`。
- 结论：gate 对二者都把质量从 prompt 移向视觉；Hall mouse 的 target 平均视觉质量反而比 Real cat 更高（39.40% vs 33.79%），说明总视觉质量不能单独作为真假判据，pair 的区分仍来自空间 target/source 对齐。raw/target 的 prompt 质量主要由 BOS `<s>` 占据；source 的 prompt 绝对质量仅约 2.4%，但其中用户指令约占 cat 28.63%、mouse 26.02%。
- PNG/PDF、逐层 CSV、逐 prompt-token CSV、JSON 和 Markdown 位于 `analysis/paper_heatmap_pairs_top32_softmax_prob_gauss_vpfix_auto/252573/vp/prompt_mass_pair_cat_mouse_softmax_prob_gauss/`。30 条 scope 记录均验证总质量为 1、视觉+prompt=1、user+template=prompt；两 token 和 VP 边界断言通过，并人工检查图像无标题/图例重叠。

## 2026-07-24 ENDAC-SOFT 同图 Real/Hall pair 扩展筛选

- 从 `COCO4000-512-ENDAC-SOFT` 的 12,686 条特征中重算 11,383 个可定位 token；3,954 张候选图里有 1,917 张同时包含 Real 与 Hall token，原 curated 目录只有一组 pair 并非候选不足。
- 用 `hpre_softmax_prob_gauss` target、Top-32、第 18/21/24/27/30 层生成 10 组 VV/VP pair 面板；VP 使用已核准的精确视觉列 `5:581`，而旧 curated manifest 记录的是 `4:580`。结果位于 `analysis/paper_heatmap_pairs_top32_softmax_prob_gauss_vpfix_auto/`，包含 20 张 PNG、20 张 PDF、两张 contact sheet 和完整候选指标。
- 人工查看后，新增候选中 `image 252573: Real cat / Hall mouse` 最直观；`image 47263: Real truck / Hall frisbee` 与 `image 540288: Real sandwich / Hall cup` 也较好。原 `image 122602: Real scissors / Hall mouse` 已在新目录以正确 VP 映射重画为 pair04。
- 验证：后台脚本正常退出；候选评分保留 11,383 条；paired 目录严格有 20 PNG + 20 PDF；VV/VP contact sheet 均生成。未改动模型特征或训练代码。

## 2026-07-24 image 164475 幻觉 `cell phone` 的 VP prompt 质量分布

- 直接读取 `COCO4000-512-ENDAC-SOFT/features.pkl` 的完整 12,686 条记录，定位 image `164475`、response token index `17`、Hall 标签的 `cell phone`。分析第 18/21/24/27/30 层，raw attention 使用 `dgst_t_vp_attention_support_per_layer`，target 严格按 `normalize(raw_attention * dgst_t_vp_hpre_softmax_prob_gauss_gate_per_layer)` 重构，source 使用 `dgst_t_vp_source_dist_per_layer`。
- 用本地 LLaVA processor 和实际 COCO 图重构出精确 VP support：593 个位置 = 576 个视觉 token（support index `5:581`）+ 17 个文本 prompt token。当前 wrapper 模板为 `USER: <image>\nDescribe this image.\nASSISTANT:`，没有字面 `SYSTEM:` 消息，因此“只删除 SYSTEM”与 full-template 完全相同；另提供有意义的 user-instruction-only 视角，只保留最短上下文 span `583:588`，即解码为 `Describe this image.` 的 5 个 token。
- 五层平均绝对质量：raw attention 的 visual/full-prompt/user-instruction 为 `0.201582/0.798418/0.001751`；softmax-Gaussian target 为 `0.524015/0.475985/0.003167`；source 为 `0.969574/0.030426/0.010272`。raw/target 的 prompt 质量分别平均有 `98.30%/92.97%` 落在 `<s>`，而 source 的 `<s>` 占 prompt 质量仅 `4.81%`；source 虽然 prompt 总质量低，但用户指令占其 prompt 质量平均 `31.80%`。
- PNG/PDF、逐层 scope CSV、逐 token CSV、JSON 与中文说明位于 `analysis/.../164475/vp/prompt_mass_cell_phone_softmax_prob_gauss/`。五张 PNG 已人工检查；所有三种分布均逐层归一到 1，形状均为 `[32,593]`，输出层号、token边界和数值均通过断言。
- 首次一次性分析在用户指令span定位处失败：按扫描顺序先命中了包含前导空格/换行的 7-token span `581:588`；修正为与 wrapper 一致的“最短匹配、再按起点”规则后严格得到 5-token span `583:588`，正式产物均来自修正后的重跑。

## 2026-07-23 CLEVR 9K 严格 8:2 校验单位修复

- 修复 `train_qa_probes.py` 对 CLEVR 9K 的二次划分校验：准备阶段严格抽取 7200/1800 个问题，且官方 train/val 图片命名空间完全隔离；同一图片存在多个问题，因此 6896/1733 个唯一图片不应被错误要求再次满足精确 8:2。现在 CLEVR 按问题行检查精确 8:2，同时仍检查任一图片不得跨 probe split。
- POPE 与 AMBER 的协议没有放宽，继续按物理图片数量检查精确 8:2；新增反向回归用例，确保即使问题行恰好 8:2，只要物理图片为 3:2 仍会拒绝。

## 2026-07-23 QA VP-only 特征提取修复

- 修复 `support_modes: [vp]` 时 QA `_compact_dgst()` 仍强制读取 VV 无前缀共享矩阵的问题。根特征 schema 升级为 `qa-prompt-last-token-v6`：VV 继续使用无前缀分支，VP 使用 `vp_*` 分支；VP-only 的主 `matrices` 指向 VP，同时 `matrices_by_scope` 明确保存全部启用作用域，双模式不再丢弃 VP。
- QA compact 现在同时保存当前 `cost_modes` 生成的 matched/state-update risk 曲线及 Prompt CAFE 标量/逐层曲线；probe 解析器支持 `vp_` 方法、`risk_sqrt_stateupd_alpha01..09`、matched/cosine/geo risk 和直接 compact 字段。此前 VP-only 的 10 条样本会全部报缺少 `dgst_t_attention_support_per_layer`，生成 5-byte 空 `features.pkl`，随后 resume 才表现为 `feature_parts is missing`。
- 真实 LLaVA-1.5-7B 双卡 CLEVR 10 问题严格 8/2 冒烟测试通过：generation/label/root feature/baseline feature 均为 10 条，generation/extraction failure 均为 0，baseline MetaToken/SVAR/DHCP/ProjectAway 和 2 个分片完整，manifest=`complete`。当前 VP-only YAML 解析出的 22 个 method feature 对全部 10 条样本均能构建有限矩阵；定向 QA 回归 20/20 通过。

## 2026-07-23 CLEVR-Exist 扩展为可配置 9K 严格 8:2

- CLEVR 数据规模和数据集 slug 不再埋在准备代码中：两份活动 YAML 新增 `clevr_dataset_name`、`clevr_train_questions`、`clevr_val_questions`、`clevr_test_questions`，当前默认分别为 `clevr_exist_9k`、`7200/0/1800`。`scripts/prepare_qa_benchmarks.py` 同时提供同名 CLI 覆盖参数，并把实际 dataset 名传入每条 question 的 `dataset`、`key`、prepared 目录和 manifest 描述，避免 9K 内容继续伪装成 5K。
- `run_clevr.sh`、统一 `run_qa.sh`、QA generation/extraction/probe/baseline/comparison CLI、全量 coordinator 和 smoke 入口均已接入 `clevr_exist_9k`；coordinator 完整性检查从 5000 改为 9000。为读取已有历史产物，相关 CLI 仍接受 `clevr_exist_5k`，但新默认入口和两份活动配置只生成独立的 `clevr_exist_9k` 目录。
- seed 42 的真实 CLEVR_v1.0 准备验证生成 9000 条问题，严格为 train/val/test=`7200/0/1800`，question metadata 全部为 `clevr_exist_9k`；对应唯一图片数为 `6896/0/1733`，官方 train/val 物理命名空间保证 train/test 无图片泄漏。Python 编译、四个 shell 入口 `bash -n`、五个 QA CLI 的 9K 参数解析、两份 YAML 解析、CLEVR dataset-name 定向测试、QA baseline/parallel/comparison 19/19 和 `git diff --check` 通过。`tests.test_qa_answer_only_config` 的其余 6 项中 5 项通过；旧的 active-YAML 断言仍要求已关闭的 `hmid_softmax_prob` 组合而失败，与本次 CLEVR 改动无关，本轮没有改回用户当前特征配置。

## 2026-07-22 InsLen Prompt CAFE 特征

- compact four-gate 抽取新增 Prompt CAFE：对每层的 post-visual instruction hpre 状态 (z_j) 使用 LM Head/反嵌入矩阵，按 `softmax(W_u z_j / temperature)` 计算固定生成物体 `target_token_id` 的概率，再严格只在 instruction 位置维取最大值；不是在词表维取最大值。当前 COCO object span 沿用既有因果目标协议，使用该物体 mention 的首个生成 token ID。
- 两份活动 YAML 均启用 `compute_prompt_cafe: true`、`prompt_cafe_temperature: 10.0`、`prompt_cafe_layer: 22`。这与 InsLen 官方代码的 `scale: 0.1` 和 `(scale * logits).softmax()` 数值等价，即论文公式中的温度 `tau=10`；第 22 层也对应其 LLaVA-1.5-7B 配置。为便于本项目后续画逐层曲线，抽取同时保存 `dgst_t_prompt_cafe_per_layer`；论文式指定层标量保存为 `dgst_t_prompt_cafe`，训练 alias `prompt_cafe` 使用该单值，另提供 `prompt_cafe_per_layer` 全层 alias。
- CAFE 与 VV/VP transport scope 正交，只计算一次并共享给同一物体的分支；所有 caption wrapper 和 prompt-target wrapper 均已透传独立 CAFE 开关、温度和层号。serializer 保存公式、温度、实际/请求层、instruction token 数量和位置范围 provenance。两份 YAML 的 method feature set 从 82 增为 83，新增项为独立 `prompt_cafe`，未自动与 risk/EV 拼接。
- 文献核对：原始 GLSim 的 Visual Logit Lens 公式直接使用 `softmax(VLL_l(v_i))[o]`，没有在 softmax 内加入 temperature；其 `tau` 是最后 real/hall 判定阈值。InsLen 官方评测代码会用共同的 `scale=0.1` 计算其 GLSim/ILS 概率，这是 InsLen 复现代码的实现选择，不能反写成 GLSim 原论文的 softmax 温度。
- 验证：新增 CAFE 温度 softmax 手算、instruction 位置最大值、pre-visual 排除、指定层标量/逐层曲线、serializer 和两个训练 alias 测试；`prompt_cafe + stateupd_alpha_sweep` 6/6，通过 wrapper/因果位置/QA 抽取 21/21，通过 pipeline 中直接相关配置路由 10/10。相关 Python 全部 `py_compile`、两份 YAML 83 项解析和 `git diff --check` 通过。本轮没有启动 COCO4000 正式重提取；旧 `features.pkl` 不包含 CAFE 字段。

## 2026-07-22 YAML 启用 hpre softmax-prob Gaussian risk

- 两份活动 YAML 的 `feature_extraction.dgst_t.four_gate_methods` 与 `branches` 已同时启用 `hpre_softmax_prob_gauss`，确保抽取阶段实际执行 hpre hidden state 的全词表 softmax、读取目标 token 概率、构造 Gaussian gate，并保存默认 `sqrt-hpre` transport risk。不能只在训练列表写 alias，否则旧特征中没有底层字段。
- 当前 `target_modes: [raw_logit_gauss]` 保持不变：该便利开关只负责 hpre raw-logit 的 Gaussian/Relative-VLL 二选一，与 softmax-prob 分支正交。当前 `support_modes: [vv, vp]` 会让同一次 forward 同时保存 VV 的 `dgst_t_hpre_softmax_prob_gauss_risk_sqrt_hpre_per_layer`/对应 mass×cosine，以及 VP 对应字段。训练列表对 VV 和 VP 各保留 matched risk、mass×cosine 单独及 matched risk+mass×cosine 三项，并新增 `sqrt_stateupd_alpha01` 至 `alpha09` 的 risk 与 risk+同作用域 mass×cosine；没有 VV risk+VP mass 或反向交叉组合。
- 全局 `cost_modes` 已包含 matched-state 与九个 alpha，因此无需再改特征计算代码；本轮把已经能够抽取和解析的 softmax-prob alpha 字段正式加入 YAML 训练对比。softmax-prob 每个作用域为 3 个基础项 + 9×2 个 alpha 项，共 21 项；VV/VP 合计 42 项。原 raw-logit 40 项保持原顺序，活动 method feature set 总数由 46 增至 82。回归测试同步验证两份 YAML 的 resolved method、82 项命令展开、合成抽取的 VV/VP 九组 softmax alpha 字段，以及全部训练 alias 均可构建。初次只加入 risk 时的定向测试曾两次为 32/33：第一次是测试误以为 resolved 顺序与 YAML 相同，而 `target_modes` 设计上会在解析末尾追加 raw-logit；第二次是服务器 YAML 原有 `extraction_mode=all` 会在 method 项后合法追加 `ads/cgc/ads+cgc`。两条断言均按真实协议修正，且没有覆盖服务器的 `all`。最终 `stateupd_alpha_sweep + extraction_modes + four_gate_dgst` 33/33、pipeline 中直接相关的 10/10 均通过；两份 YAML 均验证为 82 个方法组合、其中 softmax-prob 42 个，VV/VP 交叉拼接为 0，`git diff --check` 通过。完整 `tests.test_pipeline_config` 此前仍有 6 failure + 4 error，均来自本轮未修改的 manifest/resume 契约：4 项找不到测试期望的 `pipeline_manifest.json`，其余 6 项未触发旧断言期待的 resume 拒绝；与新增 softmax 分支无关。本轮只改配置与测试，不启动 COCO4000 正式重提取，已有 ENDAC/VVVP `features.pkl` 不会凭配置修改自动出现新字段。

## 2026-07-22 COCO baseline 原方法与 YAML 三层 MLP 对比

- `training.baseline` 从单个 `trainer` 扩展为有序 `trainers`，两份活动 YAML 默认同时运行 `native_paper` 与 `shared_torch_mlp`；CLI 保留 `--trainer` 单头兼容入口，并新增 `--trainers` 临时覆盖。默认比较范围严格为两种 head 都支持的 MetaToken、SVAR、ProjectAway，不把 DHCP/HalLoc 强行改造成不对应其原方法的 dense MLP。
- 原生 MetaToken-LR/GB、SVAR 单层 MLP和 training-free ProjectAway 现在与方法 probe 使用相同的报告 schema：同一权重同时评估固定 `0.5` 与 train-F1 搜索阈值，每套均保存 Accuracy，以及 Real/Hall 两种正类各自的 Precision、Recall、F1、AUROC、AUPR。旧 headline 扁平字段继续保留，避免已有汇总读取器失效。
- 双头运行严格共用同一份 baseline 特征、image split、token cohort、seeds 和指标实现；原生结果隔离到 `baseline/results/native_paper/` 与 `baseline/checkpoints/native_paper/`，共享三层 MLP 保持原隔离目录。新的 JSON/Markdown 对比表写入 `baseline/results/comparison/`，逐行标明原方法 head 或 YAML 三层 MLP，不覆盖已有实验。
- YAML 三层 MLP 继续直接读取 `training.torch_probe`：`128→64→32`、Linear-BatchNorm-ReLU-Dropout、dropout `0.3`、`drop_last=true`、Adam `lr=1e-3`、weight decay `1e-5`、batch `256`、最多 `100` epochs、minimum-train-loss checkpoint；没有在脚本中复制第二套隐藏超参。
- 兼容性：只有 COCO 双头运行显式启用 native 目录 namespace；QA baseline 和 `--trainer native_paper` 单头旧调用仍沿用原路径。验证已覆盖 trainer 列表解析、双阈值双正类指标、三 seed 聚合、Markdown 和 native-vs-MLP JSON/Markdown 生成；`tests.test_train_baseline_reporting` 8/8、`tests.test_svar_protocols` 11/11、两份生产 YAML 加载、CLI help、`py_compile` 与 `git diff --check` 通过。`td` 环境第一次尝试 `pytest` 在收集前报 `No module named pytest`，随后改用仓库兼容的 `unittest` 全部通过，不是实现失败。第一次正式命令又在训练前暴露 `trainer_namespace` 误加到 shared 签名，报 `unexpected keyword argument`；修正到 native 签名并增加双签名回归断言后，正式重跑完成。
- 已在当前 `run.sh` 活动实验 `outputs/llava_1_5_7b/COCO4000-512-ENDAC` 上完成 seeds `43/44/45` 正式双头训练；复用严格 3200/800 image split 和 10109/2577 token 样本，没有重提特征。train-F1 阈值下，原生 SVAR 的 Accuracy/Real-F1/Hall-F1/AUROC 为 `0.8439/0.8964/0.6830/0.9063`，对应 YAML 三层 MLP 为 `0.8293/0.8862/0.6578/0.8851`；MetaToken 原生 LR 为 `0.8386/0.8947/0.6539/0.9002`，三层 MLP 为 `0.8252/0.8893/0.5852/0.8827`；ProjectAway 原规则为 `0.7621/0.8583/0.2606/0.7652`，三层 MLP 为 `0.7691/0.8616/0.3040/0.8121`。完整两阈值、双正类报告位于 `baseline/results/comparison/llava_1_5_7b_baselines_native_vs_shared_mlp_3seed_summary.md`；14 个 trainer×method×threshold 行的所有指标均为有限值，GPU 训练结束后已释放。

## 2026-07-21 hpre raw-logit Gaussian 的 VV/VP state-update alpha sweep

- 两份活动 YAML 均收敛到 `method_only`、唯一 target 方法 `hpre_raw_logit_gauss`、`target_modes=[raw_logit_gauss]` 与 `support_modes=[vv,vp]`；Relative-VLL、hmid、softmax-prob、raw-attention 和 FAD 均未启用。主 `cost_mode` 固定为 `sqrt_matched_state`，无后缀主 risk 继续解析到 matched-state 字段。
- state-update cost 已从仅支持 `alpha05` 扩展为 `sqrt_stateupd_alpha01` 到 `alpha09`。每个模式严格计算 `C_alpha=(1-alpha)*sqrt((1-cos(hpre_i,hpre_j))/2)+alpha*sqrt((1-cos(o_ffn_i,o_ffn_j))/2)`；九个 alpha 共用同一次模型 forward/capture，但分别求精确 EMD。`alpha05` 字段和主模式为 alpha05 时的旧标量 provenance 保持兼容；多 alpha 结果新增 `dgst_t_cost_alphas` 映射，避免用单个 0.5 描述整组 sweep。
- serializer 与训练 alias 已覆盖 VV/VP 的九个独立 alpha 字段。YAML 训练列表严格为 40 组：2 个 support ×（1 个 matched 基准 + 9 个 alpha）×（risk 单独、risk + 同范围 `mass×cosine`）；不存在 VV risk + VP EV 或相反方向的交叉组合。训练超参继续使用当前三层 MLP 与 `drop_last: true`。
- 新增 `tests/test_stateupd_alpha_sweep.py`：验证 0.1/0.5/0.9 cost 矩阵的手算凸组合、九个名称/权重、VV/VP 全字段与 provenance 序列化、全部 40 个训练 alias，以及两份 YAML/三 seed 命令展开。第一次 4 项测试为 3 pass + 1 failure，原因是统一 YAML 仍为 `extraction_mode=all` 而额外带入 ADS/CGC；改为 `method_only` 后 4/4 通过。原 `tests.test_four_gate_dgst + tests.test_extraction_modes` 29/29 通过；相关 `py_compile`、生产 YAML dry-run（3 seeds、alpha01-alpha09、无 hmid/Relative-VLL/FAD）和 `git diff --check` 通过。
- 本轮只完成实现、配置和合成验证，没有启动 COCO4000 正式重提取或 120 次 probe 训练。现有 `COCO4000-512-VVVP/features.pkl` 只有旧 cost 字段，必须使用新配置重新提取后才能训练/画 alpha sweep 结果；resume fingerprint 会阻止把旧特征误当成新产物。

## 2026-07-21 默认训练 DataLoader 启用 `drop_last`

- `configs/model_configs_unified.yaml` 与服务器活动配置 `configs/model_configs_server_fj01.yaml` 的 `training.torch_probe` 均新增 `drop_last: true`；原 `dropout: 0.3` 保持不变。该开关只丢弃训练集最后一个不完整 batch，测试/评估预测仍保留全部样本。
- `TorchProbeConfig`、主训练 CLI 和 `train_and_eval.py` 的 YAML 参数展开已接通布尔 `--drop-last/--no-drop-last`，且非布尔 YAML 值会明确报错；实际训练 DataLoader 不再使用“仅余数为 1 时丢弃”的隐式规则，而是严格服从该开关。共享三层 baseline 与 source-target divergence probe 同步读取同一配置；结果的 `best_params` 记录 `drop_last` provenance。
- 验证：相关 5 个 Python 文件及定向测试 `py_compile` 通过；两个 YAML 均解析为布尔 `true`，主 probe 命令得到 `--drop-last`，共享 baseline 配置得到 `True`；新增布尔开关单测 1/1 通过；5 条训练样本、batch 4 的真实 CPU smoke 验证训练 DataLoader 仅保留 1 个完整 batch；`git diff --check` 通过。完整 `tests.test_train_and_eval_stage` 仍有本轮开始前的 2 failure + 1 error：旧断言要求当前配置已关闭的 `raw_attention` 分支，与 `drop_last` 改动无关。

## 2026-07-21 LLaVA VV/VP `mass × cosine` 幻觉/非幻觉逐层曲线

- 复用 `outputs/llava_1_5_7b/COCO4000-512-VVVP/features.pkl`，严格按当前特征定义绘制 `target-distribution top-k mass × matched hpre target cosine`；覆盖 Gaussian/Relative-VLL × VV/VP 四个分支，没有重新抽取特征或训练 probe。
- 新增可复现脚本 `scripts/plot_vv_vp_mass_cosine_by_label.py`，按原始 token 标签汇总 12,686 行、3,968 张图片、32 层；hallucination 3,146 行、real 9,540 行。曲线展示 class 内 token 均值与 95% CI。
- 四个分支的全层均值 `Hall/Real/Hall-Real` 分别为：Gaussian-VV `0.118587/0.139632/-0.021045`，Gaussian-VP `0.165977/0.195085/-0.029108`，Relative-VLL-VV `0.121211/0.142112/-0.020901`，Relative-VLL-VP `0.165420/0.194447/-0.029027`。四者整体均为 Real 更高；最大绝对差分别位于第 29/31/29/31 层，差值 `-0.051051/-0.060745/-0.050504/-0.060219`。
- PNG/PDF、128 行逐层 CSV、JSON 和 Markdown 摘要位于 `outputs/llava_1_5_7b/COCO4000-512-VVVP/results/relative_vll_gaussian_vv_vp_mass_x_cosine_by_label/`。脚本 `py_compile`、四字段/全有限值检查、CSV 4 分支 × 32 层、5 个非空产物、人工图像检查和 `git diff --check` 均通过。

## 2026-07-21 LLaVA-1.5 COCO4000 baseline 统一三层 MLP 双正类报告

- 在 `outputs/llava_1_5_7b/COCO4000-512-AC/baseline/features.pkl` 上完成 MetaToken、SVAR、ProjectAway 的统一三层 MLP 训练；DHCP 仍不属于 shared dense-vector MLP 支持范围，没有混入本轮。输入维度分别为 42、448（SVAR 第 5–18 层）和 33。
- 正式协议为严格图片级 8:2、无 validation，3200/0/800 张图片，对应 10109/0/2577 个 token 样本；seeds `43/44/45`。网络严格为 `input→128→64→32→1`、BatchNorm-ReLU-Dropout 0.3、Kaiming-uniform ReLU initialization、Adam `lr=1e-3`、weight decay `1e-5`、batch 256、最多 100 epochs、BCEWithLogitsLoss、train-loss scheduler/early stopping、minimum-train-loss checkpoint，同时报告固定 0.5 与 train Real-F1 阈值。
- 旧根 `image_splits.json` 是 8:1:1，第一次启动在任何训练前被 strict-82 validator 拒绝，日志保留为 `baseline/shared_torch_mlp_train_initial_split_failure.log`。随后 `train_baselines.py` 新增可选 `--split-path`，并兼容统一实验已保存的 `strict_train80_test20_no_validation` 报告 schema；复用 `unified_hallucination_positive_probe_train80_test20_no_validation_relu/outer80_split.json`，已验证覆盖原来完全相同的 4000 张图片，没有重新随机划分或覆盖根 split。
- 首次结果虽然读取了活动 YAML，但当时未识别出该 dirty YAML 的三项 probe 超参已从正式值被改为 `dropout=0.2/lr=3e-4/weight_decay=1e-4`；该组不能作为正式结果，已连同 checkpoint 完整归档到 `baseline/{results,checkpoints}/shared_torch_mlp_drop02_lr3e4_wd1e4/`。随后只将活动 YAML 这三项恢复为用户指定的 `0.3/1e-3/1e-5`，逐 seed 保存的 trainer config 也已反向核验一致，再完成正式重跑。
- 正式 train Real-F1 阈值的 Test 三 seed mean±population-std：MetaToken Accuracy/Real-F1/AUROC/Hall-F1=`0.8260±0.0020/0.8898±0.0012/0.8833±0.0001/0.5865±0.0066`；SVAR=`0.8280±0.0032/0.8858±0.0023/0.8858±0.0024/0.6517±0.0096`；ProjectAway=`0.7716±0.0010/0.8637±0.0006/0.8112±0.0007/0.2940±0.0123`。
- 正式固定 0.5 的 Test 三 seed mean±population-std：MetaToken Accuracy/Real-F1/Hall-F1=`0.8290±0.0005/0.8889±0.0014/0.6276±0.0147`；SVAR=`0.8220±0.0041/0.8827±0.0027/0.6283±0.0340`；ProjectAway=`0.7721±0.0021/0.8509±0.0009/0.5166±0.0091`。AUROC/AUPR 与阈值无关；完整报告还列出 Real/Hall 两种正类的 Precision、Recall、F1 和各自 AUPR。
- 新增可复用 `scripts/summarize_baseline_dual_positive.py`；完整 MD/CSV/JSON 位于 `baseline/results/shared_torch_mlp/llava_1_5_7b_baselines_shared_torch_mlp_3seed_dual_positive.*`，逐 seed JSON 与 9 个 checkpoint 位于同目录的 `seed{43,44,45}/` 和 `baseline/checkpoints/shared_torch_mlp/`。
- 验证：`tests.test_train_baseline_reporting` 5/5 通过；两个脚本 `py_compile`、strict split adapter、6 行双正类汇总、3 seeds × 3 methods、所有有限指标、正式/归档超参辨识和 9/9 checkpoint Linear weight shape 检查通过；`git diff --check` 通过，训练后两张 GPU 均为空闲。

## 2026-07-21 LLaVA Relative-VLL/Gaussian 三种 cost 的 FAD×risk 曲线

- 继续复用 `outputs/llava_1_5_7b/COCO4000-512-VVVP/features.pkl`，按训练代码的 `fad` alias 使用 `ffn_fad = dgst_t_ffn_attn_dominance_per_layer`；组合严格为同层 float32 Hadamard product `ffn_fad * risk`，没有改变 risk、标签或样本集合。
- `scripts/plot_relative_vll_gauss_three_cost_risk.py` 新增可选 `--multiply-fad` 与 `--stem`；默认不传开关时仍绘制原始 risk，并保持原来的 float64 统计口径。新图覆盖 Gaussian/Relative-VLL × VV/VP × `sqrt_matched`/`1-cos(hpre)`/`sqrt_stateupd_alpha05` 共 12 组 Hall/Real 曲线和 95% CI。
- PNG/PDF、384 行逐层 CSV、JSON 与摘要位于 `outputs/llava_1_5_7b/COCO4000-512-VVVP/results/relative_vll_gaussian_three_cost_fad_x_risk_by_label/`。统计仍为 12,686 行、3,968 张图片、32 层，hallucination 3,146、real 9,540。
- 乘 FAD 后 12 组的全层平均 `Hall-Real` 全部反转为负。Gaussian VV 三种 cost 分别为 `-0.012518/-0.013374/-0.012714`，Gaussian VP 为 `-0.025394/-0.033495/-0.025993`；Relative-VLL VV 为 `-0.012366/-0.013154/-0.012557`，Relative-VLL VP 为 `-0.024405/-0.031650/-0.025012`。VP 反转普遍约为 VV 的两倍，绝对分离最大的分支是 Gaussian/VP `1-cos(hpre)`，平均差 `-0.033495`。
- 峰值绝对差位置：VV 的两个 sqrt cost 在第 32 层、`1-cos(hpre)` 在第 1 层；VP 三种 cost 均在第 28 层。图已人工检查；脚本 `py_compile`、12 series × 32 layers、有限值、标签计数、transform provenance 和 `git diff --check` 均通过。

## 2026-07-21 LLaVA Relative-VLL/Gaussian 三种 cost 曲线与 probe

- 复用 `outputs/llava_1_5_7b/COCO4000-512-VVVP/features.pkl`，没有重新做模型抽取。完整产物包含 12,686 个 object-token 行、3,968 张图片、32 层；标签为 hallucination 3,146、real 9,540。12 个 risk 字段（2 target × VV/VP × 3 cost）和 4 个对应 EV 字段均存在且全部有限。
- 新增可复现画图脚本 `scripts/plot_relative_vll_gauss_three_cost_risk.py`。曲线 PNG/PDF、384 行逐层 CSV、JSON 和摘要写入 `outputs/llava_1_5_7b/COCO4000-512-VVVP/results/relative_vll_gaussian_three_cost_risk_by_label/`。
- 曲线结论：Gaussian 与 Relative-VLL 的整体形状非常接近；VV 的全层 Hall-Real 均值差约 `+0.0150` 到 `+0.0228`，明显大于 VP 的 `+0.0047` 到 `+0.0080`。`1-cos(hpre)` 在 VV 上的平均分离最大（Gaussian `+0.021944`、Relative-VLL `+0.022827`），峰值都在第 8 层；VP 三种 cost 的最大绝对差都位于第 2 层且方向反转。
- 完成 24 个 feature set × seeds `43/44/45` = 72 次 Torch probe：每个 target/scope/cost 分别训练 risk 与 `risk + 对应 target 的 mass×cosine`。协议为严格 image-level 8:2、无 validation、Real 正类、hidden `[128,64,32]`、BatchNorm-ReLU-Dropout `0.2`、Adam `lr=3e-4`、weight decay `1e-4`、batch `256`、最多 100 epochs、train-loss early stopping patience 10，并同时报告固定 0.5 与训练集 Real-F1 阈值。
- 训练集阈值结果中，最高 AUC 是 Gaussian/VV `sqrt_stateupd_alpha05 risk + EV`：`0.8859±0.0002`；最高 Real-F1 是 Gaussian/VV `sqrt_matched_state risk + EV`：`0.8842±0.0010`。12 个分支加入对应 EV 后 AUC 全部提升 `+0.0240` 到 `+0.0298`，Real-F1 全部提升 `+0.0078` 到 `+0.0136`。固定 0.5 下最佳仍是 Gaussian/VV `sqrt_stateupd_alpha05 risk + EV`，Accuracy/Real-F1/Hall-F1 为 `0.8258±0.0018 / 0.8826±0.0008 / 0.6623±0.0095`。
- 三 seed 汇总位于 `outputs/llava_1_5_7b/COCO4000-512-VVVP/results/llava_1_5_7b_relative_vll_gaussian_three_cost_risk_ev_3seed_summary.{md,csv,json}`；逐 seed 结果位于 `results/relative_vll_gaussian_three_cost_seed{43,44,45}/`。已校验 3 个 JSON 均严格包含相同的 24 个 feature set、Real 正类、两套阈值报告、正确 seed 与 32/64 输入维度；画图脚本 `py_compile`、曲线行数/层数/有限值检查和 `git diff --check` 通过。

## 2026-07-21 hpre `1-cosine` cost 与 Relative-VLL/Gaussian target 开关

- compact four-gate 新增可选 cost `cosine_matched_state`：严格使用 `C_ij=1-cos(h_state_i,h_state_j)`，不除以 2、不开平方；hpre 分支的 matched state 为 `h_prev`。原 `sqrt_matched_state` 与 `sqrt_stateupd_alpha05` 保留，可通过 YAML `cost_modes` 在同一次 capture/source/target 上并行提取。
- 新增 hpre target 分支 `hpre_raw_logit_relative_vll`，与 `hpre_raw_logit_gauss` 共用同一组 hpre raw target logits、attention、source distribution、support 和 EMD。Relative-VLL gate 使用 `sigmoid((logit-median)/(1.0*MAD+eps))`，Gaussian gate 使用 `sigmoid((logit-median)/(1.4826*MAD+eps))`；两者不是同一 target，前者通常更尖锐，后者更平滑。
- 两份活动 YAML 新增独立 `target_modes`，支持 `relative_vll`、`raw_logit_gauss` 或同时选择；该列表只控制上述两个 hpre/raw-logit target，不影响其他历史分支。当前默认两者同时启用，并同时提取 `sqrt_matched_state`、`cosine_matched_state`、`sqrt_stateupd_alpha05`。显式 CLI `--dgst-branches` 仍具有最高优先级，会移除 YAML target-mode 便利开关，避免被静默重新启用。
- serializer、VV/VP 字段、训练 alias 和 pipeline 训练项过滤均已接通。新增四个活动 risk 对比项：VV/VP 下 Gaussian 与 Relative-VLL 各自的 `sqrt_matched_state` 和 `cosine_matched_state`；结果同时保存每个 target method 的 MAD scale provenance。
- 验证：相关 Python `py_compile` 与两份 YAML 解析通过；`tests.test_extraction_modes + tests.test_four_gate_dgst` 共 29/29 通过；CLI override、禁用分支和 target-mode 训练过滤 3/3 通过；两份 YAML 的 relative-only/Gaussian-only 展开均无另一 target 泄漏；服务器配置 `train_and_eval.py --dry-run` 已列出新增 VV/VP risk 对比项；`git diff --check` 通过。本轮没有启动正式模型重抽取或训练，旧 `features.pkl` 不包含新增字段，不能直接训练这些新项。

## 2026-07-21 Qwen2.5-VL VV/VP hpre-risk 与 FFAD 乘积曲线

- 对 `outputs/qwen2_5_vl_7b/COCO4000-512-VVVP/features.pkl` 的完整 7,924 条 object-token 记录按原始标签分组：幻觉 `label=0` 共 972 条，非幻觉/real `label=1` 共 6,952 条；没有使用仅含 2/3 条记录的残留 `features.part0/1.pkl`。hpre-risk 严格读取当前主 cost `sqrt_cosine_matched_state` 的 VV/VP 训练字段，FFAD 组合严格按训练代码做逐层 Hadamard 乘积。
- 新增可复现脚本 `scripts/plot_vv_vp_fad_risk_by_label.py`，输出 28 层、1-based 横轴、class 内 token 均值与 95% CI 的 2×2 PNG/PDF，并同时写逐层 CSV、机器可读 JSON 和 Markdown 摘要。产物位于 `outputs/qwen2_5_vl_7b/COCO4000-512-VVVP/analysis/vv_vp_fad_risk_by_label/`。
- 全层均值 Hall/Real/Hall-Real：VV hpre-risk=`0.276035/0.254535/+0.021500`，VP hpre-risk=`0.410546/0.382826/+0.027721`；两者最大绝对差均在第 21 层，分别为 `+0.049516/+0.055561`。乘 FFAD 后全层差几乎消失：FFAD×VV=`0.084322/0.084546/-0.000225`，FFAD×VP=`0.148562/0.147675/+0.000887`；最大局部差分别在第 28 层 `-0.038404` 和第 27 层 `+0.042182`，且多层发生符号翻转。
- 脚本 Python 编译、完整 artifact 运行、PNG/PDF 类型检查、CSV/JSON/Markdown 生成与输出 SHA256 校验均通过；本轮只做统计绘图，没有启动 GPU 抽取或训练。

## 2026-07-21 SVAR 改为全层抽取、训练仅使用第 5–18 层

- SVAR 生产抽取不再按配置提前裁掉 decoder 层；controlled/official、COCO/QA 统一保存完整 `[layer, head]` visual-attention-ratio、全层扁平向量及绝对层范围 `[0,L)`。抽取 provenance 固定记录 `extraction_layers=all`，`layer_start/layer_end` 改为纯训练参数，修改训练层范围不会再误判为需要重抽其他 baseline。
- 原生单层 SVAR MLP 与统一三层 Torch MLP 的 COCO/QA 路径均在组装训练矩阵时按绝对层号切片。当前两份活动 YAML 保持 `layer_start=5`、`layer_end=19`，即严格训练第 5–18 层；checkpoint、结果 JSON 与 QA 训练指纹均记录该范围。
- 新增统一的 `svar_training_vector()`：新全层 payload 可选择任意已保存层范围；历史只保存 5–18 层但带绝对层元数据的 payload 仍可按 `[5,19)` 训练；更老的无层元数据 vector 保持原样兼容。越界范围、层元数据不一致及非有限值会明确报错。
- 验证：相关 Python 文件 `py_compile` 通过；`tests.test_baselines + tests.test_qa_baselines` 共 22/22 通过；SVAR 全层/旧缓存/QA 切片与两项抽取指纹的定向回归 4/4 通过；`tests.test_train_baseline_reporting + tests.test_svar_protocols` 相关路径通过；`git diff --check` 通过。扩展组合 30 项中 27 项通过，另外 3 项仍是本轮开始前已存在的 generation/label manifest resume 断言不一致，与 SVAR 改动无关；本轮未启动正式模型抽取或训练。

## 2026-07-21 fj01 `run.sh` 合并冲突收尾

- 已读取 `run.sh` 的 base/ours/theirs 三个 stage，并保留当前工作区中已经合并好的服务器入口：默认 `qwen2_5_vl_7b`、输出 `COCO4000-512-VVVP`、服务器统一配置、可覆盖的活动环境 Python、服务器 NLTK 路径和 `CUBLAS_WORKSPACE_CONFIG`；三阶段仍严格为生成标注、特征抽取、训练评估。
- `bash -n run.sh` 与 `git diff --cached --check -- run.sh` 通过，`run.sh` 已暂存并解除 `UU`；`git ls-files -u` 为空。`configs/model_configs_server_fj01.yaml` 当时为 `MM`，本次没有修改或覆盖其工作区内容。
- 附加验证命令 `CUDA_VISIBLE_DEVICES= /opt/conda/envs/td/bin/python -m unittest -v tests.test_pipeline_config` 共运行 32 项，结果为 6 failures、4 errors。失败集中在 pipeline manifest 旧断言：部分用例预期抛出 resume/provenance `ValueError` 但当前实现未抛出，另有 4 项因未生成临时 `pipeline_manifest.json` 报 `FileNotFoundError`；shell 入口、Python 环境、mode/feature-set 路由等相关用例通过。该失败不影响本次 `run.sh` 的语法与冲突解除，但后续需要单独统一 manifest 实现和测试预期。

## 2026-07-20 compact four-gate 新增 VP hpre_raw 与 ffn_fad

- unified 配置新增显式 `support_modes` 开关：`["vv"]` 只抽视觉支持，`["vp"]` 只抽“视觉 token ∪ prompt token”，`["vv", "vp"]` 则在同一次 decoder forward/capture 后计算两套特征。当前启用第三种；VP 独立保存为 `dgst_t_vp_*`，不会覆盖或误读 VV。
- compact 路径重新接入历史 `ffn_fad`：每层预测位置计算 `log((||o_ffn||_2+eps)/(||o_attn||_2+eps))`，保存为 `dgst_t_ffn_attn_dominance_per_layer`。仅恢复该曲线，不重新计算 EIF-dose、logit-lift 等较重的旧 FFN diagnostics；wrapper 仅在启用 FAD 时保留 MHSA 更新张量。
- 主 cost 已确认为 `sqrt_matched_state`，`sqrt_stateupd_alpha05` 仅作为显式后缀的并行对照，两者仍在一次抽取中计算。FAD 与 VV/VP risk 共用同一次 forward/capture，不存在独立重跑。默认 FAD 训练项为：FAD 单独 32 维、`FAD*VV-hpre_raw-matched-risk` 逐层乘积 32 维、该 VV 乘积再与 VV mass×cosine(EV) 拼接的 64 维输入，以及 `FAD*VP-hpre_raw-matched-risk` 逐层乘积 32 维；`*` 是 Hadamard 逐层乘法，`+` 是向量拼接。VV/VP 的 alpha05 risk 与 risk+EV 保留为消融对照。
- 训练 alias、compact serializer、单 token/批量/prompt-target 和全部模型共享 capture 路径均已接通。分支过滤同时识别 `support_modes`、`vp_` 前缀及乘积两侧的分支依赖，关闭 VV/hpre 时不会误留 FAD×VV-risk 训练项。
- 合成回归验证 VV/VP 支持矩阵分别为 3/4 位置、FAD 手算为 `log(1.0/0.5)`，且 `ffn_fad+vp_hpre_raw...risk` 可组成训练矩阵；`tests.test_four_gate_dgst` 17/17 通过，相关 Python 编译通过。系统与 vicr 均无 pytest，改用标准库 unittest；混跑旧 pipeline/stage 测试仍有 manifest 与已关闭 raw-attention/hmid 配置断言失败，属于开始前已有的不一致。
- LLaVA-NeXT 真实 1 图 smoke 已完成 7 个 object token：FAD、VV risk、VP risk/EV 均为 32 层且全部 finite；VV/VP 支持数为 984/1039。误启动的完整抽取已按用户要求中止，GPU 均已释放；隔离目录 `outputs/llava_next_8b/COCO4000-512-AC-all-fad-vp/` 留有约 158 张图的未合并分片，原 `COCO4000-512-AC-all/features.pkl` 未覆盖。用户确认前不再启动真实抽取。
- 新开关及三项 FAD 训练配置相关 Python 编译通过。随后运行完整 four-gate unittest 及单独的 `test_dual_scope_hpre_raw_and_ffn_fad_are_serialized_and_trainable` 时，进程都在导入标准库 `unittest` 阶段进入 NFS `rpc_wait_bit_killable`，数分钟无进展后已发送中断；这是文件系统读取等待，测试体尚未开始，因而本轮暂时没有新的 unittest 结果。此前同一合成用例在加入显式开关前已通过。

## 2026-07-20 COCO baseline 统一三层 MLP

- `train_baselines.py` 新增 `native_paper | shared_torch_mlp` 训练器选择；unified 的 `training.baseline.trainer` 已启用 `shared_torch_mlp`。MetaToken、SVAR、ProjectAway 复用 method probe 的同一 `input→128→64→32→1`、Linear-BatchNorm-ReLU-Dropout(0.3)、Adam、plain BCE、train-loss scheduler/early stopping/minimum-loss checkpoint，以及固定 0.5 + train Real-F1 双阈值协议。
- 输入保持各 baseline 自身定义：MetaToken=`42` 维、SVAR=`448` 维、ProjectAway=`1` 个 global internal confidence + `32` 层曲线=`33` 维；不做额外 z-score 或类别加权。DHCP 不在当前 `training.baseline.methods`，因此没有加入本轮三层 MLP。
- LLaVA-NeXT COCO4000 seeds `43/44/45` 已完成 3 方法 × 3 seed。train-F1 阈值 Test 的 Real-F1/AUROC/Hall-F1：MetaToken-MLP=`0.9082±0.0001/0.8253±0.0012/0.0404±0.0235`，SVAR=`0.9376±0.0020/0.9123±0.0017/0.6724±0.0198`，ProjectAway=`0.9309±0.0010/0.8725±0.0027/0.5558±0.0240`。
- 相对原生 head：MetaToken-MLP 对最佳 MetaToken-GB 的 Real-F1/AUROC/Hall-F1 分别下降 `0.0195/0.0612/0.5451`，几乎全部预测 real；SVAR 分别变化 `+0.0013/-0.0069/+0.0156`；ProjectAway 分别提升 `+0.0081/+0.0699/+0.1478`。统一 MLP 下 SVAR 仍为最好方法。
- 新产物隔离在 `baseline/results/shared_torch_mlp/` 与 `baseline/checkpoints/shared_torch_mlp/`，原生 baseline 结果未覆盖。9/9 checkpoint 均核验为正确的四个 Linear weight shape，三 seed 汇总为 `llava_next_8b_baselines_shared_torch_mlp_3seed_summary.md`。
- 验证：Python 编译通过；shared-MLP vector/trainer/config、baseline 双阈值汇总与 Torch probe reporting 共 8 项 unittest 通过。联合运行旧 `tests.test_train_and_eval_stage` 时另有 1 failure + 2 errors：旧测试仍要求当前 YAML 启用 raw-attention/hmid feature sets，而当前用户配置已主动关闭这些分支；与本轮 baseline MLP 无关。正式 9 次 GPU 训练、汇总生成、checkpoint shape 审计、统一训练 dry-run 和 `git diff --check` 通过。

## 2026-07-20 alpha05 risk + EV 联合训练

- unified 训练列表新增 `hpre_raw_logit_gauss_risk_sqrt_stateupd_alpha05+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine`；两个 32 层向量直接拼接为 64 维输入，未重新提取特征。
- 使用正式 Torch probe 配置（`128→64→32`、dropout 0.3、strict image-level 8:2、Real-positive、train-F1 阈值）完成 seeds `43/44/45`，结果隔离写入 `results/alpha05_ev_seed{seed}`，未覆盖已有结果或重跑 baseline。
- train-F1 阈值三 seed 均值：Accuracy=`0.867±0.002`、Real F1=`0.921±0.002`、AUROC=`0.886±0.005`、Real AUPR=`0.973±0.002`、Hall F1=`0.582±0.010`。相对单独 alpha05 risk，Real F1/AUROC/Hall F1 分别提高约 `0.005/0.045/0.066`；相对单独 EV 分别提高约 `0.003/0.029/0.108`。
- 该结果与旧的无后缀 `hpre_raw_logit_gauss_risk+...EV` 完全相同：当前 artifact 的 primary `dgst_t_cost` 是 `sqrt_stateupd_alpha05`，因此无后缀 risk 别名与显式 alpha05 risk 读取同一字段。这是同一输入的别名复现，不是两个不同 cost 的独立增益；训练列表已停用重复的无后缀组合，避免后续重复训练。
- 三 seed 汇总为 `outputs/llava_next_8b/COCO4000-512-AC-all/results/llava_next_8b_alpha05_ev_3seed_summary.{md,csv,json}`；YAML dry-run、三次 64 维训练、汇总生成及 `git diff --check` 通过。

## 2026-07-20 compact four-gate 多 cost 同提取与 matched-state 修正

- compact four-gate 新增 `cost_modes`，可在同一次模型 capture/source/target/top-K support 上同时构造多张 cost matrix 并分别求精确 EMD；`cost_mode` 保留为 primary，旧 `<method>_risk` 别名继续指向 primary，保证兼容。
- unified 当前 primary 为 `sqrt_stateupd_alpha05`，并同时启用 `sqrt_stateupd_alpha05`、`sqrt_cosine_matched_state`。两个显式训练别名分别为 `<method>_risk_sqrt_stateupd_alpha05`、`<method>_risk_sqrt_matched_state`；按用户最新要求不再提取或训练 `geo_stateupd_lu1`。
- 按用户公式修正 `sqrt_stateupd_alpha05` 的 state 项为 branch-matched state：hpre 方法使用 `h_prev`，hmid 方法使用 `h_mid`；update 项两者都使用 `h_out-h_mid=o_ffn`。公式为 `C=0.5*sqrt((1-cos(h_state_i,h_state_j))/2)+0.5*sqrt((1-cos(o_ffn_i,o_ffn_j))/2)`。
- 新结果保存 `dgst_t_cost_modes` 以及每个 method/cost 的独立 risk 字段；serializer、全部 wrapper、prompt-target 路径和训练 alias 均已透传。单 cost 旧产物继续兼容。
- 现有 LLaVA-NeXT `COCO4000-512-AC-all/features.pkl` 的 9,668 条记录本来就只含上述两种 cost；此前训练列表额外请求 lu1，导致前两项训练完成后在缺失字段处报错。现已移除该训练项，抽样验证当前 7 个活动 feature sets 均可从现有记录解析，无需重提取。
- 验证：相关 Python 编译通过；逐项手算分别验证 hpre/hmid matched state、0.5/0.5 混合以及多 cost 同时输出；four-gate、prompt-target、extraction、source-target-JS、cost-variant 共 51 项 unittest 全部通过；unified 两 cost 配置和两个显式训练别名通过。

## 2026-07-20 unified 新增并启用 sqrt_stateupd_alpha05 cost

- 新增 compact four-gate canonical cost `sqrt_stateupd_alpha05`，严格实现 `d_state=sqrt((1-cos(h_mid_i,h_mid_j))/2)`、`delta_h_i=h_out_i-h_mid_i=o_ffn_i`、`d_upd=sqrt((1-cos(delta_h_i,delta_h_j))/2)`、`C=(1-alpha)d_state+alpha*d_upd`，本版固定 `alpha=0.5`。
- `configs/model_configs_unified.yaml` 已从 `geo_stateupd_lu1` 切换到新 cost；旧 `geo_stateupd_lu1` 和 `sqrt_cosine_matched_state` 均继续保留可选。新字段为 `dgst_t_<method>_risk_sqrt_stateupd_alpha05_per_layer`，结果同时保存 `dgst_t_cost=sqrt_stateupd_alpha05` 与 `dgst_t_cost_alpha=0.5`。
- compact feature serializer 与现有 `<method>_risk` 训练别名已接入新字段，因此无需改 training feature-set 名称。逐项手算测试独立构造两张 sqrt-cosine 距离矩阵及 0.5/0.5 cost，再与 EMD 输出对齐。
- 验证：相关 Python 编译通过；four-gate、prompt-target、extraction、source-target-JS、cost-variant 共 51 项 unittest 全部通过；unified YAML 加载与 canonical normalize 通过。

## 2026-07-20 unified 启用 geo_stateupd_lu1 cost

- `configs/model_configs_unified.yaml` 的活动 `four_gate` profile 已将 `cost_mode` 切换为 `geo_stateupd_lu1`。此前 compact four-gate 快路径会忽略 YAML 的 cost 选择；现在配置值会经过各模型 wrapper/共享 capture 路径传入并真实参与 OT cost 构造。
- 逐层 support-token cost 为 `C_ij=(1-cos(h_mid_i,h_mid_j))+(1-cos(o_ffn_i,o_ffn_j))`：`h_out=h_mid+o_ffn`，因此 state-update 向量就是 `h_out-h_mid=o_ffn`，`lambda_u=1.0`。不使用旧活动配置的 `sqrt((1-cos)/2)`。风险仍在 source/target top-K union 上用精确 EMD 计算。
- 新结果字段为 `dgst_t_<method>_risk_geo_stateupd_lu1_per_layer`，并保存 `dgst_t_cost=geo_stateupd_lu1`；现有 `<method>_risk` 训练别名会按该 metadata 自动读取新字段，旧 `sqrt_cosine_matched_state` 产物继续读取原字段。
- 验证：相关 Python 文件编译、unified YAML 加载/normalize、`git diff --check` 通过；four-gate/prompt-target/extraction/source-target-JS/cost-variant 共 50 项 unittest 全部通过，其中新增手算同路径 EMD 回归覆盖 hmid 与 FFN update 两项距离。
- 额外组合运行 `tests.test_extraction_modes tests.test_pipeline_config tests.test_raw_attention_integration` 时，45 项中出现 7 failures + 4 errors：pipeline manifest 旧断言与当前未提交的 pipeline 改动不一致，另一个旧测试要求活动 YAML 启用 `raw_attention`，但本轮开始前该分支已被关闭。这些与本次 cost 路径无关，未为通过旧断言而改回用户现有配置。

## 2026-07-20 LLaVA-NeXT Llama-3 EOT 对齐修复

- LLaVA-NeXT 完成 COCO4000 双卡生成后，CHAIR labeling 在 image `578522` 的最后一个 response token 报错。该 token 为 `128009=<|eot_id|>`：tokenizer backend 与 `added_tokens_decoder` 均将其标为 special，`skip_special_tokens=True` 也会移除它，但该 checkpoint 的 `all_special_ids` 仅包含 128000/128001，旧对齐器因只读取后者而误判为可见 token。
- `utils/token_alignment.py` 现在合并 `all_special_ids` 与所有 `AddedToken.special=True` 的 ID；decoder fallback 与 SentencePiece visible-index 路径共享同一集合。原始 `response_token_ids` 长度和索引完全保留，EOT 只映射为 `None`，不删除或重编码 token。
- 新增 `all_special_ids` 不完整的 Llama-3 EOT 回归用例。16/16 项 COCO token alignment unittest 通过；真实失败样本最后 offset 为 `None`；现有 4000/4000 条 LLaVA-NeXT 生成全量只读对齐扫描通过、0 failures。已有生成可直接复用并从 CHAIR labeling 续跑。

## 2026-07-19 默认三层 Torch MLP 与双阈值报告

- 两份活动统一配置已同步切换到同一正式协议：COCO4000 严格按物理图片划分为 train/test=`3200/800`，validation 为空；网络为 `input→128→64→32→1`，每个隐藏层严格执行 Linear-BatchNorm-ReLU-Dropout(0.3)，Linear 使用 Kaiming-uniform-ReLU 初始化。
- 训练默认值为 Adam、`lr=1e-3`、`weight_decay=1e-5`、batch 256、最多 100 epochs、plain BCEWithLogitsLoss。ReduceLROnPlateau 只监控 train loss（factor 0.5、patience 5）；early stopping 同样只监控 train loss（patience 10），并恢复 minimum-train-loss checkpoint。默认不做额外 z-score 或类别加权，seeds 改为 `43/44/45`。
- 同一个 minimum-train-loss checkpoint 现在同时报告两套结果：固定阈值 `0.5`，以及只用训练集 Real-F1 搜索的阈值；test 从不参与阈值搜索。为兼容旧 JSON 消费者，顶层 `threshold/train_metrics/test_metrics` 仍对应 `train_f1`，完整双报告写在 `threshold_reports.fixed_0.5` 与 `threshold_reports.train_f1`。
- COCO Torch probe、QA DGST/ADS/CGC probe，以及默认 `shared_torch_mlp` 的 MetaToken/SVAR/ProjectAway baseline 已统一到上述架构、checkpoint 和双阈值语义；逐 seed JSON、baseline 汇总、Torch probe 汇总和 QA comparison Markdown 都会同时展示两套阈值结果。旧结果的训练指纹与新配置不一致，不能静默复用。
- 验证：41/41 项针对性 unittest 通过；三 seed CPU 小数据端到端 smoke 确认一个 checkpoint 可同时转换并聚合两套阈值，固定阈值汇总均值严格为 0.5；双 YAML 参数审计、`train_and_eval.py --dry-run`（43/44/45）、Python 编译、shell 语法和 `git diff --check` 通过。默认正式 8B 实验尚未重训，现有历史结果不应当作新配置结果。

## 2026-07-19 AMBER 固定 BN-ReLU MLP 消融

- 新增隔离入口 `scripts/train_qa_fixed_mlp_experiment.py`，在不改动正式 QA probe/baseline 训练语义的前提下联合读取协议专属 MetaToken/SVAR baseline 特征与根 DGST 特征。输入 cohort、标签、物理图片 split、baseline manifest 和文件 SHA 均在训练前校验，结果指纹为 `b3cde613...`。
- 本次按 `answer_correctness_all`、seeds 42/43/44 训练 MetaToken、SVAR、`hpre_raw_logit_gauss_risk+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine@prompt_last_token`。沿用 AMBER 现有 803/0/201 张图片、11385/0/2831 条样本；train/test 的 hall/real 分别为 1335/10050、292/2539。
- 网络严格为 `input→128→64→32→1`，每个隐藏层使用 Linear-BatchNorm-ReLU-Dropout(0.3)，Linear 用 Kaiming-uniform-ReLU 初始化；plain BCEWithLogitsLoss、Adam(lr=1e-3, weight_decay=1e-5)、batch 256。ReduceLROnPlateau 与 early stopping 均只监控 train loss，patience 分别为 5/10，恢复 minimum-train-loss checkpoint；最多 100 epochs，阈值固定 0.5，不做额外 z-score 或类别加权。
- 三 seed Test AUROC/Real-F1/Hall-F1/Acc：MetaToken=`0.8284/0.9456/0.0000/0.8969`，SVAR=`0.7849/0.9212/0.3660/0.8604`，hpre risk+EV=`0.8441/0.9369/0.3663/0.8858`。MetaToken 固定阈值下全部预测为 real；hpre risk+EV 平均 AUROC 最好，但 seed 标准差 0.0278，明显不如正式配置的 0.0040 稳定。
- 复用同一批 checkpoint，新增按每个 feature/seed 的 train Real-F1 精确搜索阈值的隔离评估；test 仍不参与阈值选择。搜索后阈值均值 MetaToken/SVAR/hpre risk+EV=`0.5289/0.3778/0.5117`，Test AUROC/Real-F1/Hall-F1/Acc 分别为 `0.8284/0.9457/0.0424/0.8972`、`0.7849/0.9320/0.3520/0.8770`、`0.8441/0.9439/0.3884/0.8972`。相对固定 0.5，hpre risk+EV 的 Real-F1/Hall-F1/Acc 提升 `+0.0069/+0.0221/+0.0114`；SVAR 的 Hall-F1 下降 `0.0140`；MetaToken 仍基本预测为 real。
- 9/9 checkpoint 的层形状、固定阈值、输入指纹、best epoch 与 history 最小 train loss 均已核对。新增 2 项架构/协议单测通过，Python 编译与 `git diff --check` 通过；正式结果位于 `outputs/qa_benchmarks/qwen3_vl_8b/amber_discriminative/experiments/mlp_bn_relu_trainloss_es_fixed05/answer_correctness_all/`。
- train-threshold 结果位于 `outputs/qa_benchmarks/qwen3_vl_8b/amber_discriminative/experiments/mlp_bn_relu_trainloss_es_train_real_f1_threshold/answer_correctness_all/`；入口为 `scripts/evaluate_qa_fixed_mlp_train_threshold.py`，只引用源 checkpoint，不复制或覆盖模型权重。

## 2026-07-19 LLaVA-NeXT-8B wrapper 接入

- 新增 `models/llava_next_wrapper.py`，完整接入 `lmms-lab/llama3-llava-next-8b` 的生成、单/批 token 特征、prompt-target、DGST、ADS/CGC 和 baseline 共用路径。NeXT 复用现有 LLaVA 抽取主干，但使用官方 `llava_llama_3` system/user 模板和 AnyRes 动态连续视觉 span，不再错误继承 LLaVA-1.5 固定 576 token / 24x24 布局。
- 该公开 checkpoint 是原始 LLaVA 平铺 config/state-dict，而不是原生 Transformers `LlavaNextConfig`。wrapper 在内存中构造 Llama-3 + CLIP-L/14-336 的嵌套配置并映射权重键，不修改磁盘上的 16 GB 权重。验证 index 的 687/687 个键均有唯一目标；当前已存在分片的 604 个张量 shape 全部匹配，无 missing/unexpected/mismatch。
- 原 checkpoint 将 `<image>` 设为 ID 128256，但 embedding 只有 128256 行；原始 LLaVA 会在查 embedding 前替换占位符，而 Transformers NeXT 会先查 embedding。processor 因此只把 image placeholder 重映射到未使用的保留 ID 128255，模型同样以 128255 做 masked scatter；视觉向量本身不经过该 token embedding，语言 token 与全部模型权重保持不变。
- AnyRes processor smoke 覆盖 `336x336`、`640x320`、`320x640`：展开 image token 数分别为 `1176/1752/1776`，与 `LlavaNextModel.pack_image_features()` 的实际打包长度逐一相等；原 tokenizer image ID 已全部替换，不残留越界 ID。
- 模型工厂注册 `llava_next_8b` 和兼容别名 `llava_next_llama3_8b`；本机/服务器统一 YAML、POPE prompt、QA provenance source hash 均已接入。`huggingface-hub` 从与 Transformers 4.57.6 不兼容的 1.8.0 恢复为 0.36.2，并在 `requirements.txt` 固定 `>=0.34,<1.0`。
- 新增 5 项 wrapper 单测；与 prompt-target 回归合计 14/14 通过。`py_compile`、YAML 加载、`git diff --check`、权重键/shape 审计通过。
- 完整 CPU unittest 命令 `CUDA_VISIBLE_DEVICES='' /opt/conda/private/envs/vicr/bin/python -m unittest discover -v tests` 共 183 项，出现 11 failures + 4 errors：其中 QA YAML 失败来自本轮开始前未提交的 feature sets 已含 `_target_cosine` 而旧断言禁止；其余 pipeline/stage/training provenance 失败来自现有未提交的 `run.sh` / training 相关改动，表现为旧测试期待 manifest 拒绝或落盘但当前逻辑未执行。本次针对性 LLaVA/NeXT 测试全部通过，未修改这些并行开发文件。
- 真实 GPU forward 尚未执行：本地模型缺 `model-00001-of-00004.safetensors`。镜像单连接已安全下载前 `549134336` 字节到同目录 `.part`；8/4 路 Range 因镜像 SSL EOF 失败，损坏的 `.parallel` 已删除，正式分片没有伪造。可用单连接 `curl -L --fail -C - -o ...safetensors.part <镜像URL>` 继续，达到官方大小 `4976706872` 且 SHA-256 为 `eda352b5dd159390824859f8a31fb1c015bdf161ae44678602e88e3b4b631e1b` 后再改正式文件名。

## 2026-07-19 AMBER baseline 原生训练器对照

- `train_qa_baselines.py` 新增 `--trainer native_paper|shared_torch_mlp` 临时覆盖参数，可在不修改 YAML 默认训练器的情况下补跑另一种 head；两类结果继续使用隔离路径。
- AMBER 原生训练器已按 seeds 42/43/44 完成两种标签协议：MetaToken 使用 LR 与 GB，SVAR 使用原生单隐层 MLP，ProjectAway 保持 training-free。样本、物理图片级严格 8:2、train Real-F1 阈值与统一 MLP 对照完全一致。
- `object_hallucination_yes_only` Test AUC/Real-F1/Hall-F1/Acc：MetaToken-LR=`0.8244/0.9169/0.3793/0.8534`，MetaToken-GB=`0.8267/0.9211/0.5378/0.8652`，SVAR=`0.8649/0.9403/0.6085/0.8965`，ProjectAway=`0.5500/0.9089/0.0000/0.8330`。
- `answer_correctness_all` Test AUC/Real-F1/Hall-F1/Acc：MetaToken-LR=`0.8527/0.9458/0.1321/0.8979`，MetaToken-GB=`0.8629/0.9431/0.3660/0.8956`，SVAR=`0.7487/0.9444/0.0277/0.8947`，ProjectAway=`0.3908/0.9456/0.0000/0.8969`。
- 原生与统一 MLP 的跨 family comparison 分别写入 `comparison_native_paper/` 与原 `comparison/`，未覆盖任何正式结果。验证：相关 baseline 单测 11/11、Python 编译、shell 语法和 `git diff --check` 通过。

## 2026-07-19 AMBER baseline 统一三层 MLP

- QA baseline 训练与 COCO baseline 对齐：由 `training.baseline.trainers` 同时运行 `native_paper` 和 `shared_torch_mlp`，在相同严格 8:2 划分与 seeds 上报告 fixed-0.5/train-F1 阈值、real/hall 双正类指标及双训练器对照；训练特征包含 MetaToken/SVAR/DHCP/ProjectAway。`qa_benchmarks.baseline_trainer` 仅选择跨 family QA 总表中的 headline baseline。
- 统一输入保持各方法特征定义：MetaToken 为 42 维 canonical `10+H`；SVAR 为 448 维中层 `layer×head`；ProjectAway 为 1 个 global internal confidence 加 36 层曲线，共 37 维。ProjectAway 的 `1-confidence` 未重复拼接。所有方法仍使用物理图片互斥的严格 8:2、最后一轮权重及 train Real-F1 阈值。
- 结果显式命名 `shared_torch_mlp` 并与原生 baseline 文件隔离。`summarize_qa_comparison.py` 会按 YAML 选择对应汇总，避免将统一 MLP 结果误称为论文原生 head。
- AMBER `object_hallucination_yes_only` 三 seed Test：MetaToken AUC/Real-F1/Hall-F1/Acc=`0.8771/0.9314/0.6007/0.8829`；SVAR=`0.8793/0.9411/0.6250/0.8982`；ProjectAway=`0.5154/0.9085/0.0000/0.8323`。
- AMBER `answer_correctness_all` 三 seed Test：MetaToken AUC/Real-F1/Hall-F1/Acc=`0.8862/0.9423/0.4584/0.8957`；SVAR=`0.7782/0.9400/0.3281/0.8898`；ProjectAway=`0.5622/0.9440/0.0086/0.8939`。ProjectAway 的高 Real-F1 伴随几乎为零的 Hall-F1，主要反映类别不平衡，不能当作有效幻觉检出。
- 2026-07-19 按当前代码指纹 `fc60ba76...` 对 `answer_correctness_all` 重新完成 3 方法 × 3 seeds 的正式 GPU 训练；9/9 checkpoint 均为 `input→256→128→64→1` 且跑满 120 epochs，新汇总与旧指纹结果逐项一致。旧产物完整保留在 `results/shared_torch_mlp_fingerprint_3d0b13e8/`，当前有效产物位于 `results/shared_torch_mlp/`。
- 将 train Real-F1 阈值搜索从逐候选全量扫描的 O(N²) 改为排序+前缀计数的等价 O(N log N)；12,000 条样本由数十秒降到约 0.007 秒，暴力对齐测试确认阈值与 F1 不变。
- 验证：全仓 173 项 unittest 通过；新增 baseline 向量、协议标签和 trainer 规范化测试通过；快速阈值与暴力版本等价；Python 编译、shell 语法、`git diff --check` 通过。两套正式三 seed 汇总及跨方法 comparison 均已生成。

## 2026-07-18 POPE random-only MetaToken / SVAR 实验

- 从现有 `object_hallucination_yes_only/baseline/features.pkl` 只筛选 POPE `random` strategy，不重新抽取 LVLM 特征；全部选中记录的 `response_token_idx=0`，与当前 `prompt_last_token` 协议数值等价。结果隔离写入 `baseline/object_hallucination_yes_only/random_only/`，未覆盖三种 strategy 合并结果。
- 沿用现有严格 image-level 8:2：train/test=`1020/258` 条、`398/98` 张物理图；train 的 real/hall=`1006/14`，test=`250/8`。seeds=`42/43/44`，阈值只按 train Real-F1 选择，test 只评估。
- 三 seed Test：MetaToken-LR Accuracy/Real-F1/AUROC/Hall-F1=`0.9690/0.9843/0.9130/0.0000`；MetaToken-GB=`0.9341±0.0032/0.9658±0.0017/0.8712±0.0016/0.1055±0.0045`；SVAR=`0.9690/0.9843/0.6163±0.0061/0.0000`。
- MetaToken-LR 与 SVAR 在 train-F1 阈值下均把 258 个 test 样本全部预测为 real，所以高 Accuracy/Real-F1 来自 `250:8` 的严重类别失衡，不能解释为已检测到幻觉；AUROC 不依赖该阈值，MetaToken-LR 的排序能力明显高于 SVAR。

## 2026-07-18 QA 判断位置改为 prompt 最后一个 token

- QA 活动位置协议从 `answer_pre_token` 改为 `prompt_last_token`。DGST、ADS+CGC 与 MetaToken/SVAR/DHCP/ProjectAway/HalLoc 共享同一次 forward，固定请求 `response_token_index=0`；该输出对应完整 prompt 的最后一个因果状态，并预测 `response_token_ids[0]`。
- 语义 yes/no 的 `answer_token_index` 仍会生成、保存和校验，只用于回答解析与标签正确性审计，不再决定特征抽取位置。因此即使模型以后先输出解释或前缀词，判断特征也不会从 prompt 末位置后移。
- 根特征 schema 升级为 `qa-prompt-last-token-v5`，baseline protocol 升级为 `qa_prompt_last_token_image_level_probe_split_v3`；YAML、QA probe 特征名、完整性检查和跨方法 Markdown 汇总均同步使用 `prompt_last_token`，防止旧 answer-position 特征被静默当成新协议。
- 当前 Qwen3 POPE 的 9000/9000 条生成中，语义 yes/no 都是 `response index=0`，因此本批旧特征与新协议的输入状态在数值上等价；协议升级主要消除未来出现回答前缀时的位置歧义。未覆盖或删除任何现有正式特征/训练结果。
- 验证：新增的非首位 yes/no 测试使用 `response_ids=[99,7]`、`answer_token_index=1`，确认 wrapper 仍收到 `response_token_indices=[0]`、`target_token_ids=[99]`。禁用 CUDA 后 170 项 unittest 全部通过；另手动执行 10 项 pytest 风格 QA 测试全部通过；相关 Python 文件编译通过。当前环境未安装 pytest，直接运行 `python -m pytest` 报 `No module named pytest`，未修改环境。

## 2026-07-18 QA 改为一次 forward 联合抽取所有特征族

- `qa_benchmarks.extraction_mode` 现与 COCO 使用相同四种语义：`all | method_only | ads_cgc_only | baseline_only`；两份统一 YAML 的 QA 默认均为 `all`。family 的 `enabled` 开关仍会与 mode 共同生效。
- `features/qa_extractor.py` 会先判断每道题缺失的根特征和 baseline protocol，再合并 DGST、ADS+CGC、MetaToken/SVAR/DHCP/ProjectAway/HalLoc 的 `ExtractionRequirements`。答案 token 只执行一次 LVLM forward，同一个 `ModelOutput` 同时提供给所有启用消费者。
- baseline 特征仍隔离写入 `outputs/qa_benchmarks/<model>/<dataset>/baseline/<label_protocol>/`；`baseline_only` 不创建或覆盖根 `features.pkl`。resume 可以只补缺失 family：根特征已完整时只补 baseline，baseline 已完整时只补根特征。
- 双卡 worker 继续按问题互斥分片；根特征写 worker 隔离目录，baseline 写 `part-workerNNN-*` 分片，父进程在所有 worker 成功后统一验证 cohort、cache、严格 8:2 split 和 manifest，再原子合并。
- `run_qa.sh` 已移除第二次 `extract_qa_baselines.py` 调用。当前流程为：准备数据；生成+标注+联合抽取；训练 DGST/ADS+CGC；训练已抽取的 baseline；仅在 `all` 模式生成跨 family 汇总。部分模式会自动跳过不适用的训练器。
- QA 根特征 schema 升级为 `qa-answer-token-v4`，记录实际 `feature_families`；fingerprint 与完整性检查包含 method/ADS+CGC 组合，防止不同 root schema 静默混用。
- 验证：新增/更新测试覆盖四种 mode、`all` 一次 wrapper 调用、同一输出对象被根方法和 baseline 共用、`baseline_only` 不生成根特征。`CUDA_VISIBLE_DEVICES='' /opt/conda/private/envs/vicr/bin/python -m unittest discover -v tests` 共 170 项通过；双 YAML 解析、Python 编译、三个 QA shell 的 `bash -n` 与 `git diff --check` 通过。本轮未启动 8B 正式抽取。

## 2026-07-18 QA 正式流水线改为双卡问题分片

- `run_qa.sh` 默认导出 `CUDA_VISIBLE_DEVICES=0,1`，generation 与 DGST/ADS+CGC 特征抽取分别通过 `--generation-devices cuda:0 cuda:1`、`--feature-devices cuda:0 cuda:1` 启动两个 worker；native baseline 特征抽取也使用同一双卡列表。probe 与 baseline 训练仍只在 `DEVICE=cuda:0` 上运行，避免两个训练进程覆盖同一结果目录。
- `scripts/qa_pipeline.py` 按稳定问题顺序 round-robin 划分互斥分片。generation worker 写入 `.qa_parallel/generation/workers-2/worker-*`，feature worker 写入 `.qa_parallel/features/workers-2/worker-*`；主进程只在所有 worker 正常退出后按原问题顺序原子合并 `generations.jsonl`、`features.pkl` 和失败记录，两个 GPU 不会并发改写同一主产物。
- 双卡 resume 使用固定 worker 目录和原子 shard：已完成的 generation 会从主文件播种到对应 worker，feature worker 复用自己的 `features.parts`；任一 worker 中断后，下一次运行只重试缺失项。`--no-resume` 同时检查隐藏的并行目录，禁止意外复用旧 worker 数据。
- `scripts/extract_qa_baselines.py` 增加 `--feature-devices`。worker 使用互斥问题分片以及 `part-workerNNN-*` 特征 shard；已有 `BaselineRuntime(parallel=True)` 负责将 DHCP shard 和 HalLoc cache 写入独立 worker 子目录，主进程最后统一验证 cohort、cache 和 split manifest 后生成正式 `features.pkl`。
- `features/qa_baseline.py` 的 `QABaselineFeatureStore` 新增安全 `part_prefix`，同时兼容历史单卡 `part-*`，因此双卡与单卡 resume 可以读取同一事务日志。README 已补充双卡默认行为和显式单卡回退命令。
- 新增 `tests/test_qa_parallel.py`，覆盖设备去重、稳定互斥分片以及两个 baseline worker shard 无文件名冲突并可确定性合并。验证命令：`CUDA_VISIBLE_DEVICES='' /opt/conda/private/envs/vicr/bin/python -m unittest discover -v tests`，共 168 项通过；`py_compile`、三个 QA shell 的 `bash -n` 与 `git diff --check` 均通过。本轮未启动 8B 正式数据运行。

## 2026-07-17 COCO 与 QA 改为纯严格 8:2

- 活动 COCO 与 QA 配置统一改为 train/test=`80%/20%`，顶层 `val=[]` 只作为旧产物结构的兼容字段；训练过程中不再创建任何内部验证集。
- COCO4000 固定为 `3200/0/800` 张图片；POPE 固定为 `400/0/100` 张图片及 `7200/0/1800` 条问题；CLEVR-Exist 5K 固定从官方 train 取 4000 条、从官方 val 取 1000 条，官方 source split 保证物理图像不跨 train/test。
- DGST、ADS+CGC、QA probe 与各 baseline 均固定训练 YAML 指定的完整 epochs，保存最后一轮 checkpoint；分类阈值只在训练集上按 Real-positive F1 选择，不 early stop，不根据 test 调参或选阈值。
- baseline 为兼容旧训练函数可在训练集上额外计算诊断 loss/metrics，字段统一报告为 `train_monitor_loss` / `train_metrics`；它们不参与 checkpoint 或超参选择，阈值仅按前述 train-F1 规则确定。
- 旧 8:1:1 `image_splits.json` 在下次流水线启动时自动备份后替换；特征本身与 split 无关，可以复用，但所有 probe/baseline 训练结果必须按新划分重跑。
- YAML 显式设置 `training.threshold_selection=train_f1`；Torch、sklearn、QA 与全部 baseline 都保存 `threshold_selection=train_f1`，旧的固定 0.5/validation 阈值结果不能静默 resume。
- 验证：166 项 unittest 与 13 项 pytest 风格 QA 纯函数测试通过；真实 POPE 准备结果为 7200/0/1800 条、400/0/100 图，真实 CLEVR 为 4000/0/1000 条、3900/0/975 个 source-image identity；Python/Shell 编译、双 YAML 断言及 `git diff --check` 均通过。未启动 8B 正式训练。

## 2026-07-17 POPE/CLEVR 首轮改为 answer-token 单位置

- POPE 与 CLEVR-Exist 的首轮 VQA 对比默认只启用 `answer_pre_token`；若实际生成的 yes/no token 位于 response index `i`，forward 严格使用 `response_ids[:i]`，最后一行因果状态预测保存的真实 token `response_ids[i]`。
- QA 配置已合并进 `configs/model_configs_unified.yaml` 与 fj01 镜像；`qa_benchmarks.position_protocols` 只保留 `answer_pre_token`。特征提取每题只运行一次 answer-position forward，不运行 object forward；trainer 复用同一 YAML 的 `training.torch_probe`，因此默认仅训练 9 组：6 个 DGST 的 `risk+target_cosine+EV`、ADS、CGC、ADS+CGC。
- 已删除独立的 `qa_benchmarks_unified.yaml` 和 `qa_benchmarks_server_fj01.yaml`；所有 QA 脚本、`run_qa.sh` 与服务器 coordinator 默认读取对应的统一模型 YAML。caption 的 `max_new_tokens=512` 保持不变，QA 仅通过 `qa_benchmarks.generation` 覆盖为 8。
- 问题中 object word 的上下文化定位能力仍保留为可选消融；以后只需在 YAML 追加 `question_object_pre_token`。该路径使用问题字符区间、完整模板实际 token span，以及图像 token 展开后的 `j-1` prediction row，不会单独编码 object。
- 修复可选 object 路径的重复词风险：模板解码仅改变空白时按整段问题做归一化映射；有精确字符区间却无法映射时直接报错，禁止静默回退到同词第一次出现。
- answer-only 位置列表写入每条 feature 和 summary，resume 会校验位置配置；从 answer-only 改为双位置时不能错误复用缺少 object 的旧特征。
- 取消旧 generation 缺位置时的“第一个普通 token”回退：generation 必须保存合法的 `answer_token_index/answer_token_id`，抽取器与 baseline 会重新定位并核验它确为实际生成的第一个 yes/no token；否则立即停止，不会拿错误位置继续训练。
- QA 特征 schema 升级为 `qa-answer-token-v3`。resume 同时校验 generation、label、question、图片内容、image-level split、模型与 DGST/ADS/CGC 配置；probe 训练前再次核对内嵌标签/划分，旧结果也必须匹配训练输入指纹。
- 对比汇总只描述 YAML 实际启用的位置；answer-only 报告不会再显示未运行的 object 消融说明或覆盖率。
- `run_pope.sh` / `run_clevr.sh` 共用三阶段 `run_qa.sh`，并在训练后按实际 baseline label protocol 生成对比汇总。
- 验证：完整 `unittest discover` 163 项通过；另有 7 项 pytest 风格 QA probe 纯函数测试手动执行通过；Shell、Python 编译与 `git diff --check` 通过。未启动任何 8B 模型正式实验。

## 2026-07-17 VQA prompt object 因果预测 API

- 新增独立 `PromptTargetRequest` 和五模型统一 `extract_prompt_target_features()`；它在真实完整问题的上下文 tokenization 中定位 object surface，多词对象保留完整 span、以首个实际子 token 为目标，并严格截断到该 token 之前。
- 图像 placeholder 展开后重新计算目标位置，attention、hidden states、logits 和 DGST capture 全部来自同一个 `target_expanded_position - 1` prediction row；实际 target ID 可显式校验，并写入 `baseline_capture.prompt_target_alignment`。
- 原 `validate_causal_batch_request()` 未修改，仍只允许 saved response index 对应的实际生成 token ID，问题中的 object token 不再伪装成 response target。
- 新增 mock 测试覆盖 contextual BPE ID、实际 target ID 校验、多 token span、`j-1`、图像展开以及五 wrapper 接口；与旧因果测试共 13 项通过。
- 完整 CPU unittest 共 152 项，151 项通过；唯一失败是并行开发中的 `detection/qa_probe.py:245` 存在 `IndentationError: expected an indented block`，与本 API 修改无关，待 QA probe 合并完成后重跑。

## 2026-07-17 Baseline 正类切换为 real

- `training.baseline.positive_class` 默认设为 `real`；后续 MetaToken、SVAR、DHCP、ProjectAway、HalLoc 都在 validation 上使用 real probability 最大化 Real-F1 选择阈值，并同时保留 real/hallucination 两套指标。
- baseline 汇总表以 Real Precision/Recall/F1/AUPR 为 headline，AUROC 在同时翻转标签和 score 后保持不变，最后一列报告 Hallucination F1。
- CPU 首轮 46 项相关测试中两项命令数量断言失败，原因是当前用户配置已将 `run.extraction_mode` 改为 `method_only`，旧测试仍假定 YAML 为 `all`；测试现显式构造 `all` 模式，不修改用户当前运行选择。
- 首轮 131 项完整 CPU 测试另有 1 项旧断言仍强制 active YAML 为 `all`；已改为验证值属于四种合法 extraction mode，继续保留用户当前 `method_only`。
- 用户当前正在占用 GPU，本轮尚未启动 baseline 重训；先完成 CPU 代码验证，待 GPU 可用后再用 seeds 42/43/44 重算指定汇总。
- 已立即将指定 `qwen3_vl_8b_baselines_3seed_summary.md` 改为 Real-positive 表格；它复用旧 JSON 中已保存的 real metrics，并明确注明旧阈值仍按 validation Hallucination-F1 选择。后续真正重跑后会由新逻辑改为 validation Real-F1 阈值。
- 验证：禁用 CUDA 后完整 131 项 unittest 通过；补充 threshold score-class/checkpoint 元数据后，baseline/reporting/SVAR protocol 相关 24 项再次通过；`py_compile` 与 `git diff --check` 通过。

## 2026-07-17 Qwen3 SVAR 关闭 early stopping 实验

- 复用 `COCO4000-512-AC/baseline/features.pkl` 和严格 image-level 8:1:1，只训练 SVAR；seeds=`42/43/44`，每个 seed 固定运行 50 epochs，不提前终止，仍按最低 validation loss 选择 checkpoint。
- 固定 50 epochs 的 test AUC/AUPR/F1/Acc=`0.8777±0.0015 / 0.6776±0.0026 / 0.6673±0.0050 / 0.8314±0.0026`；原 patience=5 为 `0.8739±0.0023 / 0.6736±0.0023 / 0.6531±0.0141 / 0.8155±0.0185`。
- 差值分别为 AUC `+0.0038`、AUPR `+0.0040`、F1 `+0.0142`、Acc `+0.0159`。seed 42 最佳 epoch 从 16 后移到 38，seed 44 从 22 后移到 28；seed 43 仍为 epoch 20，结果不变。
- 独立结果位于 `baseline/results/svar_no_early_stop_seed{42,43,44}/`，汇总为 `baseline/results/qwen3_vl_8b_svar_no_early_stop_3seed_summary.md`，未覆盖原 baseline 结果。

## 2026-07-17 EV 改为 target-dist 区域质量乘原始 target-cosine

- 活动 DGST 六个 target 分支统一使用各自 target-dist 的 top-32 区域：`mass=sum(target_dist[topK])`，`target_cosine=mean(cosine[topK])`，最终 `EV=mass*target_cosine`。
- EV 不再逐 token 计算 `support-attention*(1+cosine)/2`，也不再将 cosine 从 `[-1,1]` 平移到 `[0,1]`；因此新版 EV 允许为负数。
- hpre 分支继续匹配 prediction/visual hpre，hmid 分支继续匹配 prediction/visual hmid；source-dist、OT cost 和 EMD risk 不直接参与 EV。
- 特征记录新增 `dgst_t_ev_definition=target_dist_topk_mass_x_mean_target_cosine`；新版数值字段显式命名为 `dgst_t_{method}_ev_target_dist_mass_x_cosine_topk32_{state}_per_layer`，训练别名为 `{method}_ev_target_dist_mass_x_cosine`。旧 `{method}_ev` 别名只兼容读取旧 `..._ev_topk32...` 字段，避免新旧公式静默混用。
- 验证：`/opt/conda/private/envs/vicr/bin/python -m unittest discover -v tests` 共 131 项通过；大于 32 个视觉 token 的测试逐分支手算验证了 top-K mass、matched-state cosine 和 EV。
- 字段改名后的首轮 51 项相关测试有 1 项失败：旧测试强制要求训练列表包含 `hpre_softmax_prob_direct_risk`，但当前 YAML 有意只训练所选子集；已将断言收敛为验证 direct 抽取分支开启、新 raw-attention EV 组合可训练，随后重跑验证。

## 2026-07-16 run.sh 阶段级 resume 修复

- `coco-labeling/label_coco.py` 现在会核对当前 4000 image IDs、generation caption 与完整 labeling schema；全部一致时直接跳过 tokenizer/model、CHAIR evaluator 和逐图 labeling，不再每次重算 `labeling.json`。
- 只有主 `generations.json` 不完整时才扫描 generation shards；正常完整续跑不会重复合并 4000 条 shard。
- 联合特征抽取与 baseline-only 抽取都会先排除无 object-token span 或 token index 越界、因此不可能产生特征的图片；这些图片不再永久显示为 pending。
- `--resume` 会在加载 LVLM、CLIP/VisualBERT 或启动多 GPU worker 之前检查 root/baseline 已完成 image IDs；全部可提取图片均覆盖时直接退出特征阶段。
- 当前 Qwen3 `COCO4000-512` 实测：4000 条 generation/labeling 直接复用；3948 张可提取图片的 root+baseline 特征全部覆盖，52 张无有效 object-token span 被正确排除，特征阶段未加载模型。
- 新增 `tests/test_stage_resume.py`，覆盖完整 labeling、caption 不一致、空/越界 span 及 root+baseline 交集续跑；与 extraction mode 测试共 10 项通过。

## 2026-07-16 fj01 服务器统一配置同步

- 将 `configs/model_configs_server_fj01.yaml` 从旧的 COCO4000 visual-prompt/cost-variant 配置升级为当前 `model_configs_unified.yaml` 的服务器路径镜像。
- 两份配置的实验逻辑、五模型参数、四分支 DGST + raw-attention、baseline、ADS+CGC、严格 8:1:1、三随机种子训练和自动汇总设置完全一致。
- fj01 专属配置只替换模型路径为 `/root/rivermind-fs/xiongbo/models/...`，COCO 路径为 `/root/rivermind-data/dataset/coco`；相对输出和 shared split 路径保持不变。

## 2026-07-15 四分支 DGST、baseline 与统一流水线

- Active DGST profile is now the compact four-gate path: hpre/hmid raw target logits and vocabulary-softmax probabilities each share one chunked vocabulary projection, then independently produce Gaussian-MAD gate, exact-EMD sqrt-hpre risk, top-32 hpre target cosine, and EV. Only the eight compact capture inputs and final matrices/curves survive; full vocabulary matrices and decoder captures are released promptly.
- All five wrappers accept the shared prompt and automatic `ExtractionRequirements`. Qwen2.5/Qwen3/OneVision use sequential object-prefix extraction, while LLaVA1.5/InternVL reuse a full-caption forward. Qwen3 DeepStack residual reconstruction was corrected, and all/standalone hidden/attention parity was verified exactly.
- The active YAML keeps the original processor preprocessing and contains no `max_pixels`. `run.sh` exposes an optional runtime `MAX_PIXELS` override and passes it consistently to generation/extraction. On the largest sampled COCO image (P=529), uncapped four-branch method-only extraction OOMed on a 32 GiB GPU, while a single uncapped branch passed at about 17.6/18.4 GiB for Qwen2.5/OneVision; reducing image count alone does not lower this per-image peak.
- Added isolated MetaToken, SVAR, DHCP, ProjectAway, and HalLoc feature/training paths under `OUTPUT/baseline/`; joint all-mode shares wrapper outputs, while baseline-only never rewrites root `features.pkl`. ProjectAway projects raw intermediate hidden states; HalLoc uses frozen pretrained CLIP ViT-B/32, pretrained VisualBERT, one object head, AdamW and cosine scheduling.
- `run.sh` directly orchestrates the historical three stages (generation+labeling, feature extraction, train+eval). Model/output/devices/prompt/stage/mode choices remain shell variables; YAML keeps experiment definitions. To fit uncapped Qwen3 on 32 GiB, the safe default is `method_only` plus `hpre_raw_logit_gauss`; `DGST_BRANCHES=""` explicitly selects all four. Modes are `all`, `method_only`, `ads_cgc_only`, and `baseline_only`. A sidecar manifest fingerprints generation/root/baseline inputs, and pre-manifest outputs require one explicit `ADOPT_LEGACY_ARTIFACTS=true` registration before resume.
- Strict shared seed-42 image splits are 3200/400/400, mutually exclusive across train/val/test and across models. Train fits only, val selects checkpoints/thresholds, and test is final evaluation only; replaced legacy splits are backed up.
- Validation: full compile, 54 unit tests, four-mode dry-runs, five real wrapper smokes, three dynamic-model maximum-image smokes, and final read-only integration audit all passed. Known protocol limits are explicit: DHCP is the cross-backbone object-step/fixed-grid adaptation, and labels use local CHAIR object spans without the paper's GPT-4o semantic review.

## 2026-07-15 COCO100 softmax-relative-vll / legacy_prob 三模型实验
- 新增 `gate_comparison` / `gate_comparison_vv` 匹配对比路径。三种 target gate 共享 VV source、exact EMD、top-k union=64 与 `sqrt((1-cos(h_pre))/2)` ground cost；target cosine 统一使用 h_pre、top-k=32，relative logit source 为 h_mid。
- `relative-vll` 对各视觉 token 的目标词 raw logit 跨视觉 token 做 median/MAD；`softmax-relative-vll` 先在每个视觉 token 的 vocabulary 维 softmax、取目标词概率，再跨视觉 token 做 median/MAD；两者均使用 `1.4826*MAD` 和 sigmoid。`legacy_prob` 使用相同目标词概率直接乘 attention，不做 MAD/sigmoid。产物元数据明确记录 `softmax_axis=vocabulary`、`mad_axis=visual_tokens`；“gauss”只描述前两种 MAD gate，legacy 本身没有 Gaussian/MAD。
- 新增配置 `configs/model_configs_coco100_gate_comparison.yaml`、入口 `scripts/run_coco100_gate_comparison.py`、单模型与三模型绘图脚本。runner 支持 `prepare/extract/probe_split/train/plot/summarize`、模型子集、多 devices 和 resume。
- 从三模型 `outputs/{model}/COCO4000/labeling.json` 完全一致的 ID 顺序取 canonical 前 100 张，generation 按该顺序重排，seed=42 共享 80/10/10 split。提取完成：LLaVA 696 rows（H/N=`97/599`，99 张有效图，32 层）；InternVL 899 rows（`71/828`，99 张，32 层）；Qwen 424 rows（`32/392`，92 张，28 层）。少数图片没有可用 object-token span，因此不会产生 feature row，并非漏跑。
- 风险曲线以 `H-N = hallucination mean - non-hallucination mean` 统计。全层 signed avg：LLaVA relative/softmax/legacy=`+0.01913/+0.01639/+0.01446`；InternVL=`-0.00110/+0.00146/-0.00518`；Qwen=`+0.01291/+0.01639/-0.00453`。对应 mean absolute gap：LLaVA=`0.02223/0.02249/0.02117`，InternVL=`0.00883/0.00745/0.01104`，Qwen=`0.02059/0.02202/0.02785`。softmax+MAD 在 Qwen 的 signed 分离略好；LLaVA 与 raw-logit MAD 接近但 signed avg 稍低；InternVL 三者分离均弱。Qwen/InternVL 的 legacy 平均方向反转；鉴于 COCO100 且 Qwen 只有 32 个 hallucination token，只解释为小样本趋势。
- 三模型 relative 与 softmax-relative 的 H-N gap 曲线高度相关：LLaVA/InternVL/Qwen Pearson `r=0.9525/0.8997/0.9337`。三模型三方法的 h_pre target-cosine H-N 全层均值均为负：LLaVA=`-0.05375/-0.05446/-0.06388`，InternVL=`-0.02360/-0.02544/-0.02185`，Qwen=`-0.03501/-0.03369/-0.02857`。
- 已补做标准 Torch MLP probe：三种 gate 各训练 risk-only、对应 h_pre target-cosine-only、同 gate risk+cosine concat，共 9 个 feature sets；架构 `128/64/32`、dropout=0.3、batch=256、100 epochs、positive class=real，probe seeds=`42/43/44`，按最低 validation loss 选 checkpoint。`scripts/train_feature_sets.py` 新增 6 个不与旧 relative-VLL alias 混淆的 `gate_*` blocks。
- 原 seed=42 图像 split 的 Qwen validation 为 H/N=`0/38`，会触发训练器 train-as-validation fallback，因此没有用于 probe。另存三模型共享、同一 canonical 100 张、严格 80/10/10 的 `probe_image_splits.json`；使用 canonical 顺序的首个三模型 val/test 均双类随机 seed=1。probe train/val/test H/N：LLaVA=`81/484, 7/67, 9/48`，InternVL=`53/648, 11/108, 7/72`，Qwen=`27/303, 3/47, 2/42`，训练日志未出现 fallback、NaN 或单类告警。
- 三 seed test AUC 最佳：LLaVA 为 softmax-relative hpre-cosine-only `0.8279+/-0.0244`；InternVL 为 relative risk+cosine `0.8783+/-0.0160`；Qwen 为 legacy hpre-cosine-only `0.7262+/-0.1146`，softmax risk+cosine 为 `0.7103+/-0.0780`。risk-only AUC（relative/softmax/legacy）：LLaVA=`0.4915/0.5802/0.5633`，InternVL=`0.6792/0.5589/0.5205`，Qwen=`0.4921/0.5476/0.6349`。LLaVA/InternVL 的主要信号来自 h_pre cosine 或与其拼接；risk-only 整体较弱。PR/RC/F1/AUPR 以多数类 non-hallucination 为正类，不能用高 F1 声称幻觉检测好，主比较只看 AUC。Qwen test 仅 2 个幻觉 token且来自同一张图；三个 seed 共用同一 split，std 只反映训练初始化而非抽样不确定性；汇总中的 test-AUC 排名只能作为 smoke/exploratory 结果，不能当无偏模型选择或最终性能结论。
- 主产物位于 `outputs/{model}/COCO100-gate-comparison/`，单模型 PNG/PDF/CSV 在各自 `results/`，probe seed 产物在 `torchmlp-seed3-probe-split1-811/`；合并曲线为 `outputs/coco100-gate-comparison-summary/three_model_coco100_gate_risk_by_label.{png,pdf,csv}`，probe 汇总为同目录 `three_model_seed3_probe_split1_summary.{md,csv,json}`。首版误将 softmax 用在视觉位置维，已归档到三模型的 `COCO100-position-softmax-gate-comparison-archived/` 及 `coco100-position-softmax-gate-comparison-summary-archived/`，不作为主结果。
- 验证：`py_compile` 覆盖修改模块、三个 wrapper、四个相关脚本和测试；`/opt/conda/private/envs/vicr/bin/python -m unittest discover -v tests` 共 12 项通过；三个 `features.pkl` 的 22 字段、label、32/32/28 层、六组 risk/cosine key、axis/cost/top-k 元数据及所有数值 finite 检查通过；9 个 probe run 均含 9 个 feature sets，PR/RC/F1/Acc/AUC/AUPR 全部 finite，模型/history/config artifacts 和 seed/维度元数据检查通过；`git diff --check` 通过；四张 PNG 已目视检查。中途首次 `py_compile` 因 `_link_input` 被补丁误插入列表推导式报 `SyntaxError: invalid syntax`，修正函数位置后相同命令及 12 项测试通过。特征提取主进程均 exit 0；多进程退出时出现一次 `resource_tracker` leaked semaphore warning，但合并产物和完整性校验均正常。系统默认 `python` 的配置加载检查曾因缺少 PyYAML 报 `ModuleNotFoundError: No module named 'yaml'`，未修改环境，随后使用已有 Conda 环境重跑通过。

## 2026-07-15 合并 `-zccocochair` 到 `cocochair`
- 合并前先将当前 Qwen3 caption、Attention-TK32 JS/KL 与空间热力图工作提交为 `475e647` 并确认已推送到远端 `cocochair`；随后合并远端 `-zccocochair` 的 `ba9b74c`。
- 合并仅在 `docs/CURRENT_TASK.md` 与 `scripts/train_feature_sets.py` 出现内容冲突；交接文档两侧章节全部保留。
- `scripts/train_feature_sets.py` 同时保留两套新增能力：本地的 VV/VP Attention-TK32 JS、`KL(attention||source)`、`KL(source||attention)` 动态特征，以及远端十种 cost-variant aliases 和通用 `risk*hprecosine` 逐层 Hadamard product 语法。
- Qwen3 独立 wrapper、`qwen3_vl_8b` 注册/配置和 7 个可视化脚本均保留；远端新增的 COCO cost-variant、exact EMD 并行/Sinkhorn 实验、POPE/CLEVR QA pipeline、probe、服务器配置和测试也全部纳入。
- 依赖约束按合并后的实际能力整合：保留远端新增 POT/timm/einops 等依赖，但将 PyTorch/Torchvision 上限放宽以兼容当前 RTX 5090 环境，并将 Transformers 下限提高到 Qwen3-VL 所需且已验证的 `4.57.0`；当前环境为 torch 2.8.0、torchvision 0.23.0、Transformers 4.57.6。
- 验证：所有合并后新增/修改 Python 文件通过 `py_compile`；联合 smoke 验证三项 Attention-TK32 divergence、十种 risk alias、`risk-geo*hprecosine` 数值和 shape 均正确。
- 测试环境说明：`/opt/conda/private/envs/vicr/bin/python -m pytest -q tests` 因当前环境没有安装 pytest，报 `No module named pytest`；未擅自修改环境。改用 `unittest discover` 跑完 8 个 unittest，并以临时目录模拟 `tmp_path` 执行其余 11 个 pytest 风格测试函数，总计 19 项全部通过。

## 2026-07-10 COCO4000-all Qwen/InternVL VV 与 VP geo risk 曲线
- 用户要求绘制 Qwen2.5-VL-7B 与 InternVL2.5-8B 的 `risk_geo_raw`（V source / V target）和 `risk_visual_prompt_relative_vll_cost_geo`（VP source / VP target），按 hallucination/non-hallucination 分组。
- 数据来自各模型 `COCO4000-all/features.part{0,1}.pkl`；使用分片流式累计 mean、sample SEM，避免一次性展开 15.9/29.5GB 的合并 pickle。
- 输出：
  - `outputs/qwen2_5_vl_7b/COCO4000-all/results/qwen2_5_vl_7b_coco4000_all_vv_vs_vp_geo_risk_by_label.{png,pdf,csv}`
  - `outputs/internvl_2_5_8b/COCO4000-all/results/internvl_2_5_8b_coco4000_all_vv_vs_vp_geo_risk_by_label.{png,pdf,csv}`
  - 合并图与摘要：`outputs/coco4000_all_vv_vp_geo_risk_comparison/qwen_internvl_coco4000_all_vv_vs_vp_geo_risk_by_label.{png,pdf,csv,md,json}`
- 曲线摘要（H-N 为 hallucination mean - non-hallucination mean）：
  - Qwen VV：hall/non 全层均值 `0.260230/0.243665`，H-N avg `+0.016565`，峰值 L15 `+0.055825`。
  - Qwen VP：`0.342379/0.348071`，H-N avg `-0.005692`，峰值 L1 `-0.050949`。
  - InternVL VV：`0.257980/0.252527`，H-N avg `+0.005453`，峰值 L13 `+0.023806`。
  - InternVL VP：`0.527449/0.520299`，H-N avg `+0.007150`，峰值 L4 `+0.030312`。
- 验证：三个 CSV 均检查为非空且所有统计值 finite；合并 PNG 以及 InternVL 单模型上下两排图已目视检查，无空白、裁切或图例重叠。
- 中途失败记录：首次直接运行 `scripts/plot_layerwise_feature_comparison.py` 读取 Qwen `COCO4000-all/features.pkl` 时，进程在整文件反序列化阶段被终止且未生成文件；原因是该脚本会一次性展开大 pickle。随后改为读取 part 文件的流式统计，Qwen/InternVL 均成功完成。

## 2026-07-13 COCO4000 risk * hprecosine 逐层乘积实验
- `scripts/train_feature_sets.py` 新增通用 `*` feature-set 语义：两个别名对应的逐层向量做 Hadamard product，输入维度保持 LLaVA/InternVL 32、Qwen 28；`+` 仍表示向量拼接。已有显式乘积别名优先解析，保持历史行为。
- 新增单元测试验证 `risk-geo*hprecosine` 的逐层数值、输出维度和 label，且历史 `ffn_fad*risk_geo_raw` 别名不变；`tests.test_cost_variants` 共 6 项通过。
- 直接复用 COCO4000 严格 8:1:1 features/split，在原 seed 结果中追加 10 个 product feature sets；三模型 seeds=`42/43/44` 均完整，每个结果 JSON 现含 31 个 feature sets。
- LLaVA product best 为 `risk-rawAttention-hpre*hprecosine`：AUC/F1=`0.9094+/-0.0003 / 0.9308+/-0.0009`。它相对 `hprecosine` AUC 仅 `+0.0002`，比对应 concat 低 `0.0142`；其余 8 个非 raw product 低于 hprecosine `0.0081~0.0097`。
- InternVL product best 为 `risk-rawAttention-hpre*hprecosine`：AUC/F1=`0.8074+/-0.0025 / 0.9503+/-0.0009`，相对 hprecosine AUC `+0.0064`，但比 concat 低 `0.0325`。raw-attention product 略有互补，其他乘积大多持平或下降。
- Qwen product best 为 `risk-sqrt-hpre*hprecosine`：AUC/F1=`0.8644+/-0.0037 / 0.9560+/-0.0006`，相对 hprecosine AUC `-0.0170`，比 concat 低 `0.0374`；所有 10 个 product 都低于 hprecosine 与对应 concat。
- 总结：逐层乘积会把 risk 与 hprecosine 压成单一通道，无法让 MLP 分别学习两者的权重与交互，整体明显弱于 `risk+hprecosine` 拼接。后续主结果保留 concat；product 仅作为负向 ablation。

## 2026-07-13 COCO4000 risk + hprecosine 组合训练结果
- 基于严格共享 8:1:1 split 与 Torch MLP seeds=`42/43/44`，汇总 10 个 `risk+hprecosine` feature sets，并与 `hprecosine` 及对应 risk-only 比较。
- 三模型所有 10 个组合的 mean AUC 都高于 `hprecosine` 单特征；相对提升范围：LLaVA `+0.0109~+0.0154`，InternVL `+0.0355~+0.0433`，Qwen `+0.0150~+0.0206`。F1 提升较小，主要因为 real 类占比高且单特征 recall 已接近饱和。
- LLaVA best-by-AUC 为 `risk-rawAttention-hmid+hprecosine`：PR/RC/F1/Acc/AUC/AUPR=`0.9271/0.9464/0.9367/0.8926/0.9246/0.9846`，AUC 相对 `hprecosine=0.9092` 提升 `+0.0154`，相对 risk-only 提升 `+0.0420`。best-by-F1 为 `gauss-risk-geo+hprecosine`，F1=`0.9373`。
- InternVL best 为 `risk-sqrt-hmid+hprecosine`：PR/RC/F1/Acc/AUC/AUPR=`0.9142/0.9926/0.9518/0.9090/0.8443/0.9787`，AUC 相对 `hprecosine=0.8010` 提升 `+0.0433`，相对 risk-only 提升 `+0.0820`。InternVL 的单条 risk 曲线最弱，但与 hprecosine 的互补收益最大。
- Qwen best-by-AUC 为 `risk-sqrt-hmid+hprecosine`：PR/RC/F1/Acc/AUC/AUPR=`0.9274/0.9932/0.9592/0.9226/0.9019/0.9884`，AUC 相对 `hprecosine=0.8813` 提升 `+0.0206`，相对 risk-only 提升 `+0.0505`。best-by-F1 为 `risk-rawAttention-hmid+hprecosine`，F1=`0.9605`。
- Gaussian 组合没有形成稳定优势：三个模型的 best-by-AUC 均为非 Gaussian 组合。hmid/hpre 组合差异普遍较小，但 LLaVA 更偏 raw-attention，InternVL/Qwen 更偏 sqrt-hmid。

## 2026-07-13 COCO4000 cost-variant 三模型逐层曲线
- 三模型 `coco4000-costvariant/features.pkl` 与严格 8:1:1 三 seed 训练均已完成；当前无提取或训练进程。object-token 行数按曲线 label 统计为：LLaVA hall/non=`3721/21766`，InternVL=`2612/30918`，Qwen=`1107/13602`。
- `scripts/plot_coco500_costvariant.py` 新增兼容参数 `--dataset-label`，默认仍为 `COCO500`；本轮使用 `--dataset-label COCO4000`，生成正确的 COCO4000 标题和文件名。
- 曲线输出：`outputs/{model}/coco4000-costvariant/results/{model}_coco4000_costvariant_layerwise_by_label.{csv,png,pdf}`。
- LLaVA 分离最强：10 条 risk 的 mean absolute hall/non gap 约 `0.0272~0.0427`，geo/cosine-hpre 峰值在第 30 层约 `+0.109`，raw-attention 峰值在第 22 层约 `+0.096`；hall risk 整体高于 non-hall。`hprecosine` 全层方向一致，signed/absolute average gap=`-0.0588`，峰值第 27 层 `-0.1038`。
- Qwen 分离居中：risk mean absolute gap 约 `0.0190~0.0223`，主要峰值在第 15 层约 `+0.049~+0.056`，但部分层发生交叉；`hprecosine` 全层方向一致，average gap=`-0.0384`，峰值第 9 层 `-0.0639`。
- InternVL 仍最弱：risk mean absolute gap 仅 `0.0080~0.0098`，主要峰值在第 13 层约 `+0.022~+0.026`，标准差带高度重叠且多条曲线换向；`hprecosine` 相对更稳定，average gap=`-0.0270`，峰值第 8 层 `-0.0455`。
- Gaussian 校准与普通 relative-VLL 风险曲线几乎重合；hmid/hpre cost 的差异也很小。sqrt 主要压缩数值尺度，没有明显增强标准化分离。模型间总体排序为 LLaVA > Qwen > InternVL。
- 3-seed risk-only best AUC：LLaVA `gauss-risk-sqrt-hpre=0.8847+/-0.0036`，Qwen `risk-rawAttention-hpre=0.8685+/-0.0022`，InternVL `gauss-risk-sqrt-hmid=0.7747+/-0.0016`；三模型 `hprecosine` AUC 分别为 `0.9092/0.8010/0.8813`。

## 2026-07-12 恢复 InternVL legacy 单块模式
- 按用户要求将 `outputs/internvl_2_5_8b/coco500-costvariant-legacy-single-tile` 恢复为当前 `outputs/internvl_2_5_8b/coco500-costvariant`；动态切图全量结果完整保留在 `outputs/internvl_2_5_8b/coco500-costvariant-dynamic-tiles2`。
- `models/internvl_wrapper.py` 恢复 legacy 行为：所有图片缩放为单张 `448x448`、固定 256 个 `<IMG_CONTEXT>` tokens，并恢复手工英文 system/user prompt 拼接。保留与数值结果无关的 compact cost-variant 显存优化。
- `configs/model_configs_coco500_costvariant.yaml` 移除动态切图参数；后续该实验默认继续走 legacy 单块路径。动态切图实现与结果已归档，不再作为当前三模型 summary 的 InternVL 数据。
- legacy restore smoke 使用同一张宽图 `153976` 成功生成 8 条记录，support 固定为 256，四类 raw VV tensor shape 均为 `(32, 256)`；输出位于 `outputs/smoke-internvl-legacy-restore/features.pkl`。
- 动态切图较慢的原因：InternVL 每增加一个 tile 都会额外执行一次完整 `448x448` ViT 编码，并把 decoder visual span 从 256 增加到 512；当前 DGST capture 必须使用 eager attention，decoder 的 attention/capture 显存与计算量随序列长度近似二次增长。Qwen 使用原生 processor 的 patch grid 和 spatial merge 生成可变视觉 token，不是复制若干完整 448 方块，且 decoder 只有 28 层而 InternVL 为 32 层，因此两者的动态分辨率代价不能直接等同。

## 2026-07-12 InternVL wrapper 动态切图与原生模板修复
- 修复 `models/internvl_wrapper.py`：不再把所有图片强制缩放成单张 `448x448`；改为按 InternVL 官方动态宽高比分块逻辑预处理，并从实际连续 `<IMG_CONTEXT>` token 定位 visual span。
- prompt 改为复用模型自身的 `internvl2_5` conversation template、system message 和 role，而不再手工拼接与模型原生模板不同的英文 system prompt。
- `configs/model_configs_coco500_costvariant.yaml` 的 InternVL 配置启用 dynamic image size，并在当前 RTX 4090 + eager attention 环境限制为最多 2 个 local tiles、关闭额外 thumbnail；wrapper 本身仍支持配置 thumbnail。该限制将宽图 support 从旧实现固定的 256 tokens 扩展到 512 tokens，同时避免 3 tiles 的显存溢出。
- cost-variant 专用路径跳过最终不会保存、也不参与 relative-logit gate 的 legacy full-vocabulary support/prompt probability 计算；普通 DGST-T 模式保持原行为。这样消除了动态切图 smoke 中的额外 full-vocabulary GEMM 显存峰值。
- 新增 `tests/test_internvl_wrapper.py`，验证宽图动态分块和方图单块行为；与 cost-variant 数值测试合计 7 项通过。
- 真实 smoke：COCO image `153976`（`640x223`，宽高比约 `2.87`）成功提取 8 条 object-token records；support positions 为 512，support-attention/source-dist/普通 gate/Gaussian gate shape 均为 `(32, 512)`，所有数值有限，记录仅含 27 个白名单字段。输出位于 `outputs/smoke-internvl-dynamic-costvariant/features.pkl`。
- 旧的固定单块全量结果已完整归档到 `outputs/internvl_2_5_8b/coco500-costvariant-legacy-single-tile`，不再作为三模型最终汇总中的 InternVL 结果。

## 2026-07-12 InternVL 动态切图 COCO500 全量重跑
- 使用两张 RTX 4090 完成 500 张图的动态切图特征提取，耗时约 32.5 分钟；输出 `outputs/internvl_2_5_8b/coco500-costvariant/features.pkl` 共 4338 rows、约 778 MiB，label 分布为 real/hall=`3999/339`，覆盖 496 张有 object-token row 的图片。
- 动态 support 分布：按图片计 312 张为 256 tokens、184 张为 512 tokens；按 object-token row 计分别为 2649/1689。旧 wrapper 的 496 张图片全部固定为 256 tokens。
- full validator 通过 27 字段白名单、32 层 shape、有限值和共享严格 split 检查；split SHA-256 仍为 `fb9d1ef92873396444ab186d0c3c8c0c7ecb8718b0016cd145e7ca44bc8e0e2d`。提取过程未发生 OOM，退出时仍有一次 multiprocessing leaked semaphore warning，但主进程退出码为 0 且合并文件完整。
- 全新完成 Torch MLP seeds=`42/43/44` × 21 feature sets，并重新生成 InternVL CSV/PNG/PDF 曲线及三模型统一 summary。
- 修复后 InternVL 最佳 AUC 为 `gauss-risk-sqrt-hpre+hprecosine`：AUC=`0.8916+/-0.0123`、F1=`0.9612+/-0.0026`；旧单块结果为 AUC=`0.8736+/-0.0071`、F1=`0.9626+/-0.0013`，即 AUC `+0.0180`，F1 基本持平。
- `hprecosine` 单特征 AUC 从约 `0.8450` 升至 `0.8630+/-0.0034`。修复后 risk-only 最好为 `gauss-risk-cosine-hpre`，AUC=`0.7596+/-0.0167`；旧结果的 risk-only 最好为 `risk-rawAttention-hmid`，AUC=`0.7379+/-0.0308`。
- 逐层均值曲线仍未明显拉开：10 条 risk 的修复后 mean absolute hall/non gap 约为 `0.0072~0.0079`，旧结果约为 `0.0081~0.0087`；`hprecosine` 为 `0.02246`，旧结果为 `0.02346`。因此改进主要来自 MLP 对完整 32 层联合形状的利用，而不是某个单层均值差显著增大。
- 新结果 10 条 risk 的逐值相关性仍很高：pairwise correlation median=`0.9725`、max=`0.9977`。这解释了 risk 变体曲线仍相似；wrapper 修复改善了输入和多层联合 AUC，但没有消除这些 cost/target 变体之间的冗余。
- 最终输出：InternVL 曲线位于 `outputs/internvl_2_5_8b/coco500-costvariant/results/`；三模型汇总已恢复为 `outputs/coco500-costvariant-summary/three_model_seed3_summary.{md,csv,json}`。

## 2026-07-12 COCO500 VV cost-variant 三模型实验
- 新增专用模式 `target_gate_mode: cost_variants`，仅在 VV/visual support 下启用；普通 relative-VLL、dual 和历史配置保持原行为。
- 同一次 feature extraction 同时计算 10 条 risk：
  - 普通 relative-VLL target：`risk-geo`、`risk-cosine-hpre`、`risk-sqrt-hmid`、`risk-sqrt-hpre`。
  - raw support-attention target：`risk-rawAttention-hmid`、`risk-rawAttention-hpre`。
  - Gaussian-consistent MAD target：`gauss-risk-geo`、`gauss-risk-cosine-hpre`、`gauss-risk-sqrt-hmid`、`gauss-risk-sqrt-hpre`。
- Gaussian 分支使用 `z=(logit-median)/(1.4826*MAD+eps)`、`gate=sigmoid(z)`、`target=normalize(attention*gate)`；raw-attention 分支只对 support attention 做非负归一化。
- cost 定义：hmid/hpre 分别使用对应 support state 的 `1-cosine(i,j)`；sqrt 版本使用 `sqrt(clamp((1-cosine(i,j))/2,min=0))`。所有分支沿用 source/target top-k union，`K=64`，source 固定为 legacy FFN distribution。
- 新增精简保存 profile `costvariant_vv`：每条记录严格保存 27 个字段，包括必要 identity/label、10 条 risk、非 Gaussian `hprecosine`、support positions、support-attention、source-dist、普通/Gaussian semantic gate 和公式参数；不保存 VP、capped、entropy、source ablation 或 FFN injection 字段。
- 新增文件：
  - 配置：`configs/model_configs_coco500_costvariant.yaml`。
  - 一键流程：`scripts/run_coco500_costvariant.py`，覆盖 prepare/label/extract/validate/train/plot/summarize，三个 seed 在两张 GPU 上并行。
  - 曲线：`scripts/plot_coco500_costvariant.py`。
  - 单元测试：`tests/test_cost_variants.py`。
- `requirements.txt` 新增 `matplotlib>=3.7.0`，并已安装到 Conda 环境 `/opt/conda/envs/td`；`pip check` 无冲突。
- 三模型共享严格互斥的 COCO500 split：train/val/test=`400/50/50`，三个 `image_splits.json` SHA-256 均为 `fb9d1ef92873396444ab186d0c3c8c0c7ecb8718b0016cd145e7ca44bc8e0e2d`。
- full feature extraction 已完成并通过字段白名单、层数、shape、有限值检查：
  - LLaVA：3311 rows，32 层，`features.pkl` 约 952 MiB。
  - InternVL：4338 rows，32 层，`features.pkl` 约 565 MiB。
  - Qwen：1958 rows，28 层，`features.pkl` 约 301 MiB。
  - 路径统一为 `outputs/{model}/coco500-costvariant/`。
- 21 个 feature sets 均完成 Torch MLP seeds=`42/43/44`：hprecosine 单特征、10 个 risk 单特征、10 个 `risk+hprecosine` 组合；batch=256、epochs=100、hidden=`128/64/32`、positive class=`real`。
- 三 seed mean+/-std 的 best AUC：
  - LLaVA overall：`hprecosine`，F1=`0.9351+/-0.0023`，AUC=`0.8949+/-0.0079`；risk-only best 为 `risk-rawAttention-hpre`，AUC=`0.8191+/-0.0082`。
  - InternVL overall：`gauss-risk-sqrt-hpre+hprecosine`，F1=`0.9626+/-0.0013`，AUC=`0.8736+/-0.0071`；risk-only best 为 `risk-rawAttention-hmid`，AUC=`0.7379+/-0.0308`。
  - Qwen overall：`risk-rawAttention-hpre+hprecosine`，F1=`0.9613+/-0.0014`，AUC=`0.8980+/-0.0293`；risk-only best 为 `risk-cosine-hpre`，AUC=`0.8740+/-0.0206`。
- 汇总输出：`outputs/coco500-costvariant-summary/three_model_seed3_summary.{md,csv,json}`；每模型曲线输出位于各自 `coco500-costvariant/results/*_layerwise_by_label.{csv,png,pdf}`。
- 验证：`python -m unittest -v tests.test_cost_variants` 共 5 项通过；三模型 1-image smoke 分别验证 LLaVA `32x576`、InternVL `32x256`、Qwen `28x299` 的四类 raw VV tensor shape；三模型 full 结果均确认 `3 seeds x 21 feature sets` 完整。

## 2026-07-11 fj01 服务器数据、模型、环境与三模型 smoke test
- 已将 COCO 2014 解压到当前服务器持久化目录：
  - 图片：`/root/rivermind-data/dataset/coco/val2014`，共 40504 张 JPG。
  - 标注：`/root/rivermind-data/dataset/coco/annotations/instances_val2014.json`。
  - captions：`/root/rivermind-data/dataset/coco/annotations/captions_val2014.json`。
- 当前三模型实际目录：
  - LLaVA-1.5-7B：`/root/rivermind-fs/xiongbo/models/llava-1.5-7b-hf`，约 14G。
  - Qwen2.5-VL-7B-Instruct：`/root/rivermind-fs/xiongbo/models/Qwen2.5-VL-7B-Instruct`，约 16G。
  - InternVL2.5-8B：`/root/rivermind-fs/xiongbo/models/InternVL2_5-8B`，约 16G；本轮通过 Hugging Face 镜像新下载。
  - 原有 `InternVL3-8B` 保留，未覆盖；当前项目 wrapper 与主实验仍使用 InternVL2.5-8B。
- 新增当前服务器专用配置 `configs/model_configs_server_fj01.yaml`：
  - 从 `model_configs_coco4000_all.yaml` 派生，不修改历史实验 YAML。
  - 模型路径统一指向 `/root/rivermind-fs/xiongbo/models`。
  - COCO 路径统一指向 `/root/rivermind-data/dataset/coco`。
- 新建运行环境 `/root/rivermind-data/envs/token-detector`，验证版本为 PyTorch 2.5.1+cu124、Transformers 4.51.3，CUDA 可用。
- 补齐并写回 `requirements.txt` 的实际依赖：
  - 限定 PyTorch 2.5 / torchvision 0.20 和 Transformers 4.51.x~4.x，避免宽松依赖安装到不兼容的新主版本。
  - 新增 POT（EMD solver）、sentencepiece、timm、einops。
- NLTK CHAIR 资源安装到 `/root/nltk_data`：punkt、averaged_perceptron_tagger、wordnet、omw-1.4。
- 三模型各用 1 张 COCO 图片完成生成、CHAIR 标注和 DGST-T 特征抽取：
  - LLaVA：`outputs/smoke-fj01/llava_1_5_7b/features.pkl`，9 个 object token。
  - Qwen：`outputs/smoke-fj01/qwen2_5_vl_7b/features.pkl`，7 个 object token。
  - InternVL：`outputs/smoke-fj01/internvl_2_5_8b/features.pkl`，5 个 object token。
- 本轮失败与修复：
  - CHAIR 首次运行缺 NLTK 数据；服务器访问 NLTK/GitHub raw 超时，改为本机下载后上传。
  - 初始宽松 requirements 安装到 PyTorch 2.13/Transformers 5.13，LLaVA 生成触发 Triton 编译失败；收敛到 PyTorch 2.5.1/Transformers 4.51.3 后通过。
  - Transformers 4.45.2 不包含 Qwen2.5-VL 类；升级到 4.51.3 后通过。
  - DGST-T EMD 首次抽取缺 POT；安装 POT 后通过。
  - InternVL 首次加载缺 sentencepiece；补齐 sentencepiece/timm/einops 后通过。
- 验证命令：
  - `find /root/rivermind-data/dataset/coco/val2014 -maxdepth 1 -type f -name '*.jpg' | awk 'END {print NR}'` -> 40504。
  - 三模型分别运行 `coco-labeling/label_coco.py --num-images 1`。
  - 三模型分别运行 `scripts/extract_features.py --num-images 1`。

## 2026-07-10 COCO4000-all 主实验三 seed 与严格 8:1:1 准备
- 用户要求在三模型 `COCO4000-all` 的 42 个主实验 feature set 上运行 torch MLP，seeds=`42/43/44`、batch size=`256`、positive class=`real`。
- 检查发现原 `COCO4000-all/image_splits.json` 不是严格 8:1:1：train/val/test 数量为 `3600/400/400`，其中 val 与 test 是完全相同的 400 张图；该结果记为 9:1，并计划在各模型完成后归档到 `COCO4000-all/torchmlp-main-seed3-91/`。
- 已准备严格 8:1:1 目录：
  - `outputs/{model}/COCO4000-all/torchmlp-main-seed3-811/image_splits.json`
  - `outputs/{model}/COCO4000-all/torchmlp-main-seed3-811/seed{42,43,44}/`
  - 每个 seed 目录的 `features.pkl` 只读链接到 `COCO4000-all/features.pkl`，`image_splits.json` 链接到同实验目录的严格 split。
- 严格 split 直接复用已验证的 `COCO4000-8-2/image_splits.json`：train/val/test=`3200/400/400`，三者互斥；三模型 split 内容 SHA-256 一致。
- 新增 `scripts/summarize_torch_probe_seed_runs.py`：
  - 严格检查每个模型的三 seed 结果文件、共同 feature set、指标完整性和 seed metadata。
  - 按模型和 feature set 汇总 PR/RC/F1/Acc/AUC/AUPR 的 population mean+/-std。
  - 输出 Markdown、CSV、JSON，并在 Markdown 中给出各模型 best-by-AUC 及完整排名。
- 新增 `scripts/run_coco4000_all_main_811_torchmlp.sh`：传入单个模型名后，按已准备的严格 split 顺序运行 seeds 42/43/44 的 42 个主实验 feature set，并在完成后自动生成该模型三 seed 汇总。
- 新增 `scripts/archive_coco4000_all_main_91_runs.sh`：只有确认旧 9:1 的三个 seed 均包含 42 个完整结果且 seed metadata 正确时，才移动到 `torchmlp-main-seed3-91/seed{42,43,44}`；移动后修正 features/split 链接并自动生成 9:1 汇总。
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile scripts/summarize_torch_probe_seed_runs.py`
  - 使用临时三 seed JSON 端到端生成 `summary.md/.csv/.json`，检查通过。
  - 读取三模型严格 split，确认均为 `3200/400/400`、无交集、union=4000，且内容哈希一致。
  - `bash -n scripts/run_coco4000_all_main_811_torchmlp.sh scripts/archive_coco4000_all_main_91_runs.sh` 检查通过。
- 当前运行状态：Qwen 原 9:1 主实验 seed 42 已完成，seed 43 在 2026-07-10 本轮检查时仍在运行；未移动活动目录。

## Goal
比较 Qwen2.5-VL-7B 与 InternVL2.5-8B 在 DGST-T `support_scope=visual` 时，幻觉/非幻觉 object token 的逐层 transport risk 曲线。

## Last status
- 2026-07-07 按用户要求新增并完成 Source vs Target(relative) 的 JS/KL 快速实验，覆盖 LLaVA-1.5-7B、InternVL2.5-8B、Qwen2.5-VL-7B 的 COCO500，VV 与 VP 都看，KL 两个方向都看：
  - 新增配置 `configs/model_configs_kl_js.yaml`，使用 `dataset.num_images=500`、`seed=42`、`target_gate_mode=dual`、`relative_cost_mode=geo`、`dgst_t_dual_scope=true`，同一次 feature extraction 同时输出 VV/VP 的 JS、`KL(Target||Source)`、`KL(Source||Target)` 与 cosine。
  - 新增 DGST-T 字段：`dgst_t_js_relative_vll_per_layer`、`dgst_t_kl_target_source_relative_vll_per_layer`、`dgst_t_kl_source_target_relative_vll_per_layer`、`dgst_t_js_visual_prompt_relative_vll_per_layer`、`dgst_t_kl_target_source_visual_prompt_relative_vll_per_layer`、`dgst_t_kl_source_target_visual_prompt_relative_vll_per_layer`。
  - 更新 feature extraction 保存前缀、train aliases、plot aliases，使上述 divergence 字段可以直接用于曲线与 torch probe。
  - 三模型 `COCO500-JS` 特征已完成并检查字段完整性：LLaVA 3311 token rows、InternVL 4301 rows、Qwen 1958 rows；六个 JS/KL 字段与两个 cosine 字段均存在且无非有限值。`COCO500-KL` 复用同一份 features 与 labeling/split。
  - 曲线输出：
    - JS：`outputs/{model}/COCO500-JS/results/{model}_coco500_js_vv_vp_by_label.{png,pdf,csv}`
    - KL：`outputs/{model}/COCO500-KL/results/{model}_coco500_kl_vv_vp_by_label.{png,pdf,csv}`
  - torch MLP/probe 已完成。JS best：LLaVA `js_relative_vll+visualcosine_raw` F1/AUC=0.930/0.907；InternVL `js_relative_vll` F1/AUC=0.943/0.720；Qwen `js_visual_prompt_relative_vll+VP cosine` F1/AUC=0.952/0.895。
  - KL best：LLaVA `kl_source_target_relative_vll+visualcosine_raw` F1/AUC=0.933/0.903；InternVL 多组 F1 约 0.942，best AUC 为 `kl_source_target_visual_prompt_relative_vll+VP cosine` AUC=0.770；Qwen `kl_target_source_relative_vll` 与 `+visualcosine_raw` F1=0.952，AUC 分别为 0.834/0.867，最高 AUC 为 `kl_source_target_visual_prompt_relative_vll+VP cosine` AUC=0.883。
  - 曲线粗结论：InternVL 的 VP-KL 两方向在 hall/non 均值差上最明显；Qwen 的 VP-KL 差异也大但符号反向；LLaVA 的 KL/JS 加 cosine 后通常更稳。COCO500 上类别不均衡较明显，很多结果 recall 接近/等于 1.0，因此本轮结论需要同时看 AUC。
- 已实现 relative VLL target 构造，按用户最新公式使用无 bias raw target logit：`l_m^l = W_U^T v_m^l[w*]`，其中 `v_m^l` 为视觉 token 的 `h_mid`。
- 新增配置项：`target_gate_mode: "legacy_prob" | "relative_vll" | "dual"` 与 `relative_vll_mad_epsilon`。当前 `configs/model_configs.yaml` 默认 `legacy_prob`，`configs/model_configs_visualonly.yaml` 设置为 `dual`。
- relative VLL 新增字段包括：
  - `dgst_t_transport_risk_relative_vll_per_layer`
  - `dgst_t_transport_risk_relative_vll_capped_topmass_085_per_layer`
  - `dgst_t_target_visual_hidden_cosine_relative_vll_per_layer`
  - `dgst_t_target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer`
- 旧 DGST-T 字段保持不变；`dual` 模式会同时输出旧字段和 relative VLL 新字段。`visual_prompt` support 下 median/MAD 仍只在视觉 token 上统计，prompt token 的 relative gate 为 0。
- 用户澄清还要看 visual 和 prompt 同时参与 VLL 的效果；已新增 `visual_prompt_relative_vll` 并保持旧 `relative_vll` 不变。`visual_prompt_relative_vll` 在 visual+prompt support 全体 token 上同时计算 median/MAD、`q_i=sigmoid(z_i)` 和 `T_i=A_i q_i/sum(Aq)`，不是 prompt-only。
- 已新增 `configs/model_configs_visualonly.yaml`，只在 Qwen/InternVL 模型配置中设置 `dgst_t_support_scope: "visual"`，并修正 COCO 数据路径为 `/home/apulis-dev/userdata/DGST/token-grounding-detector/data/coco`。
- 用户要求将 visual-only 的 DGST-T `cost_mode` 改为 `decomposed` 后重跑；已确认并将 `configs/model_configs_visualonly.yaml` 设置为 `cost_mode: "decomposed"`。
- 已完成 InternVL COCO500 visual-only 特征抽取：`outputs/internvl_2_5_8b/COCO500-visualonly/features.pkl`，3715 行，hall=3064，true=651，support_size=256。
- 已完成 Qwen COCO500 visual-only 特征抽取：`outputs/qwen2_5_vl_7b/COCO500-visualonly/features.pkl`，2190 行，hall=1636，true=554。Qwen visual token 数随样本变化，首层 support_size 范围约 63-529，均值约 348.96。
- 已生成逐层 risk 曲线与 CSV：
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly/results/qwen2_5_vl_7b_visualonly_decomposed_risk_layerwise_by_label.{png,pdf,csv}`
  - `outputs/internvl_2_5_8b/COCO500-visualonly/results/internvl_2_5_8b_visualonly_decomposed_risk_layerwise_by_label.{png,pdf,csv}`
  - `outputs/visualonly_support_risk_comparison/qwen_internvl_visualonly_decomposed_risk_layerwise_by_label.{png,pdf}`
- decomposed 初步结果：Qwen visual-only 下 hallucination 与 non-hallucination 的平均 risk 曲线仍高度重合，diff_avg 约 0.0011，最大绝对差在第 12 层约 -0.0889；InternVL visual-only 下早期层差异更明显，diff_avg 约 0.0151，最大绝对差在第 3 层约 0.169。
- 与 direct 版本相比，decomposed 主要使 risk 绝对值整体下移，曲线形状和类别差异趋势基本一致。
- 已生成 capped topmass 0.85 risk 曲线：
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly/results/qwen2_5_vl_7b_visualonly_risk_vs_capped_topmass_085_by_label.{png,pdf,csv}`
  - `outputs/internvl_2_5_8b/COCO500-visualonly/results/internvl_2_5_8b_visualonly_risk_vs_capped_topmass_085_by_label.{png,pdf,csv}`
  - `outputs/visualonly_support_risk_comparison/qwen_internvl_visualonly_capped_topmass_085_risk_layerwise_by_label.{png,pdf,csv}`
- capped risk 结果：Qwen hallucination/non-hallucination 仍高度重合，diff_avg 约 0.0018，最大绝对差在第 7 层约 0.0906；InternVL capped 后整体差异缩小，diff_avg 约 0.0054，最大绝对差在第 25 层约 -0.1106。
- 已在 visual-only + decomposed 特征上补跑 feature-set 分类实验：
  - Qwen `risk` 最佳 xgb：F1=0.901，Acc=0.845，AUC=0.789。
  - Qwen `risk+target_visual_hidden_cosine_capped_topmass_085` 最佳 xgb：F1=0.914，Acc=0.870，AUC=0.861。
  - Qwen `risk_capped_topmass_085+target_visual_hidden_cosine_capped_topmass_085` 最佳 xgb：F1=0.913，Acc=0.870，AUC=0.868。
  - InternVL `risk` 最佳 xgb：F1=0.938，Acc=0.892，AUC=0.862。
  - InternVL `risk+target_visual_hidden_cosine_capped_topmass_085` 最佳 xgb：F1=0.954，Acc=0.922，AUC=0.932。
  - InternVL `risk_capped_topmass_085+target_visual_hidden_cosine_capped_topmass_085` 最佳 AUC 为 rf：F1=0.936，Acc=0.889，AUC=0.924；最佳 F1 为 xgb：F1=0.954，Acc=0.922，AUC=0.921。
- 已补齐并对比原 `COCO500` visual+prompt/direct 与 `COCO500-visualonly` visual-only/decomposed 的同名 feature-set 分类结果：
  - 对比表：`outputs/visualonly_support_risk_comparison/direct_vs_visualonly_decomposed_feature_sets.{md,csv}`。
  - 单独 `risk`：Qwen visual-only/decomposed 对 xgb AUC 从 0.811 降到 0.789，rf/MLP 基本持平；InternVL 三个分类器 AUC 均下降（xgb 0.888->0.862，rf 0.886->0.833，mlp 0.854->0.822）。
  - `risk+target_visual_hidden_cosine_capped_topmass_085`：Qwen xgb AUC 持平 0.861，rf 下降 0.862->0.831，mlp 小升 0.843->0.849；InternVL 明显提升（xgb 0.885->0.932，rf 0.881->0.912，mlp 0.843->0.929）。
  - `risk_capped_topmass_085+target_visual_hidden_cosine_capped_topmass_085`：Qwen xgb/MLP 提升（0.838->0.868，0.833->0.860），rf 下降；InternVL 三个分类器 AUC 均提升（xgb 0.860->0.921，rf 0.892->0.924，mlp 0.829->0.897）。
- Word all-token risk mean 实验效果不佳，已按用户要求从当前代码路径移除：
  - 删除 `target_token_aggregation` 配置开关和 `risk_mean` 聚合逻辑。
  - 删除 `configs/model_configs_visualprompt_relativevll_riskmean.yaml`。
  - 特征抽取恢复为每个 object word 只使用 `token_indices[0]`，保留 relative VLL 与 visual+prompt relative VLL 实现。
- 已新增 final norm relative VLL ablation：
  - 新配置项 `relative_vll_logit_source: "h_mid" | "final_norm_h_mid"`，默认/旧配置显式保持 `"h_mid"`。
  - `"final_norm_h_mid"` 只影响 relative VLL 的 target raw logit 投影输入，即 `l_m^l = W_U^T final_norm(h_mid_m^l)[w*]`，仍不使用 LM-head bias；legacy probability/path 不变。
  - 新增 `configs/model_configs_visualprompt_relativevll_finalnorm.yaml`，输出目录为 `COCO500-visualprompt-relativevll-finalnorm`。
  - Qwen/InternVL full feature extraction 已完成，并确认 `dgst_t_relative_vll_logit_source == "final_norm_h_mid"`。
  - final norm 曲线效果：visual-only relative VLL 分支基本不变；visual+prompt relative VLL risk 的 hall-non 平均差明显变大。Qwen `risk_visual_prompt_relative_vll` diff_avg 约 0.038->0.069，InternVL 约 0.049->0.078。cosine 分支变化很小。
  - 分类效果不是单调提升：Qwen final norm 后单独 visual+prompt risk AUC 下降（`risk_visual_prompt_relative_vll` 最佳 AUC 约 0.902->0.857），但 visual+prompt cosine AUC 上升（约 0.897->0.933）；InternVL 多数 feature set 小幅提升，`risk_visual_prompt_relative_vll` 最佳 AUC 约 0.901->0.917。
- 2026-07-02 新增并完成 Source Delta + Gamma 机制消融的主要结果记录：
  - 新 source `delta_src` 按公式 `softmax((cos(h_out_t^l, s_i^l) - cos(h_mid_t^l, s_i^l)) / tau)` 计算，visual-only 的 support 为 visual tokens，visual-prompt 的 support 为 visual+prompt tokens；visual-prompt 下 prompt token 参与 source softmax，不再置 0。
  - 同一次 feature extraction 同时输出旧 source baseline 与 `delta_src` 下 `gamma=0/0.5/1` 的 target 机制消融；短字段包括 `rvll_delta_g0/g05/g1`、`vp_rvll_delta_g0/g05/g1` 及对应 `_cap085`、`_cos`。
  - 四组 features 已完成：Qwen visual-only 2190 行、Qwen visual-prompt 2190 行、InternVL visual-only 3715 行、InternVL visual-prompt 3715 行。字段检查通过：visual-only 只有 `rvll_delta_*`，visual-prompt 同时有 `rvll_delta_*` 与 `vp_rvll_delta_*`，旧字段保留。
  - 四组逐层图已完成，包含 gamma=0/0.5/1 同图、legacy source gamma=1 vs `delta_src_g1` 同图、capped risk 与 cosine/capped cosine。
  - 曲线结论：新设计的 Offn/delta source 整体不如 legacy source。Qwen visual-only 中 `rvll_delta` risk 的 hall-non `diff_avg` 分别约为 g0=-0.002、g05=-0.010、g1=0.001，而 legacy source g1 约为 0.036；capped 下 legacy 约 0.030，delta g1 约 -0.002。
  - Qwen visual-prompt 中 `vp_rvll_delta` risk 的 `diff_avg` 分别约为 g0=-0.005、g05=0.000、g1=0.012，仍明显低于 legacy source g1 的约 0.038；capped 下 legacy 约 0.037，delta g1 约 0.010。
  - InternVL visual-only 中 `rvll_delta` risk 基本接近 0（g0 约 0.000、g05 约 -0.007、g1 约 0.000），低于 legacy source g1 的约 0.016；capped 下 legacy 约 0.011，delta g1 约 -0.003。
  - InternVL visual-prompt 中 `vp_rvll_delta` 有一定信号，g05/g1 的 `diff_avg` 约 0.036/0.026，但仍低于 legacy source g1 的约 0.049；capped 下 legacy 约 0.048，delta g1 约 0.025。
  - Qwen 分类消融也支持该判断：visual-only 最好 legacy 组合 `risk_relative_vll+target_visual_hidden_cosine_relative_vll` AUC 约 0.942，而最好 delta 组合约 0.910；visual-prompt 最好 legacy/visual-prompt 组合 AUC 约 0.941，而最好 `vp_rvll_delta` 组合约 0.928。
  - 当前结论：`delta_src`/新 Offn source 不适合作为默认替换，最多保留为 negative/mechanism ablation。性能主要仍来自 relative VLL target、attention-guided target 与 cosine 分支，而不是这个 source 重构。
- 2026-07-02 用户决定：后续实验暂时不再使用 deltaOffn/`delta_src`，也不再继续做 `gamma=0/0.5/1` 的 attention 机制消融。已有 Source Delta + Gamma 结果只作为记录/负结果保留，不进入后续主线实验。
- 2026-07-02 新增并完成 relative VLL transport cost 三路重构对比：
  - 新增 `relative_cost_mode` 与 `relative_cost_modes`。默认不设置时沿用旧 `cost_mode`，向后兼容；本轮使用一次 feature extraction 同时输出三种 cost，避免重复抽特征。
  - 三种 cost 为 `geo`、`target_barrier_geo`/`tbar`、`symmetric_barrier_geo`/`sbar`。barrier 参数为 `relative_barrier_lambda=1.0`、`relative_barrier_margin=0.5`、`relative_barrier_max=3.0`。
  - 新增字段包括 `dgst_t_transport_risk_relative_vll_cost_{geo,tbar,sbar}_per_layer`、对应 `_capped_topmass_085_per_layer`，以及 `dgst_t_transport_risk_visual_prompt_relative_vll_cost_{geo,tbar,sbar}_per_layer`、对应 capped 字段。
  - Qwen/InternVL full 3-way features 已完成：
    - `outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll-cost-3way/features.pkl`：2190 行，labels `{0:554,1:1636}`，每条 risk 28 层。
    - `outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll-cost-3way/features.pkl`：3715 行，labels `{0:651,1:3064}`，每条 risk 32 层。
  - 两个模型均已完成 24 组 XGB-only feature-set 对比；总表与曲线汇总位于 `outputs/relative_cost_3way_comparison/`。
  - Qwen 分类结论：`geo` 是最稳默认；`tbar` 在少数 raw/visual+prompt combo 上有小幅收益，最高为 `risk_visual_prompt_relative_vll_cost_tbar_capped_topmass_085+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085`，XGB AUC 约 0.947；`sbar` 虽抬高曲线分离，但分类通常下降。
  - InternVL 分类结论：visual+prompt risk-only 更偏向 `tbar`，`risk_visual_prompt_relative_vll_cost_tbar` AUC 约 0.943；加 cosine 后 `geo/tbar` 非常接近，最高为 `risk_visual_prompt_relative_vll_cost_tbar+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll`，AUC 约 0.960，`geo` 同组约 0.959；`sbar` 基本弱于前两者。
  - 曲线结论：barrier cost 会抬高 risk 绝对值，`sbar` 抬升最大且常给出更大的 hall-non 均值差，但该差异不稳定转化为分类收益；建议主线仍保留 `geo`，把 `tbar` 作为轻量 ablation/候选，不建议默认使用 `sbar`。
- 2026-07-03 补齐 LLaVA-1.5-7B 的 relative VLL cost 3-way 对比：
  - 使用 LLaVA 原 `COCO500` 的 `labeling.json/generations.json/image_splits.json`，一次 feature extraction 同时输出 `geo/tbar/sbar`。
  - 输出目录：`outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-3way/`，features 共 1448 行，labels `{0:757,1:691}`，每条 risk 32 层。
  - 24 组 XGB-only 分类已完成，结果已合并进 `outputs/relative_cost_3way_comparison/relative_cost_3way_summary.md` 与 CSV/图。
  - LLaVA 分类结论：单独 visual risk 中 `geo/tbar` 接近，raw 最好 `geo` AUC 约 0.818，capped 最好 `tbar` AUC 约 0.819；加 visual cosine 后 `tbar` 最强，`risk_relative_vll_cost_tbar+target_visual_hidden_cosine_relative_vll` AUC 约 0.853，是本轮 LLaVA 最高；visual+prompt combo 则 `geo` 略高，AUC 约 0.842。
  - LLaVA 曲线结论：visual branch 的 hall-non diff 为正，`sbar` 抬高均值差最大；visual+prompt branch 的 diff 反向为负（non-hall risk 更高），这和 Qwen/InternVL 不同，也解释了 visual+prompt risk-only 分类不如 visual branch 稳。
- 2026-07-03 补齐 cost 3-way 的 MLP 分类对比：
  - Qwen/InternVL/LLaVA 三个 `COCO500-visualprompt-relativevll-cost-3way` 输出目录均已在同一套 24 个 feature set 上补跑 MLP-only，并和已有 XGB 结果合并。
  - 新增汇总：`outputs/relative_cost_3way_comparison/relative_cost_3way_mlp_summary.md`、`relative_cost_3way_xgb_mlp_all.csv`、`relative_cost_3way_xgb_vs_mlp_best_by_family.csv`、`relative_cost_3way_xgb_mlp_overall_best.csv`。
  - MLP 训练中出现 `ConvergenceWarning`，表示部分网格配置 500 iter 未完全收敛；本轮仍按验证 AUC 选最优配置，作为 classifier ablation 使用。
  - Qwen：MLP 对单独 visual risk 有局部收益，`risk_relative_vll_cost_geo_capped_topmass_085` AUC 约 0.911，高于对应 XGB 约 0.898；但组合项最高 MLP 为 `risk_visual_prompt_relative_vll_cost_sbar+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll`，AUC 约 0.940，仍低于 XGB 总最高约 0.947。
  - InternVL：MLP 总体未超过 XGB；最高为 `risk_relative_vll_cost_sbar+target_visual_hidden_cosine_relative_vll`，AUC 约 0.950，低于 XGB 总最高 `risk_visual_prompt_relative_vll_cost_tbar+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll` 的约 0.960。
  - LLaVA：MLP 最高为 `risk_visual_prompt_relative_vll_cost_sbar_capped_topmass_085+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085`，AUC 约 0.852，几乎追平但略低于 XGB 最高 `risk_relative_vll_cost_tbar+target_visual_hidden_cosine_relative_vll` 的约 0.853。
  - 当前结论：MLP 可作为 sanity check/补充，未改写 cost 3-way 主结论；`sbar` 偶尔在 MLP 组合里变强，但不稳定，默认仍优先 `geo`，`tbar` 作为 ablation/候选。
- 2026-07-03 补齐 cost 3-way 的 torch MLP / torch probe 对比，并和 DGST 原仓库 COCO500 baseline 对照：
  - 当前 3-way 已跑两类 torch probe 组合：visual-only risk + visual-only cosine，以及 VP risk + VP cosine。两类都覆盖 `geo/tbar/sbar` 的 raw 与 cap085，共 12 组/模型；使用 `scripts/train_torch_probe_feature_sets.py` 默认配置：hidden=(128,64,32)、dropout=0.3、batch=16、epochs=100、lr=1e-3、wd=1e-5、seed=42、positive=hallucination。
  - DGST 路径 `/home/apulis-dev/userdata/DGST/token-grounding-detector/outputs/*/COCO500` 没有 VP/cosine 字段，只有老的 `ads_per_layer/cgc_per_layer` 等字段；因此用同一个 torch probe 结构在 `ads+cgc` 训练矩阵上补跑 baseline，结果保存在当前项目 `outputs/relative_cost_3way_comparison/dgst_coco500_torch_probe_ads_cgc/`，未改动 DGST 仓库。
  - 新增汇总：`outputs/relative_cost_3way_comparison/relative_cost_3way_torch_mlp_vs_dgst_coco500.md` 与 `.csv`。
  - Qwen：DGST `ads+cgc` baseline AUC/F1 约 0.906/0.904；visual-only 最优为 `visual_geo_raw+visualcosine_raw`，AUC/F1 约 0.951/0.921，明显强于 VP 最优 `vp_tbar_raw+vpcosine_raw` 的约 0.929/0.929。
  - InternVL：DGST baseline AUC/F1 约 0.934/0.944；visual-only 与 VP 的 best AUC 基本打平，均约 0.952；VP `geo raw` 的 F1 约 0.965，高于 visual-only best-AUC 行的约 0.958。
  - LLaVA：DGST baseline AUC/F1 约 0.803/0.722；visual-only `visual_geo_raw+visualcosine_raw` AUC 约 0.849，略高于 VP best AUC `vp_tbar_cap085+vpcosine_cap085` 的约 0.847；visual-only `visual_geo_cap085+visualcosine_cap085` 给出最高 F1 约 0.789。
  - 当前结论：torch MLP 口径下，三模型 visual-only family 和 VP family 的 best AUC 都高于 DGST 原始 ADS+CGC baseline；Qwen 明显更偏 visual-only，InternVL 两者接近，LLaVA visual-only 略优。
- 2026-07-03 补齐 `risk/cosine/vprisk/vpcosine` 单特征消融：
  - 新增汇总：`outputs/relative_cost_3way_comparison/relative_cost_3way_single_feature_ablation.md`、`relative_cost_3way_single_feature_ablation.csv`、`relative_cost_3way_single_feature_ablation_best_by_family.csv`。
  - 四类 family：visual-only `risk`、visual-only `cosine`、visual+prompt `vprisk`、visual+prompt `vpcosine`。risk/vprisk 覆盖 `geo/tbar/sbar` raw 与 cap085，cosine/vpcosine 覆盖 raw 与 cap085。
  - 已补齐三模型所有单特征的 torch probe；同时为此前缺失的 cosine/vpcosine 单特征补跑 XGB 与 sklearn-MLP。全量结果共 144 行，即 3 模型 × 16 feature sets × 3 classifiers。
  - torch probe 单特征最优：Qwen 为 `vpcosine_cap085`，AUC/F1 约 0.929/0.924；InternVL 为 visual-only `cosine_raw`，AUC/F1 约 0.943/0.945；LLaVA 为 visual-only `cosine_cap085`，AUC/F1 约 0.816/0.748。
  - 单特征结论：cosine-only 通常和 risk-only 持平或更强；Qwen 的单特征最佳是 VP cosine，InternVL/LLaVA 的单特征最佳是 visual-only cosine。risk 单独使用明显弱于 risk+cosine 组合，说明前面组合收益主要来自 cosine 与 risk 的互补，而不是 risk 单项。
- 2026-07-03 补齐 visual-only risk 与 DGST ADS/CGC 的 torch MLP 互补性实验：
  - 当前 3-way risk features 与 DGST 原仓库 `ADS/CGC` features 已按 `(image_id, response_token_idx, token_str, label)` 完整一一对齐；三模型都无重复 key、无缺失 key。
  - 新增汇总：`outputs/relative_cost_3way_comparison/relative_cost_3way_risk_ads_cgc_torch_mlp.md`、`.csv`、`_summary.csv`，原始 torch artifacts 在 `outputs/relative_cost_3way_comparison/risk_ads_cgc_torch_probe/`。
  - 仅使用 torch MLP/probe，feature sets 包括 `ADS`、`CGC`、`ADS+CGC`，以及 6 个 visual-only risk variant 分别加 `ADS` / `CGC`：`geo/tbar/sbar` raw 与 cap085。
  - ADS/CGC 单项消融：三模型都是 `CGC` 明显强于 `ADS`；本次 torch run 中 Qwen/InternVL 的 `CGC` 也强于 `ADS+CGC`。Qwen `CGC` AUC/F1 约 0.945/0.906，InternVL 约 0.947/0.948，LLaVA 约 0.809/0.748。
  - 相对 `ADS+CGC`，best risk mix 三模型都有提升：Qwen `risk_sbar_cap085+CGC` AUC 约 0.943，较 `ADS+CGC` +0.039；InternVL `risk_sbar_raw+CGC` AUC 约 0.966，+0.036；LLaVA `risk_geo_raw+ADS` AUC 约 0.847，+0.054。
  - 若相对最强旧 baseline `CGC`，Qwen best risk mix 约低 0.001 AUC，基本打平；InternVL +0.019 AUC，LLaVA +0.038 AUC。结论：risk 对 InternVL/LLaVA 的 ADS/CGC 有明确补充，对 Qwen 主要是超过 `ADS+CGC` 但没有超过 `CGC-only`。
- 2026-07-03 新增并完成 additive barrier cost 三变体实验：
  - 新增 cost aliases：`target_additive_barrier_geo`/`tadd`、`source_additive_barrier_geo`/`sadd`、`two_end_additive_barrier_geo`/`tsadd`。公式分别为 `lambda_d*d_ij + lambda_t*b_j`、`lambda_d*d_ij + lambda_s*b_i`、`lambda_d*d_ij + lambda_s*b_i + lambda_t*b_j`；旧 `geo/tbar/sbar` 保持不变。
  - 新增配置：`configs/model_configs_visualprompt_relativevll_cost_additive3.yaml`，同一次 feature extraction 输出 `geo/tadd/sadd/tsadd` 四套 visual 与 visual+prompt risk，raw/cap085 都覆盖。
  - 三模型 full features 完成：Qwen 2190 行、InternVL 3715 行、LLaVA 1448 行；新增字段包括 `dgst_t_transport_risk_relative_vll_cost_{tadd,sadd,tsadd}_per_layer`、对应 capped 字段，以及 `risk_visual_prompt_relative_vll_cost_{tadd,sadd,tsadd}` 对应字段。
  - 三模型均完成 32 个 feature set 的 XGB、sklearn MLP、torch probe；汇总目录：`outputs/relative_cost_additive3_comparison/`，包括 `relative_cost_additive3_summary.md`、`relative_cost_additive3_xgb_mlp_all.csv`、`relative_cost_additive3_torch_mlp_all.csv`、`relative_cost_additive3_best_by_family.csv`、`relative_cost_additive3_layerwise_summary.csv` 与 12 组 layerwise png/pdf/csv。
  - 总体最好 AUC：Qwen 仍偏 `geo`，torch probe 最好为 visual risk+visual cosine `geo`，AUC 约 0.951；InternVL 最强为 `tadd` visual risk cap085+visual cosine cap085，XGB/torch AUC 约 0.960/0.959；LLaVA 中 `tadd` 对 VP combo 有收益，torch probe 最好 `vp tadd raw + vpcosine raw` AUC 约 0.858。
  - 分类层面 additive 胜过 `geo` 的 family 数：XGB 13/24、MLP 10/24、torch probe 12/24；其中 `tadd` 最稳定，`sadd/tsadd` 更多表现为抬高 risk 绝对值和曲线均值差，但分类收益不稳定。当前建议：主线默认仍以 `geo` 为 baseline，`tadd` 作为有希望的 ablation/candidate，`sadd/tsadd` 仅保留为机制对照。
- 2026-07-03 补跑 source 与 target 统一为 visual-only 的 geo cost risk 曲线：
  - 新增配置：`configs/model_configs_visualonly_relativevll_cost_geo.yaml`，三模型均显式 `dgst_t_support_scope: "visual"`，`relative_cost_mode: "geo"`，`relative_cost_modes: ["geo"]`。
  - 三模型 full features 已完成：Qwen 2190 行、InternVL 3715 行、LLaVA 1448 行；字段 `dgst_t_transport_risk_relative_vll_cost_geo_per_layer` 层数分别为 28/32/32，无 NaN/inf。
  - 统一汇总目录：`outputs/vsource_vtarget_geo_risk_comparison/`，包含三模型单独曲线和合并图 `vsource_vtarget_geo_risk_3models_by_label.{png,pdf}`。
  - 曲线结果：Qwen hall/non mean 约 0.249/0.228，diff_avg 约 +0.021，peak layer 19 diff 约 +0.080；InternVL 约 0.255/0.247，diff_avg 约 +0.008，peak layer 2 diff 约 +0.043；LLaVA 约 0.478/0.451，diff_avg 约 +0.027，peak layer 22 diff 约 +0.087。
  - 已补跑 V source / V target 的 geo torch MLP，并与 additive3 中 VP source / VP target 的 geo torch MLP 对比：汇总表为 `outputs/vsource_vtarget_geo_risk_comparison/vsource_vtarget_geo_torch_mlp_vs_vp.md`，CSV 为 `vsource_vtarget_geo_torch_mlp_vs_vp_all.csv` 与 `vsource_vtarget_geo_torch_mlp_vs_vp_delta.csv`。
  - torch MLP 对比结论：Qwen V/V 更强，best AUC 0.952 vs VP/VP 0.923；InternVL VP/VP 更强，0.952 vs V/V 0.942；LLaVA VP/VP 略强，0.844 vs V/V 0.827。matched feature 上 Qwen 四组 V/V AUC 均高于 VP/VP，InternVL 四组均低于 VP/VP，LLaVA 只有 risk cap085 的 V/V AUC 小幅高于 VP/VP。
  - 已从保存的 torch probe checkpoint 回填 AUPR/average precision，并将 `scripts/train_torch_probe_feature_sets.py` 更新为后续默认输出 `aupr`。best AUPR：Qwen V/V 0.984 vs VP/VP 0.966；InternVL VP/VP 0.987 vs V/V 0.982；LLaVA VP/VP 0.844 vs V/V 0.782。
  - 2026-07-04 补跑 InternVL V/V geo source softmax tau ablation：新增 tau=0.05 与 tau=0.10，两者与基线 tau=0.07 使用同一标注/split。汇总目录：`outputs/internvl_source_tau_ablation_vsource_vtarget_geo/`。结论：tau 主要改变 risk 绝对值（tau 越小整体 risk 越高），hall/non 差值基本不变；raw diff_avg 约 0.0076/0.0082/0.0090，cap085 diff_avg 约 0.0053/0.0055/0.0057。
  - 2026-07-04 补跑 InternVL V/V geo source no-softmax ablation：新增 `source_distribution_mode: "softmax" | "relu_norm"`，默认仍为 `softmax`；`relu_norm` 为 cosine 分数 clamp 到非负后直接归一化，全非正时退回均匀分布。新增配置 `configs/model_configs_visualonly_relativevll_cost_geo_source_relu.yaml`，输出目录 `outputs/internvl_2_5_8b/COCO500-visualonly-relativevll-cost-geo-source-relu/`，features 共 3715 行，labels `{1:3064,0:651}`，所有样本 `dgst_t_source_distribution_mode=relu_norm`，raw/cap085 字段无 NaN/inf。对比汇总：`outputs/internvl_source_nosoftmax_vsource_vtarget_geo/`。结论：相对 baseline softmax tau=0.07，no-softmax/relu_norm 提升 hall-non risk gap；raw diff_avg 从 0.0082 提到 0.0177，L16-32 从 0.0013 提到 0.0098；cap085 diff_avg 从 0.0055 提到 0.0130，L16-32 从 -0.0014 提到 0.0055。
  - 同步补跑 InternVL V/V geo no-softmax 的 torch MLP/probe 四组 feature set，汇总 `outputs/internvl_source_nosoftmax_vsource_vtarget_geo/internvl_vv_geo_source_nosoftmax_torch_mlp.md`。分类结论：risk-only 明显提升，raw AUC 0.840->0.867，cap085 0.848->0.866；加 cosine 后 AUC 基本持平/小升，raw 0.942->0.943，cap085 0.936->0.942。VP/VP geo 仍保持最强 combo，raw risk+cosine AUC 约 0.952、F1 约 0.965。
  - 2026-07-04 补跑 LLaVA VP/VP geo source no-softmax risk 曲线：新增配置 `configs/model_configs_visualprompt_relativevll_cost_geo_source_relu.yaml`，输出目录 `outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-geo-source-relu/`，features 共 1448 行，labels `{0:757,1:691}`，所有样本 `dgst_t_source_distribution_mode=relu_norm`，VP risk raw/cap085 字段无 NaN/inf。对比汇总：`outputs/llava_source_nosoftmax_vp_geo/`。结论：LLaVA VP/VP 下 no-softmax 会抬高 risk 绝对值，但 hall/non 仍主要反向（non-hall risk 更高）；raw H-N diff_avg 从 -0.0284 变为 -0.0169，cap085 从 -0.0359 变为 -0.0241，反向差值有所减弱但没有转成稳定正向。
  - 后续命名约定：需要明确写成 `V source / V target` 或 `VP source / VP target`，避免再用 `visual_raw` 这种会和 source/target 混淆的简称。
- 2026-07-05 按用户要求检查 LLaVA `Raw SVD EIF Dose * risk_geo_raw`：
  - 在 `scripts/train_feature_sets.py` 新增 computed feature alias `ffn_eifdose_svd_x_risk_geo_raw`，训练时逐层计算 `ffn_eifdose_svd * risk_geo_raw`；不改已有 `ffn_eifdose_svd`、`risk_geo_raw` 或 features.pkl 原字段含义。
  - 已生成乘积曲线：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_eifdose_svd_x_risk_geo_raw_by_label.{png,pdf,csv}`，SEM 版本为 `llava_ffn_eifdose_svd_x_risk_geo_raw_sem_by_label.{png,pdf,csv}`，曲线摘要为 `llava_ffn_eifdose_svd_x_risk_geo_raw_summary.md`。
  - 曲线结果：features 共 1448 行，non=757、hall=691、32 层；乘积曲线 hall avg 约 0.182931，non avg 约 0.173482，H-N diff_avg 约 +0.009449；最大绝对差在第 1 层且为反向，H-N gap 约 -0.033289。
  - 在原独立 8:2 split 目录 `outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20/` 上补跑 torch probe；仍是 train=400 images、val=[]、test=100 images，token rows train=1122/test=326，val 为空时 trainer 使用 train 作为 val。
  - 训练汇总：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_eifdose_svd_x_risk_geo_raw_torch_probe_summary.md` 与 `.csv`。乘积单特征 AUC/F1/AUPR 约 0.790/0.720/0.761，弱于单独 `ffn_eifdose_svd` 的 0.827/0.718/0.816，但强于单独 `risk_geo_raw` 的 0.749/0.658/0.732。
  - 组合结果：`ffn_eifdose_svd_x_risk_geo_raw+visualcosine_raw` AUC/F1 约 0.836/0.764，低于 `ffn_eifdose_svd+visualcosine_raw` 的 0.868/0.785；`ffn_eifdose_svd_x_risk_geo_raw+risk_geo_raw+visualcosine_raw` AUC/F1 约 0.859/0.776，低于 `ffn_eifdose_svd+risk_geo_raw+visualcosine_raw` 的 0.883/0.814；把乘积作为额外交互项加入 `ffn_eifdose_svd+risk_geo_raw+visualcosine_raw` 后 AUC 约 0.882987，基本打平旧组合 0.882572，但 F1 降到约 0.801。
  - 当前结论：乘法曲线会放大一部分中后层正向 H-N gap，但作为替代特征不如保留原始 EIF Dose；作为额外交互项没有明确实质收益，最多作为补充 ablation。
- 2026-07-05 按用户要求检查 LLaVA `FAD * risk_geo_raw`：
  - 在 `scripts/train_feature_sets.py` 新增 computed feature alias `ffn_fad_x_risk_geo_raw`，训练时逐层计算 `ffn_fad * risk_geo_raw`；不改已有 `ffn_fad`、`risk_geo_raw` 或 features.pkl 原字段含义。
  - 已生成乘积曲线：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_fad_x_risk_geo_raw_by_label.{png,pdf,csv}`，SEM 版本为 `llava_ffn_fad_x_risk_geo_raw_sem_by_label.{png,pdf,csv}`，曲线摘要为 `llava_ffn_fad_x_risk_geo_raw_summary.md`。
  - 曲线结果：features 共 1448 行，non=757、hall=691、32 层；乘积曲线 hall avg 约 0.289963，non avg 约 0.257258，H-N diff_avg 约 +0.032705；最大绝对差在第 22 层，H-N gap 约 +0.093990。
  - 在同一独立 8:2 split 目录 `outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20/` 上补跑 torch probe；仍是 train=400 images、val=[]、test=100 images，token rows train=1122/test=326，val 为空时 trainer 使用 train 作为 val。
  - 训练汇总：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_fad_x_risk_geo_raw_torch_probe_summary.md` 与 `.csv`。乘积单特征 AUC/F1/AUPR 约 0.820/0.741/0.787，弱于单独 `ffn_fad` 的 0.846/0.764/0.830，但强于单独 `risk_geo_raw` 的 0.749/0.658/0.732。
  - 组合结果：`ffn_fad+risk_geo_raw` AUC/F1 约 0.870/0.809；`ffn_fad+risk_geo_raw+visualcosine_raw` AUC/F1 约 0.881/0.809；`ffn_fad_x_risk_geo_raw+risk_geo_raw+visualcosine_raw` AUC/F1 约 0.846/0.791，低于保留原始 FAD 与 risk 的组合。把乘积作为额外交互项加入 `ffn_fad+risk_geo_raw+visualcosine_raw` 后 AUC 约 0.881404，基本打平原组合 0.881028，但 F1 降到约 0.799。
  - 当前结论：`FAD * risk` 曲线的类别分离明显强于 `Raw SVD EIF Dose * risk`，但训练上仍不适合作为替代特征；更稳的方式是保留 FAD 与 risk 两个原始因子，乘积最多作为 interaction ablation。
- 2026-07-05 按用户要求为论文展示画 LLaVA `FAD * risk_geo_raw` 区分度图：
  - 新增主图：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_fad_x_risk_geo_raw_paper_separability.{png,pdf,csv}`，包含三 panel：hall/non mean + image-level bootstrap 95% CI、signed H-N gap + CI、per-layer ROC-AUC + CI。
  - 新增分布图：`llava_ffn_fad_x_risk_geo_raw_selected_layer_violin.{png,pdf}`，选择 L1/L8/L22/L30/L32 展示分布重叠。
  - 新增对比图：`llava_ffn_product_gap_auc_comparison.{png,pdf,csv}`，对比 `FAD * risk` 与 `Raw SVD EIF Dose * risk` 的逐层 H-N gap 和 AUC。
  - 绘图统计：使用 1448 token rows、468 unique images；L22 是 `FAD * risk` 的 peak layer，H-N gap 约 +0.093990，单层 AUC 约 0.678520。主图显示 mean curve 看起来接近，但 gap/AUC panel 能更清楚表达中后层区分度。

## Files changed recently
- 新增：`configs/model_configs_visualonly.yaml`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualonly/`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualonly/`
- 新增输出目录：`outputs/visualonly_support_risk_comparison/`
- 备份错误中间结果：`outputs/qwen2_5_vl_7b/COCO500-visualonly/bad_internvl_labels_20260630_194840/`
- 新增：`configs/model_configs_visualprompt_relativevll.yaml`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll/`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll/`
- 新增：`configs/model_configs_visualprompt_relativevll_finalnorm.yaml`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll-finalnorm/`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll-finalnorm/`
- 新增对比输出：`outputs/finalnorm_relativevll_comparison/`
- 新增：`configs/model_configs_visualonly_source_delta_gamma.yaml`
- 新增：`configs/model_configs_visualprompt_source_delta_gamma.yaml`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualonly-source-delta-gamma/`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-source-delta-gamma/`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualonly-source-delta-gamma/`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualprompt-source-delta-gamma/`
- 新增：`configs/model_configs_visualprompt_relativevll_cost_geo.yaml`
- 新增：`configs/model_configs_visualprompt_relativevll_cost_tbar.yaml`
- 新增：`configs/model_configs_visualprompt_relativevll_cost_sbar.yaml`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll-cost-3way/`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll-cost-3way/`
- 新增输出目录：`outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-3way/`
- 新增对比输出：`outputs/relative_cost_3way_comparison/`
- 新增：`configs/model_configs_visualprompt_relativevll_cost_additive3.yaml`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll-cost-additive3/`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll-cost-additive3/`
- 新增输出目录：`outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-additive3/`
- 新增对比输出：`outputs/relative_cost_additive3_comparison/`
- 新增：`configs/model_configs_visualonly_relativevll_cost_geo.yaml`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualonly-relativevll-cost-geo/`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualonly-relativevll-cost-geo/`
- 新增输出目录：`outputs/llava_1_5_7b/COCO500-visualonly-relativevll-cost-geo/`
- 新增对比输出：`outputs/vsource_vtarget_geo_risk_comparison/`
- 新增：`configs/model_configs_visualonly_relativevll_cost_geo_source_relu.yaml`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualonly-relativevll-cost-geo-source-relu/`
- 新增对比输出：`outputs/internvl_source_nosoftmax_vsource_vtarget_geo/`
- 新增：`configs/model_configs_visualprompt_relativevll_cost_geo_source_relu.yaml`
- 新增输出目录：`outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-geo-source-relu/`
- 新增对比输出：`outputs/llava_source_nosoftmax_vp_geo/`
- 更新：`scripts/train_feature_sets.py` 新增 `ffn_eifdose_svd_x_risk_geo_raw` 乘积特征别名。
- 新增诊断输出：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_eifdose_svd_x_risk_geo_raw_*`。
- 更新：`scripts/train_feature_sets.py` 新增 `ffn_fad_x_risk_geo_raw` 乘积特征别名。
- 新增诊断输出：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_fad_x_risk_geo_raw_*`。
- 新增论文展示图输出：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_fad_x_risk_geo_raw_paper_separability.*`、`llava_ffn_fad_x_risk_geo_raw_selected_layer_violin.*`、`llava_ffn_product_gap_auc_comparison.*`。
- 更新训练输出：`outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20/results/` 增加乘积特征 torch probe 结果与 artifacts。

## Commands run
- 2026-07-05 乘积特征检查：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile scripts/train_feature_sets.py scripts/train_torch_probe_feature_sets.py`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 验证 `ffn_eifdose_svd_x_risk_geo_raw` alias 可解析，train/test 矩阵分别为 `(1122, 32)` 与 `(326, 32)`，乘积与手工逐层相乘最大误差为 0，且无 NaN/inf。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 读取 `outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly/features.pkl`，生成 `llava_ffn_eifdose_svd_x_risk_geo_raw_by_label` 与 `_sem_by_label` 曲线、CSV 和摘要。
  - `CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/train_torch_probe_feature_sets.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_evidence_variants_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20 --feature-sets ffn_eifdose_svd+risk_geo_raw ffn_eifdose_svd_x_risk_geo_raw ffn_eifdose_svd_x_risk_geo_raw+risk_geo_raw ffn_eifdose_svd_x_risk_geo_raw+ffn_eifdose_svd ffn_eifdose_svd_x_risk_geo_raw+ffn_eifdose_svd+risk_geo_raw ffn_eifdose_svd_x_risk_geo_raw+visualcosine_raw ffn_eifdose_svd_x_risk_geo_raw+risk_geo_raw+visualcosine_raw ffn_eifdose_svd_x_risk_geo_raw+ffn_eifdose_svd+risk_geo_raw+visualcosine_raw --device cuda:0`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 从 `llava_1_5_7b_selected_feature_sets.json` 生成乘积特征 torch probe 汇总 `llava_ffn_eifdose_svd_x_risk_geo_raw_torch_probe_summary.md/.csv`。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 验证 `ffn_fad_x_risk_geo_raw` alias 可解析，train/test 矩阵分别为 `(1122, 32)` 与 `(326, 32)`，乘积与手工逐层相乘最大误差为 0，且无 NaN/inf。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 读取 `outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly/features.pkl`，生成 `llava_ffn_fad_x_risk_geo_raw_by_label` 与 `_sem_by_label` 曲线、CSV 和摘要。
  - `CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/train_torch_probe_feature_sets.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_evidence_variants_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20 --feature-sets ffn_fad+risk_geo_raw ffn_fad+visualcosine_raw ffn_fad+risk_geo_raw+visualcosine_raw ffn_fad_x_risk_geo_raw ffn_fad_x_risk_geo_raw+risk_geo_raw ffn_fad_x_risk_geo_raw+ffn_fad ffn_fad_x_risk_geo_raw+ffn_fad+risk_geo_raw ffn_fad_x_risk_geo_raw+visualcosine_raw ffn_fad_x_risk_geo_raw+risk_geo_raw+visualcosine_raw ffn_fad_x_risk_geo_raw+ffn_fad+risk_geo_raw+visualcosine_raw --device cuda:0`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 从 `llava_1_5_7b_selected_feature_sets.json` 生成 FAD 乘积特征 torch probe 汇总 `llava_ffn_fad_x_risk_geo_raw_torch_probe_summary.md/.csv`。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 使用 image-level bootstrap 生成论文展示图 `llava_ffn_fad_x_risk_geo_raw_paper_separability`、selected-layer violin 和 `FAD * risk` vs `Raw SVD EIF Dose * risk` gap/AUC 对比图。
- `CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model internvl_2_5_8b --config configs/model_configs_visualonly.yaml --output-dir outputs/internvl_2_5_8b/COCO500-visualonly --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`
- `CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model qwen2_5_vl_7b --config configs/model_configs_visualonly.yaml --output-dir outputs/qwen2_5_vl_7b/COCO500-visualonly --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`
- 使用 `/opt/conda/private/envs/vicr/bin/python` 读取 `features.pkl` 并生成逐层 risk 的 png/pdf/csv。
- 2026-06-30 再次运行上述两条 `extract_features.py` 命令，用 `cost_mode: "decomposed"` 重算 visual-only 特征和图。
- 2026-06-30 使用 `scripts/plot_layerwise_feature_comparison.py --features risk risk_capped_topmass_085` 生成 Qwen/InternVL 单模型 risk-vs-capped 图，并额外生成双模型 capped risk 合并图。
- 2026-06-30 使用 `scripts/train_feature_sets.py` 在两个 visual-only 输出目录上补跑 `risk`、`risk+target_visual_hidden_cosine_capped_topmass_085`、`risk_capped_topmass_085+target_visual_hidden_cosine_capped_topmass_085`，分类器为 xgb/rf/mlp。
- 2026-06-30 使用 `scripts/train_feature_sets.py` 在原 `COCO500` visual+prompt/direct 输出上补跑缺失的 `risk+target_visual_hidden_cosine_capped_topmass_085` 和 InternVL 的 capped 组合，并生成 direct vs visual-only/decomposed 对比表。
- 2026-07-01 实现 relative VLL 代码后运行：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py models/dgst_capture.py features/extractor.py models/qwen_wrapper.py models/internvl_wrapper.py models/llava_wrapper.py scripts/train_feature_sets.py scripts/train_torch_probe_feature_sets.py scripts/plot_layerwise_feature_comparison.py`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 小张量验证 median/MAD、prompt gate=0、MAD=0 稳定、`target_logits_multi` 不使用 bias、`legacy_prob` 与 `dual` 旧 risk 一致。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 验证新增 train/plot aliases 可解析。
- 2026-07-01 InternVL `COCO500-visualonly-relativevll` 已完成后生成图：
  - `outputs/internvl_2_5_8b/COCO500-visualonly-relativevll/results/internvl_2_5_8b_visualonly_relativevll_risk_vs_legacy_by_label.{png,pdf,csv}`
  - `outputs/internvl_2_5_8b/COCO500-visualonly-relativevll/results/internvl_2_5_8b_visualonly_relativevll_capped_risk_vs_legacy_by_label.{png,pdf,csv}`
  - `outputs/internvl_2_5_8b/COCO500-visualonly-relativevll/results/internvl_2_5_8b_visualonly_relativevll_target_visual_cosine_vs_legacy_by_label.{png,pdf,csv}`
  - `outputs/internvl_2_5_8b/COCO500-visualonly-relativevll/results/internvl_2_5_8b_visualonly_relativevll_capped_target_visual_cosine_vs_legacy_by_label.{png,pdf,csv}`
  - 结果摘要：relative VLL risk 平均值明显低于 legacy risk（hall/non 均值约 0.436/0.420 vs 0.928/0.912），hall-non 平均差仍约 0.016；relative VLL capped risk 的 hall-non 平均差约 0.011，高于 legacy capped risk 的约 0.005；relative target visual cosine 仍表现为 non-hallucination 高于 hallucination。
- 2026-07-01 已在 InternVL `COCO500-visualonly-relativevll` 上补跑 feature-set 消融，结果表：
  - `outputs/internvl_2_5_8b/COCO500-visualonly-relativevll/results/internvl_2_5_8b_selected_feature_sets_table.md`
  - 单独 `risk_relative_vll` 最佳为 MLP：F1=0.940，Acc=0.895，AUC=0.860；单独 `risk_relative_vll_capped_topmass_085` 最佳 AUC 为 MLP：AUC=0.866。
  - 单独 `target_visual_hidden_cosine_relative_vll_capped_topmass_085` 已很强，最佳 AUC 为 RF：AUC=0.932；最佳 F1 为 XGB：F1=0.950，AUC=0.928。
  - 最强组合为 `risk_relative_vll_capped_topmass_085+target_visual_hidden_cosine_relative_vll_capped_topmass_085` 的 XGB：Precision=0.930，Recall=0.977，F1=0.953，Acc=0.919，AUC=0.951。
  - `risk_relative_vll+target_visual_hidden_cosine_relative_vll_capped_topmass_085` 的 XGB 也较强：F1=0.948，Acc=0.911，AUC=0.943。
- 2026-07-01 Qwen `COCO500-visualonly-relativevll` 已完成后生成图：
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly-relativevll/results/qwen2_5_vl_7b_visualonly_relativevll_risk_vs_legacy_by_label.{png,pdf,csv}`
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly-relativevll/results/qwen2_5_vl_7b_visualonly_relativevll_capped_risk_vs_legacy_by_label.{png,pdf,csv}`
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly-relativevll/results/qwen2_5_vl_7b_visualonly_relativevll_target_visual_cosine_vs_legacy_by_label.{png,pdf,csv}`
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly-relativevll/results/qwen2_5_vl_7b_visualonly_relativevll_capped_target_visual_cosine_vs_legacy_by_label.{png,pdf,csv}`
  - 结果摘要：legacy risk 的 hall/non 平均值几乎重合（约 0.778/0.777，diff_avg 约 0.001），relative VLL risk 的 hall/non 平均差提升到约 0.036；legacy capped risk diff_avg 约 0.002，relative VLL capped risk diff_avg 约 0.030。relative target visual cosine 中 non-hallucination 高于 hallucination 的差异也更明显（diff_avg 约 -0.047；capped 后约 -0.040）。
- 2026-07-01 已在 Qwen `COCO500-visualonly-relativevll` 上补跑 feature-set 消融，结果表：
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly-relativevll/results/qwen2_5_vl_7b_selected_feature_sets_table.md`
  - legacy `risk` 最佳为 XGB：F1=0.901，Acc=0.845，AUC=0.789。
  - 单独 `risk_relative_vll` 最佳 AUC 为 MLP：AUC=0.904，F1=0.902；最佳 F1 为 XGB：F1=0.918，AUC=0.893。
  - 单独 `risk_relative_vll_capped_topmass_085` 最佳 AUC 为 MLP：AUC=0.903，F1=0.907；最佳 F1 为 XGB：F1=0.920，AUC=0.878。
  - 单独 `target_visual_hidden_cosine_relative_vll` 最佳 F1 为 XGB：F1=0.921，Acc=0.879，AUC=0.881。
  - 组合 `risk_relative_vll+target_visual_hidden_cosine_relative_vll` 最佳 AUC 为 MLP：Precision=0.907，Recall=0.954，F1=0.930，Acc=0.894，AUC=0.942。
  - 最高 F1/Acc 为 `risk_relative_vll_capped_topmass_085+target_visual_hidden_cosine_relative_vll_capped_topmass_085` 的 MLP：Precision=0.935，Recall=0.935，F1=0.935，Acc=0.903，AUC=0.923。
- 2026-07-01 新增并验证 visual+prompt simultaneous VLL：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py features/extractor.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 小张量验证 `visual_prompt_relative_vll` 在全 support 上归一化、`relative_vll` 仍只给视觉 token target mass、输出字段存在且无 prompt-only 字段。
  - 新增字段：`dgst_t_transport_risk_visual_prompt_relative_vll_per_layer`、`dgst_t_transport_risk_visual_prompt_relative_vll_capped_topmass_085_per_layer`、`dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer`、`dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer`。
- 2026-07-01 Qwen `COCO500-visualprompt-relativevll` 已完成：
  - 特征抽取命令：`CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model qwen2_5_vl_7b --config configs/model_configs_visualprompt_relativevll.yaml --output-dir outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`
  - features 共 2190 行，结果目录：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll/results/`
  - 图：`qwen2_5_vl_7b_visualprompt_relativevll_risk_vs_visual_relativevll_by_label.{png,pdf,csv}`、`qwen2_5_vl_7b_visualprompt_relativevll_capped_risk_vs_visual_relativevll_by_label.{png,pdf,csv}`、`qwen2_5_vl_7b_visualprompt_relativevll_target_cosine_vs_visual_relativevll_by_label.{png,pdf,csv}`、`qwen2_5_vl_7b_visualprompt_relativevll_capped_target_cosine_vs_visual_relativevll_by_label.{png,pdf,csv}`。
  - 曲线摘要：visual+prompt simultaneous risk 的 hall/non diff_avg 约 0.038，visual relative risk 约 0.040；capped 后 visual+prompt diff_avg 约 0.037，高于 visual relative capped 的约 0.027；target cosine 中 non-hall 高于 hall，visual+prompt capped diff_avg 约 -0.045，略强于 visual relative capped 的约 -0.040。
  - 消融表：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll/results/qwen2_5_vl_7b_selected_feature_sets_table.md`
  - 单独 `risk_visual_prompt_relative_vll`：XGB F1=0.908，Acc=0.860，AUC=0.902；单独 capped risk：XGB F1=0.915，AUC=0.878。
  - 单独 `target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085`：XGB F1=0.917，AUC=0.896；RF AUC=0.900。
  - 最强 AUC 为 `risk_visual_prompt_relative_vll+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll` 的 MLP：F1=0.935，Acc=0.903，AUC=0.941。
  - 最高 F1 为 visual+prompt 组合的 XGB：`risk_visual_prompt_relative_vll+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll` F1=0.939，AUC=0.935；capped+capped 组合 XGB F1=0.939，AUC=0.936。
- 2026-07-01 InternVL `COCO500-visualprompt-relativevll` 已完成：
  - 特征抽取命令：`CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model internvl_2_5_8b --config configs/model_configs_visualprompt_relativevll.yaml --output-dir outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`
  - features 共 3715 行，结果目录：`outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll/results/`
  - 图：`internvl_2_5_8b_visualprompt_relativevll_risk_vs_visual_relativevll_by_label.{png,pdf,csv}`、`internvl_2_5_8b_visualprompt_relativevll_capped_risk_vs_visual_relativevll_by_label.{png,pdf,csv}`、`internvl_2_5_8b_visualprompt_relativevll_target_cosine_vs_visual_relativevll_by_label.{png,pdf,csv}`、`internvl_2_5_8b_visualprompt_relativevll_capped_target_cosine_vs_visual_relativevll_by_label.{png,pdf,csv}`。
  - 曲线摘要：visual+prompt simultaneous risk 的 hall/non diff_avg 约 0.049，明显强于 visual relative risk 在 visual+prompt support 下的约 -0.019；capped 后 visual+prompt diff_avg 约 0.048。target cosine 中 non-hall 高于 hall，visual+prompt capped diff_avg 约 -0.038，高于 visual relative capped 的约 -0.029。
  - 消融表：`outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll/results/internvl_2_5_8b_selected_feature_sets_table.md`
  - 单独 `risk_visual_prompt_relative_vll`：XGB F1=0.956，Acc=0.924，AUC=0.901；单独 capped risk：XGB F1=0.955，AUC=0.912。
  - 单独 `target_visual_prompt_hidden_cosine_visual_prompt_relative_vll`：XGB F1=0.948，AUC=0.920；capped 版本 RF AUC=0.918。
  - visual+prompt simultaneous 最强 F1 为 `risk_visual_prompt_relative_vll+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085` 的 XGB：Precision=0.950，Recall=0.974，F1=0.962，Acc=0.935，AUC=0.937。
  - visual+prompt simultaneous 最强 AUC 为 capped+capped 组合 XGB：F1=0.959，Acc=0.930，AUC=0.944。当前总最高 AUC 仍是 visual branch 组合 `risk_relative_vll+target_visual_hidden_cosine_relative_vll` 的 MLP：F1=0.958，Acc=0.930，AUC=0.950。
- 2026-07-02 已移除 word all-token risk mean 代码路径：
  - 删除 `target_token_aggregation` 配置项、`risk_mean` 聚合逻辑和 riskmean 专用配置文件。
  - 保留 relative VLL 与 visual+prompt relative VLL。
- 2026-07-02 新增并验证 final norm relative VLL ablation：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py models/dgst_capture.py features/extractor.py models/base_wrapper.py models/qwen_wrapper.py models/internvl_wrapper.py models/llava_wrapper.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py`
  - 小张量验证 `final_norm_h_mid` 会改变 `relative_vll_logits`，且不加入 LM-head bias。
  - Qwen/InternVL/LLaVA `--num-images 2` smoke test 均通过，feature rows 的 `dgst_t_relative_vll_logit_source` 为 `final_norm_h_mid`。
  - Qwen full extraction：`outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll-finalnorm/features.pkl`，2190 行。
  - InternVL full extraction：`outputs/internvl_2_5_8b/COCO500-visualprompt-relativevll-finalnorm/features.pkl`，3715 行。
  - 生成 final-norm 单目录图：`outputs/{qwen2_5_vl_7b,internvl_2_5_8b}/COCO500-visualprompt-relativevll-finalnorm/results/*finalnorm*{png,pdf,csv}`。
  - 生成 h_mid vs final_norm_h_mid 跨目录对比图与表：`outputs/finalnorm_relativevll_comparison/finalnorm_vs_hmid_summary.{md,csv}`、`outputs/finalnorm_relativevll_comparison/finalnorm_vs_hmid_feature_sets_best.{md,csv}`。
  - 在 Qwen/InternVL finalnorm 输出目录上跑完 xgb/rf/mlp feature-set 消融。
- 2026-07-02 新增并验证 Source Delta + Gamma 机制消融：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py features/extractor.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py`
  - 小张量验证 `delta_src` softmax、visual-prompt prompt source mass 非零且总和为 1、`gamma=0/0.5/1` target 分布无 NaN/inf、默认旧配置不输出 delta 字段。
  - Qwen/InternVL 的 visual-only、visual-prompt 四组 `--num-images 2` smoke test 通过。
  - Qwen/InternVL 的 visual-only、visual-prompt 四组 full feature extraction 完成。
  - 四组逐层 plot 完成，结果位于各自 `COCO500-*-source-delta-gamma/results/`。
  - Qwen visual-only 与 Qwen visual-prompt 的 xgb/rf/mlp feature-set 消融完成；InternVL feature 与 plot 已完成，分类训练未作为本轮结论依据继续等待。
- 2026-07-03 新增并验证 additive barrier cost 3-variant：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py features/extractor.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py scripts/train_torch_probe_feature_sets.py`
  - 小张量验证 `tadd/sadd/tsadd`：zero barrier 等于 `geo`、target additive 只按列变、source additive 只按行变、two-end additive 等于 source+target-geo，且 cost finite/nonnegative。
  - Qwen `--num-images 2` smoke test 通过，确认 16 个 additive3 risk 字段存在、层数正确、无 NaN/inf。
  - 三模型 full feature extraction、32 feature sets 的 XGB/MLP/torch probe、12 组 layerwise plots 与统一 summary 均已完成。
- 2026-07-02 新增并验证 relative VLL cost 3-way 重构：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py features/extractor.py models/qwen_wrapper.py models/internvl_wrapper.py models/llava_wrapper.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py`
  - 小张量验证 `geo/tbar/sbar` cost matrix 有差异、`relative_cost_modes` 会同时输出三套字段、primary old visual-prompt risk 字段等于 `geo`。
  - Qwen/InternVL 3-way smoke test 通过，字段 `geo/tbar/sbar` 均存在。
  - 使用 `configs/model_configs_visualprompt_relativevll_cost_geo.yaml` 一次抽取 Qwen/InternVL full features，并通过 `relative_cost_modes: ["geo", "target_barrier_geo", "symmetric_barrier_geo"]` 同时生成三种 cost。
  - 在两个 3-way 输出目录上跑完 24 组 XGB-only feature-set 分类；过慢的 Qwen xgb/rf/mlp partial 结果已另存为 `qwen2_5_vl_7b_selected_feature_sets.partial_slow.json`，正式结论使用 XGB-only 完整表。
  - 生成汇总表与逐层图：`outputs/relative_cost_3way_comparison/relative_cost_3way_summary.md`、`relative_cost_3way_xgb_all.csv`、`relative_cost_3way_xgb_best_by_family.csv`、`relative_cost_3way_layerwise_summary.csv` 以及 Qwen/InternVL 的 visual/vp raw/cap085 三路曲线图。
- 2026-07-03 补跑 LLaVA relative VLL cost 3-way：
  - 复用 LLaVA `COCO500` 的 `labeling.json/generations.json/image_splits.json` 到 `outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-3way/`。
  - 特征抽取命令：`CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model llava_1_5_7b --config configs/model_configs_visualprompt_relativevll_cost_geo.yaml --output-dir outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-3way --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`
  - 字段检查通过：`dgst_t_transport_risk_relative_vll_cost_{geo,tbar,sbar}_per_layer` 与 `dgst_t_transport_risk_visual_prompt_relative_vll_cost_{geo,tbar,sbar}_per_layer` 均为 1448 行、32 层。
  - 在 LLaVA 3-way 输出目录上跑完 24 组 XGB-only feature-set 分类，并重新生成三模型总表/曲线。
- 2026-07-03 补跑 MLP classifier：
  - 在 Qwen/InternVL/LLaVA 的 `COCO500-visualprompt-relativevll-cost-3way` 输出目录上，使用同一套 24 个 feature set 跑完 `--classifiers mlp --scoring auc`。
  - 重新生成 MLP vs XGB 汇总表：`outputs/relative_cost_3way_comparison/relative_cost_3way_mlp_summary.md`。
- 2026-07-03 补跑 torch MLP / torch probe：
  - 在 Qwen/InternVL/LLaVA 的 `COCO500-visualprompt-relativevll-cost-3way` 输出目录上，使用 `scripts/train_torch_probe_feature_sets.py` 跑完 6 个 VP+VPcosine feature set 与 6 个 visual-only risk+visual-only cosine feature set，并写入各自 selected feature JSON 的 `torch_probe` 字段。
  - 用同一 torch probe 配置在 `/home/apulis-dev/userdata/DGST/token-grounding-detector/outputs/{qwen2_5_vl_7b,internvl_2_5_8b,llava_1_5_7b}/COCO500` 的 `ads+cgc` 特征上补跑 DGST baseline，结果写入 `outputs/relative_cost_3way_comparison/dgst_coco500_torch_probe_ads_cgc/`。
  - 生成对比汇总：`outputs/relative_cost_3way_comparison/relative_cost_3way_torch_mlp_vs_dgst_coco500.md` 与 `.csv`。
- 2026-07-03 补跑单特征消融：
  - 使用 `scripts/train_torch_probe_feature_sets.py` 在三模型 3-way 输出目录上补齐 `risk/cosine/vprisk/vpcosine` 单特征 torch probe。
  - 使用 `scripts/train_feature_sets.py --classifiers xgb mlp --scoring auc` 为缺失的 `cosine/vpcosine` 单特征补齐 XGB 与 sklearn-MLP。
  - 生成单特征汇总：`outputs/relative_cost_3way_comparison/relative_cost_3way_single_feature_ablation.md`、`.csv`、`_best_by_family.csv`。
- 2026-07-03 补跑 risk + ADS/CGC torch MLP：
  - 自定义 inline trainer 将当前 3-way visual-only risk features 与 DGST `ADS/CGC` features 通过 token key 对齐后训练同一套 torch probe。
  - 生成汇总：`outputs/relative_cost_3way_comparison/relative_cost_3way_risk_ads_cgc_torch_mlp.md`、`.csv`、`_summary.csv`。

## Known issues
- 第一次 Qwen visual-only 抽取误用了 InternVL 的 `labeling.json/generations.json`，产物已移动到 `outputs/qwen2_5_vl_7b/COCO500-visualonly/bad_internvl_labels_20260630_194840/`，不要用于分析。
- 默认 `/opt/conda/bin/python` 缺少 torch/transformers/scipy/sklearn/openai/pyyaml；本次运行使用的是 `/opt/conda/private/envs/vicr/bin/python`。
- `run.sh` 中的疑似 API key 示例字符串已改回占位符；如该字符串曾经是真实密钥，仍建议在 OpenAI 控制台轮换。
- Source Delta + Gamma 中 InternVL 的 feature extraction 与逐层图已完成，但 xgb/rf/mlp 分类训练在当前结论已经清楚后停止；按当前决策，后续先不补这组 deltaOffn/attention-gamma 分类表，除非专门需要负结果附录。
- 本轮曾中断过一次旧思路的单独 Qwen `cost-geo` 抽取，目录 `outputs/qwen2_5_vl_7b/COCO500-visualprompt-relativevll-cost-geo/` 只作为废弃 scratch，不用于分析。正式 cost 对比使用 `COCO500-visualprompt-relativevll-cost-3way`。

## Next steps
- relative VLL visual-only/decomposed、visual+prompt simultaneous VLL 的 Qwen 和 InternVL 逐层图与 xgb/rf/mlp feature-set 消融已完成。
- 当前 word all-token risk mean 已从代码中移除，不再作为后续主线。
- final norm ablation 已完成；当前结论是 final norm 能放大 visual+prompt risk 的均值曲线分离，但 Qwen 的 risk AUC 下降，InternVL 小幅提升，因此不建议直接替换默认 h_mid，需要作为 ablation 保留。
- 后续实验先排除 deltaOffn/`delta_src` 与 `gamma=0/0.5/1` attention 机制消融；不要默认使用 `source_delta_gamma` 配置或 `rvll_delta*` / `vp_rvll_delta*` aliases。
- relative VLL cost 主线建议先用 `geo`；如果需要追求 InternVL visual+prompt、Qwen capped VP combo 或 LLaVA visual+cosine 的小幅 AUC，可把 `tbar` 作为 ablation 候选；`sbar` 不建议作为默认。
- 下一步可只挑最有希望的 finalnorm 分支（Qwen cosine、InternVL visual+prompt risk）补跑 torch probe 或换 split seed 做稳定性检查；source 侧回到 legacy source 与当前默认 attention-guided target。
- 如需和原 visual+prompt/direct 直接对比，可把原 `COCO500/results/*risk_layerwise_by_label.csv` 与本次 `COCO500-visualprompt-relativevll` CSV 做同图差值分析。

## 2026-07-05 LLaVA FFN injection diagnostic
- 本轮目标：在 LLaVA object-token 预测位置增加三条 FFN 注入诊断逐层特征，并用 visual-only relative VLL cost-geo 主线验证曲线和 torch probe。
- 新增特征：
  - FAD / `dgst_t_ffn_attn_dominance_per_layer`：`log((||O_ffn||+eps)/(||O_attn||+eps))`。
  - EIFDose / `dgst_t_ffn_evidence_orthogonal_dose_per_layer`：基于 visual relative VLL target distribution 的 top-K 视觉 token 证据子空间，计算 `||(I-Pi_E) O_ffn||/(||h_mid||+eps)`；默认 `top_k=32`、`rank=8`、`eps=1e-12`。
  - LogitLift / `dgst_t_ffn_logit_lift_per_layer`：只用 LM head / output embedding 中目标 token 行向量，计算 `W_U[target_token_id]^T O_ffn`，不加 bias、不做 softmax。
- 代码变更：
  - `models/dgst_capture.py`：raw capture 增加 `source_attn_states` 和 `target_unembedding`，供 FFN/attention 对比与目标 token unembedding 投影使用。
  - `features/dgst_t.py`：实现三条 FFN injection 特征、逐层字段、layer stats（`ffn_attn_dominance`、`ffn_evidence_orthogonal_dose`、`ffn_logit_lift`、`ffn_evidence_subspace_rank_used`），并保留旧 raw fallback。
  - `features/extractor.py`、`models/{llava_wrapper,internvl_wrapper,qwen_wrapper}.py`：透传 `feature_extraction.dgst_t` 中的 FFN injection 配置，保存新增 per-layer 字段和元信息。
  - `scripts/train_feature_sets.py`、`scripts/plot_layerwise_feature_comparison.py`：增加 `ffn_fad`、`ffn_eifdose`、`ffn_logitlift`、`risk_geo_raw`、`visualcosine_raw` 等 aliases。
  - 新增 `configs/model_configs_llava_ffn_injection_visualonly.yaml`、`scripts/plot_llava_ffn_injection_diagnostic.py`、`scripts/summarize_llava_ffn_injection_probe.py`。
- 已运行命令：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py models/dgst_capture.py features/extractor.py models/llava_wrapper.py models/qwen_wrapper.py models/internvl_wrapper.py scripts/train_torch_probe_feature_sets.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py scripts/plot_llava_ffn_injection_diagnostic.py scripts/summarize_llava_ffn_injection_probe.py`
  - 小张量 inline 测试：验证 FAD 正负、LogitLift dot product、EIFDose 子空间内接近 0 / 正交方向为正、旧 raw 缺少 `source_attn_states` 时不破坏 legacy risk/cosine。
  - 准备 LLaVA 输出目录并复用旧 COCO500 metadata：`mkdir -p outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly`，`cp -n outputs/llava_1_5_7b/COCO500-visualonly-relativevll-cost-geo/{labeling.json,generations.json,image_splits.json} outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly/`。
  - smoke extraction：`CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_injection_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly --device cuda:0 --resume --num-images 2`。
  - full extraction：`CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_injection_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`。
  - 曲线绘制：`/opt/conda/private/envs/vicr/bin/python scripts/plot_llava_ffn_injection_diagnostic.py --features-pkl outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly/features.pkl --output-dir outputs/llava_ffn_injection_diagnostic`。
  - torch probe：`CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/train_torch_probe_feature_sets.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_injection_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly --feature-sets ffn_fad ffn_eifdose ffn_logitlift risk_geo_raw visualcosine_raw risk_geo_raw+visualcosine_raw ffn_fad+visualcosine_raw ffn_eifdose+visualcosine_raw ffn_logitlift+visualcosine_raw ffn_eifdose+risk_geo_raw ffn_eifdose+risk_geo_raw+visualcosine_raw --device cuda:0`。
  - probe 汇总与 artifact 复制：`/opt/conda/private/envs/vicr/bin/python scripts/summarize_llava_ffn_injection_probe.py --results-json outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly/results/llava_1_5_7b_selected_feature_sets.json --output-dir outputs/llava_ffn_injection_diagnostic`，并复制 `results/torch_probe/` 到 `outputs/llava_ffn_injection_diagnostic/torch_probe/`。
- 输出产物：
  - LLaVA 特征：`outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly/features.pkl`，共 1448 行，label 分布 `{0: 757, 1: 691}`；三条 FFN injection 字段均为 1448 行、32 层、无 NaN/inf。
  - 逐层曲线：`outputs/llava_ffn_injection_diagnostic/llava_ffn_attn_dominance_by_label.{png,pdf,csv}`、`llava_ffn_evidence_orthogonal_dose_by_label.{png,pdf,csv}`、`llava_ffn_logit_lift_by_label.{png,pdf,csv}`、`llava_ffn_injection_3features_by_label.{png,pdf,csv}`。
  - 曲线摘要：`outputs/llava_ffn_injection_diagnostic/llava_ffn_injection_summary.md`。
  - torch probe 汇总：`outputs/llava_ffn_injection_diagnostic/llava_ffn_injection_torch_probe_summary.md`、`llava_ffn_injection_torch_probe.csv`，并已复制逐 run artifacts 到 `outputs/llava_ffn_injection_diagnostic/torch_probe/`。
- 初步结果：
  - 曲线均值分离较小但方向总体为 hallucination 更高：FAD hall-non diff_avg 约 `+0.027`，EIFDose 约 `+0.002`，LogitLift 约 `+0.033`。
  - 单 FFN 特征已有强信号：`ffn_fad` AUC=`0.867843`、F1=`0.773723`；`ffn_eifdose` AUC=`0.829412`；`ffn_logitlift` AUC=`0.821765`。
  - baseline 中 `risk_geo_raw+visualcosine_raw` AUC=`0.838235`、F1=`0.750000`；单独 `visualcosine_raw` AUC=`0.807059`，单独 `risk_geo_raw` AUC=`0.737255`。
  - 最佳组合是 `ffn_eifdose+risk_geo_raw+visualcosine_raw`：AUC=`0.883333`、F1=`0.753623`、AUPR=`0.869921`；最高 F1 是 `ffn_fad+visualcosine_raw`：F1=`0.789116`、AUC=`0.874706`。
- 2026-07-05 追加 training-free 检查：
  - 输出：`outputs/llava_ffn_injection_diagnostic/llava_ffn_injection_trainingfree_summary.md`、`llava_ffn_injection_trainingfree_mean_scores.csv`、`llava_ffn_injection_trainingfree_val_threshold.csv`、`llava_ffn_injection_trainingfree_train_threshold.csv`。
  - 注意：当前 `image_splits.json` 中 val/test image IDs 完全相同（50/50 overlap），因此之前 torch probe 与 val-threshold training-free 都不是独立 held-out；没有 train/test overlap。
  - 固定 mean-over-layers raw score 的 test AUC 较弱：`ffn_fad` AUC=`0.580196`、`ffn_eifdose` AUC=`0.509412`、`ffn_logitlift` AUC=`0.587059`。F1@0 基本退化为 almost/all-positive baseline（约 `0.64455`）。
  - 用 train split 选择 reducer/direction/threshold 后再报 test：`ffn_fad` AUC=`0.550000`、F1=`0.653266`；`ffn_eifdose` AUC=`0.551569`、F1=`0.640000`；`ffn_logitlift` AUC=`0.562353`、F1=`0.650718`；三特征 z-avg AUC=`0.562353`、F1=`0.644550`。
  - 结论：三条 FFN 特征的 training-free scalar 检测效果一般，主要是弱排序或高召回低精度；torch probe 好很多，说明有效信息主要来自多层联合权重/形状，而不是简单均值或单阈值。
- 2026-07-05 追加 train/test 8:2、无验证集 torch probe：
  - 新目录：`outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly-train80-test20/`，复用同一份 `features.pkl` symlink，重新生成 `image_splits.json`：train/test=`400/100` images，`val=[]`，train-test overlap=`0`。
  - 实际有 object-token rows 的 split：train=`1198` rows（label 0/1=`621/577`），test=`250` rows（label 0/1=`136/114`）。
  - 运行命令：`CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/train_torch_probe_feature_sets.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_injection_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly-train80-test20 --feature-sets ffn_fad ffn_eifdose ffn_logitlift risk_geo_raw visualcosine_raw risk_geo_raw+visualcosine_raw ffn_fad+visualcosine_raw ffn_eifdose+visualcosine_raw ffn_logitlift+visualcosine_raw ffn_eifdose+risk_geo_raw ffn_eifdose+risk_geo_raw+visualcosine_raw --device cuda:0`。
  - 因 `val=[]`，现有训练脚本会提示 `val split unusable ... using train as val`；因此 checkpoint/epoch 选择在 train 上完成，但最终指标在独立 test split 上评估。
  - 汇总输出：`outputs/llava_ffn_injection_diagnostic/train80_test20/llava_ffn_injection_torch_probe_summary.md`、`.csv`，并复制 artifacts 到 `outputs/llava_ffn_injection_diagnostic/train80_test20/torch_probe/`。
  - 结果：`ffn_fad` 单特征 AUC=`0.886481`、F1=`0.798246`；`ffn_eifdose` AUC=`0.883578`、F1=`0.771930`；`ffn_logitlift` AUC=`0.785991`、F1=`0.721030`。
  - baseline：`risk_geo_raw+visualcosine_raw` AUC=`0.838751`、F1=`0.727273`；单独 `visualcosine_raw` AUC=`0.805018`；单独 `risk_geo_raw` AUC=`0.795472`。
  - 最佳组合：`ffn_fad+visualcosine_raw` AUC=`0.928083`、F1=`0.841202`、AUPR=`0.915318`；`ffn_eifdose+risk_geo_raw+visualcosine_raw` AUC=`0.902993`、F1=`0.787330`。
  - 结论：去掉 val/test 重合后，FFN torch probe 仍然很强，且 `ffn_fad` 单特征明显超过 risk/cosine baseline；不过因为无独立 val，模型选择仍建议后续用 train/val/test 三分或固定 epoch 复查。
- 2026-07-05 追加 risk/cosine 按 label 逐层图：
  - 按 `outputs/llava_ffn_injection_diagnostic` 中 FFN 图同样方法（每层按 label 求 mean，阴影为 mean±std）绘制 `risk_geo_raw` 与 `visualcosine_raw`。
  - 输出：`outputs/llava_ffn_injection_diagnostic/llava_risk_geo_raw_by_label.{png,pdf,csv}`、`llava_visualcosine_raw_by_label.{png,pdf,csv}`、`llava_risk_geo_raw_visualcosine_raw_by_label.{png,pdf,csv}`、`llava_risk_geo_visualcosine_summary.md`。
  - 曲线摘要：`risk_geo_raw` hall/non 平均约 `0.478108/0.450728`，H-N diff_avg=`+0.0273802`；`visualcosine_raw` hall/non 平均约 `0.212542/0.255011`，H-N diff_avg=`-0.0424693`。
- 2026-07-05 追加 FFN 信号乘以 risk 的逐层图：
  - 按同样方法绘制逐层 element-wise product：`risk_geo_raw * ffn_fad`、`risk_geo_raw * ffn_eifdose`、`risk_geo_raw * ffn_logitlift`。
  - 输出：`outputs/llava_ffn_injection_diagnostic/llava_ffn_fad_x_risk_geo_raw_by_label.{png,pdf,csv}`、`llava_ffn_eifdose_x_risk_geo_raw_by_label.{png,pdf,csv}`、`llava_ffn_logitlift_x_risk_geo_raw_by_label.{png,pdf,csv}`、`llava_ffn_x_risk_geo_raw_3features_by_label.{png,pdf,csv}`、`llava_ffn_x_risk_geo_raw_summary.md`。
  - 曲线摘要：`ffn_fad_x_risk_geo_raw` hall/non 平均约 `0.289963/0.257258`，H-N diff_avg=`+0.0327054`；`ffn_eifdose_x_risk_geo_raw` 约 `0.183119/0.173710`，diff_avg=`+0.00940887`；`ffn_logitlift_x_risk_geo_raw` 约 `0.556280/0.496885`，diff_avg=`+0.0593954`。
- 2026-07-05 追加“两种 FFN evidence subspace”变体：
  - 目标：不覆盖旧 `ffn_eifdose` / `dgst_t_ffn_evidence_orthogonal_dose_per_layer`，新增 Raw SVD 与 Centered PCA 两种 evidence subspace，各自输出 EIF Fraction 和 EIF Dose。
  - 新增 per-layer 字段：`dgst_t_ffn_eif_fraction_svd_per_layer`、`dgst_t_ffn_eif_dose_svd_per_layer`、`dgst_t_ffn_eif_fraction_pca_per_layer`、`dgst_t_ffn_eif_dose_pca_per_layer`。
  - 新增 layer stats：`ffn_eif_fraction_svd`、`ffn_eif_dose_svd`、`ffn_eif_fraction_pca`、`ffn_eif_dose_pca`、`ffn_evidence_svd_rank_used`、`ffn_evidence_pca_rank_used`。
  - 训练/绘图 alias：`ffn_eiffrac_svd`、`ffn_eifdose_svd`、`ffn_eiffrac_pca`、`ffn_eifdose_pca`；旧 `ffn_fad`、`ffn_eifdose`、`ffn_logitlift` alias 未改名。
  - 实现口径：evidence anchors 继续使用 relative-VLL visual target distribution 的 top-K visual tokens，`K=32`、`rank=8`、`eps=1e-8`；anchor state 使用 visual support token 的 `h_mid`。Raw SVD 直接对 `S[K,d]` 做 SVD；Centered PCA 先减去 anchor 均值再 SVD。Fraction 为 `||f_orth||^2/(||O_ffn||^2+eps)`，Dose 为 `||f_orth||/(||h_mid_t||+eps)`，其中 `h_mid_t=prediction_hidden-O_ffn`。
  - 新增配置/脚本：`configs/model_configs_llava_ffn_evidence_variants_visualonly.yaml`、`scripts/plot_llava_ffn_evidence_variants.py`、`scripts/summarize_llava_ffn_evidence_variants_probe.py`。
  - 静态检查与小张量测试已通过：
    - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py features/extractor.py models/dgst_capture.py models/llava_wrapper.py models/qwen_wrapper.py models/internvl_wrapper.py scripts/train_torch_probe_feature_sets.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py scripts/plot_llava_ffn_evidence_variants.py scripts/summarize_llava_ffn_evidence_variants_probe.py scripts/plot_llava_ffn_injection_diagnostic.py scripts/summarize_llava_ffn_injection_probe.py`
    - inline tensor tests 覆盖：Raw SVD 在 `O_ffn` 落入 anchor span 时 fraction/dose 接近 0、正交时 fraction 接近 1；Centered PCA 使用 `K-1` rank cap；identical anchors 时 PCA rank=0 且 projection 为 0 维子空间；旧 FFN/risk/cosine aliases 仍可解析。
  - 特征抽取：
    - smoke：`CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_evidence_variants_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly --device cuda:0 --resume --num-images 2`，生成 10 rows，四个新字段均为 32 层、无 NaN/inf。
    - full：`CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_evidence_variants_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`。
    - full 输出：`outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly/features.pkl`，共 1448 object-token rows，labels `{0:757,1:691}`，覆盖 468 张有 object-token row 的图片；四个新字段均为 1448 行、32 层、无 NaN/inf；`ffn_evidence_svd_rank_used` 与 `ffn_evidence_pca_rank_used` 在 46336 个 layer stats 上均为 8。
    - 新字段全局均值/范围：`eif_fraction_svd` mean=`0.984482`、min/max=`0.447743/0.999761`；`eif_dose_svd` mean=`0.388989`、min/max=`0.128346/1.625233`；`eif_fraction_pca` mean=`0.989467`、min/max=`0.583439/0.999834`；`eif_dose_pca` mean=`0.390497`、min/max=`0.128424/1.637382`。
  - 曲线输出目录：`outputs/llava_ffn_injection_diagnostic/evidence_variants/`。
    - 单特征图：`llava_ffn_eif_fraction_svd_by_label.{png,pdf,csv}`、`llava_ffn_eif_dose_svd_by_label.{png,pdf,csv}`、`llava_ffn_eif_fraction_pca_by_label.{png,pdf,csv}`、`llava_ffn_eif_dose_pca_by_label.{png,pdf,csv}`。
    - 合并图与摘要：`llava_ffn_evidence_variants_4features_by_label.{png,pdf,csv}`、`llava_ffn_evidence_variants_summary.md`。
    - 曲线摘要：四条新曲线 hall/non 平均差都很小；`ffn_eif_fraction_svd` diff_avg=`+0.001769`，`ffn_eif_dose_svd` diff_avg=`+0.002076`，`ffn_eif_fraction_pca` diff_avg=`+0.001716`，`ffn_eif_dose_pca` diff_avg=`+0.002191`。Fraction 几乎贴近 1，说明 rank=8 visual evidence subspace 对 FFN update 的解释能量很少。
  - 独立 train/test=8:2 torch probe：
    - 新目录：`outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20/`，复用完整 `features.pkl` symlink，重新生成 `image_splits.json`：train/test=`400/100` images，`val=[]`，train-test overlap=`0`。
    - 实际有 object-token rows 的 split：train=`1122` rows、test=`326` rows；feature images 覆盖 train/test=`373/95`。
    - 运行命令：`CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/train_torch_probe_feature_sets.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_evidence_variants_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20 --feature-sets risk_geo_raw visualcosine_raw risk_geo_raw+visualcosine_raw ffn_fad ffn_eifdose ffn_logitlift ffn_eiffrac_svd ffn_eifdose_svd ffn_eiffrac_pca ffn_eifdose_pca ffn_eiffrac_svd+visualcosine_raw ffn_eifdose_svd+visualcosine_raw ffn_eiffrac_pca+visualcosine_raw ffn_eifdose_pca+visualcosine_raw ffn_eiffrac_svd+risk_geo_raw+visualcosine_raw ffn_eifdose_svd+risk_geo_raw+visualcosine_raw ffn_eiffrac_pca+risk_geo_raw+visualcosine_raw ffn_eifdose_pca+risk_geo_raw+visualcosine_raw --device cuda:0`。
    - 汇总输出：`outputs/llava_ffn_injection_diagnostic/evidence_variants/llava_ffn_evidence_variants_torch_probe_summary.md`、`llava_ffn_evidence_variants_torch_probe.csv`；完整 artifacts 已复制到 `outputs/llava_ffn_injection_diagnostic/evidence_variants/torch_probe/`。
    - baseline：`risk_geo_raw` AUC/F1=`0.748870/0.658307`，`visualcosine_raw`=`0.778820/0.721408`，`risk_geo_raw+visualcosine_raw`=`0.807904/0.746082`。
    - 旧 FFN 单项：`ffn_fad` AUC/F1=`0.845577/0.763975`，`ffn_eifdose`=`0.825686/0.714286`，`ffn_logitlift`=`0.756706/0.666667`。
    - 新单项：最好 AUC 是 `ffn_eifdose_pca` AUC=`0.830018`、F1=`0.708197`；最好 F1 是 `ffn_eiffrac_svd` F1=`0.724036`、AUC=`0.784019`。整体与旧 `ffn_eifdose` 接近，但未超过 `ffn_fad`。
    - 新 + cosine：最好为 `ffn_eifdose_pca+visualcosine_raw`，AUC/F1=`0.873305/0.788060`；`ffn_eifdose_svd+visualcosine_raw` AUC/F1=`0.867880/0.784884`。
    - 新 + risk + cosine：最好为 `ffn_eifdose_svd+risk_geo_raw+visualcosine_raw`，AUC/F1/AUPR=`0.882572/0.813953/0.859939`；`ffn_eifdose_pca+risk_geo_raw+visualcosine_raw` AUC/F1=`0.875603/0.808260`。当前新变体的最强组合 F1 明显高于 baseline 与旧 FFN 单项，但 AUC 与旧 injection 最佳组合接近。
  - 注意：本轮 full extraction 结束仍出现一次 multiprocessing `resource_tracker` leaked semaphore warning；主进程退出码为 0，`features.pkl`、plot 和 torch probe 均成功生成并通过字段检查。由于 `val=[]`，torch probe 的 checkpoint/epoch 选择使用 train split；test split 独立，但严格泛化比较仍建议后续改为 train/val/test 三分或固定 epoch。当前只跑 LLaVA，Qwen/InternVL 还未扩展。
- 2026-07-05 追加 FFN gate ratio / FGR 方案：
  - 公式：`a_l = ||O_ffn,t^l|| / (||O_ffn,t^l|| + ||h_mid,t^l|| + eps)`，其中 `h_mid,t^l = prediction_hidden - O_ffn,t^l`；`FGR_l = a_l * risk_geo_raw_l`。
  - 新增 per-layer 字段：`dgst_t_ffn_gate_ratio_per_layer`、`dgst_t_ffn_fgr_per_layer`；新增 layer stats：`ffn_gate_ratio`、`ffn_fgr`；训练 alias：`ffn_gate`、`ffn_fgr`。已有 `ffn_fad`、`ffn_eifdose`、`ffn_logitlift`、risk/cosine 字段未改名。
  - 新增配置：`configs/model_configs_llava_ffn_gate_ratio_visualonly.yaml`；输出目录：`outputs/llava_1_5_7b/COCO500-ffn-gate-ratio-visualonly/`。
  - 静态与小张量检查已通过：
    - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py features/extractor.py scripts/train_feature_sets.py scripts/train_torch_probe_feature_sets.py`
    - inline test 验证 `a_l` 数值等于 `||O_ffn||/(||O_ffn||+||h_mid||+eps)`，`ffn_gate`/`ffn_fgr` alias 可解析，fake row 的逐层取值正确。
  - 特征抽取：
    - smoke：`CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_gate_ratio_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-gate-ratio-visualonly --device cuda:0 --num-images 2`，10 rows，`a_l`、`FGR`、risk 均为 32 层、无 NaN/inf，`FGR=a_l*risk` 最大误差约 `1e-8`。
    - full：`CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_gate_ratio_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-gate-ratio-visualonly --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`。
    - full 输出：1448 rows，labels `{0:757,1:691}`，覆盖 468 张有 object-token row 的图片；`dgst_t_ffn_gate_ratio_per_layer` 与 `dgst_t_ffn_fgr_per_layer` 均为 1448 行、32 层、无 NaN/inf；`a_l` 范围约 `0.1141~0.6286`，`FGR` 范围约 `0.0102~0.3925`。
  - 曲线输出：`outputs/llava_ffn_injection_diagnostic/ffn_gate_ratio/llava_ffn_gate_ratio_by_label.{png,pdf,csv}`、`llava_ffn_fgr_by_label.{png,pdf,csv}`、`llava_ffn_gate_ratio_fgr_by_label.{png,pdf,csv}`、`llava_ffn_gate_ratio_fgr_summary.md`。
  - 曲线摘要：`a_l` hall/non 平均约 `0.273037/0.271746`，H-N diff_avg 约 `+0.001292`，两类几乎重合；`FGR` hall/non 平均约 `0.129074/0.122261`，H-N diff_avg 约 `+0.006813`，peak layer 为第 23 层，H-N gap 约 `+0.017657`。乘以 risk 后曲线分离略增强，但仍弱于之前 `FAD*risk` 的 diff_avg 约 `+0.032705`。
  - 独立 train/test=8:2 torch probe：
    - 新目录：`outputs/llava_1_5_7b/COCO500-ffn-gate-ratio-visualonly-train80-test20/`，复用完整 `features.pkl` symlink，`image_splits.json` 为 train/test=`400/100` images，`val=[]`，train-test overlap=`0`。
    - 实际有 object-token rows 的 split：train=`1122` rows（label 0/1=`589/533`），test=`326` rows（label 0/1=`168/158`），feature images train/test=`373/95`。
    - 运行命令：`CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/train_torch_probe_feature_sets.py --model llava_1_5_7b --config configs/model_configs_llava_ffn_gate_ratio_visualonly.yaml --output-dir outputs/llava_1_5_7b/COCO500-ffn-gate-ratio-visualonly-train80-test20 --feature-sets risk_geo_raw visualcosine_raw risk_geo_raw+visualcosine_raw ffn_fad ffn_fad+risk_geo_raw+visualcosine_raw ffn_gate ffn_fgr ffn_gate+risk_geo_raw ffn_fgr+risk_geo_raw ffn_gate+visualcosine_raw ffn_fgr+visualcosine_raw ffn_gate+risk_geo_raw+visualcosine_raw ffn_fgr+risk_geo_raw+visualcosine_raw ffn_gate+ffn_fgr ffn_gate+ffn_fgr+risk_geo_raw ffn_gate+ffn_fgr+risk_geo_raw+visualcosine_raw --device cuda:0`。
    - 汇总输出：`outputs/llava_ffn_injection_diagnostic/ffn_gate_ratio/llava_ffn_gate_ratio_fgr_torch_probe_summary.md`、`.csv`，完整 artifacts 已复制到 `outputs/llava_ffn_injection_diagnostic/ffn_gate_ratio/torch_probe/`。
    - baseline：`risk_geo_raw` AUC/F1=`0.748870/0.658307`，`visualcosine_raw`=`0.778820/0.721408`，`risk_geo_raw+visualcosine_raw`=`0.807904/0.746082`；旧强 baseline `ffn_fad` AUC/F1=`0.845577/0.763975`，`ffn_fad+risk_geo_raw+visualcosine_raw`=`0.881028/0.808511`。
    - 新单项：`ffn_gate` AUC/F1=`0.830885/0.738170`，明显强于 `ffn_fgr` 的 `0.764165/0.716763`；说明直接给 MLP 看 `a_l` 比只看乘积 `a_l*risk` 更有信息。
    - 新组合：`ffn_gate+risk_geo_raw` AUC/F1=`0.853187/0.771014`，`ffn_fgr+risk_geo_raw`=`0.807452/0.759420`；`ffn_gate+risk_geo_raw+visualcosine_raw` AUC/F1=`0.870630/0.812121`，是本轮新方案 F1 最高组合；`ffn_gate+ffn_fgr+risk_geo_raw+visualcosine_raw` AUC/F1=`0.874548/0.781250`，AUPR=`0.876234`。
    - 结论：`FGR` 曲线确实比单独 `a_l` 更有可见 H-N gap，但训练上不如保留 `a_l` 与 risk 两个因子分开给 MLP；当前最稳用法是 `ffn_gate + risk_geo_raw + visualcosine_raw`。它相对 `risk_geo_raw+visualcosine_raw` 提升 AUC `+0.062726`、F1 `+0.066040`，但 AUC 仍低于 `ffn_fad+risk_geo_raw+visualcosine_raw` `0.010398`，F1 略高 `0.003611`。
- 注意事项 / 风险：
  - full extraction 结束时出现一次 Python multiprocessing `resource_tracker` leaked semaphore warning；`features.pkl` 与后续 plot/probe 均已成功生成，字段检查通过。
  - 当前 LLaVA 输出目录的 val/test split 重合会让模型选择后的 test 指标偏乐观；严谨比较需要重新生成独立 val/test split 或使用 train-calibrated threshold / fixed score。
  - 旧 raw capture 不含 `source_attn_states` / `target_unembedding`，只能走兼容 fallback，不能得到真实 FFN injection 值；需要重新 capture/extract 后再分析这些特征。
  - 当前只完成 LLaVA、一个 COCO500 split；结论应视为首轮证据，后续建议补 Qwen/InternVL 或换 split seed 做稳定性检查。
- 下一步建议：
  - 先把 `ffn_fad` 作为单特征强 baseline，`ffn_eifdose+risk_geo_raw+visualcosine_raw` 作为当前最强组合继续跟 `geo/tbar/sbar`、ADS/CGC baseline 对齐比较。
  - 补一张跨特征 summary 图或表，把 `ffn_fad`、`ffn_eifdose`、`risk_geo_raw`、`visualcosine_raw` 与组合的 AUC/F1 放在同一页，方便论文/汇报使用。
  - 若扩展到 Qwen/InternVL，优先复用本轮 config 和 scripts，只替换 model/output-dir；不要复用旧 features.pkl 直接训练 FFN injection 特征。

## 2026-07-06 COCO-CHAIR object-word 标注与 risk/target_cosine 流程
- 本轮目标：在 `test-cocochair/token-detector` 中接入 ZhangqiJiang07/middle_layers_indicating_hallucinations 的 CHAIR 口径，实现 generation -> COCO-CHAIR object-word labeling -> feature extraction -> `risk`/`target_cosine` 训练流程。
- 依赖处理：
  - `/opt/conda/bin/python` 无全局写权限且禁用 user site，已改用项目历史运行环境 `/opt/conda/private/envs/vicr/bin/python`。
  - 已在 `vicr` 环境安装 `nltk==3.8.1` 与 `pycocotools==2.0.11`，并下载 NLTK 数据 `punkt`、`averaged_perceptron_tagger`、`wordnet`、`omw-1.4`。
  - `requirements.txt` 已新增 `nltk==3.8.1` 和 `pycocotools`；`run.sh` 默认使用 `PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python`，可用环境变量覆盖。
- 主要代码变更：
  - `coco-labeling/coco_chair.py` 替换为 NLTK/WordNet 版 CHAIR evaluator：使用 `synonyms_txt`、COCO double-word 合并、synonym canonicalization，并用 `instances` segmentation objects 与 GT captions 中抽取出的 objects 合并为每张图的 `gt_objects`。
  - `compute_chair_token(image_id, caption)` 现在输出 `mscoco_hallucinated_words`/`hallucination_idxs` 以及新增的 `mscoco_real_words`/`real_idxs`、`object_mentions`、`word_labels`。
  - 标签语义全链路改为：`label=0` 表示 hallucinated object，`label=1` 表示 real/non-hallucinated object，非 COCO object word 为 `-100`。
  - `coco-labeling/label_coco.py` 不再按 canonical object 去重，所有 COCO object mention 都输出为 `object_token_spans`，包含 `surface_word`、`canonical_object`、`word_idx`、`char_start`、`char_end`、`token_indices`、`label`。
  - 三个 wrapper 的 generation 上限统一为默认 `512`：`BaseLVLMWrapper.generation_max_new_tokens` 读取 `cfg["max_new_tokens"]`，缺省 512；LLaVA/Qwen 的 `model.generate` 和 InternVL 手写解码循环均接入该 helper。
  - `scripts/train_feature_sets.py` 默认 feature sets 改为 `risk`、`target_cosine`、`risk+target_cosine`；新增 `target_cosine` alias，优先读 `dgst_t_target_visual_hidden_cosine_relative_vll_per_layer`，缺失时回退到 visual hidden cosine。
  - `detection/train.py`、`detection/evaluate.py`、`scripts/train_torch_probe_feature_sets.py` 已按新标签语义报告 hallucination 正类；部分 COCO 诊断/绘图脚本也同步修正 hall/non 映射。
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 验证 `nltk==3.8.1`、`pycocotools` 可导入，NLTK 四个资源可找到。
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile coco-labeling/coco_chair.py coco-labeling/label_coco.py detection/evaluate.py detection/train.py models/base_wrapper.py models/llava_wrapper.py models/internvl_wrapper.py models/qwen_wrapper.py scripts/train_feature_sets.py scripts/train_torch_probe_feature_sets.py scripts/evaluate_detection_only.py scripts/plot_layerwise_feature_comparison.py scripts/plot_llava_ffn_evidence_variants.py scripts/plot_llava_ffn_injection_diagnostic.py scripts/summarize_cost_state_variants.py scripts/summarize_llava_userprompt_vp.py scripts/diagnose_llava_target_distribution.py scripts/diagnose_vp_target_distribution.py`
  - `bash -n run.sh`
  - inline CHAIR smoke：临时 COCO fixture 中 segmentation `traffic light` 与 GT caption 的 `cell phone`/`bike` 合并进 `gt_objects`，生成 caption 中 `traffic light`、`cell phone` 标为 `1`，`dog` 标为 `0`，非 COCO word 标为 `-100`。
  - inline training smoke：fake features 上 `risk+target_cosine` 可构建矩阵，`evaluate_classifier` 按 hallucination 正类报告 F1/AUC。

## 2026-07-06 LLaVA-OneVision-1.5-8B-Instruct VP geo 接入
- 本轮目标：支持本地新模型 `/home/apulis-dev/userdata/models/LLaVA-OneVision-1.5-8B-Instruct`，用于 token-detector 在 visual+prompt support、relative VLL、cost=geo 下跑 COCO-CHAIR object-token 检测。
- 主要代码变更：
  - 新增 `models/llava_onevision_wrapper.py`：使用 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 加载 OneVision remote-code 模型；processor 走 Qwen2.5-VL 风格 chat template；动态定位连续 `<|image_pad|>` visual token span；复用 DGST-T capture 与 `compute_dgst_t` 流程。
  - `models/__init__.py` 注册 `llava_onevision_1_5_8b` 和 `llava_onevision_1_5_8b_instruct` 两个 model key。
  - 新增专用配置 `configs/model_configs_llava_onevision_vp_relativevll_cost_geo.yaml`，默认 `experiment.mode: vp`，feature sets 为 `risk_visual_prompt_relative_vll_cost_geo`、`target_visual_prompt_hidden_cosine_visual_prompt_relative_vll` 及二者组合，`relative_cost_modes: ["geo"]`。
  - 兼容性补充：`configs/model_configs_visualprompt_relativevll_cost_geo.yaml` 与 `configs/model_configs_clean_vv_vp_geo.yaml` 也加入 `llava_onevision_1_5_8b` 模型条目。
- 建议运行命令：
  - `MODEL=llava_onevision_1_5_8b CONFIG=configs/model_configs_llava_onevision_vp_relativevll_cost_geo.yaml OUTPUT=outputs/llava_onevision_1_5_8b/COCO500-vp-relativevll-cost-geo bash run.sh`
- 已完成轻量校验：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile models/llava_onevision_wrapper.py models/__init__.py`
  - inline config parse：确认 mode=`vp`、`dgst_t_support_scope=visual_prompt`、`relative_cost_modes=["geo"]`、registry 指向 `LLaVAOneVisionWrapper`。
  - processor-only smoke：336x336 测试图得到 input_len=169，`<|image_pad|>` span=(15,159)，共 144 个 visual tokens；未加载完整 8B 模型。
- 2026-07-06 备注：曾短暂尝试让 generation 使用 Transformers 默认 attention backend、不强制 eager；用户实测加速不明显，已回退。当前各 wrapper 仍显式使用 `attn_implementation="eager"`，保证 generation 与 feature extraction 使用同一加载路径。

## 2026-07-06 配置文件整理草案
- 本轮目标：先不删除 `configs/` 下已有实验配置，新增一个统一模板，方便后续把 VV/VP、relative VLL、geo/barrier/additive cost、state-update、source-delta、user-prompt、FFN diagnostic 等开关收敛到单个配置文件。
- 新增文件：`configs/model_configs_unified.yaml`。
  - 默认可直接跑当前重点实验：`experiment.mode: "vp"`，OneVision VP + relative VLL + `relative_cost_modes: ["geo"]`。
  - 保留 `llava_1_5_7b`、`llava_onevision_1_5_8b`、`llava_onevision_1_5_8b_instruct`、`internvl_2_5_8b`、`qwen2_5_vl_7b` 五个 model key。
  - 在文件头注释中集中列出可选开关和值：`mode`、`dgst_t_support_scope`、`dgst_t_dual_scope`、`target_gate_mode`、`relative_vll_logit_source`、`cost_mode`、`relative_cost_mode(s)`、`relative_cost_state_modes`、`relative_cost_update_lambdas`、`source_distribution_mode`、`source_modes`、`target_attention_gammas`、`dgst_t_prompt_support_mode`、`compute_ffn_injection_features`、cap085/topmass，以及常用 feature-set alias。
  - `run.sh` 无需修改：它会继续按 `experiment.mode` 读取 `experiment.feature_sets[mode]`；如需临时训练 FFN 或其他组合，仍可用 shell 变量 `FEATURE_SETS="..." bash run.sh` 覆盖。
- 已完成轻量校验：
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY`：确认 `configs/model_configs_unified.yaml` 可由 PyYAML 解析，模型 keys、默认 mode、默认 feature sets 与 DGST-T 参数可读取。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY`：确认 `get_model_cfg(..., "llava_onevision_1_5_8b")` 会因 `experiment.mode: "vp"` 得到 `dgst_t_support_scope="visual_prompt"`，且 OneVision `image_size=336`。
  - `bash -n run.sh`：通过。
  - `git diff --no-index --check /dev/null configs/model_configs_unified.yaml`：无 whitespace warning（命令因 no-index 新文件 diff 本身返回 1）。

## 2026-07-09 COCO500 mass-dist-topk 与 TGD ADS+CGC AUC 对比筛选
- 本轮目标：在 `outputs/` 下筛选三模型 `COCO500-mass-dist-topk` 的单特征/组合特征，找出 AUC 超过 TGD `ADS+CGC` baseline 的结果。
- 使用的 TGD COCO500 ADS+CGC baseline 来自 `token-grounding-detector/outputs/{model}/COCO500/results/torch_mlp_ads_cgc/ads__cgc/metrics.json` 的 `real_auc`。
- 初始筛选只看主目录：
  - 输入：`outputs/{model}/COCO500-mass-dist-topk/results/{model}_selected_feature_sets.json`
  - 输出：`outputs/coco500_mass_dist_topk_vs_tgd_ads_cgc_auc_filter.md`
  - 输出：`outputs/coco500_mass_dist_topk_vs_tgd_ads_cgc_auc_filter.csv`
  - AUC-only 超过 ADS+CGC 的数量：LLaVA `241`（单特征 21、组合 220），Qwen `12`（组合 12），InternVL `33`（单特征 1、组合 32）。
  - 主目录 best：LLaVA `vp_target_entropy+cosine16` AUC=`0.922`；Qwen `c_vp+cosine16` AUC=`0.938`；InternVL `c_vp+cosine16` AUC=`0.830`。
- 用户提醒后补充检查 `risk-mass-entropy/` 子目录：
  - 输入：`outputs/{model}/COCO500-mass-dist-topk/risk-mass-entropy/results/{model}_selected_feature_sets.json`
  - 输出：`outputs/coco500_mass_dist_topk_risk_mass_entropy_vs_tgd_ads_cgc_auc_filter.md`
  - 输出：`outputs/coco500_mass_dist_topk_risk_mass_entropy_vs_tgd_ads_cgc_auc_filter.csv`
  - 每个模型该子目录均有 81 个 feature sets。
  - 超过 ADS+CGC AUC 的数量：LLaVA `68`（单特征 3、组合 65），Qwen `3`（组合 3），InternVL `5`（单特征 1、组合 4）。
  - `risk-mass-entropy` best：LLaVA `vp_hprev_cos_r_union_la+vp_evidence_entropy+t_p` AUC=`0.907`；Qwen `c_vp+vv_source_topk_entropy+m_p` AUC=`0.937`；InternVL `vv_source_topk_entropy` AUC=`0.828`。
- 最终合并筛选只保留“超过 TGD ADS+CGC AUC”一个基线：
  - 输出：`outputs/coco500_mass_dist_topk_vs_tgd_ads_cgc_auc_only.md`
  - 输出：`outputs/coco500_mass_dist_topk_vs_tgd_ads_cgc_auc_only.csv`
  - 合并来源：主目录 `results/` 与子目录 `risk-mass-entropy/results/`。
  - 合并后超过 ADS+CGC AUC 的数量：LLaVA `309`（单特征 24、组合 285），Qwen `15`（组合 15），InternVL `38`（单特征 2、组合 36）。
  - 合并 best：LLaVA `vp_target_entropy+cosine16` AUC=`0.922`；Qwen `c_vp+cosine16` AUC=`0.938`；InternVL `c_vp+cosine16` AUC=`0.830`。

## 2026-07-09 COCO4000 token-detector torch MLP 3 seeds（原 9:1 split）
- 本轮目标：在 `token-detector/outputs/{model}/COCO4000` 上跑三模型 torch MLP，seeds=`42/43/44`，并汇总 mean±std。
- 重要口径：
  - 当时 `image_splits.json` 为 train/val/test=`3600/400/400` images，且 `val` 与 `test` 是同一组 400 张图；因此这是 9:1 split，不是 8:2。
  - 训练脚本当前默认 `batch_size=256`、`num_epochs=100`、`positive_class="real"`。
  - 使用原 `COCO4000/results/{model}_selected_feature_sets.json` 中的 10 个 feature set。
- token rows 统计：
  - LLaVA：total=`25487`，train=`23008`，val/test=`2479`。
  - Qwen：total=`14709`，train=`13270`，val/test=`1439`。
  - InternVL：total=`33530`，train=`30224`，val/test=`3306`。
- 输出：
  - 总表：`outputs/coco4000_torch_mlp_3seeds_bs256_summary.md`
  - CSV：`outputs/coco4000_torch_mlp_3seeds_bs256_summary.csv`
  - JSON：`outputs/coco4000_torch_mlp_3seeds_bs256_summary.json`
  - 单模型表：`outputs/{model}/COCO4000/results/{model}_torch_mlp_3seeds_bs256_summary.{md,csv}`
  - artifacts：`outputs/{model}/COCO4000/results/torch_probe_3seeds_bs256/seed_{42,43,44}/...`
- best by AUC：
  - LLaVA：`risk_visual_prompt_relative_vll_cost_geo+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll`，PR/RC/F1/AUC/AUPR=`0.930/0.946/0.938/0.919±0.001/0.985`。
  - Qwen：`risk_visual_prompt_relative_vll_cost_geo+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll`，PR/RC/F1/AUC/AUPR=`0.949/0.982/0.965/0.881±0.009/0.988`。
  - InternVL：`risk_visual_prompt_relative_vll_cost_geo_stateupd_lu1+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll`，PR/RC/F1/AUC/AUPR=`0.946/0.989/0.967/0.872±0.005/0.986`。
- 完整性检查：总表 30 行，三模型各 10 个 feature set，每行 3 seeds。
- 注意事项：
  - 曾中断过早期 batch=16 与 batch=64 的尝试，留下部分 artifacts：
    - LLaVA `COCO4000/results/torch_probe_3seeds/`、`torch_probe_3seeds_bs64/`
    - Qwen `COCO4000/results/torch_probe_3seeds_bs64/`
  - 上述 partial 不作为最终汇总使用；最终 9:1 结果以 `*_bs256_summary` 为准。

## 2026-07-09 TGD COCO4000 ADS/CGC 3 seeds baseline 汇总（跨项目记录）
- 本轮也汇总了相邻项目 `token-grounding-detector/outputs` 的 COCO4000 ADS/CGC torch MLP 三 seed baseline，用于和 token-detector 结果对照。
- 输入目录：
  - `/home/apulis-dev/userdata/CODEX/test-cocochair/token-grounding-detector/outputs/{model}/COCO4000/results/torch_mlp_ads_cgc_seed{42,43,44}/`
  - 每个 seed 下有 `ads/`、`cgc/`、`ads__cgc/` 的 `metrics.json`。
- 输出：
  - `/home/apulis-dev/userdata/CODEX/test-cocochair/token-grounding-detector/outputs/coco4000_tgd_ads_cgc_torch_mlp_3seeds_average.md`
  - `.csv`、`.json`、`_latex.tex`
- 主表使用 `real_*` 指标，mean±std。
- best by AUC 均为 CGC：
  - LLaVA CGC：PR/RC/F1/AUC=`0.928±0.003/0.952±0.002/0.940±0.001/0.921±0.001`
  - Qwen CGC：PR/RC/F1/AUC=`0.939±0.001/0.995±0.002/0.966±0.001/0.877±0.004`
  - InternVL CGC：PR/RC/F1/AUC=`0.938±0.003/0.993±0.001/0.965±0.001/0.866±0.003`

## 2026-07-09 COCO4000-8-2 / 8:1:1 split 与 token-detector torch MLP 3 seeds
- 用户要求将三模型 COCO4000 重新划分为 train/val/test=`8:1:1` 并跑 seeds=`42/43/44` 的 torch MLP。
- 新建目录：
  - `outputs/llava_1_5_7b/COCO4000-8-2/`
  - `outputs/qwen2_5_vl_7b/COCO4000-8-2/`
  - `outputs/internvl_2_5_8b/COCO4000-8-2/`
- split 生成口径：
  - 从原 `COCO4000/image_splits.json` 的 4000 张 image ids 中，用 `random.Random(42)` shuffle。
  - `train=3200`，`val=400`，`test=400`，三者互斥。
  - `features.pkl`、`labeling.json`、`generations.json`、`chair_summary.json`、`coco_ground_truth.jsonl` 通过 symlink 复用原 `COCO4000/` 文件，避免复制大文件。
  - `split_metadata.json` 记录 `split_seed=42` 与 `train_val_test_ratio="8:1:1"`。
- 实际 token rows：
  - LLaVA：train=`20397`，val=`2538`，test=`2552`。
  - Qwen：train=`11782`，val=`1432`，test=`1495`。
  - InternVL：train=`26884`，val=`3330`，test=`3316`。
- 训练口径：
  - `scripts/train_torch_probe_feature_sets.py` 默认 `batch_size=256`、`num_epochs=100`、`positive_class="real"`。
  - feature sets 仍取原 `COCO4000/results/{model}_selected_feature_sets.json` 的 10 个组合。
  - artifacts 放在 `outputs/{model}/COCO4000-8-2/results/torch_probe_3seeds_811_bs256/seed_{42,43,44}/...`。
- 输出：
  - 总表：`outputs/coco4000_8_2_torch_mlp_3seeds_811_bs256_summary.md`
  - CSV：`outputs/coco4000_8_2_torch_mlp_3seeds_811_bs256_summary.csv`
  - JSON：`outputs/coco4000_8_2_torch_mlp_3seeds_811_bs256_summary.json`
  - 单模型表：`outputs/{model}/COCO4000-8-2/results/{model}_torch_mlp_3seeds_811_bs256_summary.{md,csv}`
- best by AUC：
  - LLaVA：`risk_geo_raw+visualcosine_raw`，PR/RC/F1/AUC/AUPR=`0.922/0.952/0.937/0.921±0.002/0.984`。
  - Qwen：`risk_geo_raw+visualcosine_raw`，PR/RC/F1/AUC/AUPR=`0.932/0.991/0.960/0.898±0.005/0.987`。
  - InternVL：`risk_visual_prompt_relative_vll_cost_geo+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll`，PR/RC/F1/AUC/AUPR=`0.921/0.989/0.954/0.861±0.011/0.980`。
- 完整性检查：总表 30 行，三模型各 10 个 feature set，每行 3 seeds；最后确认 GPU 0/1 均空闲。

## 2026-07-09 COCO4000-all 特征提取准备
- 用户要求在三模型上提取 `COCO4000-all`，先只提特征，不训练。
- 新增配置：`configs/model_configs_coco4000_all.yaml`
  - `dataset.num_images=4000`
  - `target_gate_mode=relative`
  - `relative_cost_modes=["geo"]`
  - `dgst_t_dual_scope=true`
  - `source_modes=["legacy_ffn","hmid_proj","hprev_cos","hprev_proj"]`
  - `compute_capped_topmass_085=true`
- 本轮代码新增 capped085 support 落盘：
  - 每层保存局部 support 下标：`*_capped_topmass_085_support_indices_per_layer`
  - 每层保存原始 token 位置：`*_capped_topmass_085_support_positions_per_layer`
  - 覆盖 stem：`dgst_t`、`dgst_t_vv`、`dgst_t_vp`、`dgst_t_vv_source_{hmid_proj,hprev_cos,hprev_proj}`、`dgst_t_vp_source_{hmid_proj,hprev_cos,hprev_proj}`。
- `features/extractor.py` 已保留这些变长 list-of-list 字段到 `features.pkl`。
- 静态检查已通过：`python -m py_compile features/dgst_t.py features/extractor.py`。

## 2026-07-09 relative VLL h_prev logits 配置
- 用户询问 cosine 使用哪种 hidden state，并要求补一个用 `hpre` 计算 relative VLL logits 的对比配置。
- 口径确认：
  - `h_prev` / `hpre`：decoder block 输入，attention 之前。
  - `h_mid`：`h_prev + o_attn`，即 pre-FFN residual state。
  - `h_out`：`h_mid + o_ffn`，即当前代码中的 `layer_hidden`。
  - 现有 `visualcosine_raw`、`cosine16`、`vp_target_cosine` 的 cosine 相似度仍使用 `h_out`：预测 token 的 `prediction_hidden_states` 与 support token 的 `support_output_states`。
  - 本次新增的 `h_prev` 只改变 relative VLL target raw logits 的投影输入，即 support token 用 `h_prev` 过 unembedding 计算 `relative_vll_logits`；不改变 cosine hidden space。
- 代码更新：
  - `models/dgst_capture.py`：`relative_vll_logit_source` 新增 `"h_prev"`，并兼容别名 `"hpre"`、`"h_pre"`、`"prev"`、`"pre"`、`"raw_h_prev"`。
  - `features/dgst_t.py`：直接计算路径同步支持 `"h_prev"`，避免不同调用路径行为不一致。
  - `configs/model_configs_unified.yaml`：更新注释，明确 `h_prev/h_mid/final_norm_h_mid` 与 cosine 的 `h_out` 口径。
- 新增配置：
  - `configs/model_configs_dualscope_relativevll_cost_geo_updlu1_4000_hprevlogits.yaml`
  - 该配置复用 COCO4000 dual-scope VP/VV、geo cost、`state_update` cost state、`update_lambda=1.0` 口径，仅把 `relative_vll_logit_source` 改为 `"h_prev"`。
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile models/dgst_capture.py features/dgst_t.py`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 读取新增配置并验证 normalize 映射：`h_prev/hpre -> h_prev`，`h_mid -> h_mid`，`final_norm_h_mid -> final_norm_h_mid`。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 用 tiny fake capture 验证 `h_prev` 与 `h_mid` 得到的 `relative_vll_logits` 不同，且 `support_output_states` 仍等于 `h_prev + o_attn + o_ffn`。
- 中途有一次 tiny fake capture 验证命令参数误用为 batch 版本的 `prediction_positions`，报错 `TypeError: build_dgst_t_raw() got an unexpected keyword argument 'prediction_positions'`；已改为单样本参数 `prediction_position` 后通过。

## 2026-07-09 COCO500 cosine 平移实验（pingyicos）
- 用户希望验证：把当前 `h_out` cosine 曲线右移一层，近似作为下一层 `h_pre` cosine 后，与 risk 组合是否有效。
- 实现口径：
  - 在 `scripts/train_feature_sets.py` 新增 computed feature aliases：
    - `pingyi_cosine` / `shifted_cosine` / `visualcosine_shift1` / `target_cosine_shift1`
    - `vp_pingyi_cosine` / `vp_shifted_cosine` / `vp_target_cosine_shift1`
  - 当前已训练的是 visual cosine 版本：`pingyi_cosine = [0, visualcosine_raw[0], ..., visualcosine_raw[L-2]]`。
  - `vp_pingyi_cosine` 仅作为别名/计算能力保留，本轮未训练 VP 组合。
- 数据与目录：
  - 复用三模型 `outputs/{model}/COCO500-mass-dist-topk/{features.pkl,image_splits.json,labeling.json,generations.json}`。
  - 新建 `outputs/{model}/COCO500-pingyicos/`，上述文件均用 symlink 指向 `COCO500-mass-dist-topk`，避免复制。
- 训练命令口径：
  - 脚本：`scripts/train_torch_probe_feature_sets.py`
  - feature sets：`risk_geo_raw`、`visualcosine_raw`、`pingyi_cosine`、`risk_geo_raw+visualcosine_raw`、`risk_geo_raw+pingyi_cosine`
  - 三模型：`llava_1_5_7b`、`qwen2_5_vl_7b`、`internvl_2_5_8b`
  - seed=`42`，batch size=`256`，epochs=`100`，positive class=`real`。
- 输出：
  - 单模型结果：`outputs/{model}/COCO500-pingyicos/results/{model}_selected_feature_sets.json`
  - 单模型表：`outputs/{model}/COCO500-pingyicos/results/{model}_selected_feature_sets_table.md`
  - 总表：`outputs/coco500_pingyicos_torch_mlp_summary.md`
  - CSV/JSON：`outputs/coco500_pingyicos_torch_mlp_summary.{csv,json}`
- 主要结果（PR/RC/F1/AUC/AUPR）：
  - LLaVA `risk+cosine`：`0.926/0.907/0.916/0.896/0.978`
  - LLaVA `risk+pingyi_cosine`：`0.929/0.897/0.912/0.893/0.978`
  - Qwen `risk+cosine`：`0.918/0.994/0.954/0.857/0.981`
  - Qwen `risk+pingyi_cosine`：`0.922/0.989/0.954/0.860/0.981`
  - InternVL `risk+cosine`：`0.899/0.990/0.942/0.765/0.962`
  - InternVL `risk+pingyi_cosine`：`0.898/0.985/0.940/0.766/0.959`
- Delta（`risk+pingyi_cosine` - `risk+cosine`）：
  - LLaVA：F1 `-0.004`，AUC `-0.003`，AUPR `+0.000`
  - Qwen：F1 `-0.000`，AUC `+0.003`，AUPR `-0.000`
  - InternVL：F1 `-0.003`，AUC `+0.002`，AUPR `-0.003`
- 初步结论：
  - 平移 cosine 单特征在 Qwen/InternVL 的 AUC 上略高于原始 cosine，在 LLaVA 的 F1 上更高。
  - 与 risk 组合后，平移没有带来稳定大幅提升；Qwen/InternVL 组合 AUC 小幅上升，LLaVA 小幅下降。
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile scripts/train_feature_sets.py scripts/train_torch_probe_feature_sets.py`
  - inline smoke 验证 `pingyi_cosine` 满足 `new[0]=0` 且 `new[1:]=old[:-1]`，`risk_geo_raw+pingyi_cosine` 矩阵维度正确。
  - 训练结束后 `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits` 显示 GPU 0/1 均空闲。

## 2026-07-09 新增真实 hpre cosine 特征提取
- 用户指出第一层本身有 `hpre`，因此不应只用 `[0, h_out_cosine[:-1]]` 的平移近似；本轮按要求新增真实 `hpre cosine` 特征。
- 实现口径：
  - 预测 token：用当前层 block 输入 `h_prev`，由已有 raw 字段精确还原为 `prediction_hidden_states - source_attn_states - source_ffn_states`。
  - support token：用同层 `support_h_prev_states`。
  - target distribution 仍沿用 relative VLL evidence；也就是说只替换 cosine 的 hidden state 空间，不改变 risk/target 构造。
  - 如果旧 raw/features 没有 `source_attn_states`，不会伪造第一层值；新抽取的 raw 会正常输出第一层 hpre cosine。
- 新增 DGST-T 字段：
  - `dgst_t_target_visual_hpre_cosine_relative_vll_per_layer`
  - `dgst_t_target_visual_hpre_cosine16_relative_vll_per_layer`
  - `dgst_t_target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer`
  - `dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer`
  - `dgst_t_target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer`
  - `dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer`
- 更新文件：
  - `features/dgst_t.py`：新增真实 hpre cosine 计算、layer stats、result 字段和 visual-scope merge。
  - `features/extractor.py`：保存上述新字段到 `features.pkl`。
  - `scripts/train_feature_sets.py`：新增训练别名 `hprecosine`、`hprecosine16`、`hprecosine_cap085`、`vp_hprecosine`、`vp_hprecosine16`、`vp_hprecosine_cap085`。
  - `scripts/plot_layerwise_feature_comparison.py`：新增对应画图 alias。
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py features/extractor.py scripts/train_feature_sets.py scripts/train_torch_probe_feature_sets.py scripts/plot_layerwise_feature_comparison.py`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 构造 2 层 tiny raw，验证 6 个 hpre cosine 字段均存在、长度为 2、有限，且第一层非 0；同时验证训练 alias 可解析。
- 中途失败记录：
  - 一次 inline sanity check 直接调用 `feature_block(feat, "hprecosine")`，报错 `KeyError: 'hprecosine'`；原因是 `feature_block` 设计上接收 canonical block，实际训练路径会先通过 `FEATURE_ALIASES` 解析。改为 `feature_block(feat, FEATURE_ALIASES["hprecosine"])` 后通过。

## 2026-07-09 补齐 risk cost 变式提取配置
- 用户指出还有一些 `risk cost` 变式没有保存。检查后确认：
  - `features/extractor.py` 已经会通过通配规则保存所有 `dgst_t_transport_risk_*_per_layer` 和 `dgst_t_score_*` 字段。
  - 之前缺失的主要原因是当前主线配置只设置了 `relative_cost_modes: ["geo"]`，因此 `tbar/sbar/tadd/sadd/tsadd` 没有被生成。
- 已更新以下配置，把 `relative_cost_modes` 从单 `geo` 扩展为六种 cost sweep：
  - `configs/model_configs_mass_dist_topk.yaml`
  - `configs/model_configs_coco4000_all.yaml`
  - `configs/model_configs_unified.yaml`
- 当前会生成并保存的 risk cost 变式包括：
  - `geo`
  - `target_barrier_geo` -> 字段 slug `tbar`
  - `symmetric_barrier_geo` -> `sbar`
  - `target_additive_barrier_geo` -> `tadd`
  - `source_additive_barrier_geo` -> `sadd`
  - `two_end_additive_barrier_geo` -> `tsadd`
- 每种 cost 会同时生成 VV/VP 和 raw/cap085 字段，例如：
  - `dgst_t_transport_risk_relative_vll_cost_tbar_per_layer`
  - `dgst_t_transport_risk_relative_vll_cost_tbar_capped_topmass_085_per_layer`
  - `dgst_t_transport_risk_visual_prompt_relative_vll_cost_tbar_per_layer`
  - `dgst_t_transport_risk_visual_prompt_relative_vll_cost_tbar_capped_topmass_085_per_layer`
- 未改动 `configs/model_configs_dualscope_relativevll_cost_geo_updlu1_4000_hprevlogits.yaml`，因为该文件名和用途明确是 `cost_geo` 对照配置。
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 解析三个 YAML，确认 `relative_cost_modes` 等于六种 cost 列表。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 检查 `scripts.train_feature_sets.FEATURE_KEYS` 中 24 个 cost alias 均存在：6 cost × 2 scope(VV/VP) × 2 raw/cap085。

### 2026-07-09 修正：cost 只保留 geo 和 updlu1
- 用户进一步明确：cost 相关只需要保存 `geo` 和 `updlu1`，不需要保存 `tbar/sbar/tadd/sadd/tsadd`。
- 已将以下配置收窄为：
  - `relative_cost_modes: ["geo"]`
  - `relative_cost_state_modes: ["state_update"]`
  - `relative_cost_update_lambdas: [1.0]`
- 更新文件：
  - `configs/model_configs_mass_dist_topk.yaml`
  - `configs/model_configs_coco4000_all.yaml`
  - `configs/model_configs_unified.yaml`
- 新提取时保留的 cost 字段口径：
  - `dgst_t_transport_risk_relative_vll_cost_geo_per_layer`
  - `dgst_t_transport_risk_relative_vll_cost_geo_capped_topmass_085_per_layer`
  - `dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_per_layer`
  - `dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_capped_topmass_085_per_layer`
  - `dgst_t_transport_risk_relative_vll_cost_geo_stateupd_lu1_per_layer`
  - `dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_stateupd_lu1_per_layer`
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 解析三个 YAML，确认 `relative_cost_modes=["geo"]`、`relative_cost_state_modes=["state_update"]`、`relative_cost_update_lambdas=[1.0]`。

### 2026-07-09 新增 qmatch cost
- 用户要求新增 cost：`cost = d_ij^l + 1 - sqrt(q_i^l * q_j^l)`。
- 实现口径：
  - canonical mode：`semantic_match_geo`
  - 保存/训练 slug：`qmatch`
  - `q_i/q_j` 使用当前 relative VLL 的 `semantic_gate`，即 `_transport_risk_on_support` 里的 `semantic_probs`；实现中由 `source_penalty=1-q_i`、`target_penalty=1-q_j` 还原。
  - cost matrix：`lambda_d * d_ij + 1 - sqrt(q_i*q_j)`；默认 `lambda_d=1.0` 时即用户公式。
- 更新文件：
  - `features/dgst_t.py`：新增 `semantic_match_geo/qmatch/qadd` mode 解析、slug 映射、cost matrix 公式。
  - `scripts/train_feature_sets.py`：新增 `risk_relative_vll_cost_qmatch`、`risk_visual_prompt_relative_vll_cost_qmatch` 及 cap085 aliases。
  - `scripts/plot_layerwise_feature_comparison.py`：新增 qmatch 画图 aliases。
  - `configs/model_configs_coco4000_all.yaml`、`configs/model_configs_mass_dist_topk.yaml`、`configs/model_configs_unified.yaml`：`relative_cost_modes` 更新为 `["geo", "semantic_match_geo"]`。
- 为避免额外保存 `qmatch_stateupd_lu1`，已将 `relative_cost_state_modes` 的 state-update 变式限制为 geo cost；因此仍只保存 `geo_stateupd_lu1`。
- 新提取会保存：
  - `dgst_t_transport_risk_relative_vll_cost_qmatch_per_layer`
  - `dgst_t_transport_risk_visual_prompt_relative_vll_cost_qmatch_per_layer`
  - 若 `compute_capped_topmass_085=true`，还会保存对应 `_capped_topmass_085_per_layer`。
- 验证命令：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile features/dgst_t.py scripts/train_feature_sets.py scripts/plot_layerwise_feature_comparison.py features/extractor.py`
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 小张量验证 `qmatch/qadd/semantic_match_geo` 都满足 `d + 1 - sqrt(q_i*q_j)`，并验证 normalize/slug。
  - `/opt/conda/private/envs/vicr/bin/python - <<'PY' ... PY` 解析三个 YAML，确认 `relative_cost_modes=["geo", "semantic_match_geo"]`，并确认 train/plot qmatch aliases 存在。
- 中途失败记录：
  - 第一次公式小测试直接调用 `_build_cost_matrix(..., cost_mode="qmatch")`，报错 `ValueError: DGST-T only keeps direct/decomposed transport costs.`；原因是 build 函数未兼容短名，只在配置 normalize 路径兼容。已补 `_build_cost_matrix` 对 `qmatch/qadd` 短名的识别后通过。

## 2026-07-10 COCO4000-all 共享样本 GLSim 风格空间热力图
- 用户要求参考 GLSim 的 object-grounding 图格式，展示图像空间区域而不是此前的“layer × token index”矩阵；并要求三模型使用同一组 10 张 COCO 图。
- 新增可复用脚本：`scripts/plot_attention_gate_source_heatmaps.py` 与 `scripts/plot_spatial_attention_gate_source_heatmaps.py`。
- 最终空间图脚本的口径：
  - 仅展示 VV visual scope，因此每个 visual support token 可恢复为图像 patch；层固定为 `5/15/20/25`。
  - 三行分别为 raw attention `A`、未重新归一化的 gate-weighted attention `A * g`、source distribution。
  - 同一层中 raw 和 `A*g` 共用 raw attention 的最大值，避免将 gate 后绝对质量重新拉回 1；gate 行标注 `retained mass = sum(A*g)/sum(A)`。
  - source distribution 仅为定位目的按自身最大值映射颜色；所有图叠加在 COCO 原图上，使用 `turbo` 温度色。非幻觉词若有匹配的 COCO GT 类别则以红框显示；幻觉词通常没有对应 GT 框。
  - LLaVA 使用其 center-square 可见 crop；Qwen 根据 token 数与图像比例恢复动态 patch grid；InternVL 使用 16x16 grid。
- 共享图像：
  - 幻觉：`56433, 136722, 333772, 178156, 305159`
  - 非幻觉：`169361, 160726, 273083, 222118, 356800`
  - 共同候选来自三个 `labeling.json` 的交集（幻觉 340 张、非幻觉 3,586 张）；在固定 seed=42 的候选池中，以三模型 gate 前后 TV 变化的平均 rank 选择最终共享图。
- 输出目录：`outputs/coco4000_all_glsim_style_spatial_maps_shared10/`
  - 共 30 张 PNG 和 30 张 PDF：3 模型 ×（5 幻觉 + 5 非幻觉）。
  - 总览及同图各模型 target word 对照：`summary.md`；图像 id：`shared_image_ids.json`；完整元数据：`selected_samples.{json,csv}`。
- 验证：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile scripts/plot_spatial_attention_gate_source_heatmaps.py scripts/plot_attention_gate_source_heatmaps.py`
  - 小张量检查确认 `A*g` 未重新归一化（raw sum=`1.0`，gated sum=`0.3`）且 LLaVA/Qwen/InternVL 的 24x24、15x23、16x16 grid 推断正确。
  - 输出检查确认 30 个 PNG、30 个 PDF，三模型的每个 label image id 均与 `shared_image_ids.json` 一致；抽查 image `169361` 的三模型图，原图一致且 Qwen/InternVL `dog` 红框对齐。

## 2026-07-10 共享样本 Top-32 空间热力图与 gate 数值分析
- 用户进一步要求解释 gate 后注意力变淡，并只查看 TopK=`32` 的 Raw attention、gate attention 和 source 区域。
- 新增脚本：`scripts/plot_spatial_topk_attention_gate_source_heatmaps.py`。
- TopK 口径：
  - 精确复用上一轮 `selected_samples.json` 中三模型同一组 10 张图、30 个 target token。
  - Raw、`A * gate`、Source 三个信号分别选择自己的 Top-32 visual patches；其余 patch 严格置 0。
  - 每个信号在自己的 Top-32 内做 L1 归一化，再按该 Top-32 图最大值映射温度色；因此该版用于看 Top-32 的空间位置和相对结构，不用于比较三个信号的绝对总质量。
  - 图内同时标注 Top-32 对原始信号的 mass coverage；gate 行还保留 `attention_weighted_gate = sum(A*gate)/sum(A)`，用于解释上一版绝对色阶下的变淡程度。
- gate 数值输出：
  - `gate_values.csv/json` 共 120 行：30 token × 4 layers。
  - 每行包含 gate mean/median/std/min/max、`gate>=0.5` 比例、attention-weighted gate、Raw/Gated Top-32 上的 gate 均值、三种 Top-32 mass coverage 和 Raw/Gated Top-32 overlap。
- 主要解释：
  - 三模型跨样本/层普通 gate mean 分别约为 LLaVA `0.516`、Qwen `0.491`、InternVL `0.507`，符合 sigmoid gate 以中位数为中心的构造。
  - attention-weighted gate 仅为 LLaVA `0.271`、Qwen `0.334`、InternVL `0.460`；高 attention patch 的 gate 低于普通均值，所以共用 Raw 色阶时 `A*gate` 必然更淡，LLaVA 最明显。
  - 当前打开的 LLaVA image `169361`/`frisbee`：L5/15/20/25 的 attention-weighted gate 为 `0.128/0.221/0.384/0.376`；L5 虽然 gate mean=`0.530`，但 Raw Top-32 上 gate mean 仅 `0.168`，Raw/Gated Top-32 overlap 仅 `0.250`，说明 raw 高注意力区域被 gate 强烈重排。
- 输出目录：`outputs/coco4000_all_glsim_style_spatial_topk32_shared10/`
  - 30 PNG、30 PDF、`summary.md`、`gate_values.{csv,json}`、`selected_samples.json`。
- 验证：
  - `python -m py_compile scripts/plot_spatial_topk_attention_gate_source_heatmaps.py`。
  - 小张量验证 TopK mask、Top-2 coverage=`0.6`、TopK 内 L1 sum=`1.0`。
  - 输出检查确认 30 PNG、30 PDF、120 gate rows；抽查 image `169361` 的三模型 Top-32 图，非 TopK 区域无热度且三行空间分布可辨。

## 2026-07-10 COCO500 softmax-gate / attention / source 独立 Top-32 GLSim 热力图
- 用户要求在 COCO500 中选择 5 个幻觉与 5 个非幻觉小物体 token，对 LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL2.5-8B 画层 `5/15/20/25` 的 GLSim 风格空间热力图；三行分别为 `softmax(gate)`、attention、source distribution，各自取自己的 Top-32 并使用独立色阶。
- 新增独立脚本：`scripts/plot_coco500_glsim_softmax_gate_topk_heatmaps.py`，保留此前 COCO4000 的未提交 spatial-map 脚本不变。
- 输入统一使用三模型 `outputs/{model}/COCO500-mass-dist-topk/features.pkl` 的 VV 字段：
  - gate：`dgst_t_vv_semantic_gate_per_layer`，先沿 visual-token 维做 softmax；
  - attention：`dgst_t_vv_support_attention_per_layer`，逐层归一化；
  - source：`dgst_t_vv_source_dist_per_layer`，逐层归一化。
- TopK/色阶口径：
  - 三个信号分别选择自己的 Top-32，非 TopK patch 严格置 0，不在 TopK 内二次归一化；图中 `Top-32 mass` 因此保留对完整分布的 coverage 含义。
  - 同一张图中，每个信号使用一个独立的数值色阶，该色阶仅在本信号的 L5/L15/L20/L25 四列间共享；三个信号之间不共享 scale。
  - 三行均用 `turbo` overlay；非幻觉目标叠加匹配的 COCO GT 红框。
- 小物体样本选择：
  - 三模型使用相同 10 张图片；要求每个模型在该图均有合格同标签 token。
  - 非幻觉使用该图中最小匹配 GT 框面积占比，限制为 `0.1% <= area/image <= 2%`，避免目标太大或小到不可见；幻觉因无匹配 GT 框，使用该 COCO 类别全局 median box/image fraction，并限制 `<=2%`。
  - 排除 person/bus/train/truck/dining table 等语义上偏大的类别；优先选择三模型目标类别 signature 不重复的图片。
  - 幻觉图片：`153445, 469609, 123213, 110601, 183204`；目标包括 remote/book、car、bottle/sports ball、toothbrush、handbag/skis。
  - 非幻觉图片：`460461, 64889, 19484, 436183, 208748`；目标依次为 skateboard、frisbee、knife、sports ball、chair，实际最小 GT 框占比约 `0.10%-0.13%`。
- 输出目录：`outputs/coco500_glsim_softmax_gate_attention_source_topk32_small_shared10/`
  - 30 PNG + 30 PDF：3 模型 × 10 图片；
  - `summary.md`、`selected_samples.json`、`shared_image_ids.json`；
  - `topk_values.{json,csv}` 共 360 行：30 token × 4 layers × 3 signals，包含 TopK indices、mass coverage、独立色阶上限和样本信息。
- 验证：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile scripts/plot_coco500_glsim_softmax_gate_topk_heatmaps.py` 通过。
  - 小张量验证三种信号逐层和为 1、TopK mask 数量正确，softmax(gate) 结果符合预期。
  - 输出确认 PNG/PDF 均为 30 个、统计 360 行、每行 TopK indices=32、layers=`5/15/20/25`、所有 mass/scale 均 finite、无零字节文件。
  - 三信号 Top-32 mass 范围分别为 gate softmax `0.072-0.217`、attention `0.381-0.902`、source `0.073-0.490`；色阶上限范围明显不同，确认三行没有误共享 scale。
  - 目视检查 LLaVA hallucinated remote、Qwen hallucinated toothbrush、InternVL non-hallucinated frisbee：布局、三条独立 colorbar、TopK mask、标题与 GT 框均正常。
- 中途检查说明：第一次 `view_image` 使用相对输出路径时被解析到 `/home/apulis-dev/outputs/...`，报文件不存在；改用绝对路径后完成三张图的目视检查，生成结果本身无错误。

## 2026-07-10 COCO500 support-attention Top-32 与 source 的 JS/KL 曲线
- 用户要求以 `support_attention` 的 Top-K 作为 TK 区域，比较该区域内 attention distribution 与 source distribution 的 JS/KL，并查看三模型幻觉/非幻觉逐层曲线。
- 新增脚本：`scripts/plot_attention_topk_source_divergence.py`。
- 计算口径：
  - 每个 object-token row、每层由原始 `dgst_t_{vv/vp}_support_attention_per_layer` 选择 Top-32，不使用 semantic gate 或 relative-VLL target。
  - Attention 与 `dgst_t_{vv/vp}_source_dist_per_layer` 截取同一组 attention Top-32 位置后，各自在 TK 区域内做 L1 归一化。
  - 同时计算 JS、`KL(attention||source)`、`KL(source||attention)`；使用自然对数和 `eps=1e-12` 平滑。
  - VV（visual support）与 VP（visual+prompt support）都统计；按 object-token row 的 `label=0/1` 分别求 mean 和 SEM。
- 输入均为 `outputs/{model}/COCO500-mass-dist-topk/features.pkl`。实际样本数：LLaVA hall/non=`462/2849`，Qwen=`126/1832`，InternVL=`358/3943`。
- 输出目录：`outputs/coco500_attention_topk32_source_divergence/`：
  - VV/VP 三模型对比图：`{vv,vp}_attention_topk32_source_js_kl_by_label.{png,pdf}`；
  - 完整逐层统计：`attention_topk32_source_js_kl_layerwise.csv`；
  - 汇总：`attention_topk32_source_js_kl_summary.{csv,json}`、`summary.md`。
- 主要结果（全层 H-N 平均差）：
  - VV：LLaVA JS/forward-KL/reverse-KL=`+0.0185/+0.0841/+0.0948`；Qwen=`+0.0177/+0.0909/+0.0769`；InternVL=`+0.0055/+0.0274/+0.0228`。三模型总体均为幻觉 divergence 更高，LLaVA/Qwen 分离强于 InternVL。
  - VP：JS 三模型仍为小幅正差，LLaVA/Qwen/InternVL=`+0.0086/+0.0095/+0.0103`；`KL(source||attention)` 分离更稳定且更大，分别为 `+0.1720/+0.0726/+0.1437`；`KL(attention||source)` 分别为 `-0.0068/+0.0356/-0.0096`，LLaVA/InternVL 平均接近 0 且层间交叉明显。
  - 峰值示例：VV LLaVA reverse-KL 在 L27 的 H-N gap=`+0.2468`；VV Qwen forward-KL 在 L15=`+0.2401`；VP InternVL reverse-KL 在 L7=`+0.2839`。
- 验证：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile scripts/plot_attention_topk_source_divergence.py` 通过。
  - 小张量检查确认相同分布的三种 divergence 为 0，非同分布时均有限非负，JS 不超过 `ln(2)`。
  - 完整输出检查：逐层 CSV 1104 行、summary 18 行；所有 mean/std/SEM finite，标签计数与 features 一致；PNG/PDF 均非空。
  - 已目视检查 VV/VP 两张 PNG，无空白、裁切或图例遮挡。

## 2026-07-10 Attention-TK32 JS/KL torch probe（严格 8:1:1）
- 用户要求用上一节的 JS/KL 逐层散度训练 torch probe，比较三模型效果。
- `scripts/train_feature_sets.py` 新增六个动态训练 block：
  - `vv_attention_tk32_js`、`vv_attention_tk32_kl_attention_source`、`vv_attention_tk32_kl_source_attention`；
  - `vp_attention_tk32_js`、`vp_attention_tk32_kl_attention_source`、`vp_attention_tk32_kl_source_attention`。
- 动态计算完全复用曲线口径：raw support-attention Top-32、attention/source 在同一区域分别 L1 normalize、自然对数、`eps=1e-12`。每条 feature row 首次计算后在内存缓存 VV/VP 三指标，组合训练不重复计算；没有修改或复制原始大 pickle。
- 原 `COCO500-mass-dist-topk/image_splits.json` 的 val=test（50 张完全重合），因此没有沿用。新建：
  - `outputs/{model}/COCO500-attention-topk32-js-kl-probe/`；
  - `features.pkl` 只读软链接原 `COCO500-mass-dist-topk/features.pkl`；
  - 三模型共享严格 train/val/test=`400/50/50`、seed=`42` split，三组互斥。
- 训练设置：torch probe seed=`42`、batch size=`256`、epochs=`100`、positive class=`real`。共 12 个散度 feature sets：6 单特征、VV/VP 各自 all-3、三种同指标跨 scope 组合、全部六特征组合；另在同 split 补跑 `risk_geo_raw+visualcosine_raw` 与 `VP risk+VP cosine` 作为匹配 baseline。
- 主要结果（严格 test AUC）：
  - LLaVA：最好散度为 `VV+VP all 6`，AUC/F1/AUPR=`0.900920/0.945185/0.985271`；最好单散度为 `VP KL(A||S)` AUC=`0.856907`。最好匹配 baseline 为 VP risk+cosine AUC=`0.899341`，六散度高 `+0.001579`。
  - Qwen：最好散度也是最好单特征 `VP JS`，AUC/F1/AUPR=`0.924064/0.971429/0.994561`；匹配 VP risk+cosine baseline AUC=`0.922460`，高 `+0.001604`。VV-only 散度较弱，单特征 AUC=`0.742-0.784`。
  - InternVL：最好散度为 `VV+VP JS`，AUC/F1/AUPR=`0.834537/0.952830/0.977938`；最好单散度为 `VP KL(S||A)` AUC=`0.808155`。最好匹配 baseline 为 VV risk+cosine AUC=`0.861905`，散度低 `-0.027367`。
- 结论：Attention-TK32 divergence 对 Qwen 的 VP-JS 最有效；LLaVA 需要 VV+VP 六条逐层曲线联合后才能达到/略超主线 baseline；InternVL 有中等信号但不能替代 risk+cosine。类别不均衡使 F1 普遍很高，因此本轮以 AUC 为主。
- 汇总输出：
  - `outputs/coco500_attention_topk32_source_divergence/torch_probe_strict811_summary.md`；
  - `torch_probe_strict811_all_results.{csv,json}`；
  - 各模型完整 checkpoint/history/config 位于 `outputs/{model}/COCO500-attention-topk32-js-kl-probe/results/torch_probe/`。
- 验证：
  - `/opt/conda/private/envs/vicr/bin/python -m py_compile scripts/train_feature_sets.py scripts/train_torch_probe_feature_sets.py` 通过。
  - Qwen 首条 row 的六个动态 block 与绘图函数逐层对齐，最大绝对误差约 `1e-7`（float32 保存误差），所有值 finite。
  - 三模型各 14 个结果，共 42 条；所有 PR/RC/F1/Acc/AUC/AUPR finite，42 组 model/history/config artifacts 完整。

## 2026-07-11 TGD COCO500 ADS/CGC 严格 8:1:1 三 seed
- 用户要求检查相邻项目 `token-grounding-detector/outputs` 的 COCO500 严格 8:1:1 三 seed 结果；检查确认原先只有 val=test 的单 seed `torch_mlp_ads_cgc`，因此补跑严格实验。
- TGD 新目录：`token-grounding-detector/outputs/{model}/COCO500-8-1-1/`。`features.pkl` 软链接原 `COCO500/features.pkl`；split 精确复用上一节 token-detector Attention-TK32 probe 的 400/50/50、seed=42 严格 split，三模型 SHA-256 均为 `661d86777e73a11f264973324b8ad1d35e6c7256c65d6aa83cda2595867388ab`，val/test overlap=0。
- 训练：TGD `scripts/train_torch_ads_cgc.py`，feature sets=`ads/cgc/ads+cgc`，probe seeds=`42/43/44`，batch size=`256`，epochs=`100`，trained positive class=`real`。
- 三 seed real-positive mean±population-std：
  - LLaVA best `cgc`：PR/RC/F1/AUC/AUPR=`0.929±0.006/0.947±0.004/0.938±0.002/0.896±0.004/0.985±0.001`；`ads+cgc` AUC=`0.875±0.001`，`ads`=`0.701±0.016`。
  - Qwen best mean `cgc`：`0.944±0.000/1.000±0.000/0.971±0.000/0.912±0.019/0.994±0.002`；`ads+cgc` AUC=`0.902±0.004`，均值略低但更稳定；`ads`=`0.799±0.011`。
  - InternVL best `cgc`：`0.903±0.001/0.998±0.001/0.949±0.000/0.826±0.007/0.975±0.001`；`ads+cgc` AUC=`0.816±0.035`，`ads`=`0.704±0.016`。
- 严格 test object-token rows：LLaVA 370（H/R=45/325），Qwen 181（11/170），InternVL 451（45/406）。Qwen hallucination test rows 很少，F1/PR/RC 受类别不均衡影响明显，因此仍以 AUC 为主；三 seed 只测初始化波动，不覆盖 split 波动。
- TGD 汇总输出：
  - `token-grounding-detector/outputs/ads_cgc_coco500_8_1_1_torch_mlp_3seeds_811_bs256_summary.{md,csv,json}`；
  - seed 明细：`..._details.csv`；每模型每 seed 的 checkpoint/history/config/metrics 位于 `COCO500-8-1-1/results/torch_mlp_ads_cgc_seed{42,43,44}/`。
- 与 token-detector 同 split、同 seed42 的直接比较：TGD best AUC 为 LLaVA CGC=`0.896684`、Qwen ADS+CGC=`0.901070`、InternVL CGC=`0.826273`；Attention-TK32 divergence best 分别为 `0.900920/0.924064/0.834537`，差值 `+0.004236/+0.022995/+0.008265`；risk+cosine best 为 `0.899341/0.922460/0.861905`。
- 对比文件：`token-grounding-detector/outputs/ads_cgc_coco500_strict811_seed42_vs_token_detector.{md,csv,json}`。
- 完整性检查：3 models × 3 seeds × 3 feature sets=`27` 份 metrics，全部 summary 的 `val_test_same_images=false`，所有指标 finite，27 组 model/history/config/metrics artifacts 完整。

## 2026-07-10 COCO500 Top-32 区域内局部 softmax GLSim 热力图
- 用户指出原 softmax-gate / attention / source Top-32 图中 attention/source 偏浅，希望对各自选中的 Top-32 数值再做一次 softmax。
- 更新 `scripts/plot_coco500_glsim_softmax_gate_topk_heatmaps.py`：
  - 新增 `--softmax-within-topk`；先按原分布选择 Top-32，再只对这 32 个保留值做 softmax，并 scatter 回原 visual grid；非 TopK 仍严格为 0。
  - 新增 `--topk-softmax-temperature`，默认 `T=1.0`；本轮按用户要求使用默认温度。
  - TopK indices 和原始 mass coverage 均在局部 softmax 前计算并保留；图内标注改为 `Pre-softmax Top-32 mass`，避免把可视化用的局部 softmax 误读为原始分布。
  - `topk_values.{json,csv}` 新增 `softmax_within_topk`、temperature 和 `display_topk_sum`。
- 新输出目录：`outputs/coco500_glsim_softmax_gate_attention_source_topk32_localsoftmax_small_shared10/`；保留上一版原始 mass 图不覆盖。
  - 仍为三模型 × 10 张共享图片，共 30 PNG + 30 PDF、360 条 signal/layer stats。
  - 样本、TopK indices、softmax 前 coverage 与上一版逐条一致。
- 结果口径：局部 softmax 后每层 32 个显示值之和为 1；attention 的显示峰值范围约 `0.0338-0.0453`，source 约 `0.0313-0.0337`。因此图明显更亮，但尤其 source 的 32 个位置会接近等权，适合看 TopK 空间位置，不适合比较原始强弱。
- 验证：
  - py_compile 与小张量测试通过；示例 `[0.4,0.3]` 的局部 softmax 为 `[0.5250,0.4750]`，原 coverage=`0.7` 保持不变。
  - 完整输出检查确认 30 PNG、30 PDF、360 rows；所有 `display_topk_sum` 在浮点误差内等于 1，无零字节文件。
  - 与旧版逐条比较，TopK indices 和 pre-softmax coverage 完全相同。
  - 已目视检查 Qwen hallucinated toothbrush 与 InternVL non-hallucinated frisbee，新版 attention/source 橙红区域明显增多，布局、独立 colorbar、GT 框与标注正常。

## 2026-07-10 COCO500 Raw relative-VLL logits 与三信号四行 Top-32 对比
- 用户进一步要求查看 visual-token raw logits Top-32 热力图，并与 gate、attention、source 三行比较，覆盖三模型。
- 原 `features.pkl` 没有直接保存 VV raw-logit tensor，但可从同一条记录精确/近似反解：
  - VP semantic gate 覆盖 visual+prompt 全 support，顶层 `dgst_t_layer_stats` 保存每层 `visual_prompt_relative_vll_logit_median/MAD`；
  - 使用 `raw = median + (MAD + 1e-6) * logit(vp_gate)` 恢复全 support raw logits，再按 VP/VV support positions 映射并截取 visual tokens；
  - 只有精确饱和为 0/1 的 float32 gate 极端值无法恢复无限精度，本轮三模型最大饱和比例约 `0.7%`，这些点使用相邻浮点边界；median 误差为 0，MAD 最大绝对误差约 `1e-6`。
- 新增脚本：`scripts/plot_coco500_glsim_raw_logits_fourway_topk_heatmaps.py`。
- 四行口径：
  1. Raw relative-VLL logits：直接选最大的 Top-32，保留有符号 raw 数值，使用独立 `coolwarm` 数值色阶，不做 softmax；
  2. `softmax(gate)`：自己的 Top-32 后局部 softmax `T=1`；
  3. attention：自己的 Top-32 后局部 softmax `T=1`；
  4. source distribution：自己的 Top-32 后局部 softmax `T=1`。
- 每行色阶仅在该图的 L5/L15/L20/L25 共享，四行之间独立。沿用上一轮相同 10 张共享小物体图片、target token 和 GT 红框。
- Raw logits 与 VV gate 是逐层单调关系，因此理论 Top-32 位置应一致；实际 30 token × 4 layers 的 Raw/Gate Top-32 overlap 全部为 `1.0`。
- 输出目录：`outputs/coco500_glsim_raw_logits_gate_attention_source_topk32_localsoftmax_small_shared10/`：
  - 30 PNG + 30 PDF；
  - `summary.md`、`selected_samples.json`、`shared_image_ids.json`；
  - `topk_values.{json,csv}` 共 480 行；
  - `raw_logit_reconstruction.{json,csv}` 共 920 行，记录三模型所有 decoder layer 的 median/MAD 恢复误差与饱和比例。
- 选定四层的 Raw-logit Top-32 数值总范围：LLaVA 约 `0.23-10.49`，Qwen 约 `-32.75-32.50`，InternVL 约 `-0.64-9.44`；每张图使用自身 raw 四层范围，避免跨模型量纲直接混用。
- 验证：
  - py_compile 与合成 raw→gate→raw 反解测试通过，示例 visual 子集 `[1,3,5]` 精确恢复；
  - 输出检查确认 30 PNG、30 PDF、480 Top-K rows、920 reconstruction rows、无零字节文件；
  - 所有 Top-K indices 长度为 32，层为 `5/15/20/25`，Raw/Gate overlap 全为 1；
  - 已目视检查 LLaVA hallucinated remote、Qwen hallucinated toothbrush、InternVL non-hallucinated frisbee，四行布局、Raw 数值色条、局部-softmax三行、GT 框和标注均正常。

## 2026-07-11 VV Attention-TK32 JS/KL、risk、cosine 严格 8:1:1 三 seed
- 用户要求只做 VV scope，对 JS、双向 KL、risk、cosine 训练单特征，并分别比较散度/risk 加 cosine 的组合；不包含 VP。
- 9 个 feature sets：`vv_attention_tk32_{js,kl_attention_source,kl_source_attention}`、`risk_geo_raw`、`visualcosine_raw`、三种 `VV divergence+visualcosine_raw`、`risk_geo_raw+visualcosine_raw`。
- 输出目录：
  - 每模型每 seed：`outputs/{model}/COCO500-vv-attention-topk32-js-kl-risk-cosine-3seeds/seed{42,43,44}/`；
  - 总汇总：`outputs/coco500_vv_attention_topk32_js_kl_risk_cosine_3seeds_strict811/{summary.md,three_seed_summary.csv,three_seed_summary.json,all_seed_results.csv}`。
- 训练口径：严格共享 train/val/test=`400/50/50`、split seed=`42`，probe seeds=`42/43/44`，batch size=`256`、epochs=`100`、positive class=`real`；三模型共 81 个 probe runs。
- 三 seed mean±population-std：
  - LLaVA best `VV JS + cosine`：PR/RC/F1/AUC/AUPR=`0.940±0.006/0.943±0.002/0.941±0.002/0.902±0.004/0.985±0.001`。其后为 `KL(S||A)+cosine` AUC=`0.891±0.001`、`KL(A||S)+cosine`=`0.887±0.002`；cosine 单特征=`0.884±0.006`，risk+cosine=`0.866±0.008`。
  - Qwen best `VV JS + cosine`：`0.948±0.003/0.992±0.003/0.969±0.003/0.884±0.023/0.990±0.003`。risk+cosine=`0.870±0.041`，cosine 单特征=`0.860±0.015`；Qwen seed 波动较明显。
  - InternVL best `VV risk + cosine`：`0.908±0.001/0.988±0.003/0.946±0.001/0.842±0.024/0.976±0.005`。`JS+cosine` 排第二 AUC=`0.816±0.012`，`KL(S||A)+cosine`=`0.813±0.013`，cosine 单特征=`0.783±0.006`。
- 单散度都弱于同模型 cosine；最佳单散度分别为 LLaVA `KL(S||A)=0.833±0.005`、Qwen `JS=0.790±0.020`、InternVL `KL(S||A)=0.725±0.012`。
- cosine 对所有 base feature 均提升 mean AUC：
  - LLaVA：JS/KL(A||S)/KL(S||A)/risk 分别 `+0.072/+0.061/+0.058/+0.079`；
  - Qwen：`+0.094/+0.066/+0.069/+0.022`；
  - InternVL：`+0.105/+0.085/+0.088/+0.113`。
- 结论：VV divergence 主要作为 cosine 的互补信号，而不是独立替代项。LLaVA/Qwen 中 JS+cosine 最强；InternVL 仍以 risk+cosine 最稳，JS+cosine 有增益但未超过它。
- 验证：3 models × 3 seeds × 9 sets=`81` 条结果均存在，全部 PR/RC/F1/Acc/AUC/AUPR finite，model/history/config artifacts 完整；严格 split 三组互斥。
- 中途修正：首次创建 seed 目录时相对软链接多写了一层 `..`，在任何训练启动前已改为正确的 `../../...` 并逐一用 `test -f` 与 `readlink -f` 验证，没有产生错误训练结果。

## 2026-07-11 COCO500 GateAttention 与模型最后层五行空间对比
- 用户要求在 Raw logits / gate / attention / source 对比中增加 GateAttention，并进一步追加每个模型的真正最后一个 decoder layer。
- 更新 `scripts/plot_coco500_glsim_raw_logits_fourway_topk_heatmaps.py`：
  - 新增 `--include-gated-attention`：第五个信号使用原 DGST-T 定义 `norm(A * g)`，其中 `A` 为 VV support attention、`g` 为保存的 raw sigmoid semantic gate；没有误用 `softmax(gate)` 与 attention 相乘。
  - GateAttention 自己选择 Top-32，并只为显示在这 32 个值内做局部 softmax；图内同时标注归一化前的 `A×g retained = sum(A*g)/sum(A)`。
  - `topk_values.{csv,json}` 新增 attention/GateAttention 与 gate/GateAttention 的 Top-32 overlap，以及 GateAttention retained mass。
  - 新增 `--include-last-layer`：在用户指定的 L5/15/20/25 后自动去重追加当前模型实际最后层；LLaVA/InternVL 为 L32，Qwen 为 L28，不硬编码同一个层号。
- 第一版五行、四层输出：`outputs/coco500_glsim_raw_logits_gate_attention_gatedattention_source_topk32_localsoftmax_small_shared10/`：
  - 30 PNG + 30 PDF，600 Top-K rows，920 reconstruction rows，无零字节文件。
  - 跨 4 层与 10 个样本，Attention/GateAttention Top-32 mean overlap：LLaVA `0.706`、Qwen `0.762`、InternVL `0.784`；Gate/GateAttention 仅为 `0.224/0.188/0.292`。GateAttention 因而主要是在原 attention 上门控重排，并非复制纯 gate 热区。
  - `A×g retained` 均值分别为 `0.479/0.496/0.580`；幻觉/非幻觉总体均值 `0.521/0.515`，这 10 张可视化样本中没有明显 label 差异。
- 追加最后层的新输出：`outputs/coco500_glsim_raw_logits_gate_attention_gatedattention_source_topk32_localsoftmax_small_shared10_lastlayer/`：
  - 仍为 30 PNG + 30 PDF；5 signals × 5 layers × 30 tokens=`750` Top-K rows，reconstruction 仍为 920 rows。
  - 最后层 Attention/GateAttention mean overlap：LLaVA L32=`0.650`、Qwen L28=`0.769`、InternVL L32=`0.684`；最后层 retained mass 均值为 `0.504/0.536/0.545`。
- 验证：
  - py_compile、`git diff --check` 通过；Top-K row 信号/层数、每行 32 indices、有限值、Raw/Gate overlap=`1.0`、PNG/PDF 计数和非空性均通过断言。
  - 已目视检查 LLaVA hallucinated bottle、Qwen hallucinated toothbrush、InternVL non-hallucinated frisbee 的五行五列图；模型最后层号、布局、独立行色条、标注与 GT 框正常。
  - 中途 GateAttention 小张量测试第一次使用了乘积并列的输入，却错误断言唯一 argmax，故测试预期失败；改成无并列输入后验证 `A*g=[0.1,0.1,0,0.2]`、retained=`0.4`、归一化=`[0.25,0.25,0,0.5]` 通过，实现本身无需修正。

## 2026-07-14 Qwen3-VL-8B-Instruct COCO caption 接入
- 新增独立 `models/qwen3_vl_wrapper.py`，没有复用或修改 Qwen2.5 的 `models/qwen_wrapper.py`；当前只支持图片 caption generation，特征提取会显式抛出 `NotImplementedError`，避免误走未经验证的 Qwen2.5 内部结构。
- `models/__init__.py` 注册新 model key `qwen3_vl_8b`；`configs/model_configs_unified.yaml` 新增本地模型 `/home/apulis-dev/userdata/models/Qwen3-VL-8B-Instruct`，文本层数 36、patch size 16、`max_new_tokens=512`。
- generation 使用模型原生 `Qwen3VLProcessor.apply_chat_template`、greedy decoding 和 SDPA attention；输出继续复用项目 `GenerationOutput`，保存 caption、response token IDs 与逐 token 文本。
- 真实单图 smoke 已通过：模型完整加载到 RTX 5090，识别 36 个文本 decoder layers，图片输入、caption 解码和 response token IDs 均有效。
- COCO caption 使用 `coco-labeling/label_coco.py --model qwen3_vl_8b --num-images 4000 --generation-devices cuda:0 cuda:1 --resume`；并行 worker 会持续写入 `generation_shards/`，支持中断恢复。

## 2026-07-13 POPE / CLEVR-Exist 问答式幻觉检测

- 该阶段最初使用独立 QA 配置；现已将同一套 QA 设置合并到 `configs/model_configs_server_fj01.yaml` 的 `qa_benchmarks` 段，模型与训练参数继续和 COCO 共用。
- 固定数据协议已生成到 `/root/rivermind-data/dataset/qa_benchmarks/`：
  - POPE：官方 random/popular/adversarial 共 9000 问；seed=42 按 500 张共享图片严格切为 400/50/50，对应 7200/900/900。
  - CLEVR：只使用官方 program 末操作为 `exist` 且有 yes/no GT 的问题；官方 train 抽 4000，官方 val 先切互斥 image pool 再各抽 val/test 500，共 5000。
  - POPE questions SHA-256：`d8adf50411c7e4521f1b70e52b1e2965b4bfb0f8d2381528bb423a082264663f`。
  - CLEVR questions SHA-256：`dd14278a586c986b50bfaac607f00e1e11bab59335848657a194ea070d7e7a0a`。
- 新增统一 `Generate -> Label -> Extract -> Probe` 实现：
  - `data/qa_benchmark.py`、`scripts/prepare_qa_benchmarks.py`：统一 schema、答案规范化、标签/error_type、原子 JSONL、严格 split、哈希与图片泄漏检查。
  - `features/qa_extractor.py`、`scripts/qa_pipeline.py`：greedy/8 tokens、断点续跑、原子分片、answer-target 十条 risk+hprecosine、POPE object-target、ADS/CGC/uncertainty/SVAR/中层逐头 attention；`features.pkl` 不保存 full logits 和 GT。
  - 三个 wrapper 的特征接口新增可选 `prompt` 参数，默认 caption 行为保持不变；保证二次 forward 使用原 QA 问题而非硬编码 `Describe this image.`。
  - `detection/qa_probe.py`、`scripts/train_qa_probes.py`：Torch MLP 128/64/32、dropout=0.3、batch=256、100 epochs、class-weighted BCE、val loss checkpoint、val macro-F1 阈值、seeds=42/43/44、完整双类指标和 mean±std。
  - `scripts/validate_qa_artifacts.py`、`scripts/summarize_qa_generations.py`：完整性/有限值/无 GT 泄漏检查和原模型分组指标。
- 根据 `token-CVPR26.pdf` 另增不混入主结果的论文兼容轨道：
  - `detection/pope_paper_protocol.py`、`scripts/train_pope_paper_protocol.py`。
  - 只保留模型回答 Yes 的样本，hallucination 为正类，ADS 使用 answer-position，CGC 使用 question object-position，5-fold，报告 hallucination-F1/AUC；ADS、CGC、ADS+CGC 与 MLP/RF/XGB 分开保存。
- smoke test：三模型 × POPE 10 问 × CLEVR 10 问全部完成；generation/label/features 均为 10 条且 key 集一致。
  - LLaVA/InternVL 每个 answer/object DGST 均 32 层；Qwen 均 28 层。
  - POPE 三模型 object-position CGC 均非空；所有逐层特征有限；`features.pkl` 无 `gt_*` 字段。
  - Qwen 的 label 阶段重复运行后仍为 10 条，验证去重 resume。
- 测试：`/opt/conda/envs/td/bin/python -m pytest -q tests` -> `16 passed`；包含 split/标签、特征集合、加权 Torch probe、paper protocol 和语义 token 定位。
- 中途失败记录：
  - 两个运行环境最初均缺少 pytest，`python -m pytest` 报 `No module named pytest`；已在 `/opt/conda/envs/td` 安装 pytest 9.1.1。
  - 首次 LLaVA POPE object-CGC smoke 10/10 报 `Object position 1162 out of bounds (seq_len=604)`；旧代码硬编码 image token `-200`，与当前 processor/model config 不一致。已改为读取 `model.config.image_token_index`，重跑 10/10 通过。
  - 一次只读内联统计命令因 SSH here-doc 引号丢失报 `NameError: name 'pope' is not defined`；数据准备本身未失败，随后由 manifest 和验证脚本确认计数。
  - 首次后台 smoke 启动命令的嵌套 `bash -lc` 引号被 SSH 展开，跳过 LLaVA-CLEVR/InternVL worker；随后改为独立脚本/显式命令补跑，六组 smoke 均完成。
- 全量执行已启动：
  - 当前并行任务：GPU0 `llava_1_5_7b/pope`，GPU1 `qwen2_5_vl_7b/pope`；输出根目录 `outputs/qa_benchmarks/`。
  - `scripts/run_qa_full_worker.sh` 提供单模型/单数据集可恢复执行入口。
  - 后台 `scripts/run_qa_full_coordinator.sh` 会等待两项初始 POPE 完成并严格验证 9000 条，再依次执行 InternVL POPE、三模型 CLEVR、原模型汇总、六组三 seed probe，最后运行 LLaVA 的论文兼容 5-fold ADS+CGC。
  - 调度日志：`outputs/qa_benchmarks/logs/coordinator.log`；仅当全部上游验证通过才会出现 `[coordinator] ALL COMPLETE`。

## 2026-07-13 cost-variant exact EMD 并行加速

- 用户要求将每层十种 cost-variant EMD 并行，利用服务器多核 CPU。
- 实现位于 `features/dgst_t.py`：
  - 默认仍为串行，历史实验行为不变。
  - `DGST_COST_VARIANT_EMD_WORKERS=10` 开启十路并行。
  - `DGST_COST_VARIANT_EMD_BACKEND=process` 使用 spawn 的持久进程池，避免 CUDA fork。
  - 先在主进程完成 CUDA 上的 support/cost 构造并转为 NumPy；再把 32 层按十种 risk 组成十个批任务，每个 CPU worker 顺序求解一种 risk 的全部层，显著降低 IPC/线程池调度开销。
  - worker 内限制 BLAS/OpenMP 单线程，外层十个进程是唯一并行层级。
- `scripts/run_qa_full_worker.sh` 默认启用 `10 workers + process backend`。
- 数值验证：
  - 单元测试串行/并行 exact EMD 逐 risk 一致，当前 `pytest -q tests` 为 `18 passed`。
  - LLaVA POPE 真实 10 题串行与最终 process-batch 输出比较：answer/object 两个 target、十条 risk、32 层的 `max_abs_delta=0.0`。
  - 最终并行产物 10/10 generation/labels/features 完整、object-CGC 非空、层数正确且全部有限。
- 性能验证：
  - 纯 CPU 合成 32 层×10 EMD：串行 `1.0717s`，初版线程并行 `0.4177s`，EMD 子阶段约 `2.57x`。
  - LLaVA POPE 真实端到端稳态：串行约 `2.82s/题`，最终 10-process layer-batch 约 `2.47s/题`，约提升 `12%`；模型 forward、object-CGC 和非 EMD DGST 仍占主要剩余时间。
- 中途失败与修复：
  - 首次基准尝试 `/usr/bin/time`，系统不存在该路径，报 `env: '/usr/bin/time': No such file or directory`；改用 shell `time`/tqdm 计时。
  - 初版“每层创建十线程”虽然纯 EMD 快，但真实端到端 10 workers 为 `3.09s/题`，慢于串行 `2.82s/题`；改为跨 32 层批处理。
  - 首版 process-batch 直接传数百个 CPU torch storage，触发 `OSError: [Errno 24] Too many open files`，该次 10/10 均进入 extraction failures、未写 feature shard；改为 NumPy 普通 pickle payload 后重跑 10/10 通过。
  - 并行改造前已向两个 qa_pipeline 和旧 coordinator 发送 TERM；原子分片未损坏，恢复时会从现有 key 继续。
  - 首次恢复时发现 `labels.jsonl` 的已有行会被逐批重复原子重写；结果不重复但恢复 I/O 浪费。`JSONLCheckpointStore.add` 已改为内容相同直接 no-op，并新增 mtime/dirty 单测。
- 全量任务已用最终配置恢复：GPU0 LLaVA、GPU1 Qwen，各自持久化 10 个 EMD worker（共 20 个 spawn worker）；恢复起点分别为 18/3 个 feature shards，旧串行分片与新并行分片因数值完全一致可安全混合。

## 2026-07-13 Sinkhorn 小批量配对实验

- 为回答 Sinkhorn 是否影响结果，暂停 exact 主任务并建立独立实验目录 `outputs/qa_sinkhorn_comparison/`；主实验分片未修改。
- 固定样本：从已完成的 LLaVA POPE exact shards 中 seed=42 抽取 100 条，严格 50 real / 50 hallucination；复用相同 generation、labels、问题和图片，只替换 OT solver。
- 新增仅通过环境变量启用的实验路径：
  - `DGST_OT_SOLVER_OVERRIDE=sinkhorn`
  - GPU batched log-domain Sinkhorn，最终 risk 仍为 `<C, Pi_epsilon>`。
  - exact EMD 仍是默认路径，历史/主实验配置不变。
- 验证：`pytest -q tests` -> `19 passed`；100 条 `reg=0.10/0.05` 产物的 key、标签、object-CGC、层数和有限值通过。
- 配对结果（十条 risk 的 target 内平均；小样本 probe 为固定 5-fold class-weighted logistic，仅用于配对趋势）：
  - `reg=0.10`，100 条：
    - answer：MAE `0.04815`，Pearson/Spearman `0.99395/0.99310`，AUC exact/sinkhorn `0.75700/0.77332`，F1 `0.68946/0.71730`。
    - object：MAE `0.04492`，Pearson/Spearman `0.99420/0.99422`，AUC `0.68012/0.68612`，F1 `0.62892/0.63369`。
  - `reg=0.05`，100 条：
    - answer：MAE `0.01719`，Pearson/Spearman `0.99931/0.99916`，AUC `0.75700/0.75368`，F1 `0.68946/0.69384`。
    - object：MAE `0.01493`，Pearson/Spearman `0.99935/0.99926`，AUC `0.68012/0.68164`，F1 `0.62892/0.62276`。
  - `reg=0.02`：500 iterations 下即使 marginal tolerance 放宽到 `5e-4`，仍有 14/100 未收敛；86 条共同样本 MAE answer/object `0.00632/0.00537`，Pearson 均约 `0.9999`，probe 变化很小，但不能作为完整可靠数据。
  - hprecosine 不依赖 OT，三档与 exact 的最大差异均为 `0.0`。
- 速度结论：
  - 当前 exact 10-process layer-batch 稳态约 `2.47s/题`。
  - Sinkhorn `reg=0.10` 约 `2.56s/题`；`reg=0.05` 约 `3.3s/题`；`reg=0.02` 补跑约 `4.0s/题`。
  - 较小正则需要更多 log-domain 迭代，越接近 exact 越慢；整体还受模型 forward 和额外 object-CGC forward 主导。
- 决策：不把主实验切换到 Sinkhorn。保留 Sinkhorn 为独立可复现实验开关；主实验继续 exact EMD，避免定义变化和收敛缺失。
- 中途失败记录：
  - `reg=0.05` 在严格 `1e-4` marginal error 下有 1 条略超（`1.099e-4`）；放宽到 `5e-4` 后补齐。
  - `reg=0.02` 严格阈值时 99/100 失败；放宽到 `5e-4` 后仍有 14 条超过阈值，未伪装为有效结果。
  - 一次批量验证命令错误转义 shell 变量，使用了字面路径 `$reg`，报 `FileNotFoundError`；随后改用两个显式目录完成验证。
  - Sinkhorn 实验后 exact 主任务已从原子分片恢复：恢复时 LLaVA/Qwen 分别已有约 40/20 个 shards；默认 worker 没有 `DGST_OT_SOLVER_OVERRIDE`，继续使用 exact EMD。
  - 多次 TERM 暂停父进程后发现 20 个旧 ProcessPool worker 被 reparent 到 PID 1；第一次 awk 清理命令因远程引号丢失报语法错误，随后用 `pkill -TERM -P 1 -f spawn_main` 仅清除孤儿 worker。当前保留的 20 个 spawn worker 均分别属于正在运行的两个 qa_pipeline。

## 2026-07-14 QA benchmark results summary

- 三模型 POPE/CLEVR generation、label、feature extraction、严格 8:1:1 probe 与 LLaVA POPE ADS/CGC 五折实验均已完成，coordinator 标记为 `[coordinator] ALL COMPLETE`。
- 新增统一汇总：`outputs/qa_benchmarks/QA_RESULTS_SUMMARY.md`。
- 汇总包含原模型准确率、六组最佳 probe、九组 ADS/CGC 五折结果、协议解释、互补性诊断和限制。
- 按用户要求将同一 Markdown 扩展为完整结果文档：新增原模型 GT/source/error/question-family 分组表，并列出六个模型/数据集下全部 231 个 feature-set 聚合结果。每项包含 seeds 42/43/44 的 Accuracy、Balanced Accuracy、Macro-F1、AUROC、hallucination/real Precision、Recall、F1 与 AUPR 的 mean +/- std。
- 五折诊断确认：3850 个 LLaVA Yes-response 中 298 个 hallucination（7.74%）；ADS/CGC 层均值相关系数为 -0.023，跨 block 层对绝对相关均值为 0.041。
- 重要限制：当前论文式五折按问题行划分而非按 image 分组；各折 99.5%-100% 的测试图片也存在于训练折，因此 0.703 的 ADS+CGC MLP Hall-F1 不能视为严格 image-disjoint 结果。

## 2026-07-15 Beyond Global Scores baseline 子系统

- 新增独立 `features/baseline/`：实现 MetaToken（LR/GB 所需 10+H 特征）、SVAR、DHCP、ProjectAway detection 与 HalLoc；根特征仍保持原 schema，baseline 产物写入各实验的 `baseline/`。
- `BaselineRuntime` 可由联合抽取与 baseline-only 共用：自动合并 capture requirements，每张图共享一次 LVLM 输出、ProjectAway 目标 token union 投影和 HalLoc CLIP cache；DHCP 使用 float16 原子 shard，并在 `features.pkl` 提交前按图 flush，保证断点恢复不出现悬空引用。
- MetaToken 使用 wrapper 返回的六条 compact caption statistics，不保存 `[T,V]`；ProjectAway 同时按 visual row 与 vocabulary 分块做 exact softmax denominator，不生成或持久化 `[L,P,V]`，也不复制整张 fp16/bf16 LM head 为 fp32。
- HalLoc 使用预训练 `openai/clip-vit-base-patch32`（抽取阶段冻结并缓存）与预训练 `uclanlp/visualbert-vqa-coco-pre`，训练 text/visual projection、VisualBERT 与单 object head；存储标签仍为 `0=hallucination, 1=real`，所有 detector 显式转换为 `1=hallucination` 后训练和评估。
- 新增 `scripts/extract_baselines.py` 与 `scripts/train_baselines.py`；训练只用 train 拟合、val 选 checkpoint/阈值、test 最终报告，并验证 image-level strict 80/10/10 split。
- 合成单元测试覆盖精确公式、compact MetaToken 等价性、ProjectAway 双向分块 exact softmax、DHCP 保质量 resize/shard/resume、HalLoc 冻结与融合、标签方向、分类器结构、最小 capture requirements 和联合 runtime；`python -m unittest -v tests.test_baselines` 当前 11/11 通过；全仓 `unittest discover` 当前 40/40 通过。

## 2026-07-16 DGST target-comparison 显存修复与 raw-attention 对照

- 定位到四分支 OOM 的实际原因：hpre/hmid 的完整词表投影在启用 softmax 分支时仍处于 autograd，逐层保存 compact probability 会把 LM-head 计算图和完整词表中间量一并保留。现将 joint raw/prob 投影、target-row raw 投影和整个 compact target-comparison 路径置于 no-grad/inference-mode，并显式 detach。
- 词表投影改为按启用分支执行：`raw_attention` 不做词表投影；raw-only 只乘目标词 unembedding；同一 state 的 raw+softmax 共用一次分块完整词表投影；hpre/hmid 未启用时完全跳过。
- 新增 `raw_attention` 对照：target distribution 直接采用模型 post-softmax attention 的 head mean，截取视觉 key 后在视觉区域重新归一化；不使用 Gaussian gate，保存全 1 gate 仅用于矩阵 schema 对齐。它独立输出 sqrt-hpre risk、top-32 hpre target-cosine 和 EV，并写入 `dgst-target-comparison-v2` 特征 schema。
- YAML、`scripts/extract_features.py`、pipeline 分支过滤和 probe alias 已全部支持 `raw_attention`；活动 YAML 默认不设置 `max_pixels`，需要时可直接在模型配置中显式添加。
- 验证：全仓 `unittest discover` 59/59 通过，覆盖 ambient grad 下无计算图、按需 projection 次数、raw-attention 手算 target/top-K/cosine/EV、CLI/config/training 集成；`git diff --check` 与 `bash -n run.sh` 通过。
- Qwen3-VL-8B 真机一图 smoke（RTX 5090，COCO 640x480，未设置 max/min pixels）：原生 `grid_thw=[1,30,40]`、合并后 P=300，四个 Gaussian 分支加 raw-attention 同时完成；所有 `[36,300]` 矩阵与 `[36]` 曲线 finite、CPU、无 grad graph。峰值 allocated/reserved 为 17.012/17.121 GiB，模型加载后的抽取增量约 0.671/0.748 GiB。

## 2026-07-15 run.sh 三阶段薄入口

- 参考 `token-grounding-detector/run.sh`，当前 `run.sh` 只顺序执行三条清晰命令：`generate_and_label.py`、`extract_features.py`、`train_and_eval.py`。Shell 只保留 model、output、config、device、cache 和 `--resume` 等运行参数。
- `run.prompt`、`run.extraction_mode`、DGST branches、baseline 开关、训练器、正类方向、probe 参数和 feature sets 全部由 `configs/model_configs_unified.yaml` 管理，不再出现在 `run.sh`。
- `extract_features.py` 根据 YAML 自动路由 `all | method_only | ads_cgc_only | baseline_only`；`train_and_eval.py` 使用同一模式和 family switches，训练根 `features.pkl` 中选定特征及独立 `baseline/` 特征。
- 严格 8:1:1 仍由 generation/labeling 阶段建立并供所有方法共享；抽取的 `--resume` 与 baseline 目录隔离保持不变。

## 2026-07-17 Baseline Real-positive 三随机种子重跑

- Baseline 训练与汇总的 headline 正类改为 `real`；存储标签仍保持 `0=hallucination, 1=real`，不改变已有 labeling/features。
- MetaToken、SVAR、DHCP、ProjectAway 已在 GPU 0 上按 seeds `42,43,44` 完整重跑；未运行 HalLoc，也未重新提取特征。
- 每个方法均在 validation set 上最大化 Real-F1 选择阈值，test set 只做最终评估；逐 seed JSON 已核对 `headline_positive_class=real` 与 `threshold_score_class=real`。
- 三随机种子 Test 结果：MetaToken-LR AUROC/Real-F1 `0.8615/0.8892`，MetaToken-GB `0.8687/0.8901`，SVAR `0.8739/0.8940`，DHCP `0.8477/0.8919`，ProjectAway `0.7518/0.8636`。
- 新汇总位于 `outputs/qwen3_vl_8b/COCO4000-512-AC/baseline/results/qwen3_vl_8b_baselines_3seed_summary.md`；旧的 Hall-F1 阈值说明已消失，报告明确记录 validation Real-F1 阈值协议。

## 2026-07-17 COCO4000-EV 分标签曲线

- 新增 `scripts/plot_dgst_ev_mass_cosine_by_label.py`，一次读取根 `features.pkl`，绘制六种 `EV = target-distribution top-k mass × target cosine` 的 hallucination/real 逐层均值与 SEM。
- Qwen3 `COCO4000-EV` 总图、PDF、逐层 CSV 和 Markdown 摘要写入实验 `results/`；统计使用完整 11,751 条样本，其中 hallucination 2,625、real 9,126。
- 五条 Gaussian/raw-attention 分支的最大绝对 Hall-Real 均值差位于第 6 层；hpre direct target-probability 分支位于第 35 层。六个峰值差均为负，表示对应层 real 样本的平均 EV 更高。
- 验证：脚本 `py_compile` 通过；CSV 严格包含 6 分支 × 36 层 = 216 行，全部均值/SEM/difference 有限；PNG 已人工检查布局与图例，`git diff --check` 通过。

## 2026-07-17 Real-positive Torch probe 与 tuned risk+EV 实验

- 根 Torch probe（DGST 方法和 ADS+CGC 共用）改为以 `real` 为训练/验证阈值选择/headline 正类；标签文件仍保持 `0=hallucination, 1=real`。每次评估同时保存 `real_positive` 和 `hallucination_positive` 的 precision、recall、F1、AUC、AUPR，accuracy 共用。
- 单 seed Markdown 与三 seed 汇总均把 Real 指标放在前面，并在同一行补充 Hallucination 指标；三 seed 排名按 Real AUC、Real F1。汇总器拒绝混入旧的 Hall-positive headline 结果，避免不同阈值协议静默混算。
- `model_configs_unified.yaml` 和 fj01 配置的 Torch MLP 参数统一为：hidden `[256,128,64]`、dropout `0.1`、lr `3e-4`、batch `128`、weight decay `0`、epochs `120`、early-stop patience `30`、LR factor/patience `0.5/4`，seeds `42,43,44`。
- 在 `COCO4000-EV` 上只重跑 `hpre_softmax_prob_gauss_risk+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine`，写入独立 `results/tuned_hpre_softmax_prob_gauss_risk_ev_seed{42,43,44}/`，未覆盖原 seed 结果。
- tuned Test 三 seed mean±population-std：Accuracy `0.8370±0.0010`，Real P/R/F1 `0.8558±0.0051 / 0.9472±0.0093 / 0.8991±0.0014`，AUC `0.8867±0.0018`，Real AUPR `0.9597±0.0008`；Hall P/R/F1 `0.7335±0.0227 / 0.4736±0.0267 / 0.5745±0.0122`，Hall AUPR `0.7046±0.0132`。
- 旧参数且以 Hall 为正类/阈值目标的同一特征组合 AUC 为 `0.8847±0.0046`，新参数 AUC 约提升 `0.0020`；F1 因正类与 validation 阈值目标改变不能直接横向比较。
- 汇总：`outputs/qwen3_vl_8b/COCO4000-EV/results/qwen3_vl_8b_hpre_softmax_prob_gauss_risk_ev_tuned_3seed_summary.md`。
- 验证：首次全仓 `unittest discover -v tests` 的 135 项中，旧断言 `test_active_yaml_enables_raw_attention_control` 仍期待 `positive_class=hallucination`，因此 1 项失败；更新为 `real` 后完整重跑 `135/135` 通过，`bash -n run.sh` 与 `git diff --check` 通过。

## 2026-07-18 AMBER discriminative VQA

- 接入用户本机完整 AMBER：`/home/apulis-dev/userdata/AMBER`，读取官方 14,216 条 discriminative Yes/No 问题和 1004 张图片；按 existence 4,924、attribute 7,628、relation 1,664 保存来源维度。
- 严格 seed-42 物理图片级 8:2：803 train / 201 test，问题行分别为 11,385 / 2,831；同一图片跨三个问题维度始终落在同一 split，避免图片泄漏。
- `amber_discriminative` 已接入统一 `run_qa.sh`，并新增薄入口 `run_amber.sh`；与 POPE/CLEVR 一样只抽取完整 prompt 的最后一个 token causal row，联合支持 DGST、ADS+CGC 和启用的 baseline。
- 首次双卡正式生成暴露 AMBER 原图最高 54 MP，Qwen3 动态视觉 eager attention 对单图申请约 116--125 GiB。已中止且完整归档为 `amber_discriminative-uncapped-failed-20260718T183616Z`，没有与正式结果混用。
- YAML 新增仅对 AMBER 生效的 `dataset_model_overrides.amber_discriminative.max_pixels: 200704`；生成与抽取共享同一视觉网格，不影响 COCO/POPE/CLEVR。最大 6000x9000 图实机 smoke 得到 grid 34x22、输出 `yes`、峰值 allocated 16.459 GiB。
- 验证：全仓 `unittest discover` 170/170 通过，AMBER 的函数式 split/leakage 检查通过，`py_compile`、`bash -n` 和 `git diff --check` 通过。
- Qwen3 正式双卡生成 14,216/14,216 完成，generation failure 为 0。原始模型整体 accuracy `0.88555`；existence/attribute/relation 分别为 `0.91511/0.86995/0.86959`，严格 test 图片集为 `0.89686`。错误为 930 false-positive 与 697 false-negative；yes-only cohort 为 4,092 real / 930 hallucination。
- 随后已启动 prompt-last-token 联合抽取，DGST、ADS+CGC 与 MetaToken/SVAR/DHCP/ProjectAway 共用每题一次 LVLM forward；双卡前 125 条分片成功、extraction failure 为 0，完整任务继续断点运行。
- 用户随后要求暂不提取 DHCP。已在 500 条 root 分片后安全停止 worker，保留 DGST/ADS+CGC 原子分片；活动 YAML 的抽取和训练 baseline 列表均移除 DHCP，已产生的含 DHCP baseline 子目录整体归档，恢复后 baseline 仅抽取 MetaToken/SVAR/ProjectAway。

## 2026-07-18 Python 解释器路径可移植性

- `run.sh`、`run_qa.sh` 及仓库内辅助 shell 入口不再绑定 `/opt/conda/.../bin/python`；默认执行当前已激活环境中的 `python`。
- 所有入口仍支持 `PYTHON_BIN=/path/to/python` 显式覆盖；找不到解释器时会立即给出可操作的错误提示。
- 四个 Python 编排脚本默认使用启动自身的 `sys.executable`，同样接受 `PYTHON_BIN` 覆盖，不会因为本机恰好存在旧 vicr 环境而跳入错误环境。
- 旧 QA coordinator/worker/smoke 脚本同时改为根据脚本位置定位仓库根目录，不再绑定 `/root/rivermind-data/project/token-detector`。
- 本次修改只影响后续新启动的进程；已经运行中的 AMBER 抽取任务不会切换解释器或被中断。
- 验证：全部 shell 脚本 `bash -n`、四个 Python 编排脚本 `py_compile`、`tests.test_pipeline_config` 31/31 和 `git diff --check` 通过。首次单测误用缺少 `pyyaml` 的系统 Python，在模块导入阶段报 `ModuleNotFoundError: yaml`；切换到项目测试环境后完整通过，不是实现失败。

## 2026-07-19 QA feature-set YAML 路由修复

- 发现 `train_qa_probes.py` 未读取 YAML 的 `training.feature_sets`，而是通过 `detection.qa_probe.default_feature_sets()` 硬编码每个 DGST 分支的 `risk+target_cosine+EV`。这使 AMBER 首次三层 MLP 训练成为 108 维三向量组合，与 YAML 指定的 risk、risk+EV、EV 实验矩阵不一致。
- 已删除该硬编码默认矩阵。QA 训练现在严格读取 `training.feature_sets.method` 和 `training.feature_sets.ads_cgc`，只负责为每个 YAML 项追加 `@prompt_last_token`；缺失、空列表或未知 family 会直接报错。
- 训练 provenance 升级为 `qa-probe-training-provenance-v2`，记录实际 YAML 展开的 feature-set 列表；后续 YAML 组合改变时不会静默复用旧结果。
- unified 与 fj01 YAML 已重新对齐为 16 个 method 项和 3 个 ADS/CGC 项；两者均不含活动的 target-cosine 组合。
- Qwen3 AMBER 已用 `[256,128,64]` 三层 MLP、seeds 42/43/44 重跑两种标签协议：每套 19 项、共 114 个 seed-run。汇总中的 method 输入维度严格为 36（单项）或 72（risk+EV），两套汇总 target-cosine 项均为 0。
- 首次错误硬编码生成的 12 个三特征结果目录未删除，已移入 `results/_obsolete_hardcoded_triple_20260719/`；活动协议目录中 target-cosine 结果目录为 0，避免误读。
- 验证：YAML 路由单测 6/6、受影响的 QA feature-vector 纯函数测试 2/2、Python 编译和 `git diff --check` 通过。一次附加 `pytest` 命令因 vicr 环境未安装 pytest，在收集前报 `No module named pytest`；随后用项目 Python 直接执行对应纯函数测试并通过。

## 2026-07-25 VPend 后置 prompt support 修正

- 定位到旧 `VP` 的语义偏差：wrapper 提供的 `prompt_positions` 是全部非视觉 prompt token，旧实现直接与视觉 token 合并，因此包含 image 前面的 `<s> USER:` 等前缀；同文件的 prompt CAFE 实际一直采用 `position >= visual_end`，两者此前没有对齐。
- 新增独立 `VPend`/`vpend` support mode：support 严格定义为全部视觉位置加上 `prompt_position >= visual_end` 的后置 prompt token；序列化统一使用 `dgst_t_vpend_*`，并额外保存 `dgst_t_vpend_support_size` 和 `dgst_t_vpend_support_positions`。旧 `VP`/`dgst_t_vp_*` 保留，仅用于兼容已有产物。
- root extractor、QA compact schema、pipeline feature-set 过滤和 probe alias 均支持 `vpend_`；risk、target cosine、EV、JS、union-topK JS、`(1-mass)*cosine` 及所有现有 cost/alpha risk 名称可以按 VPend 前缀训练。
- `model_configs_server_fj01.yaml` 默认改为 `support_modes: ["vv", "vpend"]`；`model_configs_unified.yaml` 默认改为 `["vpend"]`；两份配置的活动 VP feature sets 改用 `vpend_`。`experiment.mode` 同步改名为 `vpend`，`get_model_cfg` 将其解析为 `visual_prompt_end`。
- 新回归样例显式构造 image 前 token 位置 0、视觉位置 `[1,3)`、image 后 prompt 位置 3，确认 VPend support 精确为 `[1,2,3]`，不会包含位置 0；同时验证 root/QA 序列化及 risk+EV 训练矩阵。
- 验证：受影响 Python 文件 `py_compile` 通过；四门 DGST 全部测试加两项 VPend pipeline 定向测试共 `20/20` 通过；3 项 QA compact 纯函数测试通过；两份活动 YAML 均可加载、活动 feature sets 全部可解析；`git diff --check` 通过。
- 失败记录：`/opt/conda/envs/td` 未安装 pytest，首次命令在收集前报 `No module named pytest`；改用 `unittest` 和直接调用 pytest 风格纯函数完成定向验证。完整 `tests.test_pipeline_config` 还出现 6 fail/4 error，均为仓库当前 pipeline-manifest 行为与旧测试预期不一致（缺少 `pipeline_manifest.json` 或未抛出旧 resume 异常），与本次 support/filter 改动无调用关系；本次新增的两项 pipeline 测试单独通过。

## 2026-07-25 全模型视觉 support 审计与 YAML 清理

- 逐一核对 LLaVA-1.5、LLaVA-NeXT、LLaVA-OneVision、InternVL、Qwen2.5-VL 和 Qwen3-VL 的视觉区间定位、连续性检查、视觉网格校验以及 DGST attention/hpre/hmid/source 的索引链路，未发现视觉 support 偏移或混入图像分隔符。
- 所有 wrapper 均使用半开区间 `[visual_start, visual_end)`：LLaVA-1.5/NeXT 将单个 image placeholder 映射为展开后的连续视觉 span；InternVL 只包含连续 `<IMG_CONTEXT>`；OneVision、Qwen2.5-VL、Qwen3-VL 只包含连续 `<|image_pad|>`，并校验视觉网格 token 数。attention、hpre、hmid 与 source distribution 共用同一 `support_positions`/`index_select`。
- 现有 LLaVA 产物的 VV support 实测为 `[32,576]`，与 24x24 视觉网格一致；旧 VP 的 593 列是 576 个视觉 token 加 17 个全部 prompt token，进一步确认此前问题仅在 prompt 范围，视觉区域本身没有取错。
- 新增 `tests/test_visual_support_ranges.py`，覆盖六类 wrapper 的精确视觉 span、图像分隔符排除和非连续视觉 token 拒绝行为，避免后续模型适配再次引入偏一位问题。
- 对 fj01/unified 两份 YAML 做“配置叶节点—实际读取代码”静态审计并结合运行时加载复核。每份配置从 253 个解析叶节点清理到 191 个；删除从未读取的 dtype/raw-capture 字段、严格 8:2 后失效的 train/val/test ratio 与 validation 字段、未接入当前 QA 路由的旧 `pope` 块、冗余 DGST `target_modes/branches` 及已由 compact exact 路径固定的旧 solver/source/support 参数，同时删除 wrapper 不读取的模型 metadata。注释示例不计入解析配置，仍保留作实验参考。
- `four_gate_methods` 现在是 target 构造的唯一活动选择器；`support_modes` 是 support 选择器。两份配置仍启用 raw-logit Gaussian 与 softmax-prob Gaussian；fj01 support 为 `vv+vpend`，unified 为 `vpend`，活动训练 feature sets 均可成功解析。
- `coco-labeling/label_coco.py` 同步移除 `_load_or_create_splits` 已被严格 8:2 覆盖且从不生效的 `train_ratio/resume` 参数，实际 image-disjoint 8:2 逻辑不变。
- 验证：受影响定向套件 `37/37` 通过，包含全模型视觉 support、four-gate、alpha sweep、训练命令、raw-attention 与 labeling 配置；两份 YAML 可运行时加载，所有模型都解析为 `visual_prompt_end`；受影响 Python 文件编译和 `git diff --check` 通过。
- 失败记录：第一次定向测试有 1 fail/1 error，分别是 unified 的既有 `prompt_cafe` feature-set 未写入新精确预期，以及 raw-attention 测试仍依赖已删除的 `branches` 选择器；修正测试预期并显式设置 `four_gate_methods/support_modes` 后，完整定向套件重跑通过。

## 2026-07-26 QA 命名 output 目录与 generation 复用

- `run_qa.sh` 新增 `OUTPUT`，默认值为 `default`；流水线、方法 probe、baseline 训练和对比汇总均显式接收同一个 `--output`，避免不同阶段各自拼接输出路径。旧 `--experiment` 参数仅作为兼容别名保留。
- 新目录规范为 `qa_benchmarks/{model}/{output}/{benchmark}/`。benchmark 子目录只保存 `labels.jsonl`、`features.pkl`、baseline 与训练结果；可复用 generation 位于命名 output 目录，命名为 `{benchmark}_generations.jsonl`，失败记录命名为 `{benchmark}_generation_failures.jsonl`。
- 新增 `utils/qa_paths.py` 作为唯一目录解析入口，并拒绝空 output 名、`.`/`..` 和包含路径分隔符的 output 名，防止意外跨目录写入。独立 baseline 抽取、固定 MLP 实验、train-threshold 复评、artifact 校验和 Sinkhorn 子集工具同步接入；只读工具仍可自动读取旧 benchmark 目录内的 `generations.jsonl`。
- `features/qa_extractor.py` 的生成、标注与提取入口支持显式 generation 路径；单卡和多卡流水线均写入同一个 output 级文件，多卡 worker 的临时分片仍留在对应 benchmark 的 `.qa_parallel/` 下。
- 三个 benchmark 放入同一 output 时，output 目录直接包含 `pope_generations.jsonl`、`clevr_exist_9k_generations.jsonl` 和 `amber_discriminative_generations.jsonl`；修改特征参数建立新 output 时，只需在 output 目录层复制所需 generation 文件，不再进入三个 benchmark 子目录逐个复制。
- 验证：受影响 Python 文件 `py_compile` 与 `bash -n run_qa.sh` 通过；新增路径测试 `4/4` 通过；QA extractor/parallel/fixed-MLP/summary 相关 `11/11`、answer-only/baseline/benchmark/probe 相关 `19/19` 通过。
- 失败记录：`/opt/conda/envs/td` 未安装 pytest，首次命令在测试收集前报 `No module named pytest`；新增测试改用标准库 `unittest` 后通过，未额外修改训练环境依赖。
- 全仓 `unittest discover` 共运行 227 项，结果为 212 通过、10 fail、5 error。失败项集中在此前已记录的旧 pipeline-manifest/stage-resume 预期（14 项）及既有 `test_method_and_ads_cgc_can_share_one_record` 缺少 `dgst_t_vv_support_size`（1 项）；本次直接受影响的 QA 定向套件均通过，未在本次目录重构中改动这些无关模块。

## 2026-08-18 四模型 Union Top-K hidden cosine 诊断

- 为检验当前 OT cost 中视觉 hidden-state cosine 是否过于同质，以及目标首 subtoken 的因果预测行 `q_t=toke_idx-1` 对不同视觉 token 是否同样缺乏区分，新增 `scripts/analyze_union_topk_hidden_cosine.py`。诊断严格使用活动协议的 `Top-32(P) ∪ Top-32(Q)`，最多 64 个视觉 token；不更改已有 labeling、features 或训练结果。
- `features/dgst_t.py` 新增只由环境变量启用的 streamed diagnostic：统计 visual–visual raw/centered cosine、`q_t`–visual cosine、现有 `tau=0.07` 下的归一化 entropy、top/bottom gap，以及 sqrt/cosine cost 分布。visual–visual 排除对角线和对称重复；diagnostic-only 模式跳过 EMD，不保存 hidden state 或 cost matrix。
- 四模型各按 seed=`20260818` 固定抽取 500 张；优先抽入最多 250 张 hall-only，但可用 hall-only 图片不足，因此实际 has-real/hall-only 分别为 LLaVA `461/39`、InternVL `444/56`、Qwen2.5 `451/49`、Qwen3 `472/28`。前向失败均为 0。
- 正式 InsLen 目标/唯一因果位置分别为：LLaVA `1882/1815`、InternVL `1354/1346`、Qwen2.5 `1127/1120`、Qwen3 `1772/1753`。同一因果位置的重复 span 不重复加权；LLaVA 的 5 个 REAL/HALL 冲突位置在 all 组只计一次、标签组各计一次。
- 四分支全层平均后的核心区间：
  - LLaVA：vv raw cosine `0.3514`，vv P90−P10 `0.6829`，centered vv `0.1992`；qv P90−P10 `0.2445`，entropy `0.8623`。
  - InternVL：`0.4953 / 0.3619 / 0.0986`；qv `0.2128 / 0.8775`。
  - Qwen2.5：`0.5599 / 0.3796 / 0.1166`；qv `0.2527 / 0.8773`。
  - Qwen3：`0.6098 / 0.3369 / 0.1165`；qv `0.2356 / 0.8967`。
- 解释：InternVL/Qwen 的 raw cosine 高值主要包含明显的公共各向异性方向，因为中心化后均值降到约 `0.10–0.12`；但 vv P90−P10 仍有 `0.34–0.38`（LLaVA 更大），因此不能说 union 内所有视觉 token cosine 都相同。`q_t`–visual 的 P90−P10 约 `0.21–0.25` 且归一化 entropy `0.86–0.90`，说明存在排序信号但分布偏扩散，Qwen3 最平。四个 P/Q 分支差异极小，结论不是某一个 gate 选择造成的。
- 新增 `scripts/summarize_union_topk_hidden_cosine_4model.py`，统一结果为 `outputs/union_topk_hidden_cosine_500_4model_summary.{md,csv,json,png,pdf}`；每个模型的逐层/Real-Hall CSV、JSON、图和报告位于自身实验目录的 `results/union_topk_hidden_cosine_500/`。
- 验证：`py_compile` 通过；`python -m unittest tests.test_union_cosine_diagnostics tests.test_four_gate_dgst tests.test_cost_variants` 为 `40/40` 通过；四模型 summary 行数依次为 `384/384/336/432`，所有浮点值 finite，500 张均处理完成且前向失败为 0。四模型总图已人工检查。仓库按设计没有 `.git`，因此 `git status` 返回 `not a git repository`，无法执行 git diff 类检查。

## 2026-08-19 四模型 JFFN-P 对照实验

- 新增 JFFN 源分布：对每层目标因果行构造精确视觉 attention residual write
  `a_j=W_O concat_h(alpha_hj v_hj)`，随后用 `torch.func.jvp` 计算
  `delta_j=J_{FFN(Norm)}(hpre+oattn)a_j`，以 `||delta_j||_2` 在视觉 token
  间直接归一化为 P；不使用 softmax 或温度。实现位于
  `features/visual_ffn_jacobian.py`。
- attention/FFN adapter 已覆盖：LLaVA/Qwen 的 separate QKV+GQA、InternVL
  InternLM2 packed `wqkv` 的最后 V 槽、`o_proj/wo` bias 只进入完整 attention
  重建而不重复分配给每个 token，以及两类 Norm+FFN 布局。LLaVA/InternVL
  默认全视觉方向并行；Qwen 按 1.5 GiB 增量预算在全并行、256、128、64
  间自适应并支持 OOM 回退。
- `features/dgst_t.py` 在 OT inference-only 路径外一次性构建全目标 JFFN payload；
  正式四 source 为 `old_hmid_cos`、`old_hpre_cos`、`new_jffn` 和
  `new_jffn_entropy_matched`。所有 source 共用原 InsLen 目标位置、两个 hpre gate、
  Union Top-32+Top-32 support、sqrt/cosine matched-state cost、exact EMD 和原 EV。
  单元测试确认加入 JFFN 后旧 hmid/hpre P 及其 risk bit-identical。
- 在既有 `configs/model_configs_inslen_official_target.yaml` 增加独立、默认关闭的
  `jffn_p_comparison` 段；普通 `run.sh` 不读取该段，没有新增实验 YAML，也不覆盖
  旧 `features.pkl`/结果。
- 新增 `features/jffn_experiment.py` 与 `scripts/run_jffn_p_comparison.py`：
  - 每个唯一 `(image_id,response_index)` 只保存一次 float32 的四个 P、E、I/R/S/D、
    16 条 risk、两条 EV 和分布统计；mention 通过轻量 sample table 引用唯一位置。
  - 每 50 图原子写 `.pt`；双 GPU 使用稳定交错 rank 和独立
    `features_rankXX_shard_YYYYY.pt`，不会发生同名提交。
  - 训练和分析真正逐 shard 消费；不把完整 P 再合并进内存。训练只保留 32 组
    最终 risk/risk+EV 矩阵，分析逐 shard 累计分布、同图异目标和 COCO 空间指标。
  - shard 提交前递归拒绝所有 NaN/Inf；训练前核对完整官方 cohort、mention、唯一
    position 和跨 rank 重复。正式未计算的 smoke-only 诊断以有限 0 配合
    `jffn_validation_computed=false` 保存，不在生产 shard 中写 NaN。
  - 训练严格为 32 feature sets × seeds 43/44/45、`[128,64,32]`、batch 256、
    最多 100 epochs、无归一化、train-loss checkpoint、train-F1 threshold；每 seed
    原子保存 progress，可断点恢复。主比较用 seed-平均预测做 10,000 次图片级
    paired bootstrap。
- 熵匹配：仅从各模型固定 seed=20260818 的 500 张 train 图片拟合逐层 beta，
  优化为按动态视觉宽度分组的 batched bisection；test leakage 均为 0。结果：
  - LLaVA：32 层，beta `1.8310--7.2172`，最大熵误差 `1.27e-7`。
  - InternVL：32 层，beta `1.5381--5.4188`，最大熵误差 `1.46e-7`。
  - Qwen2.5：28 层，beta `0.7039--5.2848`，最大熵误差 `7.19e-8`。
  - Qwen3：36 层，beta `0.6043--9.7503`，最大熵误差 `6.44e-8`。
- 四模型固定 5 图 smoke 全部通过：
  - LLaVA：attention 重建 max relative/min cosine `4.84e-4/0.9999998`，JVP
    线性/全并行-分块 max relative `2.02e-6/2.42e-6`，有限差分 cosine 中位数
    `1.0`，峰值 `15.30 GiB`。
  - InternVL：`3.15e-3/0.9999955`，`1.10e-6/1.74e-6`，`1.0`，峰值
    `18.08 GiB`。
  - Qwen2.5：`6.19e-3/0.9999925`，`1.16e-6/2.00e-6`，`1.0`，峰值
    `17.71 GiB`。
  - Qwen3：`4.13e-3/0.9999942`，`1.07e-6/1.73e-6`，`1.0`，峰值
    `18.46 GiB`。
  - 正式 E/P 仍使用模型 dtype 的 `torch.func.jvp`；仅 smoke 数值恒等式使用临时
    FP32 Norm+FFN 副本，避免 BF16 batch-kernel 舍入与小步长有限差分量化噪声。
- Qwen shared-caption query-row gate：发现 custom attention 在无显式 mask 路径可能
  泄漏未来 token，已无条件补严格 causal mask 并增加回归测试。修复显著降低误差，
  但 Qwen2.5/Qwen3 最坏 parity 仍约 `1.32/0.098`，未达到 `1e-3`，因此正式协议
  均按预案统一回退到原 prefix-forward，不混用快速路径。
- LLaVA 正式生产路径已用双卡完成前 100 图：rank0/1 各 50 图，合计 375 个唯一
  position、386 个 mention，0 failure，200 秒（含并发 checkpoint I/O）；两个原子
  shard 均可读取。此后去掉训练/分析从不读取的 raw attention、gate 和 Q 重复矩阵，
  后续 shard 只保留预注册接口，约节省一半磁盘。
- 新增 `scripts/run_jffn_p_comparison_4models.sh` 完整 coordinator：依次执行四模型
  smoke/reuse、校准/reuse、双卡完整抽取、96 个 seed-run、分析及跨模型汇总。当前
  持久会话已从 LLaVA 的 100 图 shard 继续恢复，日志为
  `outputs/jffn_p_comparison_logs/coordinator.log`。
- 验证：JFFN/streaming/four-gate 定向 `unittest` 为 `38/38` 通过；相关 Python
  `py_compile` 与 coordinator `bash -n` 通过。额外的既有
  `tests.test_config_mirror` 仍因 unified/fj01 的 `run.extraction_mode` 等历史配置差异
  失败，与本次独立 InsLen YAML/JFFN 路径无调用关系。仓库无 `.git`，不能执行
  `git diff --check`。

## 2026-08-19 JFFN-P 已完成模型独立审计与最终结论

- 按用户最新范围只分析已经完整完成的 LLaVA-1.5-7B 与 InternVL2.5-8B；不续跑
  Qwen2.5-VL/Qwen3-VL，也不把后两者 smoke 外推为 4000 图正式结论。
- 两个正式 cohort 均已完整：LLaVA 为 3971 图、14951 唯一位置、15463 mention；
  InternVL 为 3927 图、11630 唯一位置、11759 mention。32 feature sets × 3 seeds
  和 10000 次图片级 paired bootstrap 均存在。
- 新增 `scripts/audit_jacobian_visual_ffn.py`，在 LLaVA 3 张真实图、6 个 REAL/HALL
  目标、层 1/8/16/24/32 上独立重算 attention decomposition、per-token JVP、FP32
  epsilon sweep、非线性删除曲线、32 个 matched-norm random directions、signed
  contribution/cancellation、full-prefix/cache parity 和黑图/shuffle/无关图控制；共
  30 个 layer-target 案例、17280 个 token 诊断行。
- 独立数值结果：attention 重建 relative error mean/max 为
  `4.29e-4/5.76e-4`；FP32 token additivity mean/max 为
  `1.60e-4/3.65e-4`；JVP 标量线性 max `7.82e-7`；模型 dtype 与 FP32 P 的 JS
  约 `1e-9`。central finite difference 呈正常 U 形，epsilon `0.01--0.1` 的中位
  relative error 为 `1.91e-4--7.27e-5`。完整删除视觉写入时局部线性近似的
  relative error median/mean/max 为 `0.0844/0.1527/0.5989`，因此只支持局部解释。
- 发现并保留记录的核心审计缺陷：`features/visual_ffn_jacobian.py` 当前把
  `component_sum_relative_error` 硬编码为 `0.0`；独立真实值 mean/max 为
  `5.02e-4/6.01e-4`。该字段不参与 `a/E/P` 或训练，因此不改变现有正式结果，
  但不能将其当成分解误差证据。本轮按审计要求未静默修改核心公式。
- LLaVA 的 visual gain 相对 matched-random median 在层 1/8/16/24/32 为
  `3.85/1.49/1.05/1.03/5.30`；中层仅略高于随机方向，说明 `S` 受层 Jacobian
  scale 影响且并非每层都视觉特异。`||a_j||` 与 `E_j` Pearson mean `0.951`，
  attention 与 `E_j` Pearson/Spearman mean `0.764/0.866`；新 P 很大程度仍由
  attention-mediated write magnitude 驱动。响应 cancellation ratio median `0.557`，
  表明非负模长 P 丢失明显的方向相消。
- P 的正面证据：风险层中 new JFFN 的归一化熵为 LLaVA `0.764--0.766`、InternVL
  `0.710--0.727`，明显不再近似均匀；同图异目标 JS 为 `0.195/0.305`，均显著高于
  old-hpre 的约 `0.016`。COCO REAL box 上 patch AUPRC 从 `0.349/0.316` 提升到
  `0.451/0.445`，Top-1 pointing 从 `0.336/0.279` 提升到 `0.502/0.625`；熵匹配保持
  相同排序指标，证明空间/目标排序提升不只是分布变尖。
- 幻觉检测主结论为负：预注册 raw-logit + sqrt cost + risk+EV 下，LLaVA 的
  seed-平均预测 AUROC 从 old-hpre `0.8934` 降到 new `0.8832`，差值 `-0.0102`、
  95% CI `[-0.0198,-0.0012]`；InternVL 从 `0.8567` 到 `0.8506`，差值
  `-0.0060`、CI `[-0.0197,+0.0077]`。entropy-matched 也未优于旧 P。因此
  “空间定位更好”不能写成“幻觉检测更好”。
- 全量 I/R/S/D 诊断显示 response norm `R` 的单变量最佳方向 AUROC 为 LLaVA
  `0.6734`、InternVL `0.6787`，明显强于 gain `S` 的 `0.5529/0.5378`；后续优先
  预注册 `old risk + I/R/S/D` 或 target-logit signed Jacobian，而不是继续只替换
  归一化 P。
- 新增 `scripts/analyze_jacobian_visual_ffn_full_cohort.py`、
  `scripts/build_jacobian_visual_ffn_validation_summary.py`，产出根目录
  `jacobian_visual_ffn_validation_report.md`、summary JSON/CSV/PNG，以及 LLaVA 独立
  cases/epsilon/removal/token/prefix/image-control/efficiency CSV。机器汇总明确标记只含
  两个已完成模型。
- 验证：`tests.test_visual_ffn_jacobian`、`tests.test_jffn_experiment`、
  `tests.test_visual_support_ranges` 共 `17/17` 通过；核心文件与三个分析/汇总脚本
  `py_compile` 通过。仓库仍按设计没有 `.git`。

## 2026-08-19 JFFN 二轮增量验证（仅 LLaVA/InternVL）

- 按用户最新要求只分析两个已完成模型，不续跑 Qwen。二轮正式 cohort：
  LLaVA 3971 图/14951 unique positions/15463 mentions，InternVL
  3927/11630/11759；全量 token-map 使用全部 32 层，预注册主比较固定层 16–32。
- 修复 features/visual_ffn_jacobian.py 的 component_sum_relative_error 伪零：
  现在重建所有 source-token output-projection contribution、bias 只加一次，并保存
  真实误差。全量 mean/max 为 LLaVA 0.00120/0.00708、InternVL
  0.01242/0.09855；InternVL 高值与 production 使用 BF16 hmid-hpre 作为 reference
  的 reconstruction error 同步，不参与 a/E/P 或训练。
- 新增 second-round production payload：WRITE energy/distribution、signed Q_j、
  Q conservation/cancellation；标签与旧 risk/split 保持不变。新增/完成脚本：
  - scripts/run_jffn_second_round_extraction.py
  - scripts/analyze_jffn_second_round_scalar.py
  - scripts/analyze_jffn_second_round_token_maps.py
  - scripts/run_jffn_second_round_logit_causal.py
  - scripts/analyze_jffn_second_round_conceptual_stages.py
  - scripts/build_jffn_second_round_summary.py
- WRITE-vs-JFFN 主空间结论为负：LLaVA JFFN−WRITE 的 bbox/AUPRC 为
  -0.00117/-0.00401，InternVL 为 -0.00190/-0.00636，10000 次 image-level
  paired bootstrap 的 95% CI 均完全低于 0；Top-1 在 LLaVA 持平、InternVL
  下降 -0.00809。I_j/E_j Pearson 为 0.9806/0.9901；Top-1 agreement
  0.8668/0.8756。全 32 层 correction/regression 为
  LLaVA 7300/5916、InternVL 4430/5398。
- G_j=||Ja_j||/||a_j|| 在层 16–32 的 mean/CV 为 LLaVA
  0.3894/0.1645、InternVL 0.3687/0.1511；tiny denominator 极少且 robust
  CV 不变，但 box 内 gain 均低于 box 外，未形成对象方向的 Jacobian amplification。
- train-only balanced logistic regression 显示 [I,S]-I AUROC：
  LLaVA +0.02317 CI [-0.00300,+0.04917]，InternVL +0.00150
  CI [-0.00043,+0.00330]；均无可信增量。Residualized-R AUROC 为
  0.7511/0.5820，说明 LLaVA 有较强、InternVL 有较弱的 I-unexplained structure。
- signed Q conservation 最大相对误差为 4.00e-7/4.49e-7。CancellationRatio
  REAL/HALL 为 LLaVA 0.4781/0.4532、InternVL 0.4902/0.4900，负质量/负 token
  的 label pattern 不一致，因此只作机制诊断。Q+ 空间收益只在 LLaVA 部分复现。
- 每模型完成 25 图、200 target-layer logit/margin attribution（100 REAL/100 HALL，
  层 8/16/24/32）和 288 条 intervention。Target 始终不在 prefix；logit/margin
  additivity max error 为 LLaVA 4.65e-6/2.32e-4、InternVL
  1.33e-4/1.28e-5。A_margin_total direction-free Hall AUROC 为
  0.612/0.508，未跨模型复现；LOGIT+ 空间定位均弱于 WRITE。
- 干预基线已修为相同 no-grad hook 的 eta=0；InternVL remote attention 的
  compact capture 不兼容时改用 official eager full attention。LLaVA/InternVL
  在 eta=.05 的 zero observed fraction 为 63.5%/88.5%，最小非零 margin
  step 0.0078125/0.25 远大于 median prediction 0.000123/0.000201；因此 causal
  direction 仅 PARTIAL、magnitude FAIL，不能声称已验证 causal effect。
- 新增同一 200-case 的 conceptual-stage join/correlation 表；raw total visual
  attention mass 未在 shard 中持久化，明确为 NOT RUN，没有用 normalized map 冒充。
- 最终科学判断为 **Outcome C**：JVP 是可信 mechanistic measurement，但当前
  JFFN token-map 的空间收益主要来自 WRITE magnitude；Jacobian 不优于 WRITE，
  S 不提供稳定 hallucination increment，logit alignment/causal validation
  也尚未成立。
- 交付：
  - 根报告 jffn_second_round_incremental_validation_report.md
  - jffn_second_round_summary.json
  - jffn_second_round_metrics.csv（420 rows）
  - jffn_second_round_representative_heatmaps.png
  - 两模型各自 results/jffn_second_round/ 下完整 CSV/JSON 和 Figure 1–6。
- 验证命令：
  - python -m unittest -v tests.test_visual_ffn_jacobian tests.test_jffn_experiment tests.test_jffn_second_round_analysis：18/18 OK。
  - 所有 second-round core/analysis/summary 脚本 py_compile 通过，run.sh
    bash -n 通过。
  - 根 summary 递归 finite 检查、metrics value finite 检查、28 个必需文件/图检查、
    report 最终固定标题检查均通过。仓库没有 .git，git status/diff 不可用。
