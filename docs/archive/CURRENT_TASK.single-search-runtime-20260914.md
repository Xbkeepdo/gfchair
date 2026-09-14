# 当前任务摘要

## 当前：811 单隐藏层 MLP 固定预算搜索

- 用户要求寻找使其方法优于原生 SVAR/MetaToken 的单层 MLP 参数，已明确 AUROC 优先、HALL-AUPR 同时报告。执行探索性验证集搜索，不保证超过，不用测试分数选参或继续扩网格。
- scripts/search_single_mlp_811.py；outputs/single_mlp_search_811_v1/。复用真实 RMS K32 与冻结 RMS K50 × V/VP/G/VP+G 共8组/模型；无需重新提取 VLM。两路径 AE 相同，S 的路径/基点/Norm 处理不同。
- 固定 811 图片3200/400/400，split seed20260912，保留原3200训练及全部 mentions。原800留出集此前已查看，不能称独立盲测。三 seeds43/44/45 分别算 AUROC/HALL-AUPR 后均值±总体std；ensemble另列。用户要求每次结果均说明split/特征路径/K/分类器选择/seed与均值口径。
- 单隐藏 Linear→可选BN→ReLU/GELU→Dropout→Linear，BCE REAL=1、Adam；可选标准化只拟合train。固定24配置，候选seed20260914；宽度64/128/248/256/512/1024、标准化/BN、dropout、激活、lr、wd、batch及checkpoint monitor可变。全配置见每模型protocol.json；max150epoch、早停20、LR调度patience6/factor.5/min1e-6。
- checkpoint/调度/早停依据是预注册配置中的val_loss或val_auroc；配置排名始终val AUROC优先、AP次之、索引升序。24配置seed43→每组top3补44/45→3seed均值选参。每组30fits、每模型240、四模型960。每模型再按验证3seed均值在8组中选champion。四selection.json全冻结后才评估本轮test；每组最终3heads=四模型96，不train+val重训。
- 原生SVAR/MetaLR/MetaGB全部并列报告（均三seeds），注明其未获相同调参预算，不能声称特征公平预算优势。旧单层sklearn12候选作额外对照。测试输出阈值train REAL-F1及固定.5；原生阈值val REAL-F1，AUROC/AP不受阈值影响。

## 执行与验证

- 本机 GPU0 Qwen2.5、GPU1 LLaVA；screen single811_qwen2/single811_llava。子代理server_4090负责32678 GPU0 Qwen3/GPU1 InternVL；screen single811_qwen3/single811_internvl。四worker已启动；18:37UTC Qwen2.5完成240fits并冻结，其他三模型继续验证搜索；不跨机重复写模型。
- Python /opt/conda/private/envs/vicr/bin/python，OMP/OPENBLAS/MKL线程1，CUBLAS_WORKSPACE_CONFIG=:4096:8，TF32关闭、torch deterministic。flock模型锁，逐fit原子result.pt，断点恢复只读已完成fit；selection完成后不重写冻结时间。
- 两端唯一watcher root=outputs/single_mlp_search_811_v1，10秒原子覆写EXPERIMENT_PROGRESS_LOCAL.md / EXPERIMENT_PROGRESS_32678.md，显示阶段、fit数/240及epoch；完成时24测试heads/24。
- tests/test_single_mlp_search_811.py：6个unittest PASS（包含中断后仅补239/240 fits恢复检查），覆盖训练scaler/常量列、BN末批保留mentions、val_loss/valAUROC checkpoint重载一致、单隐藏架构/固定预算/验证排名、四模型test gate。原命令pytest失败因环境未安装pytest；改用项目已有unittest，不增依赖。3脚本py_compile通过。
- 远端首个Q3/Intern candidate00训练、checkpoint BN buffers、train/val数量及无test键已核验；无阻断错误。新汇总脚本scripts/summarize_single_mlp_search_811.py审计960候选/验证选择/训练scaler/96测试概率重载/36native指标与对齐，生成docs/SINGLE_MLP_SEARCH_811_RESULTS.md、参数和指标CSV、PNG/PDF。
- Qwen2.5验证champion=true_rms/visual，c11：width256/scaler+BN/GELU/dropout.5/lr.003/wd1e-6/batch256/val_loss。仅验证选择，尚未新test；不据此保证超过基线。
- 日志 outputs/single811_{qwen2,llava,qwen3,internvl}.log；本机全局single811_summary已启动等待四模型，报告/绘图合成临时数据smoke PASS。所有选择冻结前不读取新test指标；人可读预注册docs/SINGLE_MLP_SEARCH_811_PROTOCOL.md。待完成后更新此摘要及报告，明确哪些模型胜过哪些基线，完整保留落后组。

## 前序完成结果与继承限制

- 真实RMS区域AE811四模型324头完成：docs/REGION_AE_811_RESULTS.md、outputs/all_attention_ae_811_v1/。
- 冻结RMS区域AE811四模型144头/480候选完成，独立核验PASS：docs/FROZEN_REGION_AE_811_RESULTS.md、outputs/frozen_region_ae_811_v1/。K50纯积分通过；Qwen2/Qwen3源重构总闭合尾部最大9.51%/11.62%，本轮搜索不修提取误差。K50与真实K32均capture_full历史接口但不同历史pass，AE本轮逐值相同。
- 原生SVAR/MetaToken811四模型36头完成：docs/NATIVE_BASELINES_811_RESULTS.md、outputs/native_baselines_811_v1/。SVAR248/ReLU/Adam.001/batch32/max50/patience5/val_loss，无BN/dropout/scaler；Meta训练StandardScaler+LR(lbfgs,max2000)/GB100。
- 统一SVAR/MetaToken811四模型72头已全部完成：docs/SVAR_METATOKEN_811_RESULTS.md、outputs/baselines_811_v1/。SVAR零基[5,19)层视觉mass，MetaToken10+H且含完整回答长度/span，不是严格pretarget-only。
- 旧82区域AE288头完整：docs/ALL_ATTENTION_AE_STRENGTH_20260913.md。旧主报告多处ensemble，不能混比seed均值。
- 原B2 K4全量纯积分最大7.46%>1%尚未修复；本轮不使用B2 K4。前序细节归档docs/archive/CURRENT_TASK.before-single-search-20260914.md及其归档链；索引docs/EXPERIMENT_RESULTS_INDEX.md。
- 保留已有未提交修改，未授权提交/上传，不写入凭据。实际文件UTC时间目前2026-09-13，环境声明current_date=09-14；报告使用实际心跳时间，不据此改随机种子。
