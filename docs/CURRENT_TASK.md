# 当前任务摘要

## 已完成：发布近期代码与可复现实验结果（2026-09-14）

- 用户已授权将近期代码和实验结果上传GitHub；目标为origin/main。纳入实现、测试、中文报告、协议、CSV/JSON汇总及必要结果图。
- 不发布运行心跳、日志、PT/checkpoint、逐样本张量、缓存、账号或服务器凭据；41GB全来源输出只选取汇总和绘图数据。
- 本轮发布覆盖B1/B2全来源路径、811区域AE/原生基线/单层MLP，以及batch、标准化、SVAR层范围和模型曲线分段消融。
- 40项定向unittest、Python编译、JSON解析、链接及敏感信息检查均通过；主内容提交f9ceab9已推送origin/main。

## 最新完成：SADT风格Attention/AE/P_E/cosine图册（2026-09-14）

- 四模型各10个主案例（REAL/HALL各5）及3个高差异HALL案例，共52张；保留层10/15/20/25/30/最后层，Qwen2最后层28。每图先给模型输入图与完整生成原文并标词，再独立画Top16/32。
- 四行信号为原始视觉Attention、AE/T、P_E∝||e_m||、softmax(cos(a_m,e_m)/.2)；采用SADT式JET冷色覆盖，不画patch框。softmax cosine只表达相对空间排序，不能读出原cosine正负。
- 全共享500图候选统计显示P_E与Attention总体高度相关（rank中位数.974–.979），额外案例只展示局部/单层重排，不能据此声称二者独立。总览与52图见outputs/ffn_write_effect_sadt_gallery_v1/summary.md，脚本scripts/plot_sadt_style_write_effect_gallery.py。
- 前两版图目录已删除，释放约565 MB；新目录约103 MB。图数、分组、层、TopK统计2392行、链接、尺寸、Python编译及diff检查通过。

## 已完成补充：raw_AE_top32 单独及拼接B检测（2026-09-14）

- 用户在旧P_E/JS/OT对照讨论后要求“试试”raw_AE_top32；本次沿用该对照的原3200/800划分、四模型全部mentions、旧三隐藏MLP及seeds43/44/45，不混用下方811搜索协议。
- 两组：全层O=Top32(P_raw)与Top32(T)交集比例；[B,O]，B=[AE,log1p(I),log1p(S_E)]。O不log；原B三头直接引用，共24个新头，无调参或VLM/积分重跑。
- 脚本scripts/train_raw_ae_top32.py；输出outputs/ffn_source_composition_v1/raw_ae_top32_detection/；说明docs/RAW_AE_TOP32_DETECTION.md。24头完成、两GPU进程及汇总正常退出；三seed均值±总体std、配对差及双阈值已保存。
- 单独O AUROC(%)：Q2 64.14/LL74.53/Q3 68.19/Intern68.97。B+O相对B的ΔAUROC/AP(pp)：Q2 −2.98/−4.54、LL −0.54/−1.30、Q3 −0.73/−0.13、Intern +0.06/+0.58；不支持通用增益。
- 旧划分/mention对齐、Top32合成公式及四模型旧曲线均值核对通过（差0），24头/12汇总/72双阈值/12配对行完备，编译及diff检查通过；保留现有811及其他工作区改动，无提交上传。

## 当前任务已完成：811 单隐藏层 MLP 搜索，AUROC 优先

- 用户要求寻找优于原生SVAR/MetaToken的单层MLP参数，并明确AUROC优先、HALL-AUPR同时报告。全四模型960次训练、96测试头已于UTC 2026-09-13 19:03完成；全部核验PASS。未经新授权不继续按已见测试扩大搜索。
- 中文报告 docs/SINGLE_MLP_SEARCH_811_RESULTS.md；预注册 docs/SINGLE_MLP_SEARCH_811_PROTOCOL.md；输出 outputs/single_mlp_search_811_v1/（32组/96seed、参数、960验证记录、native及旧单层差值、两项指标PNG/PDF、validation.json）。图已目检，全部完整。
- 入口scripts/search_single_mlp_811.py；汇总scripts/summarize_single_mlp_search_811.py；tests/test_single_mlp_search_811.py 6项unittest PASS（scaler、单层/预算、checkpoint重载、四模型test gate、模拟中断后只补239fits）。环境无pytest，已按项目unittest运行，无新增依赖。审计960fit验证指标/最优epoch/scaler/选择、96测试概率CPU重载、36native指标与样本对齐全部PASS。

## 实验设置与报告口径（用户要求每次说清楚）

- 图片811=3200训练/400验证/400测试，split seed20260912，保留原3200和全部mentions；旧800此前已查看，属于探索性对照。seeds43/44/45先分别算AUROC/HALL-AUPR，再均值±总体std；不是ensemble，后者另列CSV。
- 真实RMS K32与冻结RMS K50×V/VP/G/VP+G=8组/模型。V=[AE_V,log1p(S_V)]，VP=[AE_VP,log1p(S_V+S_P)]，G=[AE_G,log1p(S_G)]，VP+G拼接后两块。AE_VP联合区域重算，空G置零但不删mention。S是逐token响应范数和gross，四组不输入残差/bias附加量。
- 真RMS路径z−A_all→z，J_(FFN∘Norm)，K32；冻结RMS端点D(z)、纯FFN沿αNorm(z)，K50全来源分解。两版AE相同，路径/基点/Norm均有区别，不能只归因RMS导数。
- 每组固定24候选seed43；val AUROC/AP/索引选top3补44/45；三seed验证均值选配置。每模型240fits，再从8组以验证均值选champion；四selection.json全冻结后才test。候选seed20260914，未看新test改参、不train+val重训。
- 严格单隐藏Linear→可选BN→ReLU/GELU→Dropout→Linear，BCE REAL=1、Adam。可选标准化只拟合train；max150epoch，早停20，ReduceLROnPlateau factor.5/patience6/min1e-6；checkpoint及早停/调度监控val_loss或val_AUROC，由候选预注册。
- 原生基线SVAR248/ReLU/Adam.001/batch32/max50/patience5/val_loss、无BN/dropout/scaler；Meta训练StandardScaler+LR(lbfgs,max2000)/GB100。未获相同HPO预算，不能称公平预算特征单独优势；Meta保留完整回答长度/span。阈值本轮train REAL-F1，native val REAL-F1；AUROC/AP不依赖阈值。

## 核心结果（验证选出的champion，不按test选）

- 四champion都是true_rms；Qwen2.5选V，其余VP+G。三seed测试AUROC/AP均值(%)：Q2 88.12/44.04，LL 90.01/71.85，Q3 92.88/72.75，Intern 88.99/60.08。
- 对native SVAR的AUROC/AP差(pp)：Q2 +.78/−3.01，LL −.44/+.88，Q3 +3.92/+10.10，Intern +2.01/+6.34。AUROC超过SVAR/MetaLR/MetaGB全部三者为3/4；LL未超过SVAR；Q2 AUROC提高但AP下降。没有显著性或独立测试推广保证。
- 对旧同组sklearn单层的AUROC/AP增益(pp)：Q2 +6.07/+11.80，LL +.90/+1.74，Q3 +3.98/+8.74，Intern +4.59/+9.56。差异包含实现、标准化、正则化、优化与搜索预算，非单因素消融。
- Q2 c11：256/GELU、scaler+BN、drop.5、lr.003/wd1e-6/batch256/val_loss，epochs30/31/22。
- LL/Q3 c12：128/ReLU、BN无scaler、drop.3、lr.001/wd.001/batch512/val_AUROC，LL epochs61/53/70、Q3 109/83/88。
- Intern c21：248/GELU、scaler+BN、drop.1、lr.0003/wd.001/batch512/val_loss，epochs48/62/62。

## 双机执行与恢复

- 本机GPU0 Q2、GPU1 LL；子代理server_4090/32678 GPU0 Q3、GPU1 Intern。四worker已结束，GPU空闲，两个EXPERIMENT_PROGRESS_*.md保留24/24 completed。无提交/上传，保留已有改动。
- 日志outputs/single811_{qwen2,llava,qwen3,internvl}.log；Intern恢复日志single811_internvl_resume1.log，summary日志single811_summary.log。全部selection冻结时间保留：Q2 18:36:58、LL18:45:50、Intern18:46:25、Q3 18:57:06 UTC。
- Intern与watcher曾D态卡在progress.json的NFS inode revalidate；旧worker退出且锁释放后同pipeline恢复，仍卡同progress inode；同字节原子重发该progress后恢复，新worker19:03:26完成。240fits/selection不变，resume无FIT，无重训。旧watcher退出/锁释放后唯一watcher刷新最终状态，未并发写同模型。

## 前序结果与未修问题

- 真实区域AE811324头：docs/REGION_AE_811_RESULTS.md、outputs/all_attention_ae_811_v1/；冻结811144头/480候选：docs/FROZEN_REGION_AE_811_RESULTS.md、outputs/frozen_region_ae_811_v1/，均完成核验。
- 冻结K50纯积分通过；Qwen2/Qwen3来源重构总闭合尾部最大9.51%/11.62%，本轮未改提取。原B2 K4纯积分7.46%>1%未修，本轮没有使用B2 K4。
- native36头 docs/NATIVE_BASELINES_811_RESULTS.md，统一baseline72头 docs/SVAR_METATOKEN_811_RESULTS.md，旧82区域AE288头 docs/ALL_ATTENTION_AE_STRENGTH_20260913.md，均已完成。旧82报告含ensemble，不能混比seed均值。
- 详细运行归档 docs/archive/CURRENT_TASK.single-search-runtime-20260914.md；前序docs/archive/CURRENT_TASK.before-single-search-20260914.md及其归档链；结果索引docs/EXPERIMENT_RESULTS_INDEX.md。
