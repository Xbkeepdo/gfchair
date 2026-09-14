# 当前任务摘要

## 最新：仅汇总我的方法与原生基线

- 用户明确选择“只汇总我的方法已有的检测结果，与原生基线比较”，本次无新增训练。docs/OURS_VS_NATIVE_BASELINES_811.md与outputs/ours_vs_native_811/{ours,native,deltas}.csv；四模型4组×3分类器=48组三seed结果及12组原生基线均从保存概率复算，mentions/标签/三split逐行一致。
- 15:08 UTC Qwen3 AE811全部81头完成，四模型AE811已全完成。VP+G三层对原生SVAR的AUROC/AP差(pp)：Q2 +0.02/−0.22，LL −0.40/+1.10，Q3 +3.34/+7.38，Intern +0.23/−0.44。AE原生分类器头尚未运行；当前是各自配置系统效果比较，不是同分类器特征消融。

## 当前：原生SVAR/MetaToken分类器811

- 用户要求“用他们原生分类器看看效果”。scripts/train_native_baselines_811.py复用detection/baselines.py已有原生头与同一811全部mentions；SVAR单隐藏248/ReLU/Adam .001/batch32/max50、验证loss最优及patience5，无标准化/BN/dropout/调度；MetaToken训练集StandardScaler+LR(lbfgs,max_iter2000)和GB100。43/44/45，9头/模型，四模型36头。
- outputs/native_baselines_811_v1/，native protocol指纹冻结，独立feature_cache避免与统一头写冲突。内部1=HALL，保存全部split概率显式转P(REAL)，阈值验证REAL-F1及0.5；AUROC/AP三seed均值/std另存ensemble。保留controlled样本及MetaToken完整回答长度/span信息；仅原生分类器与811适配，不宣称论文原始数据集复现。
- 15:02 UTC四模型原生36头全部完成，docs/NATIVE_BASELINES_811_RESULTS.md及输出根detection.csv(12组)/seed_metrics.csv(36行)。日志outputs/native811_{qwen2,llava,qwen3,internvl}.log。双机唯一watcher已确认锁释放再替换，原生完成后自动回到仍跑的统一头。
- 14项相关测试+3项原核心回归PASS；Qwen2真实恢复9头mtime未变。36头从保存分类器重算train/val/test概率和指标全部一致，independent_validation.json。首次未限BLAS线程的逐位比较出现2.22e-16舍入差；统一单线程后最大差0。原核心仅加可选epoch_callback。

## 新增：SVAR与MetaToken的同811对照

- 用户要求“那也看看svar和metatoken的”。按当前三层MLP/sklearn单层/XGB统一分类器比较两种基线特征，不是论文原生SVAR248隐藏头或MetaToken标准化LR/GB。每模型18头/60候选，共72头/240候选。
- scripts/train_baselines_811.py，输出outputs/baselines_811_v1/。复用各模型COCO4000-INSLEN-OFFICIAL-TARGET/baseline/features.pkl中的controlled样本；四模型全部记录的image/目标位置/token/label/word多重集合与AE mentions完全一致。按语义键重排、保留重复/冲突mentions，不补齐或删除。
- SVAR使用既有零基[5,19)层每head视觉attention mass（Qwen2 392维）；MetaToken使用原10+H维（Qwen2 38维），保留完整回答长度和对象span统计，信息范围非严格pre-target-only。均不额外标准化，训练协议与AE811完全相同。
- 统一基线Qwen2已18头完成，LL/Intern仍在SVAR单层搜索；Qwen3在AE811完成后自动启动。日志outputs/baseline811_{qwen2,llava,qwen3,internvl}.log。
- 单一进度watcher新增--secondary-progress-root outputs/baselines_811_v1；primary仍AE811，未完成时显示AE811，完成后接续显示基线。本机baseline811_progress_local已起；远程确认旧锁释放后启动双根watcher，共享文件14:46:33实际刷新通过。
- 已复核39个完成头的保存概率、mentions/标签/图片划分及验证集选参，阶段结果docs/SVAR_METATOKEN_811_EARLY_RESULTS.md与outputs/baselines_811_v1/early_{detection,validation}.csv。三层下SVAR有竞争力；MetaToken用XGB较两个MLP显著提高数值，但尚无标准化/原生头消融，不能断言特征本身差。
- 全局screen baseline811_summary运行同入口--summarize-all，等待四模型后生成docs/SVAR_METATOKEN_811_RESULTS.md、24组/72seed/ae_vs_baselines.csv及同分类器直接对照表。缺失/替代样本拒绝、重复/冲突保留、进度切换及811回归共11项PASS。

## 当前：区域AE＋All-attention gross的8:1:1对照

- 用户要求“看看811的效果”。按图片3200训练/400验证/400测试；默认保留原3200训练，原800以seed20260912分400验证/400测试。已异步询问是否希望重新随机划分，未收到其他选择，按默认继续。旧800曾被研究查看，明确为探索性评估，不称新独立测试。
- 沿用4组fusion：V=[AE_V,log1p(SV_all)]、VP=[AE_VP,log1p(SV_all+SP_all)]、G=[AE_G,log1p(SG_all)]、VP+G完整两块；加4组AE-only及旧视觉legacy_visual=[AE_V,log1p(S_E)]，共9组×3分类器×3seeds×4模型=324头。
- 三层MLP128/64/32、BN/dropout.3等原参数不变，改为验证loss选择checkpoint/学习率调度/早停；单层sklearn12候选、XGB18候选，seed43按验证AUROC/AP选参，最终43/44/45只拟合3200训练图，不合并验证图重训。阈值仍train-F1和固定0.5，AUROC/AP报告mean/std及ensemble两种口径。
- 新入口scripts/train_region_ae_811.py --model MODEL --device cuda:N --workers 8；输出outputs/all_attention_ae_811_v1/。每模型独占锁、原子保存、协议指纹；复用已有特征，不再VLM提取。--split-mode reshuffle可用但当前未选。
- train_torch_probe_feature_sets.py新增显式checkpoint_selection=minimum_val_loss；默认minimum_train_loss保留。3项测试PASS：图片互斥/可重复划分、train改善val变坏时选val最佳epoch、旧默认选train最佳epoch。
- 汇总screen ae811_summary运行同脚本--summarize-all，等待四模型后输出docs/REGION_AE_811_RESULTS.md、108组均值/std、324seed及vs82_same400.csv。把旧82概率限制到相同400测试图比较，不能直接减旧800图分数。状态summary_progress.json，日志outputs/ae811_summary.log。
- 实际21头val最佳checkpoint与两本机模型恢复协议已核验PASS；下一步监控任务完成；最终从概率复核指标/标签对齐/选择不接触test。2026-09-13 14:50 UTC：Qwen2.5/LL/Intern各81头完成；Qwen3原82提取4000/4000、固定24头完成，搜索182/240，正式29/48。三模型四fusion的三层早期对照见docs/REGION_AE_811_EARLY_RESULTS.md：同400图AUROC六升六降、AP两升十降；不能说811普遍更好。未经新授权不提交/上传。

## 前一轮82已完成

- 四模型全部4000图/72头，原82全局报告docs/ALL_ATTENTION_AE_STRENGTH_20260913.md、PNG/PDF及outputs/all_attention_ae_log_strength_v1/完整结果；288头概率复算、960候选和64选择验证通过。详情见docs/archive/CURRENT_TASK.before-native-811-20260913.md。
- 比较必须统一口径：旧主报告多处为三seed概率ensemble，近期为三seed指标均值。旧单层Torch+BN/dropout/48候选，本轮sklearn/12候选，不能将差距全归于S。
- AE按区域独立归一化和MAD gate，VP重算，不是AE_V+AE_P；S是真实Norm的All-attention K32。新AE复现旧执行形状（Qwen逐目标，LL/Intern完整因果回答），签名冻结不改；全量parity只证明旧AE复现，不是FP32真值。

## 更早实验与未修事项

- 原All-attention/B1/B2：outputs/ffn_all_source_paths_v1，180头及bootstrap。B2 K4全量7.46% case闭合>1%，提高求积/全层审计/重训仍待做，本轮不使用B2特征。B1-F6对F0的8个CI均跨零。
- 报告docs/ALL_SOURCE_PATHS_RESULTS_20260913.md、docs/ALL_SOURCE_SIGNALS_20260913.md、docs/ALL_SOURCE_DETECTION_COMPARISON_20260913.md；完整结果索引docs/EXPERIMENT_RESULTS_INDEX.md。
- 本次转换前的完整状态见docs/archive/CURRENT_TASK.before-811-20260913.md，更早原记录和已授权发布历史保留在其引用的归档。当前仓库有其他未提交工作，保留，不提交凭据；旧上传授权不是新任务自动发布指令。
