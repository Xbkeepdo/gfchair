# 当前任务摘要

## 当前：冻结RMS的相同区域AE四组811对照

- 用户明确选择“补跑相同V、VP、G、VP+G特征组的811对照”。scripts/train_frozen_region_ae_811.py，outputs/frozen_region_ae_811_v1/，复用AE811的区域AE及图片/mentions/标签/掩码，仅替换gross为旧K50冻结RMS全来源分解。四模型4组×3分类器×3seeds=144头，480选参候选。
- V=[AE_V,log1p(S_V)]，VP=[AE_VP,log1p(S_V+S_P)]，G=[AE_G,log1p(S_G)]，VP+G完整拼接后两块。AE_VP区域重新归一化/gate；不是旧full-prefix gated mass相加。全分解包含residual/bias，但这些四组不额外输入它们。
- 冻结来源：outputs/ffn_source_composition_v1/source_strength_detection/<model>/matrices.pt的raw_visual/prompt/generation，来自shards/k50的sum token ||c_j||；c_j=∫J_FFN(alpha*Norm(z))D(z)a_j d_alpha。局部FP32 K50，不使用B2 K4。
- 四模型prepare全部PASS：AE逐值相同、mentions/标签/三split相同、4000 K50 shards、空G零、非负有限。已有组合/811回归6测试PASS。初次prepare因AE protocol无counts键失败，已改从masks计数后四模型成功。每模型50训练图片缓存gross/因果分区复核PASS，共585targets/19020layers，最大相对差5.92e-8，cache_validation.json。
- K50全量纯积分最大相对误差Q2/LL/Q3/Intern约1.05e-6/2.38e-7/1.78e-6/2.41e-7；总闭合最大9.51%/.0545%/11.62%/.3384%，Q2/Q3保留原生来源重构误差，不能宣称完全闭合。numerics.json明确all_sample_closure_pass=false。真实S_all与冻结K50均沿capture_full接口，但不同历史capture；AE缓存本轮两版逐值相同，不把新AE exact-prefix本身当成本轮变因。
- 15:43 UTC四模型全部144头完成（每模型12三层+24搜索头/120候选/8选择）。双机日志outputs/frozen811_{qwen2,llava,qwen3,internvl}.log，原进度文件完成后回退尚在运行的统一baseline。
- 单一进度watcher primary=baselines_811_v1、secondary=frozen_region_ae_811_v1；本机确认旧锁释放后起frozen811_progress_local。原生/统一头旧任务保留。全局frozen811_summary已完成，输出docs/FROZEN_REGION_AE_811_RESULTS.md、48组/144seed、vs_true_rms.csv及差值热图PNG/PDF（summary已重启加载绘图版本）。逐组同分类器同seed同400测试图报告冻结减真实Norm差，std另存ensemble，无新bootstrap。144头独立指标重算/480候选/32选择/val checkpoint验证PASS(detector_validation.json)。末尾Q3 seed45曾FileNotFound，重读完成全部36头验证，无重训；图已目检。

## 统一实验与结果口径（用户要求每次报告都说明）

- 811按图片3200训练/400验证/400测试；保留旧3200训练，旧800用seed20260912分半，全部mentions不删。旧800曾被研究查看，只称探索性比较。seeds43/44/45，各seed先算AUROC/HALL-AUPR再平均；与概率ensemble明确分开。
- 三层Torch[128,64,32]、BN/dropout.3、Adam.001/wd1e-5/batch256/max100，验证loss最优checkpoint/调度/早停。单层sklearn12候选（64/128/256、.01/.001、adam/sgd，max500）；XGB18候选（depth4/6/8、lr.1/.05、100/200/500树）。seed43验证AUROC、AP、索引依次选参，最终仅拟合3200训练，不train+val重训。无额外标准化；分类器差异不限层数。
- 原生SVAR是248单隐藏ReLU、Adam.001/batch32/max50、val loss/patience5、无BN/dropout/scaler；MetaToken是训练StandardScaler+LR(lbfgs,max2000)与GB100。均内部1=HALL，结果保存转P(REAL)；原生阈值val REAL-F1，统一头train REAL-F1，排名指标不依赖阈值。

## 已完成与继续运行的前序任务

- 原生SVAR/MetaToken四模型36头全部完成，docs/NATIVE_BASELINES_811_RESULTS.md和outputs/native_baselines_811_v1/。14相关+3原核心测试、9头实际恢复、36头重载重算三split概率验证通过；统一线程后最大差0。未统一BLAS的首次逐位比较出现2.22e-16舍入差已记录。
- 真实Norm AE811四模型324头全部完成，docs/REGION_AE_811_RESULTS.md及outputs/all_attention_ae_811_v1/；区域AE＋真实Norm S_all K32四fusion+四AE-only+legacy_visual。全局summary_progress completed(15:08)。
- 用户上一轮明确仅汇总已有“我的方法”与原生基线，未授权/未运行AE接原生分类器；docs/OURS_VS_NATIVE_BASELINES_811.md、outputs/ours_vs_native_811/，48fusion组及12native组概率复算/对齐。VP+G三层减原生SVAR的AUROC/AP(pp)：Q2 +.02/−.22，LL −.40/+1.10，Q3 +3.34/+7.38，Intern +.23/−.44。当前冻结RMS补跑是新授权。
- 统一SVAR/MetaToken811入口scripts/train_baselines_811.py，outputs/baselines_811_v1/；每模型18头/60候选，共72头。Q2/Intern已完成；LL单层SVAR仍搜索，Q3搜索中；server_4090监控远端，全局baseline811_summary等待。共有controlled特征，SVAR零基[5,19)层head视觉mass，MetaToken10+H且含完整回答长度/span，非严格pretarget-only；无新scaler。
- 原82区域AE四模型288头已完，docs/ALL_ATTENTION_AE_STRENGTH_20260913.md及outputs/all_attention_ae_log_strength_v1/；288概率、960候选64选择验证PASS。旧主报告多处ensemble，不能与seed均值直接比；旧单层Torch48候选也不同于本轮sklearn12候选。

## 未修问题与归档

- 原All-attention/B1/B2 outputs/ffn_all_source_paths_v1已180头/bootstrap，但B2 K4全量7.46%纯积分误差>1%，提高K/审计/60头重训仍待做；本轮不使用B2 K4。B1-F6对F0的8个CI均跨零。报告docs/ALL_SOURCE_PATHS_RESULTS_20260913.md、ALL_SOURCE_SIGNALS_20260913.md、ALL_SOURCE_DETECTION_COMPARISON_20260913.md。
- 前序细节docs/archive/CURRENT_TASK.before-frozen-811-20260913.md及其中归档链，完整索引docs/EXPERIMENT_RESULTS_INDEX.md。保留已有未提交修改，未经新授权不提交/上传，不写入凭据。
