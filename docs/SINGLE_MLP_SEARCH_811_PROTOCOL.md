# 811 单隐藏层 MLP 搜索设置（测试前冻结）

目标：AUROC 优先，HALL-AUPR 同时报告。搜索是否能优于原生 SVAR、MetaToken LR/GB；不保证超过，不用测试指标选参或追加搜索。

- 图片划分：3200 train / 400 validation / 400 test，保留旧3200训练，split seed20260912。全部 mentions 保留；该留出集此前已经查看，本轮是探索性比较。
- 特征：真实 RMS K32、冻结 RMS K50，各做 V、VP、G、VP+G，共8组/模型。V=`[AE_V,log1p(S_V)]`；VP=`[AE_VP,log1p(S_V+S_P)]`；G=`[AE_G,log1p(S_G)]`；VP+G完整拼接后两块。AE_VP在联合区域重新计算。S是token响应范数之和gross。
- 真实 RMS 路径：z−A_all→z，积分 J_(FFN∘Norm)。冻结 RMS：端点D(z)，纯FFN沿αNorm(z)的K50积分；完整分解含残差、attention、bias，当前四组只取attention gross。两路径/基点也不同。冻结源重建继承Qwen2/Qwen3总闭合尾部误差，本轮不改提取。
- 分类器：一个隐藏层，Linear→可选BN→ReLU/GELU→Dropout→Linear，BCE REAL=1，Adam。标准化可选且仅拟合train。无类别重权，不重新拟合train+val。
- 固定24候选，构造seed20260914：先固定10个anchor，再按脚本固定顺序抽14个不重复配置。宽度、标准化、BN、dropout、激活、lr、wd、batch、checkpoint monitor均保存在每模型protocol.json。不能在看新测试后改配置。
- max150 epochs，early stop patience20；ReduceLROnPlateau factor.5/patience6/min_lr1e-6。checkpoint/调度/早停按候选指定val_loss或val_AUROC。
- 每组24候选先seed43，按验证AUROC、验证HALL-AUPR、索引升序排前三；前三补44/45，然后按三seed验证均值选配置。每组30fits；每模型240fits；四模型960fits。
- 每模型在8个入选组间按三seed验证AUROC/AP选一个champion，最后以固定组顺序破同分。必须四个selection.json全冻结，才允许任何新测试评估。
- 测试：8组×3seed×4模型=96头。分别计算seed43/44/45指标再取均值±总体标准差；概率ensemble另列。完整报告8组，不能按test选champion。
- 比较：原生SVAR（248单隐藏ReLU/Adam.001/batch32/max50/val_loss早停5）、MetaToken LR（训练StandardScaler+lbfgs max2000）、GB（训练StandardScaler+100树），三者分别比较。原生基线没有相同HPO预算，本轮不能证明公平预算下的特征单独优势。
- 辅助旧单层对照：同组、同811、同seed的旧sklearn12候选（max500，无额外标准化）。新旧差值包含实现、正则化、优化、checkpoint及预算，不是仅层数差异。
- 附带阈值：本方法train REAL-F1及固定.5；原生val REAL-F1。AUROC/HALL-AUPR不依赖阈值。

执行：本机GPU0 Qwen2.5、GPU1 LLaVA；32678 GPU0 Qwen3、GPU1 InternVL。模型独占锁、逐fit原子保存/恢复；两个进度文件10秒覆写。主脚本`scripts/search_single_mlp_811.py`，审计/汇总`scripts/summarize_single_mlp_search_811.py`。

输出：`outputs/single_mlp_search_811_v1/`；最终`docs/SINGLE_MLP_SEARCH_811_RESULTS.md`、全32组/96seed CSV、具体参数、与三种基线差值、图、960验证记录及独立审计。
