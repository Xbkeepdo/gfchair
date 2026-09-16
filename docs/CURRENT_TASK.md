# 当前任务摘要

## 当前进行：同分解 S_PV 与 κ_P/κ_V∪P 比较（2026-09-17）

- 用户明确`S_PV=Σ_{m∈P∪V}||u_m||`。真RMS All-attention K32已从同路径缓存复原κ_P和逐token κ_V∪P，固定811四模型48新头、各2000次配对图片bootstrap、CPU重载完成，逐层及同图κ曲线见`outputs/decomposition_pv_kappa_811_v1/all_attention/summary.md`。κ_P拼接相对AE+log1p(S_PV)的AUROC区间在Q2/Q3/Intern全正；κ_V∪P拼接四模型AUROC点估计均低于κ_P拼接。同图HALL−REAL的κ_P绝大多数层为正，κ_V∪P较弱。
- 含残差B1/B2的逐token P/V κ及B1的S_PV未保存在正式缓存；B2原K4验收失败，不能以组向量κ或K4近似冒充同口径结果。后续需新提取（数小时级GPU）并数值验收，再做B1/B2的检测比较。旧Visual-only无prompt来源，只作参考。

## 最新完成：Shikra/MiniGPT-4 全视觉 attention×概率求和（2026-09-16）

- 复用两模型 ENDAC `[L,N_V]` attention、目标首 token raw/norm softmax 概率，每层对全部视觉 token 算 `sum_v a_v p_v(y)` 并拼接 `log1p(S_E)`；固定811与43/44/45，raw/norm各24次Optuna trial，验证集选参后测试。MiniGPT-4 raw/norm AUROC/HALL-AUPR(%)为93.20/76.24、93.56/77.26；Shikra为87.90/68.45、87.49/68.05。两模型相对已有Top32无清晰优势，旧对照为冻结24候选而非本轮TPE。完整比较见`docs/PREFIX_FULL_VISUAL_PRODUCT_811_RESULTS.md`，三种子与参数见`outputs/coco4000_512_endac_prefix_full_product_optuna_811/summary.md`；12/12最终头CPU重载PASS，最大概率差1.79e-7。

## 最新完成：log1p(S_E)+κ_P^e 的811检测（2026-09-16）

- 从旧Visual-only F仅取逐层`log1p(S_E)`，拼接真RMS All-attention K32的逐层prompt响应`κ_P^e`；同cohort固定811、seeds43/44/45、128/noBN/train-only z-score/val-loss。新训logS-only和logS+κ共24头，复用κ-only和AE+logS+κ；无新VLM前向。全量对齐、24头CPU重载及2000次/模型配对图片bootstrap通过。
- logS+κ AUROC/HALL-AUPR(%)：Q2 85.61/40.29、LLaVA 89.37/64.79、Q3 91.00/68.09、Intern 87.42/54.34。相对logS-only，AUROC四模型名义区间均全正；AP仅Q3全正，LLaVA点估计下降0.59pp。AE+logS+κ仍四模型AUROC点估计更高。详见`outputs/logs_prompt_effect_kappa_811_v1/summary.md`。

## 最新完成：目标词 softmax 概率的全视觉 MAD 检测（2026-09-16）

- 每层对所有视觉 token 的目标首 token 词表概率算 `median_v |p_v−median_u p_u|`，raw/norm 两版，分别单独与拼接 `log1p(S_E)`；复用矩阵、固定811，四模型 480 次验证拟合与 48 个最终测试头完成。逐模型四组结果、数值限制见 `docs/SEMANTIC_PROBABILITY_MAD_811_RESULTS.md`，全量三种子/超参/预测见 `outputs/coco4000_512_endac_semantic_probability_mad_optuna_811/`。
- Qwen3 raw MAD+logS 测试 AUROC/HALL-AUPR 89.75/74.04%，AP 较 AE+logS 高 2.29pp、AUROC 低 0.10pp；其余模型无一致优势。Qwen2.5 raw MAD 有 19,372/221,872 个零层行，涉及 7,923/7,924 mentions；11/28 层训练标准差低于 trainer 的 `1e-12` 缩放阈值。48 头重载在显式 `1e-5` CPU/GPU 概率容差下 PASS，最大差 `5.11e-6`，指标差 `3.33e-16`；原通用 `1e-6` 门槛未通过，已在报告注明。

## 最新完成：attention Top32 / 联合 Top32 JS 与全视觉乘积（2026-09-16）

- 两个空间分布在同一区域内分别归一化后计算 base-2 JS：attention Top32，或 attention Top32 与目标词概率 Top32 的并集；另比较全部视觉 token 的 `sum_v a_v p_v(y)`。四模型 raw/norm 共六项，每层与 `log1p(S_E)` 拼接。复用已保存矩阵，无新 VLM 提取。
- 811 划分、43/44/45 种子、train-only 标准化、无 BN 单隐藏层；每项 24 次 Optuna trial，参数/checkpoint 按验证集冻结，六项特征按测试集探索性比较。四模型 720 次验证拟合、72 个最终测试头完成；CPU 重载/指标复算 PASS。
- 逐模型六项完整表见 `outputs/coco4000_512_endac_semantic_js_full_optuna_811/summary.md`，与 AE/旧 Top32 的对比见 `docs/SEMANTIC_JS_FULL_811_RESULTS.md`。Qwen2.5 最佳 JS 为 norm 联合32（85.74/49.73 AUROC/HALL-AUPR%）；LLaVA、InternVL 本轮以全视觉乘积较好（92.59/77.42、88.98/64.23）；Qwen3 最佳 JS 为 norm attention32（90.47/72.46）。Qwen2.5 raw attention32 的 3.40% mention×层行有目标概率零质量，使用均匀回退并单独注明。

## 最新完成：Prompt WRITE κ_P^a 的811检测与F拼接（2026-09-16）

- 从已有真RMS K32来源缓存还原`κ_P^a=||Σ_P a_m||/Σ_P||a_m||`，四模型全部值在(0,1]，与`κ_P^e`及F对齐；24个新头+复用三组既有头，固定811/seeds43–45/同128-noBN配置，CPU重载与各2000次配对图片bootstrap通过。
- κ_P^a单独与κ_P^e单独四模型双指标差值区间均跨零。F+κ_P^a较F宏平均AUROC/AP+1.22/+2.66pp；LLaVA/Qwen3/Intern AUROC区间全正，Intern AP全正。对F+κ_P^e，仅Qwen3 AP明确较低（−2.12pp，95% CI [−3.99,−0.16]）。
- 检测、a_m/e_m同cohort曲线与完整区间见`docs/PROMPT_WRITE_KAPPA_811_RESULTS.md`、`outputs/prompt_write_kappa_811_v1/`；无新VLM前向。

## 最新口径：四模型 Optuna 参数按验证集、特征按测试集选择（2026-09-16）

- 用户澄清仅特征在 test 上从五个已验证选参的变体中选择，不在 test 上重新搜参数。按 test AUROC 选：Qwen2.5 AE、LLaVA norm Top32、Qwen3 AE、InternVL raw Top32；仅 InternVL 与原 validation-champion 不同，test AUROC/AP 从 88.23/62.41 变为 89.61/66.88%。Qwen3 若按 test HALL-AUPR 单独选，则为 raw Top32 71.99%（AE 71.75%）。明细见 `outputs/coco4000_512_endac_four_ae_semantic_optuna_811/test_feature_selected_summary.md`；原验证选择报告保持不变。此口径是探索性 test 特征选择，不是独立测试泛化估计。

## 最新完成：四模型 AE / 语义注意力 Optuna 搜参（2026-09-16）

- `vicr` 已装 Optuna 5.0.0；Qwen2.5、LLaVA、Qwen3、InternVL 的 AE 与 raw/norm Top16/32 语义注意力组共 600 次四卡拟合、验证冻结后的 60 个测试头均完成；train-only 标准化、无 BN，核验 PASS。模型内选中测试 AUROC/AP(%)：Qwen2.5 AE 87.18/54.77、LLaVA norm Top32 92.41/76.99、Qwen3 AE 89.85/71.75、InternVL norm Top32 88.23/62.41。详见 `docs/FOUR_AE_SEMANTIC_OPTUNA_811_RESULTS.md` 与 `FOUR_AE_SEMANTIC_OPTUNA_811_PROGRESS.md`。

## 最新完成：两模型 AE / 语义注意力单层 MLP 搜参（2026-09-16）

- MiniGPT-4、Shikra 复用 ENDAC-811，比较 `AE_V+log1p(S_E)` 与 raw/norm Top16/32 `semantic_attention+log1p(S_E)`；Torch 单隐藏层固定 24 候选，强制 train-only 标准化、无 BN。两模型各 150 次验证拟合已完成，冻结选择后评估 test；30/30 最终头重算 PASS。MiniGPT-4 验证选中语义 `norm_top32` 测试 AUROC/AP 93.47/76.84%，AE 为 93.53/77.95%；Shikra 语义 `raw_top16` 87.82/68.51%，AE 为 86.22/64.43%。完整表见 `outputs/coco4000_512_endac_prefix_ae_semantic_single_mlp_811/summary.md`，进度见 `PREFIX_AE_SEMANTIC_SINGLE_MLP_811_PROGRESS.md`。

## 最新完成：MiniGPT-4 / Shikra ENDAC exact + Top-K 扩展（2026-09-16）

- 用户确认重做两模型 ENDAC exact mention/response-offset 定位，并同步提取 SVAR、MetaToken、Visual-only K4、All-attention K32 与 raw/norm Top16/32 六项特征；矩阵保留以便复原全视觉求和。进度见 `ENDAC_811_PREFIX_PROGRESS.md`。
- 新配置 `configs/model_configs_endac_811_prefix.yaml`；两模型各 4000 张 exact 标注与固定 811 图片划分完成。MiniGPT-4 全量特征已合并（3929 图/9140 行），Shikra 四份续跑并合并（3976 图/13067 行）；逐条对齐、数值、矩阵形状核对 PASS。原生/三层/单层搜索与 Top-K 六项 standalone/+logS/+S 的 811 训练、测试均已完成，见 `ENDAC_811_PREFIX_PROGRESS.md`。
- 验证集选出的语义组合：MiniGPT-4 `norm_top16/all_six+S` 测试 AUROC/HALL-AUPR `93.67/78.16%`，Shikra `raw_top32/all_six+S` 为 `87.79/69.91%`；504/504 个语义头与预测重算 PASS。原四方法与 84 组完整逐模型结果分别在 `outputs/coco4000_512_endac_811/<model>/results/summary.md` 和 `outputs/coco4000_512_endac_prefix_semantic_attention_topk_detection_811/<model>/summary.md`。

## 最新完成：AE/log1p(S)分别拼接五个几何块811检测（2026-09-16）

- 按用户澄清，项目既有`legacy_visual=[AE_V,log1p(S_E)]`分别与prompt/generation κ、B1 residual G−V balance、B1 Δcos(R,G)、visual context change拼接；固定811、seeds43/44/45、统一128/noBN/train-only z-score/val-loss，无HPO、无VLM前向、无B2。
- prompt κ宏平均增量AUROC/AP +1.36/+2.60pp，B1 Δcos(R,G) +1.31/+2.62pp，是跨模型最稳定的两个块；前者AUROC 3/4模型名义区间全正，后者AUROC 3/4模型全正。其余三个块收益更局部。
- 72个主结果头完成CPU重载、checkpoint与三划分概率复算；用户澄清到达前队列额外完成的12个五量合并头仅存supplementary，不进主表/图/结论。每模型2000次图片bootstrap完成；报告docs/LEGACY_VISUAL_GEOMETRY_FUSION_811_RESULTS.md，输出outputs/legacy_visual_geometry_fusion_811_v1/。
- κ_P单独图已用相同4000图缓存绘制全量与811测试REAL/HALL逐层均值、IQR及差值，无新训练；见`outputs/legacy_visual_geometry_fusion_811_v1/prompt_kappa/summary.md`。

## 最新完成：五个来源/残差几何标量811检测（2026-09-16）

- 复用真RMS All-attention K32与数值通过的B1缓存；单独检测prompt/generation组内κ、B1 residual G−V balance、B1 Δcos(R,G)、visual context relative change，并补五量拼接及generation κ+position；无VLM前向，不使用B2 K4。
- 固定3200/400/400、seeds43/44/45、train-only z-score、单隐藏128/noBN/val-loss；84个新头及每模型2000次图片bootstrap完成，CPU重载与三划分概率复算PASS。
- 五量拼接AUROC/AP(%)：Q2 85.54/41.01、LL90.08/68.39、Q3 90.77/66.95、Intern87.44/56.12；单量宏平均以B1 Δcos(R,G) AUROC较高，单量均弱于五量拼接的点估计。
- generation κ+position相对κ-only四模型双指标区间均跨零；κ+position相对position-only均为正，说明κ含位置之外的检测信息，但未完成等长度匹配，不能称已消除长度混杂。报告docs/SELECTED_GEOMETRY_SCALARS_811_RESULTS.md；输出outputs/selected_geometry_scalars_811_v1/。

## 最新完成：Raw attention / AE Top-K类别曲线（2026-09-15）

- 复用四模型COCO4000 K50缓存；视觉token内归一化后算Top16/32质量。50812 mentions、1024行统计及门禁通过；产物outputs/topk_raw_attention_ae_curves_v1/，无VLM或检测训练。

## 最新完成：Visual-only条件路径Torch单层搜参811（2026-09-15）

- 复用`legacy_visual=[AE_V,log1p(S_E)]`固定811缓存；沿用此前24候选Torch单层协议，seed43筛前三再补44/45，四模型120次拟合，无VLM提取。已有XGB18候选直接作对照。
- 新单层AUROC/AP(%)：Q2 87.95/45.73、LL89.74/68.37、Q3 89.81/68.45、Intern86.69/53.70。相对旧三层差pp：+2.26/+8.52、−.36/−1.15、−.31/+1.51、−.05/−.81；仅Q2同时明显受益。
- seeds42/43/44的Visual-only AUROC/AP为87.90/45.75、89.66/67.57、89.68/68.26、86.66/53.59；未标准化SVAR为86.61/43.90、90.42/71.24、89.23/64.28、87.85/55.28。
- SVAR仅加train-only z-score后为85.19/39.46、90.21/70.39、89.26/63.29、87.11/53.21；宏平均较未标准化下降.58/2.09pp，四模型AP均降。12头重载PASS；报告docs/SVAR_STANDARDIZED_SEED_424344_811_RESULTS.md，GPU已释放。
- Visual-only统一标准化宏平均88.63/58.47（较原+.15/−.32pp）；再去BN为88.59/57.86（−.04/−.62pp），三个实际去BN模型AP均降。报告docs/LEGACY_VISUAL_STANDARDIZED_SEED_424344_811_RESULTS.md及docs/LEGACY_VISUAL_STANDARDIZED_NO_BN_SEED_424344_811_RESULTS.md。

## 最新完成：生成响应除以位置的811检测（2026-09-15）

- 用户确认S_G-only、S_G/response_index-only、position-only，四模型36头；真实RMS All-attention K32，复用缓存，无VLM提取。
- 沿用统一128/noBN/log1p/train-only标准化/最低val-loss及固定811/seeds43-45，比例先除再log；position=0保留且检测填0。协议docs/GENERATION_PER_POSITION_811_PROTOCOL.md；入口scripts/evaluate_generation_per_position_811.py；输出outputs/generation_per_position_811_v1/。
- 四模型36头、各2000次图片bootstrap、CPU重载及3项测试通过，无零分母。ratio对原S_G的AUROC差pp：Q2−.29/LL−.56/Q3−1.20/Intern−2.40，仅Intern AUROC名义区间全负；Q3/Intern AP区间全负。ratio仍四模型优于position-only，不支持用比值替换总量；详见docs/GENERATION_PER_POSITION_811_RESULTS.md。

## 最新完成：四模型 COCO4000 OVIR（2026-09-15）

- 原始视觉WRITE取95%能量子空间，OVIR测量真实RMS All-attention K32作用后的子空间外能量；四模型4000图、50812 mentions、1622248 target-layer，无检测器。
- HALL−REAL：Q2 +.006474、LL+.000165、Q3−.002143、Intern+.003276；方向不一致，不支持统一机制结论。完整数值门禁、bootstrap和恢复审计见docs/OVIR_ALL_ATTENTION_4000_RESULTS.md，输出outputs/ovir_all_attention_4000_v1/。

## 最新完成：视觉AE/gross全层和training-free（2026-09-15）

- 四模型原4000图cohort全部mentions；复用真RMS All-attention K32缓存，无分类器、训练、标准化或参数拟合。
- `A_V=sum_l AE_V(l)`，`S_FREE=log(1+sum_l S_V(l))`；固定`E_w=(1-w)S_FREE+wA_V`，w=0/.2/.5/.8/1，HALL风险为`-E_w`。
- w=.2的AUROC/AP(%)：Q2 62.33/14.66、LL62.08/31.27、Q3 63.62/25.96、Intern70.97/31.75。前三模型AUROC以纯S_FREE最好，Intern w=.2相对纯S提高+2.29/+2.94pp。
- 入口scripts/evaluate_visual_layer_sum_training_free.py；报告docs/AE_LOGS_LAYER_SUM_TRAINING_FREE_4000_RESULTS.md；输出outputs/visual_layer_sum_training_free_4000_v1/。3项测试、编译、来源矩阵一致性、4000图cohort及20项指标核验通过。

## 最新完成：All-attention WRITE/SS/gain 前三项（2026-09-15）

- 用户授权先做前三项；复用真RMS全attention K32四模型4000图，无新VLM提取，不做第四项冻结RMS对照。
- P/V/G/ALL的I/S/G、token内lambda分位数/IQR/CV及同图控制；固定811、统一单层128/noBN/标准化、I/S/I+S×PVG/ALL×3seeds，共72头。协议docs/ALL_ATTENTION_WRITE_GAIN_811_PROTOCOL.md。
- 四模型4000图统计、72头、各2000次图片bootstrap及CPU重载完成。PVG I+S对I的AUROC增量pp：Q2+1.81/LL+.41/Q3+1.34/Intern+1.03，除LL外名义95%区间为正；S-only对I四区间均跨零。视觉/生成强度类别差主要随WRITE差，token内gain IQR均非零但类别差小。报告docs/ALL_ATTENTION_WRITE_GAIN_811_RESULTS.md。
- 入口scripts/analyze_all_attention_write_gain.py及summarize_all_attention_write_gain.py；输出outputs/all_attention_write_gain_811_v1/。11项测试、缓存及72头核验通过；首次Torch scalar JSON保存失败已修复并重读缓存，日志保留。既有未提交改动保留，不发布。

## 最新完成：VP/G直接比值特征811检测（2026-09-14）

- 真RMS K32每层输入 `[AE_VP/AE_G, log((S_V+S_P)/S_G)]`，无分母偏移；未定义raw存NaN、检测置0。本轮四模型实际均无零分母/非有限值。
- 固定3200/400/400、seeds43/44/45、train-only StandardScaler、无BN、batch128、150 epochs及最低val-loss checkpoint；沿用对应VP+G MLP配置。
- VP/G AUROC/AP(%)：Q2 82.84/34.16、LL88.59/69.97、Q3 88.71/63.70、Intern83.70/47.94；AUROC四模型均低于V和VP+G，详见docs/VP_OVER_G_811_RESULTS.md。
- 四模型12头CPU重载、训练统计、checkpoint、概率和指标复算PASS，最大概率差1.79e-7；双机进度最终3/3，GPU已释放。

## 实验设置与报告口径（用户要求每次说清楚）

- 图片811=3200训练/400验证/400测试，split seed20260912，保留原3200和全部mentions；旧800此前已查看，属于探索性对照。seeds43/44/45先分别算AUROC/HALL-AUPR，再均值±总体std；不是ensemble，后者另列CSV。
- 真RMS K32路径`z−A_all→z`、`J_(FFN∘Norm)`；V/VP/G输入区域AE及相应gross的log1p，VP+G拼接两块，S为逐token响应范数和。冻结RMS K50另用纯FFN路径，不能混称同一分解。
- 单层搜索为24候选seed43，验证top3补44/45，再按三seed验证均值冻结；只用train拟合标准化/权重，不合并train+validation。结构为Linear→可选BN→激活→dropout→Linear、Adam、最多150 epoch。
- 原生基线SVAR248/ReLU/Adam.001/batch32/max50/patience5/val_loss、无BN/dropout/scaler；Meta训练StandardScaler+LR(lbfgs,max2000)/GB100。未获相同HPO预算，不能称公平预算特征单独优势；Meta保留完整回答长度/span。阈值本轮train REAL-F1，native val REAL-F1；AUROC/AP不依赖阈值。

## 核心结果（验证选出的champion，不按test选）

- 四champion都是true_rms；Qwen2.5选V，其余VP+G。三seed测试AUROC/AP均值(%)：Q2 88.12/44.04，LL 90.01/71.85，Q3 92.88/72.75，Intern 88.99/60.08。
- 对native SVAR的AUROC/AP差(pp)：Q2 +.78/−3.01，LL −.44/+.88，Q3 +3.92/+10.10，Intern +2.01/+6.34。AUROC超过SVAR/MetaLR/MetaGB全部三者为3/4；LL未超过SVAR；Q2 AUROC提高但AP下降。没有显著性或独立测试推广保证。
- 赢家参数和对旧sklearn单层的完整差值见docs/SINGLE_MLP_SEARCH_811_RESULTS.md；差异同时含实现、标准化、正则化、优化和搜索预算。

## 前序结果与未修问题

- 真实区域AE811324头：docs/REGION_AE_811_RESULTS.md、outputs/all_attention_ae_811_v1/；冻结811144头/480候选：docs/FROZEN_REGION_AE_811_RESULTS.md、outputs/frozen_region_ae_811_v1/，均完成核验。
- 冻结K50纯积分通过；Qwen2/Qwen3来源重构总闭合尾部最大9.51%/11.62%，本轮未改提取。原B2 K4纯积分7.46%>1%未修，本轮没有使用B2 K4。
- native36头 docs/NATIVE_BASELINES_811_RESULTS.md，统一baseline72头 docs/SVAR_METATOKEN_811_RESULTS.md，旧82区域AE288头 docs/ALL_ATTENTION_AE_STRENGTH_20260913.md，均已完成。旧82报告含ensemble，不能混比seed均值。
- ENDAC语义注意力Top16/32、全视觉不取Top-K及原始S_E替代log1p(S_E)均完成；原始S继续train-only标准化、同cohort补S/logS/AE/AE+S/AE+logS参考，四模型228头核验PASS。六项拼接rawS相对logS无一致优势；旧legacy AE含更多mentions不可直接比较。报告docs/SEMANTIC_ATTENTION_RAW_S_811_RESULTS.md，单项表outputs/coco4000_512_endac_semantic_attention_raw_s_811/single_feature_summary.md。
