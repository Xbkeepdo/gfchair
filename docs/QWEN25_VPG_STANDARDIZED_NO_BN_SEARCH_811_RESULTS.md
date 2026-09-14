# Qwen2.5 True-RMS VP+G：StandardScaler、无BN搜参

## 实验设置

- 特征：True-RMS All-attention K32的VP+G，即`[AE_VP,log1p(S_V+S_P),AE_G,log1p(S_G)]`，112维。
- 固定811图片划分：3200 train / 400 validation / 400 test，split seed `20260912`，全部mentions。
- 每列独立StandardScaler，只拟合训练mentions；`BatchNorm=False`；BCE（REAL=1）；训练满150 epochs，从完整轨迹保留最低验证loss checkpoint。
- 将此前24个固定候选模板强制改为上述设置后去重，得到19个候选；变化参数包括width、dropout、激活、lr、wd和batch，并非笛卡尔积。
- 19候选先跑seed43，按验证AUROC、HALL-AUPR、较小索引排序；前三补seeds44/45，再按三seed验证均值冻结赢家，之后读取测试集。总训练25个头。
- 测试集此前已在其他实验访问，本轮属于探索性追加；未按本轮测试结果继续扩搜索。

## 验证选择

| 排名候选 | width | dropout | 激活 | lr | wd | batch | 最佳epoch 43/44/45 | 三seed验证 AUROC / HALL-AUPR |
|---:|---:|---:|---|---:|---:|---:|---|---:|
| 7（赢家） | 128 | 0.3 | ReLU | 0.001 | 0.001 | 512 | 47/64/43 | **85.59 / 46.52** |
| 4 | 128 | 0.3 | ReLU | 0.001 | 1e-5 | 256 | 28/26/25 | 85.40 / 45.66 |
| 3 | 248 | 0 | ReLU | 0.001 | 0 | 32 | 9/7/5 | 85.33 / **46.77** |

候选3在seed43单次验证AUROC最高，但补齐三个种子后候选7的平均AUROC最高，因此按预定规则选择候选7。

## 测试结果及对照

下列测试指标均为seeds43/44/45独立计算后的均值 ± 总体标准差，单位%。V与未搜参VP+G使用上一轮同头对照；SVAR使用相同固定811划分和seeds的原生固定分类器。

| 方法 | 测试 AUROC | 测试 HALL-AUPR |
|---|---:|---:|
| Qwen2.5 True-RMS V，当前标准化无BN头 | **87.96 ± 0.16** | 42.23 ± 0.81 |
| Qwen2.5 True-RMS VP+G，搜参前固定V头 | 86.68 ± 0.33 | 46.20 ± 0.60 |
| Qwen2.5 True-RMS VP+G，验证搜参赢家 | 87.20 ± 0.04 | **47.76 ± 0.79** |
| 原生 SVAR | 87.34 ± 0.22 | 47.06 ± 0.66 |

搜参相对固定V头训练VP+G提高`+0.51 AUROC/+1.56 HALL-AUPR pp`。赢家相对V为`-0.76/+5.54 pp`，相对SVAR为`-0.14/+0.71 pp`：AP更高，但AUROC略低。

验证集上，当前V为`88.46/50.90`，搜参VP+G为`85.59/46.52`。因此按预定验证AUROC优先规则，Qwen2.5仍应选择V；不能依据VP+G的测试AP优势反向替换正式赢家。

原生SVAR没有本轮相同HPO预算，因此与SVAR的差值比较两套完整方法，不是分类器同预算的纯特征比较。

## 复核文件

- 完整协议及19候选：`outputs/qwen25_vpg_standardized_no_bn_search_811_v1/protocol.json`
- 验证选择：`outputs/qwen25_vpg_standardized_no_bn_search_811_v1/selection.json`
- 测试汇总：`outputs/qwen25_vpg_standardized_no_bn_search_811_v1/summary.json`
- 三seed测试指标：`outputs/qwen25_vpg_standardized_no_bn_search_811_v1/seed_metrics.csv`
- 25个可恢复训练头：`outputs/qwen25_vpg_standardized_no_bn_search_811_v1/candidate*/seed*/result.pt`
- 复现实验：`scripts/search_qwen25_vpg_standardized_no_bn_811.py`
