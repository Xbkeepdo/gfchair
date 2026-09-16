# ENDAC Semantic-Attention Top-K 提取进度

最后更新：2026-09-16 14:44 CST

## 协议

- 四模型复用 `coco4000_512_endac_811` 的 exact controlled mentions、generation 和 811 split。
- 多 token mention 只使用 exact span 第一个 token ID、生成位置及该位置的 `h_prev`。
- 对每层视觉 `h_prev` 分别计算 no-Norm 与 FinalNorm 后的目标 token softmax 概率。
- 两套概率独立选 top-16/top-32；attention 为所有 head 的均值。
- 原始矩阵使用 FP32：`attention`、`object_probability_raw/norm`、`cosine_similarity`，形状均为 `[L,N_V]`。
- 每个 top-k 保存 semantic-only、attention-only、cosine-only、semantic×attention、cosine×attention、三者联合六项。
- 保存方式沿用 append-pickle：正式过程写 `semantic_attention.part0.pkl`，结束合并为 `semantic_attention.pkl`。

## 完成状态

| 模型 | 图片 | mention | 层数 | train / val / test | 最终文件 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| qwen2_5_vl_7b | 3,679 | 7,924 | 28 | 6,298 / 828 / 798 | 1.33 GB | merged + verified |
| llava_1_5_7b | 3,968 | 12,686 | 32 | 10,109 / 1,297 / 1,280 | 3.87 GB | merged + verified |
| qwen3_vl_8b | 3,948 | 11,751 | 36 | 9,384 / 1,196 / 1,171 | 1.96 GB | merged + verified |
| internvl_2_5_8b | 3,939 | 10,781 | 32 | 8,589 / 1,124 / 1,068 | 1.52 GB | merged + verified |

四架构 1 图 FP32 smoke 已通过。保存矩阵可重建 top-k 与六项聚合特征，最大复算误差约 `1.2e-7`。
最初的 FP16 smoke 因小概率下溢会破坏 top-k 复算，已移到 `fp16_smoke_discarded/`，正式任务未使用。

四模型最终文件均完成行数、sample ID 唯一性、图片数、split 数量、全量 shape/finite 检查。
从保存的原始矩阵抽样重算 top-k 六项特征，四模型最大误差均不超过 `1.2e-7`。
所有提取 worker 均正常退出，本地与远端 GPU 已释放。

输出根目录：`outputs/coco4000_512_endac_semantic_attention_topk/`

## 幻觉检测（已完成）

- 固定原 3200/400/400 图片划分和 seeds 43/44/45，test 不参与特征组选择。
- 分别评估 `raw/norm × top16/top32` 下的六个单项及 `all_six`，共 28 组/模型、84 次拟合/模型。
- 输入按 train-only 逐列 z-score；单隐藏层 128/ReLU/dropout 0.3、无 BN、Adam，最低 validation BCE loss checkpoint。
- 336/336 次拟合均完成。逐模型 validation champion 均为 `all_six`；Qwen3 选 raw-top32，其余选 norm-top32。
- 四模型 champion 的 test AUROC/HALL-AUPR 宏平均为 88.18/64.42；统一 `norm_top32/all_six` 为 88.18/63.48。
- 对 336 个头的指标和预测文件完成复算，champion checkpoint CPU 重载概率最大误差 `7.90e-7`。
- 入口：`scripts/train_semantic_attention_topk_811.py`；检测输出：`outputs/coco4000_512_endac_semantic_attention_topk_detection_811/`。
- 完整分析：`docs/SEMANTIC_ATTENTION_TOPK_811_RESULTS.md`。

## 与 Visual-only `log1p(S_E)` 融合（已完成）

- 对 28 个 semantic-attention 特征组分别拼接逐层 Visual-only 条件路径 `log1p(S_E)`，另训练 `log1p(S_E)`-only 对照。
- 保持同一 811 split、seeds 43/44/45、train-only z-score 和固定 128 单隐藏层探针。
- 共 29 头/模型、87 次拟合/模型；所有 validation 完成后冻结选择，再访问 test。
- 348/348 次拟合均完成；逐模型 champion 均为 all-six + logS，宏平均 89.57/68.06。
- 相对不加 logS 的同组逐模型 champion，四模型双指标全部提高，宏平均 +1.39/+3.64 pp。
- 348 个头的指标/预测文件复算通过，champion CPU 重载概率最大误差 `1.85e-6`。
- 入口：`scripts/train_semantic_attention_logs_fusion_811.py`；输出：`outputs/coco4000_512_endac_semantic_attention_logs_fusion_811/`。
- 六个单特征分别拼接 logS 的完整 24 行表：`outputs/coco4000_512_endac_semantic_attention_logs_fusion_811/single_feature_summary.md`。

## 全视觉 token 区域（已完成）

- 不再选 Top16/32，直接使用已保存的完整 `[L,N_V]` attention、目标词概率及 cosine 矩阵，不重新运行视觉模型。
- 六项公式沿用 Top-K 实验，仅把集合换成所有视觉 token：semantic-only 和 cosine-only 保留区域均值，其余保留区域求和；同时评估 raw/norm、单项/六项及分别拼接 `log1p(S_E)`。
- 四模型特征已对齐并保存：Qwen2.5 7,924、LLaVA 12,686、Qwen3 11,751、InternVL 10,781 条 mention；固定原 811 图片划分和 seeds 43/44/45。
- 336/336 次验证训练完成后冻结：Qwen2.5/LLaVA/InternVL 选 norm/all_six+logS，Qwen3 选 raw/all_six+logS；全部 test 与预测文件已保存，指标/预测复算 336/336 PASS，champion CPU 重载最大误差 `8.35e-7`。
- 全区域六项独立时，四模型 AUROC/AP 均超过对应 validation 选出的 Top32 六项；拼 logS 后 InternVL 双指标略低于 Top32，Qwen2.5 AP 低 0.79 pp。完整逐模型结果与协议：`docs/SEMANTIC_ATTENTION_ALL_VISUAL_811_RESULTS.md`。
- 入口：`scripts/train_semantic_attention_all_visual_811.py`；输出：`outputs/coco4000_512_endac_semantic_attention_all_visual_811/`。

## 原始 S_E（不取 log1p；已完成）

- 复用全视觉 token 六项特征与 Visual-only `[AE_V,log1p(S_E)]` 缓存，用 `expm1` 恢复原始 `S_E`；四模型恢复后数值全为有限且非负，不重新运行视觉模型。
- 比较六单项及六项拼接分别加原始 S，并增加 S-only、AE-only、AE+S、logS-only、AE+logS 同样本对照。所有输入继续仅用训练集做逐列 z-score，固定 811、种子 43/44/45 和 128/no-BN 单层 MLP；validation 完成后才冻结选择并评估 test。
- 四模型 228/228 头训练、测试与指标/预测复算均完成，champion/AE 参考头 CPU 重载最大误差 `8.94e-7`；输出 `outputs/coco4000_512_endac_semantic_attention_raw_s_811/`。
- 核对发现旧的 `legacy_visual` AE 报告有更多 mention，不能与此 ENDAC exact 结果直接相减；本轮追加同 cohort 的 AE+logS 参考头纠正该口径。
- 六项拼接+S 相对+logS：Q2 AUROC/AP −.15/−.50pp，LLaVA +.05/+.41，Q3 −.08/+.32，Intern −.10/+.11；无一致优势。完整报告：`docs/SEMANTIC_ATTENTION_RAW_S_811_RESULTS.md`，单项对照：`outputs/coco4000_512_endac_semantic_attention_raw_s_811/single_feature_summary.md`。
- 入口：`scripts/train_semantic_attention_raw_s_811.py`；输出：`outputs/coco4000_512_endac_semantic_attention_raw_s_811/`。
