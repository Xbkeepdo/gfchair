# Semantic-Attention Top-K 幻觉检测结果

## 结论

`all_six` 拼接在四模型 validation 上均为最佳，说明 attention、物体语义概率和当前生成状态相似度确实有互补信息。但是将三者直接乘成一个标量的 `semantic_attention_positive_cosine` 单独表现较弱；保留六个逐层通道让分类器自行融合更有效。

## 协议

- 固定 ENDAC 3200/400/400 图片划分，seeds 43/44/45。
- `raw/norm × top16/top32`，每个变体评估六个单项及 `all_six`，共 28 组/模型。
- 训练集逐列 z-score；单隐藏层 128/ReLU/dropout 0.3、无 BN、Adam lr=0.001/wd=1e-5/batch=128，最多 150 epochs，最低 validation BCE loss checkpoint。
- 所有 336 次拟合完成 validation 后，才按三 seed validation AUROC、HALL-AUPR 冻结选择并访问 test。

## Validation 选出的逐模型结果

| 模型 | 特征组 | Test AUROC (%) | Test HALL-AUPR (%) |
|---|---|---:|---:|
| Qwen2.5-VL-7B | norm-top32 / all-six | 82.54 ± 0.58 | 45.92 ± 2.94 |
| LLaVA-1.5-7B | norm-top32 / all-six | 91.79 ± 0.13 | 74.85 ± 0.07 |
| Qwen3-VL-8B | raw-top32 / all-six | 88.60 ± 0.04 | 72.05 ± 0.38 |
| InternVL-2.5-8B | norm-top32 / all-six | 89.79 ± 0.10 | 64.84 ± 0.22 |
| 四模型宏平均 | 逐模型 validation champion | 88.18 | 64.42 |

跨模型统一 validation champion 是 `norm_top32/all_six`，test 宏平均为 88.18/63.48。Qwen3 单独选择 raw-top32 后 AUROC 基本不变，HALL-AUPR 从 68.29 提高到 72.05。

## 与原生 baseline 比较

| 模型 | 新特征 vs SVAR AUROC / AP (pp) | 新特征 vs 较优 MetaToken AUROC / AP (pp) |
|---|---:|---:|
| Qwen2.5-VL-7B | -3.65 / +2.26 | -3.68 / -4.93 |
| LLaVA-1.5-7B | +1.11 / +2.48 | +1.59 / +2.52 |
| Qwen3-VL-8B | -0.36 / +3.80 | +0.31 / +3.17 |
| InternVL-2.5-8B | +1.33 / -1.62 | +2.65 / +1.54 |
| 宏平均 | -0.39 / +1.73 | +0.21 / +0.58 |

这些 baseline 没有获得与新特征完全相同的标准化、网络和优化预算，因此只是实验系统层面的对照，不能归因为纯特征优势。

## 消融

| 模型 | validation 选出的最佳单项 | 单项 Test AUROC / AP | all-six 增量 (pp) |
|---|---|---:|---:|
| Qwen2.5-VL-7B | norm-top32 / cosine-only | 76.25 / 33.47 | +6.29 / +12.46 |
| LLaVA-1.5-7B | norm-top32 / attention-only | 86.43 / 64.27 | +5.37 / +10.58 |
| Qwen3-VL-8B | raw-top32 / cosine×attention | 86.23 / 62.20 | +2.36 / +9.85 |
| InternVL-2.5-8B | norm-top32 / attention-only | 86.71 / 62.18 | +3.08 / +2.66 |

- 跨模型固定 `norm_top32`：all-six 为 88.18/63.48，最强单项 attention-only 为 83.96/54.99，增量 +4.22/+8.49 pp。
- norm-top32 all-six 相对 raw-top32 all-six 提高 +1.57/+2.45 pp；norm-top32 相对 norm-top16 提高 +0.98/+2.51 pp。
- 三者乘积 `semantic_attention_positive_cosine` 的最强跨模型变体只有 81.57/49.72，低于 attention-only 的 83.96/54.99。因此“同时一致”的想法有信号，但固定相乘会压缩信息，不适合作为唯一检测量。

## 产物与核验

- 入口：`scripts/train_semantic_attention_topk_811.py`
- 详细 28 组结果：`outputs/coco4000_512_endac_semantic_attention_topk_detection_811/summary.md`
- 逐 seed 结果、checkpoint 和预测：`outputs/coco4000_512_endac_semantic_attention_topk_detection_811/<model>/`
- 336/336 个头的指标复算和预测文件对齐误差为 0；12 个 champion checkpoint 在 CPU 重载后三划分概率最大误差 `7.90e-7`。

注：该 400 图 test 在项目前序实验中已多次查看，本轮内部选择虽在 test 前冻结，整体仍应视为探索性结果，不是全新独立外部测试。

## 后续：拼接 Visual-only `log1p(S_E)`

对上述 28 个 semantic-attention 组分别拼接 Visual-only 条件路径的逐层 `log1p(S_E)`，并增加 logS-only 对照。其余 split、seeds、标准化和分类器完全不变；348 次 validation 拟合全部完成后才冻结选择和访问 test。

### 六个特征分别拼接 logS

每个模型×单特征只用 validation 在 raw/norm×top16/32 中选择变体，表中为对应 test 的四模型宏平均。

| 单特征 + logS | AUROC / AP (%) | 相对同特征同变体 standalone (pp) | 相对 logS-only (pp) |
|---|---:|---:|---:|
| semantic-only + logS | 88.90 / 65.82 | +11.36 / +17.79 | +2.43 / +6.52 |
| attention-only + logS | 87.75 / 63.90 | +3.59 / +8.28 | +1.28 / +4.61 |
| cosine-only + logS | 88.44 / 64.42 | +6.04 / +11.21 | +1.97 / +5.13 |
| semantic×attention + logS | 88.39 / 63.61 | +7.08 / +14.90 | +1.93 / +4.31 |
| cosine×attention + logS | 87.83 / 63.80 | +4.04 / +8.81 | +1.37 / +4.50 |
| semantic×attention×positive-cosine + logS | 88.21 / 63.41 | +6.83 / +13.90 | +1.74 / +4.11 |

单特征中 `semantic-only + logS` 宏平均最好，其次是 `cosine-only + logS`。六项相对 logS-only 都有正增量，说明并非只是 logS 在工作。逐模型 24 行结果见 `outputs/coco4000_512_endac_semantic_attention_logs_fusion_811/single_feature_summary.md`。

### All-six 附加对照

| 模型 | Validation champion | Fusion AUROC (%) | Fusion HALL-AUPR (%) | 相对 standalone (pp) |
|---|---|---:|---:|---:|
| Qwen2.5-VL-7B | norm-top32 / all-six + logS | 84.31 ± 0.44 | 53.29 ± 1.24 | +1.77 / +7.37 |
| LLaVA-1.5-7B | norm-top32 / all-six + logS | 92.87 ± 0.15 | 78.22 ± 0.19 | +1.08 / +3.38 |
| Qwen3-VL-8B | raw-top32 / all-six + logS | 90.92 ± 0.31 | 73.64 ± 0.73 | +2.32 / +1.59 |
| InternVL-2.5-8B | norm-top32 / all-six + logS | 90.17 ± 0.21 | 67.08 ± 0.37 | +0.38 / +2.23 |
| 宏平均 | 逐模型 champion | 89.57 | 68.06 | +1.39 / +3.64 |

- 统一使用 `norm_top32/all_six + logS` 的宏平均为 89.68/67.84，相对不加 logS 的同组提高 +1.50/+4.37 pp。
- logS-only 宏平均为 86.46/59.30；逐模型 champion fusion 再高 +3.10/+8.76 pp，因此收益不是只由 logS 单独贡献。
- 四模型×28组共 112 个同配对中，加 logS 后 AUROC 和 HALL-AUPR 全部同时上升。
- fusion champion 相对 native SVAR 宏平均提高 +1.00 AUROC / +5.37 HALL-AUPR pp；Qwen2.5 AUROC 仍低 1.87 pp，但 AP 提高 9.63 pp。
- 三者乘积 + logS 的最强统一变体 `norm_top32` 为 88.22/63.27，仍低于 all-six + logS 的 89.68/67.84；保留分量再由 MLP 融合依然更好。

融合入口为 `scripts/train_semantic_attention_logs_fusion_811.py`，结果位于 `outputs/coco4000_512_endac_semantic_attention_logs_fusion_811/`。348/348 个头的指标和预测文件复算误差为 0，champion checkpoint CPU 重载概率最大误差 `1.85e-6`。
