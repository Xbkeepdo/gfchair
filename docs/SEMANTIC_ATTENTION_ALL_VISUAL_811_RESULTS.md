# 全视觉 token 区域的 Semantic-Attention 811 检测

## 协议

直接复用已保存的逐层 `[L,N_V]` attention、目标词概率（raw/norm）与 cosine 矩阵，不再按物体概率取 Top16/32，也不重新运行视觉模型。每层在该图片的**全部视觉 token** 上计算原六项：`mean(p)`、`sum(a)`、`mean(cos)`、`sum(p·a)`、`mean(cos)·sum(a)`、`sum(p·a·max(cos,0))`。原公式中的两项均值仍为均值，其余为全区域求和。分别训练六单项、六项拼接，并分别拼接逐层 Visual-only `log1p(S_E)`。

固定原 ENDAC 3200/400/400 图片划分、种子 43/44/45；特征表只包含有目标物体 mention 的图片。用 train-only z-score、单隐藏层 128/ReLU/dropout 0.3、无 BN、Adam，按最低 validation loss 选 checkpoint。先完成全部 336 次验证训练，再按三种子 validation AUROC、HALL-AUPR 冻结方案，最后评估 test。原项目的 test 已在其他实验中看过，所以结果仅为探索性对照。

## 六项拼接：不取 Top vs 原 Top-K

每个模式均只用 validation 在其候选变体中选出表中配置；表中为三种子 test 均值，单位百分比。原 Top-K 在四模型均选 Top32，Qwen3 选 raw，其余选 norm。`Δ` 是不取 Top 减原 Top-K，单位百分点。

| 模型 | 全区域单独 AUROC/AP | Top32 单独 AUROC/AP | Δ | 全区域 + logS AUROC/AP | Top32 + logS AUROC/AP | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 85.52/53.25 | 82.54/45.92 | +2.98/+7.32 | 85.99/52.51 | 84.31/53.29 | +1.67/−0.79 |
| LLaVA-1.5-7B | 92.76/78.25 | 91.79/74.85 | +0.97/+3.40 | 93.24/79.47 | 92.87/78.22 | +0.37/+1.25 |
| Qwen3-VL-8B | 91.31/76.39 | 88.60/72.05 | +2.71/+4.34 | 92.09/76.67 | 90.92/73.64 | +1.17/+3.03 |
| InternVL-2.5-8B | 89.99/66.72 | 89.79/64.84 | +0.20/+1.88 | 89.96/66.55 | 90.17/67.08 | −0.21/−0.53 |

全区域六项**单独**使用时，四模型的 AUROC 和 AP 都高于 validation 选出的 Top32 六项。拼接 `log1p(S_E)` 后并非全胜：Qwen2.5 AP 低 0.79 pp，InternVL 双指标略低。对全区域本身，logS 带来 LLaVA、Qwen3 双指标提升，但 Qwen2.5 与 InternVL 的 AP 略降。不能把 test 表中未被 validation 选中的变体追认为赢家。

全区域单独和 +logS 的 validation 赢家 raw/norm 分别均为 Qwen3 raw、其余 norm。全区域 +logS 的最终四模型 test AUROC/AP（均值±总体标准差）为：Qwen2.5 `85.99±0.14 / 52.51±1.11`，LLaVA `93.24±0.13 / 79.47±0.36`，Qwen3 `92.09±0.11 / 76.67±0.31`，InternVL `89.96±0.06 / 66.55±1.04`。

## 单项特征与完整数据

每个模型六单项 × raw/norm × 不拼/拼 logS 的全部 112 个 test 单元格见 [`summary.md`](../outputs/coco4000_512_endac_semantic_attention_all_visual_811/summary.md)；三种子明细和标准差见各模型的 `test_seed_metrics.csv` 与 `test_summary.csv`。在六单项中仅按 validation 选出的最佳项如下，未用 test 搜索：

| 模型 | 单项独立：特征及 test AUROC/AP | 单项 + logS：特征及 test AUROC/AP |
|---|---|---|
| Qwen2.5 | raw attention-only，81.59/38.11 | raw semantic-only，83.94/45.86 |
| LLaVA | raw semantic×attention，90.14/74.15 | norm semantic×attention，92.56/77.71 |
| Qwen3 | raw cosine×attention，87.90/68.93 | raw semantic-only，91.38/74.51 |
| InternVL | raw semantic×attention，86.26/59.88 | raw semantic-only，88.57/65.32 |

attention-only、cosine-only 和 cosine×attention 不使用目标词概率，所以 raw/norm 的同组结果必然相同；不是两次独立语义计算的收敛巧合。

## 产物与核验

- 入口：`scripts/train_semantic_attention_all_visual_811.py`；输出：`outputs/coco4000_512_endac_semantic_attention_all_visual_811/`。
- 每模型保存紧凑特征 `features.pt`、validation/test 三种子结果、每个训练头的 checkpoint 与全部样本预测。原 `[L,N_V]` 矩阵仍在 `outputs/coco4000_512_endac_semantic_attention_topk/<model>/semantic_attention.pkl`。
- 336/336 个头复算指标与预测文件误差均为零；四个 validation champion 在 CPU 重载后三划分概率最大误差 `8.35e-7`。
- 特征表中的 mention-bearing 图片 train/val/test 分别为：Qwen2.5 2946/368/365，LLaVA 3171/399/398，Qwen3 3158/393/397，InternVL 3150/399/390；三组互不重叠，属于原 3200/400/400 图片划分的子集。
