# Union-topK JS + mass×cosine 四模型对照

协议：gfchair InsLen official target；严格图片级 8:2；seeds 43/44/45；相同三层 MLP 与 train-Real-F1 阈值。Union-topK JS 使用 source/target 各自 Top-32 的并集（并集最多 64 个视觉 token）；mass×cosine 使用既有 target Top-32。表中均为 test AUROC 的三 seed 总体均值 ± 总体标准差。

## 两个单特征与拼接

| 模型 | Gate | Union-topK JS | mass×cosine | JS + mass×cosine | 拼接相对较好单特征 |
|---|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | raw-logit | 0.8660±0.0017 | 0.8652±0.0032 | 0.8883±0.0013 | +0.0223 |
| LLaVA-1.5-7B | softmax-prob | 0.8650±0.0016 | 0.8749±0.0010 | **0.8891±0.0017** | +0.0141 |
| InternVL2.5-8B | raw-logit | 0.7838±0.0002 | 0.8387±0.0029 | **0.8524±0.0051** | +0.0136 |
| InternVL2.5-8B | softmax-prob | 0.7964±0.0014 | 0.8299±0.0025 | 0.8473±0.0011 | +0.0175 |
| Qwen2.5-VL-7B | raw-logit | 0.7949±0.0081 | 0.7844±0.0019 | **0.8299±0.0075** | +0.0349 |
| Qwen2.5-VL-7B | softmax-prob | 0.7915±0.0076 | 0.7851±0.0049 | 0.8259±0.0086 | +0.0344 |
| Qwen3-VL-8B | raw-logit | 0.8251±0.0024 | 0.8270±0.0058 | **0.8649±0.0024** | +0.0380 |
| Qwen3-VL-8B | softmax-prob | 0.8238±0.0016 | 0.8418±0.0012 | 0.8633±0.0017 | +0.0214 |

拼接在 4 个模型、两种 gate 的 8/8 个分支上都提高 AUROC；相对较好的对应单特征，增益范围为 +0.0136 到 +0.0380。

## 每个模型最好的拼接分支

| 模型 | 最佳 gate | Test AUROC | Real-F1 | Hall-F1 | 相对该模型当前最佳 risk+mass AUROC |
|---|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | softmax-prob | **0.8891±0.0017** | 0.8938±0.0004 | 0.6378±0.0031 | +0.0000（基本持平） |
| InternVL2.5-8B | raw-logit | **0.8524±0.0051** | 0.9157±0.0013 | 0.4706±0.0231 | +0.0009 |
| Qwen2.5-VL-7B | raw-logit | **0.8299±0.0075** | 0.9341±0.0029 | 0.3648±0.0079 | -0.0069 |
| Qwen3-VL-8B | raw-logit | **0.8649±0.0024** | 0.9153±0.0019 | 0.5486±0.0048 | +0.0016 |

这里的“当前最佳 risk+mass”是在该模型现有 sqrt/cosine cost、hmid/hpre source 的全部 risk+mass 分支中取最高三 seed AUROC。结论是：JS+mass 明显优于任一单特征，但与当前最佳 risk+mass 相比，LLaVA 基本持平、InternVL/Qwen3 小幅更好、Qwen2.5 略低。

## 样本与维度

| 模型 | 全部对象样本 | Train/Test | 单特征维度 | 拼接维度 |
|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | 15,463 | 12,317 / 3,146 | 32 | 64 |
| InternVL2.5-8B | 11,759 | 9,378 / 2,381 | 32 | 64 |
| Qwen2.5-VL-7B | 8,717 | 6,985 / 1,732 | 28 | 56 |
| Qwen3-VL-8B | 14,873 | 11,846 / 3,027 | 36 | 72 |

完整 fixed-0.5 与 train-F1 双阈值结果见同目录的 `union_topk_js_mass_cosine_4model_3seed_summary.md/.csv/.json`，各模型 seed 结果保存在其 `results/union_topk_js_mass_cosine_seed{43,44,45}/`。
