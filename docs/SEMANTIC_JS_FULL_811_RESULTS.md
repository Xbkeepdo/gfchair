# ENDAC-811：视觉注意力与物体概率的 JS 散度

本轮复用四模型已保存的 `[L,N_V]` head-mean attention、目标物体首个 token 的 raw/norm 词表 softmax 概率及全视觉区域 `sum_v a_v p_v(y)`，没有重新运行 VLM。入口为 `scripts/optuna_semantic_js_full_811.py`，全量六项逐模型结果、超参数与预测位于 `outputs/coco4000_512_endac_semantic_js_full_optuna_811/`。

## 特征口径

每层令 `A32 = Top32_v(a_v)`，`P32 = Top32_v(p_v(y))`，比较两种区域：`A32` 和 `A32 ∪ P32`。在每个区域 `R` 内，**两个分布都限制到 R**，再分别按区域内总和归一化，计算 base-2 `JS(ã_R || p̃_R)`。因此 attention Top32 版本不是拿 32 个 attention 值同全视觉概率分布比较。第三种特征为未作区域归一化的 `sum_{v∈V} a_v p_v(y)`。三种特征各试 raw 和 final-Norm 两种反嵌入概率，形成六项；每层标量与相同层的 `log1p(S_E)` 拼接输入分类器。

训练沿用 3200/400/400 图片划分、43/44/45 三种子、train-only 标准化、无 BN 的单隐藏层 PyTorch MLP。每项独立运行 24 次 Optuna trial，在验证集筛出前三项并复跑 44/45，按三种子验证 AUROC、HALL-AUPR 冻结参数和 checkpoint；测试集只用于本轮六项特征的探索性比较。它不是独立测试集上的无偏泛化估计。

## 测试结果

三种子均值，单位为百分数；完整均值±标准差及验证指标见 [summary.md](../outputs/coco4000_512_endac_semantic_js_full_optuna_811/summary.md)。下表的 AE 和旧 Top32 采用相同 ENDAC-811 拆分及既有 Optuna 对照。

| 模型 | 本轮最佳 JS（AUROC / HALL-AUPR） | 全视觉乘积 AUROC 最佳 | AE+logS | 旧 Top32 raw | 旧 Top32 norm |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | norm 联合32 **85.74 / 49.73** | norm 84.05 / 42.84 | 87.18 / 54.77 | 82.74 / 41.02 | 82.38 / 43.58 |
| LLaVA-1.5-7B | norm 联合32 **91.61 / 75.95** | norm 92.59 / 77.42 | 90.39 / 72.63 | 91.68 / 75.76 | 92.41 / 76.99 |
| Qwen3-VL-8B | norm attention32 **90.47 / 72.46** | norm 90.00 / 71.98 | 89.85 / 71.75 | 89.51 / 71.99 | 89.82 / 71.21 |
| InternVL-2.5-8B | raw 联合32 **87.32 / 63.02** | raw 88.98 / 64.23 | 87.81 / 62.36 | 89.61 / 66.88 | 88.23 / 62.41 |

Qwen2.5、Qwen3 的 JS 相对各自旧 Top32 有收益，但 Qwen2.5 仍低于 AE；Qwen3 的 JS 略高于 AE。LLaVA 最好的是全视觉乘积（比旧 norm Top32 仅高 0.18 AUROC、0.43 HALL-AUPR 个百分点），InternVL 则仍以旧 raw Top32 较强。上述均为点估计，未做配对不确定性检验。

## 数值与复核

Qwen2.5 raw 概率在 attention Top32 中有 7,543/221,872 个 mention×层区域（3.40%）的概率质量全为零；此时条件分布数学上未定义，本实验以区域内均匀分布回退。raw 联合区域另有 37 行；norm 两组及其他模型无此问题。Qwen2.5 的 attention 本身没有零质量行。raw JS 结果需连同这一限制解释，不能把回退值当成真实语义分布。

四模型共 24 个特征配置、72 个最终种子头，测试预测、checkpoint 重载、标准化统计和指标复算通过；`validation.json` 记录最大 CPU 重载概率误差 `3.58e-7`，指标差为零。
