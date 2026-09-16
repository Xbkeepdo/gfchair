# Shikra / MiniGPT-4：全视觉 attention×目标词概率求和

复用两模型 ENDAC exact mention 缓存中的 `[L,N_V]` head-mean attention 和目标物体首 token 的 raw/norm 词表 softmax 概率，不再运行 VLM。每层直接计算 `sum_{v∈全部视觉token} a[l,v]p[l,v,y]`，不取 Top-K、不作视觉位置条件归一化；与同层 `log1p(S_E)` 拼接。两模型分别按固定 3200/400/400 图片划分、43/44/45 三种子训练单隐藏层无 BN MLP，train-only 标准化；每个 raw/norm 变体 24 次 Optuna TPE trial，验证集 top3 在另外两种子复跑后冻结参数/checkpoint。

## 测试结果

单元为三种子测试 AUROC / HALL-AUPR（%）。完整均值±标准差、所选参数、每种子预测见 [结果汇总](../outputs/coco4000_512_endac_prefix_full_product_optuna_811/summary.md)。AE 与旧 Top32 为同一 ENDAC-811 cohort 的已有对照；旧对照也跑 24 个训练候选，但使用冻结候选网格而非本轮 Optuna，因此差值同时包含选参流程差异。

| 模型 | raw 全视觉乘积 + logS | norm 全视觉乘积 + logS | 旧 AE+logS | 旧 raw Top32+logS | 旧 norm Top32+logS |
|---|---:|---:|---:|---:|---:|
| MiniGPT-4-7B | 93.20 / 76.24 | **93.56 / 77.26** | 93.53 / 77.95 | 93.20 / 76.07 | 93.47 / 76.84 |
| Shikra-7B | **87.90 / 68.45** | 87.49 / 68.05 | 86.22 / 64.43 | 87.90 / 69.00 | 87.34 / 67.53 |

MiniGPT-4 的 norm 全视觉乘积与 AE 的 AUROC 基本相同（+0.03 pp），但 HALL-AUPR 低 0.69 pp；相对旧 norm Top32 则高 0.09/0.42 pp。Shikra 的 raw 全视觉乘积比 AE 高 1.68/4.02 pp，但与旧 raw Top32 的 AUROC 相同、HALL-AUPR 低 0.55 pp。总体看，全视觉求和没有形成对已有 Top32 的清晰优势；这些是已经多次查看的测试集上的探索性点估计，不是独立泛化结论。

MiniGPT-4 9,140 条、Shikra 13,067 条 mention 顺序与旧 split 完全对齐；两模型 raw/norm 的乘积值均无零行，全部 32 层的训练标准差也高于既有 scaler 的 `1e-12` 下限。12/12 个最终种子头完成 CPU checkpoint 重载与指标复算，最大概率差 `1.79e-7`、指标差 0；见输出目录 `validation.json`。
