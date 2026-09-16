# MiniGPT-4 / Shikra：AE 与语义注意力单层 MLP 搜参进度

## 协议

- 复用 ENDAC exact 811 样本与固定图片划分（3200 train / 400 val / 400 test），检测器种子 43/44/45，不重新提取 VLM 特征。
- 对照特征 `ae_logs=[AE_V,log1p(S_E)]`；四个语义特征分别为 raw/norm × Top16/32 的 `[semantic_attention,log1p(S_E)]`，其中 semantic_attention 是逐层视觉 token 概率与 attention 的乘积和。两类输入每条均为 64 维。
- 单隐藏层 Torch MLP；全部候选强制 train-only z-score、无 BatchNorm。固定 24 个候选，seed43 在 validation 上选前三，再补 seeds44/45，以三种子 validation AUROC、HALL-AUPR 选配置；两个模型选择均冻结后才评估 test。
- `vicr` 环境未安装 Optuna；本轮沿用项目已有的固定候选搜索，不改动环境。测试集此前已被其他实验查看，因此结论仅作探索性比较。
- 入口：`scripts/search_prefix_ae_semantic_single_mlp_811.py`；输出：`outputs/coco4000_512_endac_prefix_ae_semantic_single_mlp_811/`。

## 进度

- 2026-09-16 12:04 UTC：两模型特征顺序、`log1p(S_E)` 数值一致性及矩阵形状核对通过；两张本地 RTX 4090 分别开始 MiniGPT-4、Shikra 搜参。每模型预计 150 次拟合；动态进度见输出目录各自的 `progress.json`。
- 12:15 UTC：两模型各 150/150 次验证拟合完成，validation 冻结选择后才运行 test。两张本地 GPU 已释放；本轮搜索完成时无需把任务再拆到远端两卡。
- 验证选中设置的测试 AUROC/HALL-AUPR：MiniGPT-4 `norm_top32` 为 `93.47±0.06/76.84±0.29%`，其 AE 对照为 `93.53±0.01/77.95±0.20%`；Shikra `raw_top16` 为 `87.82±0.12/68.51±0.25%`，AE 对照为 `86.22±0.18/64.43±0.84%`。完整 raw/norm Top16/32 三种子表见 `outputs/coco4000_512_endac_prefix_ae_semantic_single_mlp_811/summary.md`。
- 30/30 个最终头通过 train-only scaler 重算、CPU checkpoint 重载和测试指标复算；最大概率差 `2.98e-7`，`validation.json` 为 `pass`。相对已有单层 SVAR，MiniGPT-4 `94.26/80.20%`、Shikra `88.08/69.65%`，本轮验证选中的语义单项均未同时超过两项指标。
- 相同语义特征此前固定 128/noBN/标准化 MLP 的对应测试分数：MiniGPT-4 `norm_top32` 为 `93.58/76.99%`，Shikra `raw_top16` 为 `87.95/68.57%`。本轮按验证集选出的 24 候选头在两模型上均未提高测试分数；不能因单次测试差值把候选重新改回去。
