# 区域AE 811首批结果：三层MLP

2026-09-13 14:34 UTC快照。三模型四种fusion各三seed已保存；单层/XGB仍在运行，Qwen3等待前序82实验完成。本表不是完整324头验收。

按图片保留原3200训练图，原800以seed20260912分400验证/400测试。811使用验证loss选择checkpoint、调度学习率与早停；原82使用训练loss。其余三层MLP配置保持不变。

**下面两侧都评估在同一400张测试图片、同一mentions上**。旧82列从已保存的800图概率中取对应400图，不能与此前800图总表混算差值。每格AUROC / HALL-AUPR（%，三seed指标均值）。原800图历史上已被查看，此轮为探索性分析。

| 模型 | 拼接 | 旧82训练，同400图测试 | 新811训练，同400图测试 |
|---|---|---:|---:|
| Qwen2.5 | V | 88.03 / 42.69 | 86.21 / 39.25 |
| Qwen2.5 | VP | 84.04 / 38.33 | 86.21 / 40.03 |
| Qwen2.5 | G | 81.94 / 38.32 | 82.02 / 35.22 |
| Qwen2.5 | VP+G | 87.22 / 47.89 | 87.36 / 46.83 |
| LLaVA | V | 90.50 / 69.82 | 90.03 / 69.38 |
| LLaVA | VP | 89.39 / 68.62 | 89.02 / 68.10 |
| LLaVA | G | 87.95 / 65.81 | 87.31 / 63.35 |
| LLaVA | VP+G | 89.51 / 69.73 | 90.05 / 72.06 |
| InternVL | V | 87.02 / 56.58 | 87.16 / 54.93 |
| InternVL | VP | 87.17 / 52.78 | 86.74 / 51.92 |
| InternVL | G | 84.84 / 50.01 | 84.83 / 49.71 |
| InternVL | VP+G | 86.86 / 54.83 | 87.21 / 53.30 |

V=[AE_V,log1p(SV_all)]；VP=[AE_VP,log1p(SV_all+SP_all)]；G=[AE_G,log1p(SG_all)]；VP+G拼接完整VP及G两块。AE取原值、各域独立计算，gross采用真实RMS All-attention K32，全部层拼接。

这12组中，AUROC六升六降，HALL-AUPR两升十降。当前结果不支持“改成811就会普遍提高分数”。没有新增bootstrap，不能用点差宣称显著性。

旧Visual-only特征也在相同811配置中重训：Qwen2.5 legacy_visual的AUROC/AP为85.69/37.21，其旧82模型在相同400图为87.77/42.17。此项同时说明下降不只出现在All-attention特征。

训练产物见outputs/all_attention_ae_811_v1/各模型的three_hidden目录；全部分类器完成后自动生成每模型detection.csv、seed_metrics.csv、vs82_same400.csv和summary.md。四模型全完成后汇总至docs/REGION_AE_811_RESULTS.md。
