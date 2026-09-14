# MiniGPT-4 / Shikra：视觉 V 单隐藏层 811

设置：3200/400/400图片，split seed20260912，全mentions；旧800图此前已查看，属于探索性比较。每个seed单独计算后报告seeds43/44/45均值±总体标准差，不是概率ensemble。
特征：`V=[AE_V,log1p(S_V)]`，32层。这里S_V来自视觉专用真实Norm路径`z-A_V→z`的local-FP32 Gauss–Legendre K4；两个模型的K4对K64 smoke及4000图闭合审计已通过。它不是后来四模型的all-attention K32，也没有冻结RMS版。
单隐藏层设置与四模型搜索完全相同：24个固定候选seed43，验证AUROC/AP选top3补44/45，再按三seed验证均值冻结配置；两模型都冻结后才测试。原生SVAR/MetaToken使用相同811样本重新训练；SVAR为248隐藏ReLU、batch32、max50、val_loss早停5，Meta为训练StandardScaler+LR/GB100。

| 模型 | 方法 | AUROC mean±std (%) | HALL-AUPR mean±std (%) |
|---|---|---:|---:|
| MiniGPT-4 | V: AE+log1p(S_V) | 90.53 ± 0.18 | 64.83 ± 1.54 |
| MiniGPT-4 | 原生SVAR | 91.42 ± 0.16 | 70.67 ± 0.44 |
| MiniGPT-4 | MetaToken LR | 89.68 ± 0.00 | 64.22 ± 0.00 |
| MiniGPT-4 | MetaToken GB | 89.56 ± 0.04 | 61.18 ± 0.07 |
| Shikra | V: AE+log1p(S_V) | 86.25 ± 0.18 | 60.81 ± 0.30 |
| Shikra | 原生SVAR | 88.03 ± 0.14 | 65.45 ± 0.42 |
| Shikra | MetaToken LR | 87.25 ± 0.00 | 64.14 ± 0.00 |
| Shikra | MetaToken GB | 87.10 ± 0.00 | 63.07 ± 0.01 |

选择仅看验证集。原生基线与本方法的调参预算不同，因此不作公平预算下的特征单独因果结论。完整配置见各模型protocol.json和selection.json，ensemble在CSV单列。
