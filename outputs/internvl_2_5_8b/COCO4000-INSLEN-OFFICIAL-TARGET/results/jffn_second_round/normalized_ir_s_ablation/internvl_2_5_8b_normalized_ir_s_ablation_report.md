# internvl_2_5_8b: normalized I/R sum 与 aggregate S 消融

`N=I/||I||₂+R/||R||₂`：每个样本分别沿 32 个 decoder 层对 I、R 做 L2 归一化后逐层相加，不做第二次归一化。S 使用已有 aggregate `R/(I+epsilon)`。

所有分类头均为 `[128,64,32]` 三隐藏层 MLP、seeds 43/44/45、图片级 split、无特征标准化；REAL 为 AUROC 正类。未运行 bootstrap。

## 三 seed 结果

| Feature | Dim | AUROC | Hall AUPR | Hall F1 | Ensemble AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| N = L2(I) + L2(R) | 32 | 0.817755 ± 0.003241 | 0.471264 | 0.356491 | 0.822576 |
| S = R / I | 32 | 0.847344 ± 0.001919 | 0.508109 | 0.332601 | 0.852084 |
| N + S | 64 | 0.864059 ± 0.003415 | 0.555407 | 0.495597 | 0.869509 |
| Original risk + EV | 64 | 0.848825 ± 0.004339 | 0.502827 | 0.466311 | 0.856685 |
| Original risk + EV + N | 96 | 0.851431 ± 0.001600 | 0.510616 | 0.495720 | 0.860115 |
| Original risk + EV + S | 96 | 0.866745 ± 0.002137 | 0.546172 | 0.510817 | 0.874746 |
| Original risk + EV + N + S | 128 | 0.861279 ± 0.005597 | 0.554411 | 0.506169 | 0.868859 |
| P_JFFN risk + EV | 64 | 0.840525 ± 0.002801 | 0.483769 | 0.478527 | 0.850637 |
| P_JFFN risk + EV + N | 96 | 0.846111 ± 0.002536 | 0.502062 | 0.477121 | 0.855354 |
| P_JFFN risk + EV + S | 96 | 0.855042 ± 0.005471 | 0.524322 | 0.485575 | 0.864186 |
| P_JFFN risk + EV + N + S | 128 | 0.864111 ± 0.001063 | 0.553556 | 0.509602 | 0.872823 |

## 描述性均值差

- N + S − N = L2(I) + L2(R)：AUROC +0.046304；Hall AUPR +0.084143；Hall F1 +0.139106。
- N + S − S = R / I：AUROC +0.016715；Hall AUPR +0.047298；Hall F1 +0.162996。
- Original risk + EV + N − Original risk + EV：AUROC +0.002607；Hall AUPR +0.007789；Hall F1 +0.029409。
- Original risk + EV + S − Original risk + EV：AUROC +0.017920；Hall AUPR +0.043345；Hall F1 +0.044506。
- Original risk + EV + N + S − Original risk + EV：AUROC +0.012454；Hall AUPR +0.051584；Hall F1 +0.039858。
- P_JFFN risk + EV + N − P_JFFN risk + EV：AUROC +0.005586；Hall AUPR +0.018294；Hall F1 -0.001405。
- P_JFFN risk + EV + S − P_JFFN risk + EV：AUROC +0.014516；Hall AUPR +0.040554；Hall F1 +0.007048。
- P_JFFN risk + EV + N + S − P_JFFN risk + EV：AUROC +0.023585；Hall AUPR +0.069788；Hall F1 +0.031075。

## N/S 曲线摘要

- N：HALL>REAL / REAL>HALL 层数=10/22；最强 L26，Hall−Real=-0.051170，d=-0.3892，最佳方向单层 AUROC=0.6105。
- S：HALL>REAL / REAL>HALL 层数=13/19；最强 L13，Hall−Real=-0.053433，d=-0.4780，最佳方向单层 AUROC=0.6434。

## 协议核验

- train/test images：3142/785。
- train/test mentions：9378/2381。
- stored S 与 R/I 最大绝对误差：0.000e+00。
- I/R 单位范数最大误差：1.192e-07。
- 两种 risk 只改变 source P；target Q、Union Top-K、cost 与 EV 全部相同。
- 不使用跨样本均值/方差，不读取测试集统计进行归一化。
