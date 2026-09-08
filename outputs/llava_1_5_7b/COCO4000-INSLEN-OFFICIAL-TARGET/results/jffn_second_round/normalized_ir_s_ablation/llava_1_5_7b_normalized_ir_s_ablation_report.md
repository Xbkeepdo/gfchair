# llava_1_5_7b: normalized I/R sum 与 aggregate S 消融

`N=I/||I||₂+R/||R||₂`：每个样本分别沿 32 个 decoder 层对 I、R 做 L2 归一化后逐层相加，不做第二次归一化。S 使用已有 aggregate `R/(I+epsilon)`。

所有分类头均为 `[128,64,32]` 三隐藏层 MLP、seeds 43/44/45、图片级 split、无特征标准化；REAL 为 AUROC 正类。未运行 bootstrap。

## 三 seed 结果

| Feature | Dim | AUROC | Hall AUPR | Hall F1 | Ensemble AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| N = L2(I) + L2(R) | 32 | 0.871648 ± 0.001620 | 0.632786 | 0.562015 | 0.874497 |
| S = R / I | 32 | 0.864967 ± 0.000251 | 0.622850 | 0.589011 | 0.869133 |
| N + S | 64 | 0.878449 ± 0.001717 | 0.637600 | 0.626564 | 0.883761 |
| Original risk + EV | 64 | 0.888141 ± 0.001422 | 0.656175 | 0.644332 | 0.893401 |
| Original risk + EV + N | 96 | 0.892321 ± 0.001848 | 0.664836 | 0.644156 | 0.899170 |
| Original risk + EV + S | 96 | 0.893087 ± 0.002437 | 0.675664 | 0.658950 | 0.902881 |
| Original risk + EV + N + S | 128 | 0.895169 ± 0.002824 | 0.678783 | 0.653397 | 0.903889 |
| P_JFFN risk + EV | 64 | 0.876717 ± 0.004153 | 0.659444 | 0.611740 | 0.883175 |
| P_JFFN risk + EV + N | 96 | 0.886031 ± 0.003664 | 0.666376 | 0.631153 | 0.892745 |
| P_JFFN risk + EV + S | 96 | 0.883950 ± 0.002771 | 0.672963 | 0.624144 | 0.894369 |
| P_JFFN risk + EV + N + S | 128 | 0.892915 ± 0.004270 | 0.680232 | 0.647254 | 0.901139 |

## 描述性均值差

- N + S − N = L2(I) + L2(R)：AUROC +0.006802；Hall AUPR +0.004814；Hall F1 +0.064549。
- N + S − S = R / I：AUROC +0.013482；Hall AUPR +0.014751；Hall F1 +0.037553。
- Original risk + EV + N − Original risk + EV：AUROC +0.004180；Hall AUPR +0.008661；Hall F1 -0.000175。
- Original risk + EV + S − Original risk + EV：AUROC +0.004946；Hall AUPR +0.019489；Hall F1 +0.014618。
- Original risk + EV + N + S − Original risk + EV：AUROC +0.007027；Hall AUPR +0.022608；Hall F1 +0.009066。
- P_JFFN risk + EV + N − P_JFFN risk + EV：AUROC +0.009314；Hall AUPR +0.006932；Hall F1 +0.019412。
- P_JFFN risk + EV + S − P_JFFN risk + EV：AUROC +0.007233；Hall AUPR +0.013519；Hall F1 +0.012404。
- P_JFFN risk + EV + N + S − P_JFFN risk + EV：AUROC +0.016197；Hall AUPR +0.020788；Hall F1 +0.035514。

## N/S 曲线摘要

- N：HALL>REAL / REAL>HALL 层数=9/23；最强 L11，Hall−Real=-0.071124，d=-0.9696，最佳方向单层 AUROC=0.7665。
- S：HALL>REAL / REAL>HALL 层数=7/25；最强 L19，Hall−Real=-0.052086，d=-0.6829，最佳方向单层 AUROC=0.7036。

## 协议核验

- train/test images：3174/797。
- train/test mentions：12317/3146。
- stored S 与 R/I 最大绝对误差：0.000e+00。
- I/R 单位范数最大误差：1.192e-07。
- 两种 risk 只改变 source P；target Q、Union Top-K、cost 与 EV 全部相同。
- 不使用跨样本均值/方差，不读取测试集统计进行归一化。
