# llava_1_5_7b: Union-TopK 区域新 S

新 S 在每层 `Top32(P_JFFN) ∪ Top32(Q_raw)` 内计算 `ΣR_j / ΣI_j`。`tokenwise all-token S` 使用相同标量聚合但不限制区域，用于单独检验 Union 区域筛选的贡献。

## 曲线摘要

| S | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| old aggregate S | 7 | 25 | 19 | -0.052086 | -0.6829 |
| tokenwise all-token S | 5 | 27 | 19 | -0.038943 | -0.7900 |
| Union-TopK S | 4 | 28 | 19 | -0.050426 | -0.8151 |

## 三种子 MLP

| Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| old risk+EV | 0.888141 ± 0.001422 | 0.644332 | 0.656175 ± 0.005976 | 0.893401 | 0.668119 |
| old risk+EV+old aggregate S | 0.893087 ± 0.002437 | 0.658950 | 0.675664 ± 0.004283 | 0.902881 | 0.694670 |
| risk+EV+tokenwise all-token S | 0.894717 ± 0.001654 | 0.649339 | 0.668844 ± 0.008230 | 0.904363 | 0.688662 |
| risk+EV+Union-TopK S | 0.891793 ± 0.001312 | 0.640199 | 0.672379 ± 0.001749 | 0.900209 | 0.691012 |

## 图片级 paired bootstrap

- Union S vs risk+EV：ensemble AUROC Δ=+0.006808，95% CI [-0.001083,+0.014666]；Hall-AUPR Δ=+0.022893，95% CI [-0.002812,+0.049412]。
- Union S vs old aggregate S：ensemble AUROC Δ=-0.002672，95% CI [-0.008183,+0.002773]；Hall-AUPR Δ=-0.003657，95% CI [-0.026136,+0.019206]。
- Union S vs all-token S：ensemble AUROC Δ=-0.004153，95% CI [-0.008058,-0.000272]；Hall-AUPR Δ=+0.002350，95% CI [-0.012417,+0.017712]。

实际 Union 大小：min/mean/max = 32/41.49/64；train/test mentions=12317/3146。
