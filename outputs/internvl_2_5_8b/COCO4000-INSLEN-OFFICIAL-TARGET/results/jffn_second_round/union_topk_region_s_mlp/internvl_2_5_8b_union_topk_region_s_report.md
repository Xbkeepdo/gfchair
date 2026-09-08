# internvl_2_5_8b: Union-TopK 区域新 S

新 S 在每层 `Top32(P_JFFN) ∪ Top32(Q_raw)` 内计算 `ΣR_j / ΣI_j`。`tokenwise all-token S` 使用相同标量聚合但不限制区域，用于单独检验 Union 区域筛选的贡献。

## 曲线摘要

| S | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| old aggregate S | 13 | 19 | 13 | -0.053433 | -0.4780 |
| tokenwise all-token S | 15 | 17 | 21 | +0.018737 | +0.5635 |
| Union-TopK S | 13 | 19 | 11 | -0.036578 | -0.5282 |

## 三种子 MLP

| Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| old risk+EV | 0.848825 ± 0.004339 | 0.466311 | 0.502827 ± 0.002522 | 0.856685 | 0.522819 |
| old risk+EV+old aggregate S | 0.866745 ± 0.002137 | 0.510817 | 0.546172 ± 0.012493 | 0.874746 | 0.565589 |
| risk+EV+tokenwise all-token S | 0.863213 ± 0.001334 | 0.503787 | 0.545624 ± 0.009251 | 0.870371 | 0.561951 |
| risk+EV+Union-TopK S | 0.866605 ± 0.001525 | 0.500021 | 0.552289 ± 0.002817 | 0.878673 | 0.567884 |

## 图片级 paired bootstrap

- Union S vs risk+EV：ensemble AUROC Δ=+0.021988，95% CI [+0.009584,+0.034726]；Hall-AUPR Δ=+0.045065，95% CI [+0.010734,+0.080086]。
- Union S vs old aggregate S：ensemble AUROC Δ=+0.003927，95% CI [-0.004651,+0.012282]；Hall-AUPR Δ=+0.002296，95% CI [-0.022889,+0.027944]。
- Union S vs all-token S：ensemble AUROC Δ=+0.008302，95% CI [+0.001906,+0.014983]；Hall-AUPR Δ=+0.005933，95% CI [-0.008744,+0.021373]。

实际 Union 大小：min/mean/max = 32/39.71/59；train/test mentions=9378/2381。
