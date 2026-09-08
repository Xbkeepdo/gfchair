# llava_1_5_7b: Union-TopK aggregate S

本实验在 `Top32(P_JFFN) ∪ Top32(Q_raw)` 内先做向量求和，再计算 `||ΣU J_f(z)a_j|| / ||ΣU a_j||`，因此保留视觉 token 间的方向抵消。

## 曲线摘要

| S | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-token aggregate S | 7 | 25 | 7 | -0.064217 | -0.7074 |
| Union tokenwise S | 4 | 28 | 19 | -0.049119 | -0.7750 |
| Union aggregate S | 7 | 25 | 7 | -0.078661 | -0.7645 |

## 三种子 MLP

| Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| risk+EV | 0.821245 ± 0.004099 | 0.610406 | 0.549276 | 0.826405 | 0.553167 |
| risk+EV+all-token aggregate S | 0.829718 ± 0.005879 | 0.603090 | 0.570312 | 0.835802 | 0.572445 |
| risk+EV+all-token tokenwise S | 0.840155 ± 0.005531 | 0.596042 | 0.615408 | 0.846154 | 0.630171 |
| risk+EV+Union tokenwise S | 0.845628 ± 0.004829 | 0.627679 | 0.603987 | 0.848829 | 0.610713 |
| risk+EV+Union aggregate S | 0.839762 ± 0.007118 | 0.612775 | 0.576353 | 0.844784 | 0.576682 |

## 图片级 paired bootstrap

- vs risk+EV：ensemble AUROC Δ=+0.018379，95% CI [-0.004855,+0.043193]；Hall-AUPR Δ=+0.023515，95% CI [-0.033403,+0.072511]。
- vs all-token aggregate S：ensemble AUROC Δ=+0.008982，95% CI [-0.006139,+0.025114]；Hall-AUPR Δ=+0.004237，95% CI [-0.032934,+0.042315]。
- vs all-token tokenwise S：ensemble AUROC Δ=-0.001370，95% CI [-0.019257,+0.019932]；Hall-AUPR Δ=-0.053489，95% CI [-0.090651,-0.008457]。
- vs Union tokenwise S：ensemble AUROC Δ=-0.004045，95% CI [-0.016718,+0.012028]；Hall-AUPR Δ=-0.034031，95% CI [-0.067071,+0.003662]。

Union 大小 min/mean/max=32/41.54/64；positions/mentions=1861/1917。
