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
| old_risk_ev | 0.888141 ± 0.001422 | 0.644332 | 0.656175 | 0.883835 | 0.652571 |
| old_hpre_risk_sqrt_plus_ev_plus_jacobian_S | 0.893087 ± 0.002437 | 0.658950 | 0.675664 | 0.894697 | 0.696823 |
| risk+EV+Union-TopK S | 0.891793 ± 0.001312 | 0.640199 | 0.672379 | 0.901959 | 0.734553 |
| risk+EV+Union aggregate S | 0.839762 ± 0.007118 | 0.612775 | 0.576353 | 0.844784 | 0.576682 |

## 图片级 paired bootstrap

- vs risk+EV：ensemble AUROC Δ=-0.039051，95% CI [-0.067752,-0.012018]；Hall-AUPR Δ=-0.075888，95% CI [-0.154142,+0.011723]。
- vs all-token aggregate S：ensemble AUROC Δ=-0.049912，95% CI [-0.079141,-0.023912]；Hall-AUPR Δ=-0.120140，95% CI [-0.208358,-0.023983]。
- vs Union tokenwise S：ensemble AUROC Δ=-0.057175，95% CI [-0.083747,-0.032438]；Hall-AUPR Δ=-0.157871，95% CI [-0.220946,-0.078436]。

Union 大小 min/mean/max=32/41.54/64；positions/mentions=1861/1917。
