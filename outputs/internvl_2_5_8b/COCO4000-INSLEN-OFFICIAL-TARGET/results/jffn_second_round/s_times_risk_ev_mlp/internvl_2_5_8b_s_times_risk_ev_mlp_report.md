# internvl_2_5_8b: S、S×risk 曲线与 S×risk+EV 三层 MLP

S 为逐层 Jacobian sensitivity R/I；S×risk 为与同层 old-hpre raw-logit Gaussian / sqrt-matched-state Union Top-K OT risk 的逐元素乘积。

## 曲线摘要

| Feature | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 13 | 19 | 13 | -0.053433 | -0.4780 |
| S×risk | 24 | 8 | 24 | +0.011729 | +0.5121 |

阴影为 mention-level 均值的近似 95% CI；用于描述曲线，不当作图片级独立显著性检验。

## 三种子 MLP

| Seed | AUROC | Real F1 | Hall F1 | Hall AUPR | Best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 0.857727 | 0.913466 | 0.450617 | 0.517359 | 79 |
| 44 | 0.846628 | 0.907348 | 0.455988 | 0.492778 | 100 |
| 45 | 0.849477 | 0.916850 | 0.484115 | 0.523862 | 99 |

## 公平对照

| Feature | Dimensions | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| old risk+EV | 64 | 0.848825 ± 0.004339 | 0.502827 ± 0.002522 | 0.856685 | 0.522819 |
| old risk+EV+S | 96 | 0.866745 ± 0.002137 | 0.546172 ± 0.012493 | 0.874746 | 0.565589 |
| S×risk+EV | 64 | 0.851277 ± 0.004707 | 0.511333 ± 0.013386 | 0.859675 | 0.527881 |

S×risk+EV − old risk+EV 的 seed-ensemble AUROC 差值 +0.002989，图片级 95% CI [-0.008205,+0.014293]；Hall-AUPR 差值 +0.005062，95% CI [-0.027290,+0.038788]。

S×risk+EV − old risk+EV+S 的 seed-ensemble AUROC 差值 -0.015071，图片级 95% CI [-0.028068,-0.002237]；Hall-AUPR 差值 -0.037707，95% CI [-0.074887,-0.000772]。
