# llava_1_5_7b: S、S×risk 曲线与 S×risk+EV 三层 MLP

S 为逐层 Jacobian sensitivity R/I；S×risk 为与同层 old-hpre raw-logit Gaussian / sqrt-matched-state Union Top-K OT risk 的逐元素乘积。

## 曲线摘要

| Feature | Hall>Real layers | Real>Hall layers | Strongest layer | Hall−Real | Cohen's d |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 7 | 25 | 19 | -0.052086 | -0.6829 |
| S×risk | 20 | 12 | 21 | +0.019098 | +0.6306 |

阴影为 mention-level 均值的近似 95% CI；用于描述曲线，不当作图片级独立显著性检验。

## 三种子 MLP

| Seed | AUROC | Real F1 | Hall F1 | Hall AUPR | Best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 0.890545 | 0.891151 | 0.645467 | 0.670735 | 100 |
| 44 | 0.891054 | 0.894415 | 0.626263 | 0.674468 | 100 |
| 45 | 0.891081 | 0.893308 | 0.648464 | 0.673593 | 100 |

## 公平对照

| Feature | Dimensions | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| old risk+EV | 64 | 0.888141 ± 0.001422 | 0.656175 ± 0.005976 | 0.893401 | 0.668119 |
| old risk+EV+S | 96 | 0.893087 ± 0.002437 | 0.675664 ± 0.004283 | 0.902881 | 0.694670 |
| S×risk+EV | 64 | 0.890893 ± 0.000247 | 0.672932 ± 0.001594 | 0.895697 | 0.684988 |

S×risk+EV − old risk+EV 的 seed-ensemble AUROC 差值 +0.002295，图片级 95% CI [-0.004802,+0.009246]；Hall-AUPR 差值 +0.016869，95% CI [-0.009806,+0.042902]。

S×risk+EV − old risk+EV+S 的 seed-ensemble AUROC 差值 -0.007185，图片级 95% CI [-0.014471,-0.000020]；Hall-AUPR 差值 -0.009682，95% CI [-0.033461,+0.013280]。
