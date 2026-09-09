# old risk + EV + S：两模型三层 MLP 汇总

输入为逐层 old-hpre risk、mass×cosine EV 和 Jacobian S，合计 96 维；三个隐藏层为 [128,64,32]。

| Model | Feature | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | --- | ---: | ---: | ---: | ---: |
| llava_1_5_7b | old risk+EV | 0.888141 ± 0.001422 | 0.656175 ± 0.005976 | 0.893401 | 0.668119 |
| llava_1_5_7b | old risk+EV+S | 0.893087 ± 0.002437 | 0.675664 ± 0.004283 | 0.902881 | 0.694670 |
| internvl_2_5_8b | old risk+EV | 0.848825 ± 0.004339 | 0.502827 ± 0.002522 | 0.856685 | 0.522819 |
| internvl_2_5_8b | old risk+EV+S | 0.866745 ± 0.002137 | 0.546172 ± 0.012493 | 0.874746 | 0.565589 |

## Paired image bootstrap

- llava_1_5_7b：AUROC Δ=+0.009480, 95% CI [+0.001897,+0.017336]；Hall AUPR Δ=+0.026551, 95% CI [-0.000633,+0.053685]。
- internvl_2_5_8b：AUROC Δ=+0.018060, 95% CI [+0.004740,+0.031558]；Hall AUPR Δ=+0.042769, 95% CI [+0.009725,+0.076242]。
