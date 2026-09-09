# S、S×risk 曲线与 S×risk+EV：两模型汇总

S×risk 是 32 层逐元素乘积；与 32 层 EV 拼接后为 64 维三隐藏层 MLP。

| Model | Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| llava_1_5_7b | old risk+EV | 0.888141 ± 0.001422 | 0.644332 | 0.656175 | 0.893401 |
| llava_1_5_7b | old risk+EV+S | 0.893087 ± 0.002437 | 0.658950 | 0.675664 | 0.902881 |
| llava_1_5_7b | S×risk+EV | 0.890893 ± 0.000247 | 0.640065 | 0.672932 | 0.895697 |
| internvl_2_5_8b | old risk+EV | 0.848825 ± 0.004339 | 0.466311 | 0.502827 | 0.856685 |
| internvl_2_5_8b | old risk+EV+S | 0.866745 ± 0.002137 | 0.510817 | 0.546172 | 0.874746 |
| internvl_2_5_8b | S×risk+EV | 0.851277 ± 0.004707 | 0.463574 | 0.511333 | 0.859675 |

## 图片级 paired bootstrap

- llava_1_5_7b vs old risk+EV：AUROC Δ=+0.002295，95% CI [-0.004802,+0.009246]；Hall-AUPR Δ=+0.016869，95% CI [-0.009806,+0.042902]。
- llava_1_5_7b vs old risk+EV+S：AUROC Δ=-0.007185，95% CI [-0.014471,-0.000020]；Hall-AUPR Δ=-0.009682，95% CI [-0.033461,+0.013280]。
- internvl_2_5_8b vs old risk+EV：AUROC Δ=+0.002989，95% CI [-0.008205,+0.014293]；Hall-AUPR Δ=+0.005062，95% CI [-0.027290,+0.038788]。
- internvl_2_5_8b vs old risk+EV+S：AUROC Δ=-0.015071，95% CI [-0.028068,-0.002237]；Hall-AUPR Δ=-0.037707，95% CI [-0.074887,-0.000772]。
