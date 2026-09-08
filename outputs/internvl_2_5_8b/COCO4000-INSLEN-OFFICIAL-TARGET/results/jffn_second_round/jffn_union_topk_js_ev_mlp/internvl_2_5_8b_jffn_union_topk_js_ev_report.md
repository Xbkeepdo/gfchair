# internvl_2_5_8b: P_JFFN-target Union-Top32 JS + EV

每层分别取 source/target Top-32 的并集，在并集内各自重新归一化后计算自然对数 JS；32层 JS 与同 gate 的32层 mass×cosine EV 拼接为64维。

## 三种子结果

| Source | Target gate | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| old_hpre_cos | hpre_raw_logit_gauss | 0.848808 ± 0.002423 | 0.474816 | 0.507783 | 0.857859 |
| old_hpre_cos | hpre_softmax_prob_gauss | 0.844465 ± 0.001018 | 0.474098 | 0.491826 | 0.853618 |
| new_jffn | hpre_raw_logit_gauss | 0.838730 ± 0.007828 | 0.447544 | 0.458472 | 0.847685 |
| new_jffn | hpre_softmax_prob_gauss | 0.842990 ± 0.005356 | 0.445454 | 0.479301 | 0.852763 |

## 配对对照

- hpre_raw_logit_gauss，P_JFFN JS+EV − old-hpre-P JS+EV：ensemble AUROC Δ=-0.010174，95% CI [-0.023899,+0.003084]；Hall-AUPR Δ=-0.050323，95% CI [-0.086775,-0.010059]。
- hpre_raw_logit_gauss，P_JFFN JS+EV − P_JFFN OT-risk+EV：ensemble AUROC Δ=-0.002952，95% CI [-0.010010,+0.004182]；Hall-AUPR Δ=-0.024095，95% CI [-0.043405,-0.003387]。
- hpre_softmax_prob_gauss，P_JFFN JS+EV − old-hpre-P JS+EV：ensemble AUROC Δ=-0.000855，95% CI [-0.015399,+0.013373]；Hall-AUPR Δ=-0.008628，95% CI [-0.048325,+0.032160]。
- hpre_softmax_prob_gauss，P_JFFN JS+EV − P_JFFN OT-risk+EV：ensemble AUROC Δ=+0.002405，95% CI [-0.005837,+0.010712]；Hall-AUPR Δ=-0.001514，95% CI [-0.024376,+0.020546]。

## JS 曲线摘要

- hpre_raw_logit_gauss：Hall>Real 23/32 层；全层均值 Real=0.041128、Hall=0.042646；最大绝对差 L6，Hall−Real=+0.011056。
- hpre_softmax_prob_gauss：Hall>Real 14/32 层；全层均值 Real=0.029338、Hall=0.028920；最大绝对差 L6，Hall−Real=+0.006123。
