# llava_1_5_7b: P_JFFN-target Union-Top32 JS + EV

每层分别取 source/target Top-32 的并集，在并集内各自重新归一化后计算自然对数 JS；32层 JS 与同 gate 的32层 mass×cosine EV 拼接为64维。

## 三种子结果

| Source | Target gate | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| old_hpre_cos | hpre_raw_logit_gauss | 0.889554 ± 0.001292 | 0.640065 | 0.665323 | 0.896075 |
| old_hpre_cos | hpre_softmax_prob_gauss | 0.890203 ± 0.001922 | 0.652942 | 0.675934 | 0.896289 |
| new_jffn | hpre_raw_logit_gauss | 0.876057 ± 0.002547 | 0.621406 | 0.651249 | 0.881195 |
| new_jffn | hpre_softmax_prob_gauss | 0.877191 ± 0.002651 | 0.629348 | 0.655254 | 0.884187 |

## 配对对照

- hpre_raw_logit_gauss，P_JFFN JS+EV − old-hpre-P JS+EV：ensemble AUROC Δ=-0.014881，95% CI [-0.024194,-0.005572]；Hall-AUPR Δ=-0.016699，95% CI [-0.047960,+0.014904]。
- hpre_raw_logit_gauss，P_JFFN JS+EV − P_JFFN OT-risk+EV：ensemble AUROC Δ=-0.001980，95% CI [-0.006710,+0.002530]；Hall-AUPR Δ=-0.012039，95% CI [-0.028159,+0.003902]。
- hpre_softmax_prob_gauss，P_JFFN JS+EV − old-hpre-P JS+EV：ensemble AUROC Δ=-0.012102，95% CI [-0.021909,-0.002441]；Hall-AUPR Δ=-0.021350，95% CI [-0.052581,+0.010684]。
- hpre_softmax_prob_gauss，P_JFFN JS+EV − P_JFFN OT-risk+EV：ensemble AUROC Δ=+0.005109，95% CI [-0.000283,+0.010567]；Hall-AUPR Δ=+0.013138，95% CI [-0.002481,+0.028955]。

## JS 曲线摘要

- hpre_raw_logit_gauss：Hall>Real 27/32 层；全层均值 Real=0.085722、Hall=0.097283；最大绝对差 L17，Hall−Real=+0.027871。
- hpre_softmax_prob_gauss：Hall>Real 25/32 层；全层均值 Real=0.043102、Hall=0.045905；最大绝对差 L32，Hall−Real=-0.016202。
