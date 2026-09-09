# P_JFFN-target Union-Top32 JS + EV：两模型汇总

| Model | Source P | Target gate | Mean AUROC | Hall F1 | Hall AUPR | Ensemble AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| llava_1_5_7b | old_hpre_cos | hpre_raw_logit_gauss | 0.889554 ± 0.001292 | 0.640065 | 0.665323 | 0.896075 |
| llava_1_5_7b | old_hpre_cos | hpre_softmax_prob_gauss | 0.890203 ± 0.001922 | 0.652942 | 0.675934 | 0.896289 |
| llava_1_5_7b | new_jffn | hpre_raw_logit_gauss | 0.876057 ± 0.002547 | 0.621406 | 0.651249 | 0.881195 |
| llava_1_5_7b | new_jffn | hpre_softmax_prob_gauss | 0.877191 ± 0.002651 | 0.629348 | 0.655254 | 0.884187 |
| internvl_2_5_8b | old_hpre_cos | hpre_raw_logit_gauss | 0.848808 ± 0.002423 | 0.474816 | 0.507783 | 0.857859 |
| internvl_2_5_8b | old_hpre_cos | hpre_softmax_prob_gauss | 0.844465 ± 0.001018 | 0.474098 | 0.491826 | 0.853618 |
| internvl_2_5_8b | new_jffn | hpre_raw_logit_gauss | 0.838730 ± 0.007828 | 0.447544 | 0.458472 | 0.847685 |
| internvl_2_5_8b | new_jffn | hpre_softmax_prob_gauss | 0.842990 ± 0.005356 | 0.445454 | 0.479301 | 0.852763 |

## Paired image bootstrap

- llava_1_5_7b / hpre_raw_logit_gauss vs old-hpre-P JS+EV：AUROC Δ=-0.014881，95% CI [-0.024194,-0.005572]；Hall-AUPR Δ=-0.016699，95% CI [-0.047960,+0.014904]。
- llava_1_5_7b / hpre_raw_logit_gauss vs P_JFFN OT-risk+EV：AUROC Δ=-0.001980，95% CI [-0.006710,+0.002530]；Hall-AUPR Δ=-0.012039，95% CI [-0.028159,+0.003902]。
- llava_1_5_7b / hpre_softmax_prob_gauss vs old-hpre-P JS+EV：AUROC Δ=-0.012102，95% CI [-0.021909,-0.002441]；Hall-AUPR Δ=-0.021350，95% CI [-0.052581,+0.010684]。
- llava_1_5_7b / hpre_softmax_prob_gauss vs P_JFFN OT-risk+EV：AUROC Δ=+0.005109，95% CI [-0.000283,+0.010567]；Hall-AUPR Δ=+0.013138，95% CI [-0.002481,+0.028955]。
- internvl_2_5_8b / hpre_raw_logit_gauss vs old-hpre-P JS+EV：AUROC Δ=-0.010174，95% CI [-0.023899,+0.003084]；Hall-AUPR Δ=-0.050323，95% CI [-0.086775,-0.010059]。
- internvl_2_5_8b / hpre_raw_logit_gauss vs P_JFFN OT-risk+EV：AUROC Δ=-0.002952，95% CI [-0.010010,+0.004182]；Hall-AUPR Δ=-0.024095，95% CI [-0.043405,-0.003387]。
- internvl_2_5_8b / hpre_softmax_prob_gauss vs old-hpre-P JS+EV：AUROC Δ=-0.000855，95% CI [-0.015399,+0.013373]；Hall-AUPR Δ=-0.008628，95% CI [-0.048325,+0.032160]。
- internvl_2_5_8b / hpre_softmax_prob_gauss vs P_JFFN OT-risk+EV：AUROC Δ=+0.002405，95% CI [-0.005837,+0.010712]；Hall-AUPR Δ=-0.001514，95% CI [-0.024376,+0.020546]。
