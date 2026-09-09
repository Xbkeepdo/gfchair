# Union-TopK 区域新 S：两模型汇总

新 S 为 `Top32(P_JFFN) ∪ Top32(Q_raw)` 区域内的 `ΣR_j/ΣI_j`。

| Model | Feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| llava_1_5_7b | old_risk_ev | 0.888141 ± 0.001422 | 0.644332 | 0.656175 | 0.893401 |
| llava_1_5_7b | old_risk_ev_old_aggregate_s | 0.893087 ± 0.002437 | 0.658950 | 0.675664 | 0.902881 |
| llava_1_5_7b | token_all_s | 0.894717 ± 0.001654 | 0.649339 | 0.668844 | 0.904363 |
| llava_1_5_7b | union_topk_s | 0.891793 ± 0.001312 | 0.640199 | 0.672379 | 0.900209 |
| internvl_2_5_8b | old_risk_ev | 0.848825 ± 0.004339 | 0.466311 | 0.502827 | 0.856685 |
| internvl_2_5_8b | old_risk_ev_old_aggregate_s | 0.866745 ± 0.002137 | 0.510817 | 0.546172 | 0.874746 |
| internvl_2_5_8b | token_all_s | 0.863213 ± 0.001334 | 0.503787 | 0.545624 | 0.870371 |
| internvl_2_5_8b | union_topk_s | 0.866605 ± 0.001525 | 0.500021 | 0.552289 | 0.878673 |

## Union S 的 paired bootstrap

- llava_1_5_7b vs risk+EV：AUROC Δ=+0.006808，95% CI [-0.001083,+0.014666]。
- llava_1_5_7b vs old aggregate S：AUROC Δ=-0.002672，95% CI [-0.008183,+0.002773]。
- llava_1_5_7b vs all-token S：AUROC Δ=-0.004153，95% CI [-0.008058,-0.000272]。
- internvl_2_5_8b vs risk+EV：AUROC Δ=+0.021988，95% CI [+0.009584,+0.034726]。
- internvl_2_5_8b vs old aggregate S：AUROC Δ=+0.003927，95% CI [-0.004651,+0.012282]。
- internvl_2_5_8b vs all-token S：AUROC Δ=+0.008302，95% CI [+0.001906,+0.014983]。
