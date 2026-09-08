# llava_1_5_7b: risk + Union-TopK S（无 EV）

固定 `S_U=Σ_{j∈U}||J_f(z)a_j||₂ / (Σ_{j∈U}||a_j||₂+epsilon)`，`U=Top32(P_JFFN)∪Top32(Q_hpre_raw_logit_gauss)`。

| Feature | Dim | AUROC | Hall AUPR | Hall F1 |
| --- | ---: | ---: | ---: | ---: |
| Original risk only | 32 | 0.870446 ± 0.000288 | 0.630539 | 0.587302 |
| Original risk + Union S | 64 | 0.886082 ± 0.005399 | 0.655453 | 0.631205 |
| P_JFFN risk only | 32 | 0.847202 ± 0.002038 | 0.594926 | 0.524126 |
| P_JFFN risk + Union S | 64 | 0.884921 ± 0.000498 | 0.666505 | 0.622577 |

## S 的增量

- Original risk + Union S − Original risk only：AUROC +0.015636；Hall AUPR +0.024915；Hall F1 +0.043903。
- P_JFFN risk + Union S − P_JFFN risk only：AUROC +0.037719；Hall AUPR +0.071579；Hall F1 +0.098451。
