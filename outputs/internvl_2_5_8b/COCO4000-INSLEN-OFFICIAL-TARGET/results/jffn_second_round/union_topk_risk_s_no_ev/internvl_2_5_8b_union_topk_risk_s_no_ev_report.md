# internvl_2_5_8b: risk + Union-TopK S（无 EV）

固定 `S_U=Σ_{j∈U}||J_f(z)a_j||₂ / (Σ_{j∈U}||a_j||₂+epsilon)`，`U=Top32(P_JFFN)∪Top32(Q_hpre_raw_logit_gauss)`。

| Feature | Dim | AUROC | Hall AUPR | Hall F1 |
| --- | ---: | ---: | ---: | ---: |
| Original risk only | 32 | 0.762005 ± 0.002592 | 0.371828 | 0.280208 |
| Original risk + Union S | 64 | 0.847181 ± 0.007271 | 0.523135 | 0.475387 |
| P_JFFN risk only | 32 | 0.737744 ± 0.003461 | 0.323021 | 0.275036 |
| P_JFFN risk + Union S | 64 | 0.848088 ± 0.004372 | 0.508339 | 0.485323 |

## S 的增量

- Original risk + Union S − Original risk only：AUROC +0.085176；Hall AUPR +0.151307；Hall F1 +0.195180。
- P_JFFN risk + Union S − P_JFFN risk only：AUROC +0.110344；Hall AUPR +0.185317；Hall F1 +0.210287。
