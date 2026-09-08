# internvl_2_5_8b: Union-TopK 区域 I/R/S 检测消融

每层 `U=Top32(P_JFFN)∪Top32(Q_raw)`；`I_U=Σ_{j∈U}||a_j||₂`，`R_U=Σ_{j∈U}||J_f(z)a_j||₂`，`S_U=R_U/(I_U+epsilon)`。I/R/S 联合项均为特征拼接。

三隐藏层 `[128,64,32]` MLP，seeds 43/44/45、相同图片级 split、无特征标准化。未运行 bootstrap。

## 三 seed 结果

| Feature | Dim | AUROC | Hall AUPR | Hall F1 | Ensemble AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Union I only | 32 | 0.827443 ± 0.003186 | 0.476635 | 0.360788 | 0.832127 |
| Union R only | 32 | 0.832901 ± 0.003799 | 0.484103 | 0.287482 | 0.837303 |
| Union S only | 32 | 0.848491 ± 0.001019 | 0.521512 | 0.374066 | 0.853844 |
| Union I + R + S | 96 | 0.854411 ± 0.004653 | 0.520276 | 0.431676 | 0.857897 |
| Original risk + EV | 64 | 0.848825 ± 0.004339 | 0.502827 | 0.466311 | 0.856685 |
| Original risk + EV + Union I | 96 | 0.849898 ± 0.002931 | 0.520156 | 0.433004 | 0.852836 |
| Original risk + EV + Union R | 96 | 0.859690 ± 0.002624 | 0.533008 | 0.459017 | 0.864285 |
| Original risk + EV + Union S | 96 | 0.866605 ± 0.001525 | 0.552289 | 0.500021 | 0.878673 |
| Original risk + EV + Union I + R + S | 160 | 0.867020 ± 0.002442 | 0.545717 | 0.490279 | 0.872858 |
| P_JFFN risk + EV | 64 | 0.840525 ± 0.002801 | 0.483769 | 0.478527 | 0.850637 |
| P_JFFN risk + EV + Union I | 96 | 0.846019 ± 0.003254 | 0.508750 | 0.414824 | 0.849552 |
| P_JFFN risk + EV + Union R | 96 | 0.856753 ± 0.002742 | 0.519059 | 0.458675 | 0.862118 |
| P_JFFN risk + EV + Union S | 96 | 0.852447 ± 0.002957 | 0.508563 | 0.482223 | 0.866599 |
| P_JFFN risk + EV + Union I + R + S | 160 | 0.862542 ± 0.001021 | 0.532282 | 0.480706 | 0.867512 |

## 描述性均值差

- Union I + R + S − Union I only：AUROC +0.026968；Hall AUPR +0.043641；Hall F1 +0.070888。
- Union I + R + S − Union R only：AUROC +0.021510；Hall AUPR +0.036173；Hall F1 +0.144194。
- Union I + R + S − Union S only：AUROC +0.005920；Hall AUPR -0.001236；Hall F1 +0.057610。
- Original risk + EV + Union I − Original risk + EV：AUROC +0.001073；Hall AUPR +0.017329；Hall F1 -0.033306。
- Original risk + EV + Union R − Original risk + EV：AUROC +0.010866；Hall AUPR +0.030182；Hall F1 -0.007293。
- Original risk + EV + Union S − Original risk + EV：AUROC +0.017781；Hall AUPR +0.049462；Hall F1 +0.033710。
- Original risk + EV + Union I + R + S − Original risk + EV：AUROC +0.018195；Hall AUPR +0.042891；Hall F1 +0.023968。
- P_JFFN risk + EV + Union I − P_JFFN risk + EV：AUROC +0.005494；Hall AUPR +0.024982；Hall F1 -0.063702。
- P_JFFN risk + EV + Union R − P_JFFN risk + EV：AUROC +0.016228；Hall AUPR +0.035291；Hall F1 -0.019852。
- P_JFFN risk + EV + Union S − P_JFFN risk + EV：AUROC +0.011922；Hall AUPR +0.024794；Hall F1 +0.003696。
- P_JFFN risk + EV + Union I + R + S − P_JFFN risk + EV：AUROC +0.022017；Hall AUPR +0.048513；Hall F1 +0.002179。

## 曲线摘要

- I_U：HALL>REAL / REAL>HALL=0/32；最强 L9，Hall−Real=-0.761323，d=-0.7181。
- R_U：HALL>REAL / REAL>HALL=0/32；最强 L9，Hall−Real=-0.602932，d=-0.7227。
- S_U：HALL>REAL / REAL>HALL=13/19；最强 L11，Hall−Real=-0.036578，d=-0.5282。

## 审计

- train/test images：3142/785。
- train/test mentions：9378/2381。
- Union size min/mean/max：32/39.71/59。
- 输入缓存：本轮新建。
- 复用的既有训练头：old_risk_ev, jffn_risk_ev, old_risk_ev_union_s。
