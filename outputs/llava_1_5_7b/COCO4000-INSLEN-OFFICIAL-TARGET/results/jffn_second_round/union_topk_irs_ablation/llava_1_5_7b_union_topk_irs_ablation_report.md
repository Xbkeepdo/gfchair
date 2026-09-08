# llava_1_5_7b: Union-TopK 区域 I/R/S 检测消融

每层 `U=Top32(P_JFFN)∪Top32(Q_raw)`；`I_U=Σ_{j∈U}||a_j||₂`，`R_U=Σ_{j∈U}||J_f(z)a_j||₂`，`S_U=R_U/(I_U+epsilon)`。I/R/S 联合项均为特征拼接。

三隐藏层 `[128,64,32]` MLP，seeds 43/44/45、相同图片级 split、无特征标准化。未运行 bootstrap。

## 三 seed 结果

| Feature | Dim | AUROC | Hall AUPR | Hall F1 | Ensemble AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Union I only | 32 | 0.873471 ± 0.000650 | 0.641057 | 0.581208 | 0.875292 |
| Union R only | 32 | 0.878886 ± 0.001136 | 0.651995 | 0.604126 | 0.881142 |
| Union S only | 32 | 0.874894 ± 0.002045 | 0.625940 | 0.607913 | 0.877740 |
| Union I + R + S | 96 | 0.884852 ± 0.001289 | 0.650374 | 0.627091 | 0.887401 |
| Original risk + EV | 64 | 0.888141 ± 0.001422 | 0.656175 | 0.644332 | 0.893401 |
| Original risk + EV + Union I | 96 | 0.888888 ± 0.000825 | 0.667713 | 0.658052 | 0.890952 |
| Original risk + EV + Union R | 96 | 0.893297 ± 0.000901 | 0.663003 | 0.638730 | 0.896775 |
| Original risk + EV + Union S | 96 | 0.891793 ± 0.001312 | 0.672379 | 0.640199 | 0.900209 |
| Original risk + EV + Union I + R + S | 160 | 0.894870 ± 0.002304 | 0.670014 | 0.649803 | 0.898683 |
| P_JFFN risk + EV | 64 | 0.876717 ± 0.004153 | 0.659444 | 0.611740 | 0.883175 |
| P_JFFN risk + EV + Union I | 96 | 0.883542 ± 0.000424 | 0.667929 | 0.612375 | 0.886145 |
| P_JFFN risk + EV + Union R | 96 | 0.887541 ± 0.000623 | 0.664790 | 0.621674 | 0.890654 |
| P_JFFN risk + EV + Union S | 96 | 0.888278 ± 0.005211 | 0.673936 | 0.620983 | 0.897061 |
| P_JFFN risk + EV + Union I + R + S | 160 | 0.890868 ± 0.000634 | 0.674085 | 0.633513 | 0.896407 |

## 描述性均值差

- Union I + R + S − Union I only：AUROC +0.011381；Hall AUPR +0.009316；Hall F1 +0.045882。
- Union I + R + S − Union R only：AUROC +0.005966；Hall AUPR -0.001621；Hall F1 +0.022965。
- Union I + R + S − Union S only：AUROC +0.009958；Hall AUPR +0.024433；Hall F1 +0.019178。
- Original risk + EV + Union I − Original risk + EV：AUROC +0.000746；Hall AUPR +0.011538；Hall F1 +0.013720。
- Original risk + EV + Union R − Original risk + EV：AUROC +0.005156；Hall AUPR +0.006828；Hall F1 -0.005602。
- Original risk + EV + Union S − Original risk + EV：AUROC +0.003652；Hall AUPR +0.016204；Hall F1 -0.004133。
- Original risk + EV + Union I + R + S − Original risk + EV：AUROC +0.006729；Hall AUPR +0.013839；Hall F1 +0.005471。
- P_JFFN risk + EV + Union I − P_JFFN risk + EV：AUROC +0.006825；Hall AUPR +0.008485；Hall F1 +0.000634。
- P_JFFN risk + EV + Union R − P_JFFN risk + EV：AUROC +0.010824；Hall AUPR +0.005346；Hall F1 +0.009934。
- P_JFFN risk + EV + Union S − P_JFFN risk + EV：AUROC +0.011561；Hall AUPR +0.014491；Hall F1 +0.009243。
- P_JFFN risk + EV + Union I + R + S − P_JFFN risk + EV：AUROC +0.014150；Hall AUPR +0.014641；Hall F1 +0.021773。

## 曲线摘要

- I_U：HALL>REAL / REAL>HALL=7/25；最强 L1，Hall−Real=-0.062643，d=-1.2118。
- R_U：HALL>REAL / REAL>HALL=7/25；最强 L1，Hall−Real=-0.112833，d=-1.1273。
- S_U：HALL>REAL / REAL>HALL=4/28；最强 L19，Hall−Real=-0.050426，d=-0.8151。

## 审计

- train/test images：3174/797。
- train/test mentions：12317/3146。
- Union size min/mean/max：32/41.49/64。
- 输入缓存：本轮新建。
- 复用的既有训练头：old_risk_ev, jffn_risk_ev, old_risk_ev_union_s。
