# internvl_2_5_8b: Ours 特征 × S-VAR/MetaToken 原生分类头

只替换分类头；特征、InsLen cohort、图片级 8:2 split 和 seeds 43/44/45 不变。
S-VAR 保留单隐藏层 248、50 epochs、Adam；MetaToken 保留 StandardScaler+LR/GB-100。

## 三种子均值

| Feature | Head | AUROC | Hall F1 | Hall AUPR | Ensemble AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| old risk + EV | current 3-layer Torch MLP | 0.848825 ± 0.004339 | 0.466311 | 0.502827 | 0.856685 |
| old risk + EV | svar_native | 0.822749 ± 0.000884 | 0.295343 | 0.466272 | 0.823663 |
| old risk + EV | svar_native_h128 | 0.817035 ± 0.001616 | 0.269644 | 0.446114 | 0.818015 |
| old risk + EV | metatoken_lr | 0.818551 ± 0.000000 | 0.268293 | 0.436445 | 0.818552 |
| old risk + EV | metatoken_gb | 0.812016 ± 0.000033 | 0.375887 | 0.468151 | 0.812013 |
| old risk + EV + Jacobian S | current 3-layer Torch MLP | 0.866745 ± 0.002137 | 0.510817 | 0.546172 | 0.874746 |
| old risk + EV + Jacobian S | svar_native | 0.851424 ± 0.001563 | 0.324979 | 0.505964 | 0.851937 |
| old risk + EV + Jacobian S | svar_native_h128 | 0.845775 ± 0.004036 | 0.297013 | 0.491057 | 0.847003 |
| old risk + EV + Jacobian S | metatoken_lr | 0.839470 ± 0.000000 | 0.288577 | 0.472539 | 0.839470 |
| old risk + EV + Jacobian S | metatoken_gb | 0.840995 ± 0.000008 | 0.422856 | 0.518275 | 0.841005 |

## 相对当前三层 MLP

- old risk + EV / svar_native：mean AUROC -0.026075，mean Hall-AUPR -0.036555，ensemble AUROC -0.033022。
- old risk + EV / svar_native_h128：mean AUROC -0.031790，mean Hall-AUPR -0.056713，ensemble AUROC -0.038670。
- old risk + EV / metatoken_lr：mean AUROC -0.030274，mean Hall-AUPR -0.066381，ensemble AUROC -0.038134。
- old risk + EV / metatoken_gb：mean AUROC -0.036808，mean Hall-AUPR -0.034676，ensemble AUROC -0.044673。
- old risk + EV + Jacobian S / svar_native：mean AUROC -0.015321，mean Hall-AUPR -0.040208，ensemble AUROC -0.022808。
- old risk + EV + Jacobian S / svar_native_h128：mean AUROC -0.020970，mean Hall-AUPR -0.055115，ensemble AUROC -0.027743。
- old risk + EV + Jacobian S / metatoken_lr：mean AUROC -0.027275，mean Hall-AUPR -0.073633，ensemble AUROC -0.035276。
- old risk + EV + Jacobian S / metatoken_gb：mean AUROC -0.025750，mean Hall-AUPR -0.027897，ensemble AUROC -0.033740。

## 加入 S 的头内增量

- svar_native：mean AUROC +0.028674，mean Hall-AUPR +0.039692，ensemble AUROC +0.028274。
- svar_native_h128：mean AUROC +0.028740，mean Hall-AUPR +0.044943，ensemble AUROC +0.028987。
- metatoken_lr：mean AUROC +0.020919，mean Hall-AUPR +0.036094，ensemble AUROC +0.020918。
- metatoken_gb：mean AUROC +0.028979，mean Hall-AUPR +0.050124，ensemble AUROC +0.028993。
