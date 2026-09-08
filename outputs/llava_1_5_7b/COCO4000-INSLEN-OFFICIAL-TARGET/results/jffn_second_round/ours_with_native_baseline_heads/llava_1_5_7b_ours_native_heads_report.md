# llava_1_5_7b: Ours 特征 × S-VAR/MetaToken 原生分类头

只替换分类头；特征、InsLen cohort、图片级 8:2 split 和 seeds 43/44/45 不变。
S-VAR 保留单隐藏层 248、50 epochs、Adam；MetaToken 保留 StandardScaler+LR/GB-100。

## 三种子均值

| Feature | Head | AUROC | Hall F1 | Hall AUPR | Ensemble AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| old risk + EV | current 3-layer Torch MLP | 0.888141 ± 0.001422 | 0.644332 | 0.656175 | 0.893401 |
| old risk + EV | svar_native | 0.882879 ± 0.001215 | 0.593330 | 0.637729 | 0.883905 |
| old risk + EV | svar_native_h128 | 0.881584 ± 0.001588 | 0.582373 | 0.634111 | 0.882937 |
| old risk + EV | metatoken_lr | 0.874598 ± 0.000000 | 0.579114 | 0.618606 | 0.874598 |
| old risk + EV | metatoken_gb | 0.873417 ± 0.000025 | 0.543381 | 0.626063 | 0.873426 |
| old risk + EV + Jacobian S | current 3-layer Torch MLP | 0.893087 ± 0.002437 | 0.658950 | 0.675664 | 0.902881 |
| old risk + EV + Jacobian S | svar_native | 0.887273 ± 0.000665 | 0.585181 | 0.664128 | 0.888122 |
| old risk + EV + Jacobian S | svar_native_h128 | 0.885765 ± 0.001455 | 0.560051 | 0.658544 | 0.887352 |
| old risk + EV + Jacobian S | metatoken_lr | 0.880830 ± 0.000000 | 0.556837 | 0.646143 | 0.880830 |
| old risk + EV + Jacobian S | metatoken_gb | 0.888901 ± 0.000012 | 0.613707 | 0.675866 | 0.888905 |

## 相对当前三层 MLP

- old risk + EV / svar_native：mean AUROC -0.005262，mean Hall-AUPR -0.018446，ensemble AUROC -0.009496。
- old risk + EV / svar_native_h128：mean AUROC -0.006557，mean Hall-AUPR -0.022064，ensemble AUROC -0.010465。
- old risk + EV / metatoken_lr：mean AUROC -0.013543，mean Hall-AUPR -0.037569，ensemble AUROC -0.018803。
- old risk + EV / metatoken_gb：mean AUROC -0.014724，mean Hall-AUPR -0.030112，ensemble AUROC -0.019976。
- old risk + EV + Jacobian S / svar_native：mean AUROC -0.005815，mean Hall-AUPR -0.011536，ensemble AUROC -0.014759。
- old risk + EV + Jacobian S / svar_native_h128：mean AUROC -0.007322，mean Hall-AUPR -0.017120，ensemble AUROC -0.015529。
- old risk + EV + Jacobian S / metatoken_lr：mean AUROC -0.012258，mean Hall-AUPR -0.029521，ensemble AUROC -0.022051。
- old risk + EV + Jacobian S / metatoken_gb：mean AUROC -0.004187，mean Hall-AUPR +0.000202，ensemble AUROC -0.013976。

## 加入 S 的头内增量

- svar_native：mean AUROC +0.004393，mean Hall-AUPR +0.026399，ensemble AUROC +0.004217。
- svar_native_h128：mean AUROC +0.004182，mean Hall-AUPR +0.024433，ensemble AUROC +0.004415。
- metatoken_lr：mean AUROC +0.006232，mean Hall-AUPR +0.027536，ensemble AUROC +0.006232。
- metatoken_gb：mean AUROC +0.015484，mean Hall-AUPR +0.049802，ensemble AUROC +0.015479。
