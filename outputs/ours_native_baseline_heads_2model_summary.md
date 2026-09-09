# Ours 特征 × S-VAR/MetaToken 原生分类头：两模型汇总

仅更换分类头；两种特征、样本、图片级 split 和 seeds 均严格对齐。

| Model | Feature | Head | Mean AUROC | Hall F1 | Hall AUPR | Ensemble AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| llava_1_5_7b | old_risk_ev | current_3layer_torch_mlp | 0.888141 ± 0.001422 | 0.644332 | 0.656175 | 0.893401 |
| llava_1_5_7b | old_risk_ev | svar_native | 0.882879 ± 0.001215 | 0.593330 | 0.637729 | 0.883905 |
| llava_1_5_7b | old_risk_ev | svar_native_h128 | 0.881584 ± 0.001588 | 0.582373 | 0.634111 | 0.882937 |
| llava_1_5_7b | old_risk_ev | metatoken_lr | 0.874598 ± 0.000000 | 0.579114 | 0.618606 | 0.874598 |
| llava_1_5_7b | old_risk_ev | metatoken_gb | 0.873417 ± 0.000025 | 0.543381 | 0.626063 | 0.873426 |
| llava_1_5_7b | old_risk_ev_s | current_3layer_torch_mlp | 0.893087 ± 0.002437 | 0.658950 | 0.675664 | 0.902881 |
| llava_1_5_7b | old_risk_ev_s | svar_native | 0.887273 ± 0.000665 | 0.585181 | 0.664128 | 0.888122 |
| llava_1_5_7b | old_risk_ev_s | svar_native_h128 | 0.885765 ± 0.001455 | 0.560051 | 0.658544 | 0.887352 |
| llava_1_5_7b | old_risk_ev_s | metatoken_lr | 0.880830 ± 0.000000 | 0.556837 | 0.646143 | 0.880830 |
| llava_1_5_7b | old_risk_ev_s | metatoken_gb | 0.888901 ± 0.000012 | 0.613707 | 0.675866 | 0.888905 |
| internvl_2_5_8b | old_risk_ev | current_3layer_torch_mlp | 0.848825 ± 0.004339 | 0.466311 | 0.502827 | 0.856685 |
| internvl_2_5_8b | old_risk_ev | svar_native | 0.822749 ± 0.000884 | 0.295343 | 0.466272 | 0.823663 |
| internvl_2_5_8b | old_risk_ev | svar_native_h128 | 0.817035 ± 0.001616 | 0.269644 | 0.446114 | 0.818015 |
| internvl_2_5_8b | old_risk_ev | metatoken_lr | 0.818551 ± 0.000000 | 0.268293 | 0.436445 | 0.818552 |
| internvl_2_5_8b | old_risk_ev | metatoken_gb | 0.812016 ± 0.000033 | 0.375887 | 0.468151 | 0.812013 |
| internvl_2_5_8b | old_risk_ev_s | current_3layer_torch_mlp | 0.866745 ± 0.002137 | 0.510817 | 0.546172 | 0.874746 |
| internvl_2_5_8b | old_risk_ev_s | svar_native | 0.851424 ± 0.001563 | 0.324979 | 0.505964 | 0.851937 |
| internvl_2_5_8b | old_risk_ev_s | svar_native_h128 | 0.845775 ± 0.004036 | 0.297013 | 0.491057 | 0.847003 |
| internvl_2_5_8b | old_risk_ev_s | metatoken_lr | 0.839470 ± 0.000000 | 0.288577 | 0.472539 | 0.839470 |
| internvl_2_5_8b | old_risk_ev_s | metatoken_gb | 0.840995 ± 0.000008 | 0.422856 | 0.518275 | 0.841005 |

## 加入 S 的头内增量

- llava_1_5_7b / svar_native：mean AUROC +0.004393，Hall-AUPR +0.026399，ensemble AUROC +0.004217。
- llava_1_5_7b / svar_native_h128：mean AUROC +0.004182，Hall-AUPR +0.024433，ensemble AUROC +0.004415。
- llava_1_5_7b / metatoken_lr：mean AUROC +0.006232，Hall-AUPR +0.027536，ensemble AUROC +0.006232。
- llava_1_5_7b / metatoken_gb：mean AUROC +0.015484，Hall-AUPR +0.049802，ensemble AUROC +0.015479。
- internvl_2_5_8b / svar_native：mean AUROC +0.028674，Hall-AUPR +0.039692，ensemble AUROC +0.028274。
- internvl_2_5_8b / svar_native_h128：mean AUROC +0.028740，Hall-AUPR +0.044943，ensemble AUROC +0.028987。
- internvl_2_5_8b / metatoken_lr：mean AUROC +0.020919，Hall-AUPR +0.036094，ensemble AUROC +0.020918。
- internvl_2_5_8b / metatoken_gb：mean AUROC +0.028979，Hall-AUPR +0.050124，ensemble AUROC +0.028993。
