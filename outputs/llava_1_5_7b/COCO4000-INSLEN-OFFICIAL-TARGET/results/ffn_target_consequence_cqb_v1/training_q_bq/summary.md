# C / Q / B_Q 检测对照

Original 3200/800 image split; exploratory, not an independent holdout claim

F=full AE+log1p(raw S); K=kappa_vec; C/Q=concat(log1p(positive_mass),log1p(negative_mass)); B_Q=negative_Q_mass/S

| 特征 | Ensemble AUROC | REAL AUPR | HALL AUPR | Seed AUROC mean±std |
|---|---:|---:|---:|---:|
| F | 0.904960 | 0.970033 | 0.719276 | 0.900152±0.002722 |
| F_K | 0.901700 | 0.969437 | 0.709059 | 0.892433±0.002328 |
| Q | 0.887956 | 0.964761 | 0.678376 | 0.885642±0.000530 |
| B_Q | 0.822845 | 0.938368 | 0.574464 | 0.819485±0.003373 |
| F_Q | 0.900403 | 0.969452 | 0.704828 | 0.895400±0.001000 |
| F_B_Q | 0.904571 | 0.970104 | 0.720404 | 0.900159±0.001578 |
| F_K_Q | 0.902197 | 0.969423 | 0.716533 | 0.895899±0.003009 |
| F_K_B_Q | 0.902525 | 0.969656 | 0.712432 | 0.894827±0.001058 |
