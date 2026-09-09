# C / Q / B_Q 检测对照

Original 3200/800 image split; exploratory, not an independent holdout claim

F=full AE+log1p(raw S); K=kappa_vec; C/Q=concat(log1p(positive_mass),log1p(negative_mass)); B_Q=negative_Q_mass/S

| 特征 | Ensemble AUROC | REAL AUPR | HALL AUPR | Seed AUROC mean±std |
|---|---:|---:|---:|---:|
| F | 0.882138 | 0.984175 | 0.457740 | 0.874489±0.000845 |
| F_K | 0.883630 | 0.984333 | 0.463260 | 0.877980±0.003647 |
| Q | 0.862626 | 0.980824 | 0.412535 | 0.854264±0.004065 |
| B_Q | 0.740153 | 0.957153 | 0.270710 | 0.735154±0.005476 |
| F_Q | 0.879838 | 0.983600 | 0.465380 | 0.867442±0.001727 |
| F_B_Q | 0.882424 | 0.984065 | 0.462078 | 0.875432±0.002945 |
| F_K_Q | 0.876633 | 0.982955 | 0.455412 | 0.866482±0.006813 |
| F_K_B_Q | 0.886812 | 0.984381 | 0.474876 | 0.875190±0.006215 |
