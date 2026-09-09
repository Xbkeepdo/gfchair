# C / Q / B_Q 检测对照

Original 3200/800 image split; exploratory, not an independent holdout claim

F=full AE+log1p(raw S); K=kappa_vec; C/Q=concat(log1p(positive_mass),log1p(negative_mass)); B_Q=negative_Q_mass/S

| 特征 | Ensemble AUROC | REAL AUPR | HALL AUPR | Seed AUROC mean±std |
|---|---:|---:|---:|---:|
| F | 0.863812 | 0.970639 | 0.546154 | 0.858254±0.005869 |
| F_K | 0.875039 | 0.973339 | 0.559442 | 0.864396±0.005873 |
| Q | 0.847554 | 0.964889 | 0.509195 | 0.843165±0.002123 |
| B_Q | 0.716398 | 0.930280 | 0.311164 | 0.706647±0.010677 |
| F_Q | 0.865442 | 0.971204 | 0.524087 | 0.854012±0.011415 |
| F_B_Q | 0.867505 | 0.970870 | 0.547435 | 0.857550±0.006468 |
| F_K_Q | 0.878815 | 0.973234 | 0.572464 | 0.864849±0.005752 |
| F_K_B_Q | 0.872486 | 0.972353 | 0.551286 | 0.862913±0.004406 |
