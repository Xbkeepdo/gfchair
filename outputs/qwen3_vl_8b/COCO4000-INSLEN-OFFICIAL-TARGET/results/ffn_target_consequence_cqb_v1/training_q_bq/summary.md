# C / Q / B_Q 检测对照

Original 3200/800 image split; exploratory, not an independent holdout claim

F=full AE+log1p(raw S); K=kappa_vec; C/Q=concat(log1p(positive_mass),log1p(negative_mass)); B_Q=negative_Q_mass/S

| 特征 | Ensemble AUROC | REAL AUPR | HALL AUPR | Seed AUROC mean±std |
|---|---:|---:|---:|---:|
| F | 0.896510 | 0.976118 | 0.649342 | 0.886550±0.003048 |
| F_K | 0.893620 | 0.974928 | 0.653327 | 0.880061±0.007491 |
| Q | 0.881705 | 0.972816 | 0.607049 | 0.873287±0.002075 |
| B_Q | 0.764873 | 0.936976 | 0.424509 | 0.756999±0.000964 |
| F_Q | 0.898818 | 0.975777 | 0.664112 | 0.883264±0.009763 |
| F_B_Q | 0.895670 | 0.975638 | 0.660139 | 0.885001±0.007354 |
| F_K_Q | 0.892100 | 0.974421 | 0.646519 | 0.876303±0.005224 |
| F_K_B_Q | 0.895683 | 0.975331 | 0.659556 | 0.881761±0.006328 |
