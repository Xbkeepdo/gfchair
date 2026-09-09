# risk + Union-TopK S（无 EV）：两模型汇总

三 seed、相同三层 MLP 与图片级 split、无标准化、不做 bootstrap。

## llava_1_5_7b

| Feature | AUROC | Hall AUPR | Hall F1 |
| --- | ---: | ---: | ---: |
| Original risk only | 0.870446 ± 0.000288 | 0.630539 | 0.587302 |
| Original risk + Union S | 0.886082 ± 0.005399 | 0.655453 | 0.631205 |
| P_JFFN risk only | 0.847202 ± 0.002038 | 0.594926 | 0.524126 |
| P_JFFN risk + Union S | 0.884921 ± 0.000498 | 0.666505 | 0.622577 |

## internvl_2_5_8b

| Feature | AUROC | Hall AUPR | Hall F1 |
| --- | ---: | ---: | ---: |
| Original risk only | 0.762005 ± 0.002592 | 0.371828 | 0.280208 |
| Original risk + Union S | 0.847181 ± 0.007271 | 0.523135 | 0.475387 |
| P_JFFN risk only | 0.737744 ± 0.003461 | 0.323021 | 0.275036 |
| P_JFFN risk + Union S | 0.848088 ± 0.004372 | 0.508339 | 0.485323 |
