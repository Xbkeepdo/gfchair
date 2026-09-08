# internvl_2_5_8b: original risk+EV 平方根 HALL sample weight

训练集 HALL 权重=`2.394208`，REAL 权重=`1.0`；测试集不加权。

| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.848825 ± 0.004339 | 0.502827 | 0.540956 | 0.412869 | 0.466311 |
| Sqrt HALL sample weight | 0.848040 ± 0.004697 | 0.509767 | 0.527547 | 0.436997 | 0.477875 |

## 加权减无权重

- AUROC: -0.000785。
- Hall AUPR: +0.006940。
- Hall precision: -0.013410。
- Hall recall: +0.024129。
- Hall F1: +0.011564。

## 固定 0.5 阈值补充结果

| Training | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: |
| Unweighted | 0.506833 | 0.491510 | 0.484509 |
| Sqrt HALL sample weight | 0.438029 | 0.697051 | 0.532949 |
