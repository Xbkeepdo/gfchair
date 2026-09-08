# llava_1_5_7b: original risk+EV 平方根 HALL sample weight

训练集 HALL 权重=`1.853042`，REAL 权重=`1.0`；测试集不加权。

| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.888141 ± 0.001422 | 0.656175 | 0.632111 | 0.657734 | 0.644332 |
| Sqrt HALL sample weight | 0.889209 ± 0.002119 | 0.656811 | 0.641582 | 0.635637 | 0.637918 |

## 加权减无权重

- AUROC: +0.001068。
- Hall AUPR: +0.000636。
- Hall precision: +0.009471。
- Hall recall: -0.022097。
- Hall F1: -0.006414。

## 固定 0.5 阈值补充结果

| Training | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: |
| Unweighted | 0.620203 | 0.690644 | 0.651382 |
| Sqrt HALL sample weight | 0.548077 | 0.827927 | 0.658482 |
