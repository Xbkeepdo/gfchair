# original risk+EV：平方根 HALL sample weight 两模型对照

HALL 权重只由训练集计算，测试集不加权；三 seeds、相同 MLP/split，不做 bootstrap。

## llava_1_5_7b（HALL weight=1.853042）

| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.888141 ± 0.001422 | 0.656175 | 0.632111 | 0.657734 | 0.644332 |
| Sqrt HALL weight | 0.889209 ± 0.002119 | 0.656811 | 0.641582 | 0.635637 | 0.637918 |

固定 0.5 阈值：

| Training | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: |
| Unweighted | 0.620203 | 0.690644 | 0.651382 |
| Sqrt HALL weight | 0.548077 | 0.827927 | 0.658482 |

## internvl_2_5_8b（HALL weight=2.394208）

| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.848825 ± 0.004339 | 0.502827 | 0.540956 | 0.412869 | 0.466311 |
| Sqrt HALL weight | 0.848040 ± 0.004697 | 0.509767 | 0.527547 | 0.436997 | 0.477875 |

固定 0.5 阈值：

| Training | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: |
| Unweighted | 0.506833 | 0.491510 | 0.484509 |
| Sqrt HALL weight | 0.438029 | 0.697051 | 0.532949 |
