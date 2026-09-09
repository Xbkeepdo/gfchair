# DHCP sampler：risk+EV 单输出与双输出头对照

只替换输出头和二分类 loss 参数化；两模型、三 seeds，不做 bootstrap。

# llava_1_5_7b: DHCP sampler 下单输出与双输出头对照

唯一训练变量是输出参数化：`Linear(32,1)+BCEWithLogitsLoss` 对比 `Linear(32,2)+CrossEntropyLoss`。两者共享 risk+EV、DHCP sampler、三层隐藏层、split、epoch 与 seeds。

## 当前 train-REAL-F1 阈值

| Head | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 output + BCE | 0.889179 ± 0.001272 | 0.657014 | 0.652399 | 0.616831 | 0.633924 |
| 2 outputs + CE | 0.887270 ± 0.002081 | 0.661106 | 0.636341 | 0.631406 | 0.633671 |

## 固定 0.5 / 双输出 argmax

| Head | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: |
| 1 output + BCE | 0.513757 | 0.879643 | 0.648489 |
| 2 outputs + CE | 0.556350 | 0.812882 | 0.660075 |

## 双输出减单输出

- AUROC: `-0.001908`。
- Hall AUPR: `+0.004092`。
- Hall F1: `-0.000254`。

# internvl_2_5_8b: DHCP sampler 下单输出与双输出头对照

唯一训练变量是输出参数化：`Linear(32,1)+BCEWithLogitsLoss` 对比 `Linear(32,2)+CrossEntropyLoss`。两者共享 risk+EV、DHCP sampler、三层隐藏层、split、epoch 与 seeds。

## 当前 train-REAL-F1 阈值

| Head | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 output + BCE | 0.846005 ± 0.006709 | 0.506057 | 0.528139 | 0.470063 | 0.497310 |
| 2 outputs + CE | 0.846307 ± 0.002873 | 0.500854 | 0.527285 | 0.443253 | 0.481276 |

## 固定 0.5 / 双输出 argmax

| Head | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: |
| 1 output + BCE | 0.448831 | 0.674710 | 0.535904 |
| 2 outputs + CE | 0.453098 | 0.688114 | 0.544742 |

## 双输出减单输出

- AUROC: `+0.000302`。
- Hall AUPR: `-0.005202`。
- Hall F1: `-0.016034`。
