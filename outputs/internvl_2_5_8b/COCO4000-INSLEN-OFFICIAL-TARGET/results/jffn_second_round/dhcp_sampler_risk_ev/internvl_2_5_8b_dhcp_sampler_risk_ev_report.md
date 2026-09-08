# internvl_2_5_8b: original risk+EV 的 DHCP 反频率采样消融

这是只替换训练采样器的受控实验：特征、split、三层 MLP、优化器、epoch 和 seeds 均不变。DHCP sampler 使用 `1/N_class`、有放回抽样，每个 epoch 抽取与原训练集相同的行数；loss 不加权。

训练 REAL/HALL=`7985/1393`，HALL/REAL 单行抽样权重比=`5.732233`。

## 当前 train-REAL-F1 阈值

| Training | AUROC | Hall AUPR | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.848825 ± 0.004339 | 0.502827 | 0.540956 | 0.412869 | 0.466311 |
| Sqrt HALL loss weight | 0.848040 ± 0.004697 | 0.509767 | 0.527547 | 0.436997 | 0.477875 |
| DHCP balanced sampler | 0.846005 ± 0.006709 | 0.506057 | 0.528139 | 0.470063 | 0.497310 |

## 固定 0.5（等价于二分类 argmax）

| Training | Hall precision | Hall recall | Hall F1 |
| --- | ---: | ---: | ---: |
| Unweighted | 0.506833 | 0.491510 | 0.484509 |
| Sqrt HALL loss weight | 0.438029 | 0.697051 | 0.532949 |
| DHCP balanced sampler | 0.448831 | 0.674710 | 0.535904 |

## DHCP sampler 的增量

- 相对 unweighted: AUROC `-0.002819`，Hall AUPR `+0.003230`，Hall F1 `+0.031000`。
- 相对 sqrt: AUROC `-0.002034`，Hall AUPR `-0.003710`，Hall F1 `+0.019436`。
