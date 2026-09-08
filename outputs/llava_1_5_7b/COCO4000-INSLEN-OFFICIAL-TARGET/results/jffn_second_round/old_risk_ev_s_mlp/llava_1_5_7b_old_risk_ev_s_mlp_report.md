# llava_1_5_7b: old risk + EV + S 三层 MLP

输入为 32 层 old-hpre raw-logit Gaussian / sqrt-matched-state OT risk、32 层 mass×cosine EV、32 层 Jacobian sensitivity S，合计 96 维。

MLP 使用三个隐藏层 [128,64,32]；split、seeds 43/44/45、batch 256、最多 100 epochs、train-loss checkpoint、train-F1 threshold、无特征归一化均与现有 JFFN 对照实验一致。

## 三种子结果

| Seed | AUROC | Real F1 | Hall F1 | Hall AUPR | Best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 0.894659 | 0.898700 | 0.660208 | 0.679334 | 97 |
| 44 | 0.889645 | 0.900676 | 0.655784 | 0.669656 | 99 |
| 45 | 0.894958 | 0.894583 | 0.660858 | 0.678001 | 99 |

## 与 old risk + EV 对比

| Feature | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: |
| old risk + EV | 0.888141 ± 0.001422 | 0.656175 ± 0.005976 | 0.893401 | 0.668119 |
| old risk + EV + S | 0.893087 ± 0.002437 | 0.675664 ± 0.004283 | 0.902881 | 0.694670 |

Seed-ensemble paired image bootstrap：AUROC 差值 +0.009480，95% CI [+0.001897,+0.017336]；Hall AUPR 差值 +0.026551，95% CI [-0.000633,+0.053685]。

注意：baseline 直接复用协议完全相同的既有训练及预测，不重新随机训练；因此比较只新增 S 这一组输入。
