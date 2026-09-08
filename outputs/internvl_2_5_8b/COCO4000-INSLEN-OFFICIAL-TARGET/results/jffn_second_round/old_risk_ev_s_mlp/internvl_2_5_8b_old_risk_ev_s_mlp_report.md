# internvl_2_5_8b: old risk + EV + S 三层 MLP

输入为 32 层 old-hpre raw-logit Gaussian / sqrt-matched-state OT risk、32 层 mass×cosine EV、32 层 Jacobian sensitivity S，合计 96 维。

MLP 使用三个隐藏层 [128,64,32]；split、seeds 43/44/45、batch 256、最多 100 epochs、train-loss checkpoint、train-F1 threshold、无特征归一化均与现有 JFFN 对照实验一致。

## 三种子结果

| Seed | AUROC | Real F1 | Hall F1 | Hall AUPR | Best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 0.868717 | 0.914710 | 0.518828 | 0.559909 | 98 |
| 44 | 0.867741 | 0.915580 | 0.509299 | 0.529682 | 96 |
| 45 | 0.863776 | 0.915438 | 0.504323 | 0.548925 | 99 |

## 与 old risk + EV 对比

| Feature | Mean AUROC | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: |
| old risk + EV | 0.848825 ± 0.004339 | 0.502827 ± 0.002522 | 0.856685 | 0.522819 |
| old risk + EV + S | 0.866745 ± 0.002137 | 0.546172 ± 0.012493 | 0.874746 | 0.565589 |

Seed-ensemble paired image bootstrap：AUROC 差值 +0.018060，95% CI [+0.004740,+0.031558]；Hall AUPR 差值 +0.042769，95% CI [+0.009725,+0.076242]。

注意：baseline 直接复用协议完全相同的既有训练及预测，不重新随机训练；因此比较只新增 S 这一组输入。
