# llava_1_5_7b: aggregate I/R 曲线与幻觉检测消融

I/R 均为 32 层 aggregate norm。所有分类器使用相同的 `[128,64,32]` 三隐藏层 MLP、图片级 split、seeds 43/44/45、无特征标准化；REAL 为正类。

## 三 seed 结果

| Feature | Dim | AUROC | Hall AUPR | Hall F1 | Ensemble AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| I only | 32 | 0.877711 ± 0.001584 | 0.642977 | 0.591980 | 0.879825 |
| R only | 32 | 0.874441 ± 0.000459 | 0.645850 | 0.559944 | 0.877044 |
| I + R | 64 | 0.881668 ± 0.001390 | 0.648914 | 0.604165 | 0.884686 |
| Original risk + EV | 64 | 0.888141 ± 0.001422 | 0.656175 | 0.644332 | 0.893401 |
| Original risk + EV + I | 96 | 0.890281 ± 0.002556 | 0.668572 | 0.643421 | 0.893927 |
| Original risk + EV + R | 96 | 0.893901 ± 0.000307 | 0.666282 | 0.641042 | 0.896988 |
| Original risk + EV + I + R | 128 | 0.892193 ± 0.001693 | 0.660835 | 0.635218 | 0.897496 |
| P_JFFN risk + EV | 64 | 0.876717 ± 0.004153 | 0.659444 | 0.611740 | 0.883175 |
| P_JFFN risk + EV + I | 96 | 0.887550 ± 0.000801 | 0.679365 | 0.634203 | 0.890168 |
| P_JFFN risk + EV + R | 96 | 0.888169 ± 0.000450 | 0.667310 | 0.631459 | 0.892118 |
| P_JFFN risk + EV + I + R | 128 | 0.885945 ± 0.003733 | 0.661473 | 0.631002 | 0.891920 |

## 不确定性估计

- 按本次实验要求未运行 bootstrap；仅报告三个随机种子的均值与标准差。

## I/R 曲线摘要

- I：HALL>REAL / REAL>HALL 层数=6/26；最强 L1，Hall−Real=-0.098628，d=-1.0566，最佳方向单层 AUROC=0.7796。
- R：HALL>REAL / REAL>HALL 层数=5/27；最强 L11，Hall−Real=-0.958545，d=-1.0034，最佳方向单层 AUROC=0.7732。

## 协议核验

- train/test images：3174/797。
- train/test mentions：12317/3146。
- 原始 risk 与 P_JFFN risk 共享同一个 hpre raw-logit Gaussian target、sqrt-matched-state cost、Union Top-K 和 mass×cosine EV。
- 曲线置信带是 mention-level 描述性 95% CI；分类性能仅报告三个随机种子。
