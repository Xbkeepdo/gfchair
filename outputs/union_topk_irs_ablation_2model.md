# Union-TopK 区域 I/R/S：两模型汇总

`I_U=Σ||a_j||`、`R_U=Σ||Ja_j||`、`S_U=R_U/I_U`；三 seed、同一 MLP/split，无标准化，不做 bootstrap。

## llava_1_5_7b

| Feature | AUROC | Hall AUPR | Hall F1 |
| --- | ---: | ---: | ---: |
| Union I only | 0.873471 ± 0.000650 | 0.641057 | 0.581208 |
| Union R only | 0.878886 ± 0.001136 | 0.651995 | 0.604126 |
| Union S only | 0.874894 ± 0.002045 | 0.625940 | 0.607913 |
| Union I + R + S | 0.884852 ± 0.001289 | 0.650374 | 0.627091 |
| Original risk + EV | 0.888141 ± 0.001422 | 0.656175 | 0.644332 |
| Original risk + EV + Union I | 0.888888 ± 0.000825 | 0.667713 | 0.658052 |
| Original risk + EV + Union R | 0.893297 ± 0.000901 | 0.663003 | 0.638730 |
| Original risk + EV + Union S | 0.891793 ± 0.001312 | 0.672379 | 0.640199 |
| Original risk + EV + Union I + R + S | 0.894870 ± 0.002304 | 0.670014 | 0.649803 |
| P_JFFN risk + EV | 0.876717 ± 0.004153 | 0.659444 | 0.611740 |
| P_JFFN risk + EV + Union I | 0.883542 ± 0.000424 | 0.667929 | 0.612375 |
| P_JFFN risk + EV + Union R | 0.887541 ± 0.000623 | 0.664790 | 0.621674 |
| P_JFFN risk + EV + Union S | 0.888278 ± 0.005211 | 0.673936 | 0.620983 |
| P_JFFN risk + EV + Union I + R + S | 0.890868 ± 0.000634 | 0.674085 | 0.633513 |

## internvl_2_5_8b

| Feature | AUROC | Hall AUPR | Hall F1 |
| --- | ---: | ---: | ---: |
| Union I only | 0.827443 ± 0.003186 | 0.476635 | 0.360788 |
| Union R only | 0.832901 ± 0.003799 | 0.484103 | 0.287482 |
| Union S only | 0.848491 ± 0.001019 | 0.521512 | 0.374066 |
| Union I + R + S | 0.854411 ± 0.004653 | 0.520276 | 0.431676 |
| Original risk + EV | 0.848825 ± 0.004339 | 0.502827 | 0.466311 |
| Original risk + EV + Union I | 0.849898 ± 0.002931 | 0.520156 | 0.433004 |
| Original risk + EV + Union R | 0.859690 ± 0.002624 | 0.533008 | 0.459017 |
| Original risk + EV + Union S | 0.866605 ± 0.001525 | 0.552289 | 0.500021 |
| Original risk + EV + Union I + R + S | 0.867020 ± 0.002442 | 0.545717 | 0.490279 |
| P_JFFN risk + EV | 0.840525 ± 0.002801 | 0.483769 | 0.478527 |
| P_JFFN risk + EV + Union I | 0.846019 ± 0.003254 | 0.508750 | 0.414824 |
| P_JFFN risk + EV + Union R | 0.856753 ± 0.002742 | 0.519059 | 0.458675 |
| P_JFFN risk + EV + Union S | 0.852447 ± 0.002957 | 0.508563 | 0.482223 |
| P_JFFN risk + EV + Union I + R + S | 0.862542 ± 0.001021 | 0.532282 | 0.480706 |
