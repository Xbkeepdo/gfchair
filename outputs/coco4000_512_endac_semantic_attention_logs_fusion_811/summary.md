# Semantic-attention + log1p(S_E) ENDAC-811

固定 3200/400/400；seeds 43/44/45；train-only z-score；
单隐藏 128/ReLU/dropout .3、无 BN；validation 选择在 test 前冻结。

| model | validation champion | fusion AUROC (%) | fusion AP (%) |
|---|---|---:|---:|
| qwen2_5_vl_7b | norm_top32/all_six+logS | 84.31 ± 0.44 | 53.29 ± 1.24 |
| llava_1_5_7b | norm_top32/all_six+logS | 92.87 ± 0.15 | 78.22 ± 0.19 |
| qwen3_vl_8b | raw_top32/all_six+logS | 90.92 ± 0.31 | 73.64 ± 0.73 |
| internvl_2_5_8b | norm_top32/all_six+logS | 90.17 ± 0.21 | 67.08 ± 0.37 |

| model | standalone AUROC / AP | logS-only AUROC / AP | fusion−standalone (pp) | fusion−logS (pp) |
|---|---:|---:|---:|---:|
| qwen2_5_vl_7b | 82.54 / 45.92 | 81.93 / 38.69 | +1.77 / +7.37 | +2.38 / +14.61 |
| llava_1_5_7b | 91.79 / 74.85 | 88.97 / 69.84 | +1.08 / +3.38 | +3.90 / +8.38 |
| qwen3_vl_8b | 88.60 / 72.05 | 88.46 / 68.46 | +2.32 / +1.59 | +2.46 / +5.18 |
| internvl_2_5_8b | 89.79 / 64.84 | 86.49 / 60.19 | +0.38 / +2.23 | +3.68 / +6.88 |

逐模型 champion 宏平均：fusion 89.57/68.06；standalone 88.18/64.42；logS-only 86.46/59.30。
