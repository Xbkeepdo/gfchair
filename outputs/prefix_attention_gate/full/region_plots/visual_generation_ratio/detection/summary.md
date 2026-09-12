# Visual / generated-text attention ratio detection

每个目标每层先求比值，原值直接输入MLP，不取log、不截断、不加epsilon。raw attention与attention×gate分别训练。原COCO4000/3200-800/seeds43-45；均值±总体标准差（%），F1按各seed训练REAL-F1阈值。

| Model | Input | AUROC | HALL AUPR | HALL F1 |
|---|---|---:|---:|---:|
| minigpt4_7b | attention_ratio | 88.335 ± 0.373 | 57.110 ± 1.016 | 43.202 ± 3.140 |
| minigpt4_7b | gated_ratio | 88.862 ± 0.289 | 61.249 ± 0.912 | 47.510 ± 2.094 |
| minigpt4_7b | F_reference | 90.907 ± 0.148 | 65.486 ± 1.035 | 59.354 ± 1.769 |
| shikra_7b | attention_ratio | 84.130 ± 0.142 | 56.813 ± 0.044 | 42.586 ± 2.623 |
| shikra_7b | gated_ratio | 83.279 ± 0.087 | 55.498 ± 0.519 | 46.513 ± 3.374 |
| shikra_7b | F_reference | 86.265 ± 0.308 | 60.481 ± 0.794 | 55.992 ± 1.056 |
| qwen2_5_vl_7b | attention_ratio | 78.291 ± 0.659 | 25.653 ± 0.706 | 6.190 ± 0.476 |
| qwen2_5_vl_7b | gated_ratio | 81.228 ± 0.157 | 29.848 ± 0.939 | 10.635 ± 3.353 |
| qwen2_5_vl_7b | F_reference | 87.449 ± 0.085 | 43.982 ± 1.678 | 38.401 ± 3.469 |
| llava_1_5_7b | attention_ratio | 87.571 ± 0.096 | 65.296 ± 0.294 | 57.367 ± 3.865 |
| llava_1_5_7b | gated_ratio | 87.235 ± 0.260 | 64.988 ± 0.547 | 57.008 ± 0.260 |
| llava_1_5_7b | F_reference | 90.015 ± 0.272 | 70.815 ± 0.619 | 66.788 ± 0.073 |
| qwen3_vl_8b | attention_ratio | 84.639 ± 0.284 | 53.469 ± 0.566 | 37.069 ± 5.775 |
| qwen3_vl_8b | gated_ratio | 85.472 ± 0.421 | 55.617 ± 0.687 | 40.345 ± 2.601 |
| qwen3_vl_8b | F_reference | 88.655 ± 0.305 | 62.832 ± 0.353 | 58.839 ± 1.088 |
| internvl_2_5_8b | attention_ratio | 82.514 ± 0.546 | 43.720 ± 1.277 | 25.874 ± 3.490 |
| internvl_2_5_8b | gated_ratio | 83.326 ± 0.109 | 45.172 ± 0.570 | 24.772 ± 3.344 |
| internvl_2_5_8b | F_reference | 85.825 ± 0.587 | 53.034 ± 1.201 | 51.378 ± 0.362 |
