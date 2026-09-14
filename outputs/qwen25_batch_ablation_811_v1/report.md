# Qwen2.5-VL-7B：batch size 受控消融

811 图片划分：3200 train / 400 validation / 400 test；本消融只报告 validation，test 未访问。全部 mentions。
特征路径：真实 RMS all-attention `z-A_all→z`，local FP32 Gauss–Legendre K32。V 与 VP+G 各自固定原已选单层 MLP 的其他参数，只改变 batch size。三种子 43/44/45 的算术均值 ± 总体标准差，不是概率 ensemble。
训练仍按固定 epoch/patience 调度，因此 batch 改变时每个 epoch 的 optimizer steps 也随之改变。

| 特征 | batch | validation AUROC | validation HALL-AUPR |
|---|---:|---:|---:|
| visual | 32 | 88.21 ± 0.29 | 49.87 ± 0.75 |
| visual | 64 | 88.35 ± 0.47 | 50.55 ± 0.86 |
| visual | 128 | 88.05 ± 0.17 | 49.75 ± 0.63 |
| visual | 256 | 88.40 ± 0.19 | 51.22 ± 0.48 |
| visual | 512 | 88.17 ± 0.27 | 50.95 ± 0.78 |
| vp_generation | 32 | 86.76 ± 0.15 | 49.23 ± 0.65 |
| vp_generation | 64 | 86.75 ± 0.12 | 49.08 ± 0.76 |
| vp_generation | 128 | 86.96 ± 0.02 | 49.33 ± 0.67 |
| vp_generation | 256 | 86.89 ± 0.16 | 49.26 ± 0.57 |
| vp_generation | 512 | 86.75 ± 0.04 | 48.55 ± 0.46 |
