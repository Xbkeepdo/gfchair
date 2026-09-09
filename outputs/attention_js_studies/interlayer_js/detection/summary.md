# 四模型层间 JS 幻觉检测汇总

严格图片级 3200/800 split；唯一无冲突 target token；三层 MLP [128,64,32]；seeds 43/44/45；无特征归一化。

| Model | Best feature | Dim | AUROC mean±std | Ensemble AUROC | Hall AUPR | Hall prior | Hall F1 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | `adj_A_all` | 31 | 0.8540±0.0008 | 0.8575 | 0.6244 | 0.2282 | 0.5598 |
| Qwen2.5-VL-7B | `adj_A_all` | 27 | 0.7787±0.0048 | 0.7841 | 0.2780 | 0.1063 | 0.1932 |
| Qwen3-VL-8B | `adj_E_all` | 35 | 0.7908±0.0043 | 0.7962 | 0.4727 | 0.1725 | 0.3640 |
| InternVL2.5-8B | `adj_A_all` | 31 | 0.7945±0.0011 | 0.8002 | 0.3927 | 0.1580 | 0.2540 |

四组 A/E × all/Top32 特征完全独立训练；没有特征融合。单块通常是 all-token 优于 Top32。

该实验只评估层间 JS standalone detector；没有把 JS 与现有 risk+EV 拼接，因此不能据此判断增量价值。
