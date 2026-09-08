# LLaVA-1.5-7B seed 43：100 vs 200 epoch

## 口径

- 数据、图片级 8:2 split、seed 43、三层 Torch MLP、优化器和阈值协议均相同。
- 方法只比较当前 16 个 `risk` / `risk+EV` 特征组，不包含 `ADS+CGC`。
- baseline 为 SVAR、MetaToken 的 shared Torch MLP 版本；native-paper head 不受 `--num-epochs` 控制，因此未混入本表。
- Real-F1、Hall-F1 使用 train Real-F1 选择的阈值；AUROC 与阈值无关。
- `best/run` 表示最低训练损失所在 epoch / 实际执行 epoch。没有 validation set，checkpoint 依据最低 train loss 选择。

## 结论

- 16 个方法分支从 100 增至最多 200 epoch 后，平均 train loss 下降 `0.02719`、平均 train AUROC 上升 `0.01662`，但平均 test AUROC 下降 `0.00165`，Real-F1 下降 `0.00235`。继续拟合训练集没有转化成稳定的测试收益。
- risk-only 的平均 test AUROC 变化为 `-0.00040`；risk+EV 为 `-0.00291`。EV 仍然有效，但无需依赖 200 epoch 才体现效果。
- 200-epoch 方法中，最高 test AUROC 为 raw-logit、P=hmid、cosine cost 的 `risk+EV`：`0.8894`。它在第 83 epoch 已达到最低 train loss，并在第 93 epoch early-stop，所以与原 100-epoch结果完全相同。
- 200-epoch SVAR 的 train AUROC 从 `0.9949` 升至 `0.9999`，test AUROC 从 `0.8853` 降至 `0.8799`，是明显过拟合。MetaToken 在第 47 epoch 达到最低 train loss、第 57 epoch early-stop，因此 100/200 结果完全相同。

## 我们的方法

| Gate / P-source / 特征 | Best/run 100 | Best/run 200 | Train loss 100→200 | Test AUROC 100→200 | Real-F1 100→200 | Hall-F1 100→200 |
|---|---:|---:|---:|---:|---:|---:|
| raw / hmid / sqrt risk | 98/100 | 195/200 | 0.3397→0.3001 | 0.8649→0.8633 | 0.8873→0.8805 | 0.5576→0.5907 |
| raw / hmid / sqrt risk+EV | 97/100 | 137/147 | 0.2747→0.2569 | 0.8849→0.8848 | 0.8987→0.8987 | 0.6350→0.6442 |
| raw / hmid / cosine risk | 100/100 | 195/200 | 0.3445→0.3050 | 0.8619→0.8640 | 0.8846→0.8813 | 0.5491→0.5750 |
| raw / hmid / cosine risk+EV | 83/93 | 83/93 | 0.2933→0.2933 | 0.8894→0.8894 | 0.8987→0.8987 | 0.6313→0.6313 |
| raw / hpre / sqrt risk | 96/100 | 131/141 | 0.3444→0.3297 | 0.8693→0.8679 | 0.8881→0.8869 | 0.5975→0.6085 |
| raw / hpre / sqrt risk+EV | 99/100 | 191/200 | 0.2722→0.2175 | 0.8886→0.8818 | 0.8975→0.8960 | 0.6230→0.6231 |
| raw / hpre / cosine risk | 100/100 | 200/200 | 0.3429→0.2961 | 0.8630→0.8609 | 0.8862→0.8815 | 0.5712→0.5845 |
| raw / hpre / cosine risk+EV | 99/100 | 155/165 | 0.2758→0.2508 | 0.8888→0.8868 | 0.8961→0.8934 | 0.6442→0.6287 |
| softmax / hmid / sqrt risk | 98/100 | 148/158 | 0.3331→0.3178 | 0.8534→0.8528 | 0.8882→0.8825 | 0.5677→0.5828 |
| softmax / hmid / sqrt risk+EV | 100/100 | 157/167 | 0.2667→0.2239 | 0.8845→0.8780 | 0.8958→0.8918 | 0.6182→0.6101 |
| softmax / hmid / cosine risk | 85/95 | 85/95 | 0.3412→0.3412 | 0.8525→0.8525 | 0.8868→0.8868 | 0.5338→0.5338 |
| softmax / hmid / cosine risk+EV | 98/100 | 174/184 | 0.2721→0.2220 | 0.8813→0.8791 | 0.8943→0.8893 | 0.6078→0.6155 |
| softmax / hpre / sqrt risk | 98/100 | 106/116 | 0.3418→0.3383 | 0.8608→0.8611 | 0.8879→0.8878 | 0.5679→0.5766 |
| softmax / hpre / sqrt risk+EV | 99/100 | 146/156 | 0.2694→0.2346 | 0.8873→0.8838 | 0.8945→0.8924 | 0.6385→0.6380 |
| softmax / hpre / cosine risk | 69/79 | 69/79 | 0.3547→0.3547 | 0.8579→0.8579 | 0.8864→0.8864 | 0.5860→0.5860 |
| softmax / hpre / cosine risk+EV | 100/100 | 173/183 | 0.2723→0.2219 | 0.8815→0.8796 | 0.8937→0.8933 | 0.6086→0.6277 |

## Shared-MLP baselines

| Baseline | Best/run 100 | Best/run 200 | Train loss 100→200 | Train AUROC 100→200 | Test AUROC 100→200 | Real-F1 100→200 | Hall-F1 100→200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 100/100 | 200/200 | 0.1379→0.0484 | 0.9949→0.9999 | 0.8853→0.8799 | 0.8995→0.8963 | 0.6510→0.6351 |
| MetaToken | 47/57 | 47/57 | 0.3694→0.3694 | 0.8685→0.8685 | 0.8705→0.8705 | 0.9044→0.9044 | 0.5838→0.5838 |

## 200-epoch 横向结果

- 最高 AUROC 方法：raw / hmid / cosine risk+EV，AUROC `0.8894`、Real-F1 `0.8987`、Hall-F1 `0.6313`。
- Hall-F1 更均衡的方法：raw / hmid / sqrt risk+EV，AUROC `0.8848`、Real-F1 `0.8987`、Hall-F1 `0.6442`。
- SVAR：AUROC `0.8799`、Real-F1 `0.8963`、Hall-F1 `0.6351`。
- MetaToken：AUROC `0.8705`、Real-F1 `0.9044`、Hall-F1 `0.5838`。

因此，在相同 shared MLP 下，我们的方法最高 AUROC 比 SVAR 高 `0.0095`、比 MetaToken 高 `0.0189`。若选择 raw / hmid / sqrt risk+EV，则 AUROC 和 Hall-F1 都高于 SVAR，分别高 `0.0049` 和 `0.0091`。
