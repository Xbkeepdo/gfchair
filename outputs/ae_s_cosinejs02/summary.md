# AE + log1p(S) + JS(softmax(cos(e_m,G(Z)-G(Z0))/0.2), T)

每层计算softmax(cos(e_m,G(Z)-G(Z0))/0.2)与同源T的JS散度（自然对数），再与AE/log1p(S)拼接。固定COCO4000，分别计算seeds 43/44/45的指标，再报告均值±总体标准差（ddof=0），不使用ensemble。F1阈值由各seed训练集REAL-F1选择。

| Model | Input | AUROC mean ± std | HALL AUPR mean ± std | HALL F1 mean ± std (train-REAL-F1) |
|---|---|---:|---:|---:|
| minigpt4_7b | F | 0.909065 ± 0.001480 | 0.654855 ± 0.010351 | 0.593538 ± 0.017687 |
| minigpt4_7b | F_CosineJS02 | 0.901880 ± 0.003258 | 0.651935 ± 0.004541 | 0.591230 ± 0.032029 |
| shikra_7b | F | 0.862647 ± 0.003076 | 0.604807 ± 0.007945 | 0.559916 ± 0.010562 |
| shikra_7b | F_CosineJS02 | 0.856436 ± 0.005258 | 0.599189 ± 0.000569 | 0.530144 ± 0.016119 |
