# AE + log1p(S) + JS(softmax(Q/0.2), T)

每层计算softmax(raw signed Q/0.2)与同源T的JS散度（自然对数），再与AE/log1p(S)拼接。固定COCO4000，分别计算seeds 43/44/45的指标，再报告均值±总体标准差（ddof=0），不使用ensemble。F1阈值由各seed训练集REAL-F1选择。

| Model | Input | AUROC mean ± std | HALL AUPR mean ± std | HALL F1 mean ± std (train-REAL-F1) |
|---|---|---:|---:|---:|
| minigpt4_7b | F | 0.909065 ± 0.001480 | 0.654855 ± 0.010351 | 0.593538 ± 0.017687 |
| minigpt4_7b | F_QJS02 | 0.897706 ± 0.001309 | 0.638029 ± 0.005955 | 0.600570 ± 0.021785 |
| shikra_7b | F | 0.862647 ± 0.003076 | 0.604807 ± 0.007945 | 0.559916 ± 0.010562 |
| shikra_7b | F_QJS02 | 0.861438 ± 0.004240 | 0.613644 ± 0.008214 | 0.558259 ± 0.015437 |
