# minigpt4_7b method + ADS/CGC 3-seed Torch MLP summary

Seeds: `43, 44, 45`. Values are population mean+/-std.
Real is the headline positive class; hallucination-positive metrics are reported alongside it.
Strict 8:2 has no validation set: train-loss early stopping restores the minimum-train-loss checkpoint.
Every checkpoint is reported twice: fixed threshold 0.5 and a Real-F1 threshold selected on train only.
Real/Hall AUC values are equal under score inversion, while AUPR differs; hallucination metrics use the complementary prediction at the same fixed boundary.

## Best by model

| Model | Threshold mode | Best feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minigpt4_7b | fixed_0.5 | `ads+cgc` | 0.500+/-0.000 | 0.866+/-0.003 | 0.881+/-0.004 | 0.970+/-0.003 | 0.924+/-0.002 | 0.874+/-0.001 | 0.968+/-0.001 | 0.692+/-0.013 | 0.338+/-0.027 | 0.453+/-0.026 | 0.874+/-0.001 | 0.588+/-0.002 |
| minigpt4_7b | train_f1 | `ads+cgc` | 0.527+/-0.019 | 0.867+/-0.001 | 0.890+/-0.004 | 0.960+/-0.005 | 0.924+/-0.001 | 0.874+/-0.001 | 0.968+/-0.001 | 0.665+/-0.015 | 0.398+/-0.027 | 0.497+/-0.018 | 0.874+/-0.001 | 0.588+/-0.002 |

## minigpt4_7b

| Threshold mode | Rank | Feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.5 | 1 | `ads+cgc` | 0.500+/-0.000 | 0.866+/-0.003 | 0.881+/-0.004 | 0.970+/-0.003 | 0.924+/-0.002 | 0.874+/-0.001 | 0.968+/-0.001 | 0.692+/-0.013 | 0.338+/-0.027 | 0.453+/-0.026 | 0.874+/-0.001 | 0.588+/-0.002 |
| train_f1 | 1 | `ads+cgc` | 0.527+/-0.019 | 0.867+/-0.001 | 0.890+/-0.004 | 0.960+/-0.005 | 0.924+/-0.001 | 0.874+/-0.001 | 0.968+/-0.001 | 0.665+/-0.015 | 0.398+/-0.027 | 0.497+/-0.018 | 0.874+/-0.001 | 0.588+/-0.002 |
