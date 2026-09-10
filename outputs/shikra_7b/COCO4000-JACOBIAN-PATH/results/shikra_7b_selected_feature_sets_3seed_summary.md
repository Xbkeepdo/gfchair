# shikra_7b method + ADS/CGC 3-seed Torch MLP summary

Seeds: `43, 44, 45`. Values are population mean+/-std.
Real is the headline positive class; hallucination-positive metrics are reported alongside it.
Strict 8:2 has no validation set: train-loss early stopping restores the minimum-train-loss checkpoint.
Every checkpoint is reported twice: fixed threshold 0.5 and a Real-F1 threshold selected on train only.
Real/Hall AUC values are equal under score inversion, while AUPR differs; hallucination metrics use the complementary prediction at the same fixed boundary.

## Best by model

| Model | Threshold mode | Best feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shikra_7b | fixed_0.5 | `ads+cgc` | 0.500+/-0.000 | 0.790+/-0.008 | 0.852+/-0.016 | 0.884+/-0.034 | 0.867+/-0.008 | 0.798+/-0.002 | 0.928+/-0.001 | 0.545+/-0.029 | 0.468+/-0.087 | 0.496+/-0.043 | 0.798+/-0.002 | 0.514+/-0.006 |
| shikra_7b | train_f1 | `ads+cgc` | 0.501+/-0.065 | 0.789+/-0.007 | 0.847+/-0.005 | 0.889+/-0.018 | 0.867+/-0.006 | 0.798+/-0.002 | 0.928+/-0.001 | 0.540+/-0.023 | 0.445+/-0.032 | 0.487+/-0.012 | 0.798+/-0.002 | 0.514+/-0.006 |

## shikra_7b

| Threshold mode | Rank | Feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.5 | 1 | `ads+cgc` | 0.500+/-0.000 | 0.790+/-0.008 | 0.852+/-0.016 | 0.884+/-0.034 | 0.867+/-0.008 | 0.798+/-0.002 | 0.928+/-0.001 | 0.545+/-0.029 | 0.468+/-0.087 | 0.496+/-0.043 | 0.798+/-0.002 | 0.514+/-0.006 |
| train_f1 | 1 | `ads+cgc` | 0.501+/-0.065 | 0.789+/-0.007 | 0.847+/-0.005 | 0.889+/-0.018 | 0.867+/-0.006 | 0.798+/-0.002 | 0.928+/-0.001 | 0.540+/-0.023 | 0.445+/-0.032 | 0.487+/-0.012 | 0.798+/-0.002 | 0.514+/-0.006 |
