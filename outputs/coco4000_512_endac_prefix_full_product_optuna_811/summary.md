# Shikra / MiniGPT-4 ENDAC-811 full-visual attention×probability (Optuna)

Per layer: `sum_{v in all visual tokens} a_v p_v(y)`; raw and final-Norm vocabulary-softmax variants. Each is concatenated with `log1p(S_E)` and uses validation-selected hyperparameters/checkpoints.

AUROC and HALL-AUPR are three-seed test mean ± population SD, in percent. Feature comparison on this previously examined test split is exploratory.

| Model | Feature | Val AUROC | Test AUROC | Test HALL-AUPR |
|---|---|---:|---:|---:|
| minigpt4_7b | raw_full_product | 93.53 | 93.20 ± 0.04 | 76.24 ± 0.12 |
| minigpt4_7b | norm_full_product | 93.98 | 93.56 ± 0.04 | 77.26 ± 0.29 |
| shikra_7b | raw_full_product | 88.34 | 87.90 ± 0.06 | 68.45 ± 0.16 |
| shikra_7b | norm_full_product | 88.29 | 87.49 ± 0.04 | 68.05 ± 0.09 |

## Validation-selected hyperparameters

| Model | Feature | Width | Dropout | Activation | LR | WD | Batch | Monitor | Epochs 43/44/45 |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| minigpt4_7b | raw_full_product | 128 | 0.1 | relu | 0.00154 | 0 | 64 | val_loss | 20/18/14 |
| minigpt4_7b | norm_full_product | 128 | 0.1 | gelu | 0.000463 | 1e-06 | 64 | val_auroc | 60/44/47 |
| shikra_7b | raw_full_product | 1024 | 0.5 | relu | 0.000696 | 1e-06 | 256 | val_auroc | 35/44/19 |
| shikra_7b | norm_full_product | 1024 | 0.5 | relu | 0.00135 | 1e-05 | 512 | val_auroc | 36/33/31 |

## Input and checkpoint checks

- minigpt4_7b: zero full-product layer-rows raw=0, norm=0.
- shikra_7b: zero full-product layer-rows raw=0, norm=0.
- CPU reload: 12/12 heads passed; maximum probability error 1.79e-07; metric error 0.

Per-seed metrics and predictions are in each model directory. Previously tested prefix AE/Top-K features: `../coco4000_512_endac_prefix_ae_semantic_single_mlp_811/summary.md` (same 24-run budget but frozen candidate grid rather than Optuna).
