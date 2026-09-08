# internvl_2_5_8b JFFN-P comparison

## Numerical acceptance

- finite/P-normalized: `True` / max sum error `2.384e-07`
- attention reconstruction: min cosine `0.999996`, max relative error `3.150e-03`
- JVP: max linearity error `1.103e-06`, max full/chunk error `1.741e-06`, median finite-difference cosine `1.000000`
- peak memory: `18.08 GiB`

## Cohort and calibration

- entropy beta fit: 500 fixed train-only images; test leakage `0`; max layer entropy error `1.459e-07`.
- extracted official cohort: 3927 images, 11630 unique positions, 11759 mentions; complete=`True`.

## Preregistered downstream comparison

| Comparison | New AUROC | Old AUROC | ΔAUROC | 95% image-bootstrap CI |
|---|---:|---:|---:|---:|
| new_jffn_vs_old_hpre_cos | 0.8506 | 0.8567 | -0.0060 | [-0.0197, +0.0077] |
| new_jffn_entropy_matched_vs_old_hpre_cos | 0.8552 | 0.8567 | -0.0014 | [-0.0151, +0.0116] |

### All 32 feature sets (mean ± population std over seeds 43/44/45)

| Feature set | AUROC | Real F1 | Hall F1 | Real AUPR | Hall AUPR | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `old_hmid_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.7582 ± 0.0043 | 0.9042 ± 0.0014 | 0.3003 ± 0.0300 | 0.9411 ± 0.0011 | 0.3661 ± 0.0050 | 0.8316 ± 0.0014 |
| `old_hmid_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8525 ± 0.0045 | 0.9141 ± 0.0022 | 0.4947 ± 0.0130 | 0.9662 ± 0.0014 | 0.5198 ± 0.0014 | 0.8531 ± 0.0033 |
| `old_hmid_cos__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.7493 ± 0.0024 | 0.9071 ± 0.0043 | 0.2543 ± 0.0315 | 0.9392 ± 0.0004 | 0.3489 ± 0.0033 | 0.8348 ± 0.0064 |
| `old_hmid_cos__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8501 ± 0.0009 | 0.9165 ± 0.0040 | 0.4895 ± 0.0100 | 0.9659 ± 0.0002 | 0.5175 ± 0.0102 | 0.8565 ± 0.0063 |
| `old_hmid_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.7820 ± 0.0057 | 0.9073 ± 0.0042 | 0.2956 ± 0.1005 | 0.9487 ± 0.0013 | 0.3878 ± 0.0079 | 0.8368 ± 0.0041 |
| `old_hmid_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8474 ± 0.0064 | 0.9147 ± 0.0007 | 0.4699 ± 0.0174 | 0.9644 ± 0.0023 | 0.4995 ± 0.0106 | 0.8531 ± 0.0005 |
| `old_hmid_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.7738 ± 0.0041 | 0.9048 ± 0.0021 | 0.3422 ± 0.0024 | 0.9468 ± 0.0006 | 0.3731 ± 0.0042 | 0.8337 ± 0.0033 |
| `old_hmid_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8456 ± 0.0028 | 0.9132 ± 0.0010 | 0.4616 ± 0.0252 | 0.9646 ± 0.0008 | 0.4872 ± 0.0092 | 0.8505 ± 0.0012 |
| `old_hpre_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.7620 ± 0.0026 | 0.9086 ± 0.0039 | 0.2802 ± 0.0408 | 0.9420 ± 0.0017 | 0.3718 ± 0.0017 | 0.8380 ± 0.0053 |
| `old_hpre_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8488 ± 0.0043 | 0.9146 ± 0.0013 | 0.4663 ± 0.0272 | 0.9652 ± 0.0010 | 0.5028 ± 0.0025 | 0.8529 ± 0.0010 |
| `old_hpre_cos__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.7590 ± 0.0056 | 0.9076 ± 0.0005 | 0.3245 ± 0.0112 | 0.9409 ± 0.0017 | 0.3663 ± 0.0060 | 0.8375 ± 0.0010 |
| `old_hpre_cos__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8484 ± 0.0027 | 0.9155 ± 0.0009 | 0.4799 ± 0.0297 | 0.9653 ± 0.0009 | 0.5098 ± 0.0051 | 0.8547 ± 0.0009 |
| `old_hpre_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.7837 ± 0.0013 | 0.9055 ± 0.0032 | 0.3394 ± 0.0230 | 0.9474 ± 0.0006 | 0.3941 ± 0.0019 | 0.8347 ± 0.0044 |
| `old_hpre_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8421 ± 0.0037 | 0.9168 ± 0.0007 | 0.4568 ± 0.0215 | 0.9634 ± 0.0010 | 0.4919 ± 0.0037 | 0.8557 ± 0.0009 |
| `old_hpre_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.7750 ± 0.0013 | 0.9048 ± 0.0017 | 0.3592 ± 0.0086 | 0.9449 ± 0.0006 | 0.3787 ± 0.0062 | 0.8342 ± 0.0027 |
| `old_hpre_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8399 ± 0.0036 | 0.9137 ± 0.0020 | 0.4581 ± 0.0118 | 0.9631 ± 0.0009 | 0.4866 ± 0.0042 | 0.8512 ± 0.0026 |
| `new_jffn__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.7377 ± 0.0035 | 0.8963 ± 0.0025 | 0.2750 ± 0.0185 | 0.9348 ± 0.0008 | 0.3230 ± 0.0096 | 0.8186 ± 0.0043 |
| `new_jffn__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8405 ± 0.0028 | 0.9120 ± 0.0016 | 0.4785 ± 0.0154 | 0.9626 ± 0.0011 | 0.4838 ± 0.0036 | 0.8494 ± 0.0029 |
| `new_jffn__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.7418 ± 0.0031 | 0.9042 ± 0.0029 | 0.2203 ± 0.0412 | 0.9360 ± 0.0010 | 0.3277 ± 0.0099 | 0.8295 ± 0.0039 |
| `new_jffn__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8379 ± 0.0051 | 0.9092 ± 0.0010 | 0.4542 ± 0.0119 | 0.9607 ± 0.0009 | 0.4707 ± 0.0165 | 0.8443 ± 0.0019 |
| `new_jffn__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.7168 ± 0.0040 | 0.9080 ± 0.0040 | 0.2237 ± 0.0516 | 0.9262 ± 0.0006 | 0.3377 ± 0.0080 | 0.8356 ± 0.0054 |
| `new_jffn__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8398 ± 0.0022 | 0.9145 ± 0.0034 | 0.4676 ± 0.0122 | 0.9625 ± 0.0012 | 0.4830 ± 0.0080 | 0.8527 ± 0.0053 |
| `new_jffn__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.7115 ± 0.0013 | 0.9021 ± 0.0049 | 0.2563 ± 0.0271 | 0.9256 ± 0.0009 | 0.3326 ± 0.0051 | 0.8271 ± 0.0070 |
| `new_jffn__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8378 ± 0.0025 | 0.9122 ± 0.0013 | 0.4612 ± 0.0066 | 0.9624 ± 0.0003 | 0.4667 ± 0.0089 | 0.8489 ± 0.0019 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.7614 ± 0.0016 | 0.9077 ± 0.0029 | 0.3035 ± 0.0403 | 0.9388 ± 0.0004 | 0.3842 ± 0.0040 | 0.8370 ± 0.0037 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8445 ± 0.0014 | 0.9173 ± 0.0018 | 0.4753 ± 0.0393 | 0.9635 ± 0.0010 | 0.5094 ± 0.0134 | 0.8572 ± 0.0038 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.7572 ± 0.0033 | 0.9116 ± 0.0035 | 0.2317 ± 0.0476 | 0.9378 ± 0.0012 | 0.3725 ± 0.0103 | 0.8415 ± 0.0045 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8490 ± 0.0022 | 0.9198 ± 0.0026 | 0.4696 ± 0.0384 | 0.9654 ± 0.0008 | 0.5235 ± 0.0108 | 0.8607 ± 0.0048 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.7487 ± 0.0049 | 0.9162 ± 0.0008 | 0.1771 ± 0.0560 | 0.9370 ± 0.0024 | 0.3835 ± 0.0047 | 0.8480 ± 0.0009 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8337 ± 0.0027 | 0.9129 ± 0.0009 | 0.4326 ± 0.0176 | 0.9617 ± 0.0005 | 0.4696 ± 0.0093 | 0.8491 ± 0.0010 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.7373 ± 0.0026 | 0.9156 ± 0.0003 | 0.1883 ± 0.0371 | 0.9320 ± 0.0013 | 0.3731 ± 0.0041 | 0.8471 ± 0.0009 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8333 ± 0.0054 | 0.9135 ± 0.0006 | 0.4468 ± 0.0135 | 0.9609 ± 0.0020 | 0.4740 ± 0.0033 | 0.8503 ± 0.0014 |

### Preregistered per-seed AUROC deltas

| Comparison | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| new_jffn vs old_hpre_cos | -0.0097 | -0.0054 | -0.0098 |
| new_jffn_entropy_matched vs old_hpre_cos | -0.0046 | +0.0028 | -0.0111 |

机器可读的逐 seed 全指标见 `training_results.json`。

## P 本身的验证

- 空间验证：9392 个 REAL mention 有同类别 COCO box；601 个无匹配 box；0 个网格不匹配。
- 分布曲线：`p_distribution_entropy.png`。
- 同图不同目标特异性：`p_target_specificity_js.png`。
- COCO 空间富集：`p_spatial_enrichment.png`。
- 全指标面板：`p_distribution_all_metrics.png`、`p_target_specificity_all_metrics.png`、`p_spatial_all_metrics.png`。
