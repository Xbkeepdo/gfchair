# llava_1_5_7b JFFN-P comparison

## Numerical acceptance

- finite/P-normalized: `True` / max sum error `2.384e-07`
- attention reconstruction: min cosine `1.000000`, max relative error `4.835e-04`
- JVP: max linearity error `2.022e-06`, max full/chunk error `2.419e-06`, median finite-difference cosine `1.000000`
- peak memory: `15.30 GiB`

## Cohort and calibration

- entropy beta fit: 500 fixed train-only images; test leakage `0`; max layer entropy error `1.273e-07`.
- extracted official cohort: 3971 images, 14951 unique positions, 15463 mentions; complete=`True`.

## Preregistered downstream comparison

| Comparison | New AUROC | Old AUROC | ΔAUROC | 95% image-bootstrap CI |
|---|---:|---:|---:|---:|
| new_jffn_vs_old_hpre_cos | 0.8832 | 0.8934 | -0.0102 | [-0.0198, -0.0012] |
| new_jffn_entropy_matched_vs_old_hpre_cos | 0.8872 | 0.8934 | -0.0062 | [-0.0142, +0.0015] |

### All 32 feature sets (mean ± population std over seeds 43/44/45)

| Feature set | AUROC | Real F1 | Hall F1 | Real AUPR | Hall AUPR | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `old_hmid_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.8646 ± 0.0010 | 0.8917 ± 0.0019 | 0.5684 ± 0.0157 | 0.9565 ± 0.0005 | 0.6223 ± 0.0047 | 0.8269 ± 0.0037 |
| `old_hmid_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8863 ± 0.0026 | 0.8967 ± 0.0043 | 0.6262 ± 0.0089 | 0.9641 ± 0.0008 | 0.6528 ± 0.0083 | 0.8382 ± 0.0047 |
| `old_hmid_cos__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.8639 ± 0.0023 | 0.8869 ± 0.0021 | 0.5897 ± 0.0047 | 0.9567 ± 0.0010 | 0.6197 ± 0.0056 | 0.8226 ± 0.0029 |
| `old_hmid_cos__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8849 ± 0.0017 | 0.8934 ± 0.0010 | 0.6226 ± 0.0123 | 0.9639 ± 0.0004 | 0.6485 ± 0.0087 | 0.8338 ± 0.0016 |
| `old_hmid_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.8558 ± 0.0023 | 0.8868 ± 0.0018 | 0.5651 ± 0.0077 | 0.9528 ± 0.0006 | 0.6186 ± 0.0053 | 0.8204 ± 0.0020 |
| `old_hmid_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8855 ± 0.0010 | 0.8950 ± 0.0015 | 0.6272 ± 0.0097 | 0.9635 ± 0.0005 | 0.6702 ± 0.0038 | 0.8362 ± 0.0010 |
| `old_hmid_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.8527 ± 0.0011 | 0.8810 ± 0.0029 | 0.5715 ± 0.0052 | 0.9523 ± 0.0006 | 0.6058 ± 0.0039 | 0.8137 ± 0.0031 |
| `old_hmid_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8850 ± 0.0012 | 0.8945 ± 0.0025 | 0.6286 ± 0.0077 | 0.9638 ± 0.0003 | 0.6667 ± 0.0026 | 0.8358 ± 0.0023 |
| `old_hpre_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.8704 ± 0.0003 | 0.8914 ± 0.0006 | 0.5873 ± 0.0053 | 0.9591 ± 0.0001 | 0.6305 ± 0.0045 | 0.8280 ± 0.0011 |
| `old_hpre_cos__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8881 ± 0.0014 | 0.8938 ± 0.0014 | 0.6443 ± 0.0104 | 0.9652 ± 0.0002 | 0.6562 ± 0.0060 | 0.8365 ± 0.0022 |
| `old_hpre_cos__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.8675 ± 0.0016 | 0.8887 ± 0.0021 | 0.5799 ± 0.0109 | 0.9584 ± 0.0002 | 0.6205 ± 0.0077 | 0.8241 ± 0.0023 |
| `old_hpre_cos__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8846 ± 0.0018 | 0.8910 ± 0.0009 | 0.6179 ± 0.0105 | 0.9641 ± 0.0006 | 0.6513 ± 0.0061 | 0.8305 ± 0.0007 |
| `old_hpre_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.8557 ± 0.0034 | 0.8848 ± 0.0039 | 0.5670 ± 0.0070 | 0.9534 ± 0.0016 | 0.6159 ± 0.0038 | 0.8181 ± 0.0047 |
| `old_hpre_cos__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8852 ± 0.0021 | 0.8943 ± 0.0003 | 0.6207 ± 0.0087 | 0.9637 ± 0.0010 | 0.6683 ± 0.0035 | 0.8347 ± 0.0009 |
| `old_hpre_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.8549 ± 0.0011 | 0.8846 ± 0.0023 | 0.5857 ± 0.0066 | 0.9531 ± 0.0009 | 0.6154 ± 0.0013 | 0.8196 ± 0.0029 |
| `old_hpre_cos__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8852 ± 0.0018 | 0.8910 ± 0.0008 | 0.6259 ± 0.0095 | 0.9642 ± 0.0006 | 0.6674 ± 0.0030 | 0.8312 ± 0.0004 |
| `new_jffn__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.8472 ± 0.0020 | 0.8838 ± 0.0053 | 0.5241 ± 0.0198 | 0.9492 ± 0.0006 | 0.5949 ± 0.0083 | 0.8134 ± 0.0054 |
| `new_jffn__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8767 ± 0.0042 | 0.8915 ± 0.0013 | 0.6117 ± 0.0147 | 0.9584 ± 0.0013 | 0.6594 ± 0.0074 | 0.8304 ± 0.0028 |
| `new_jffn__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.8516 ± 0.0032 | 0.8842 ± 0.0006 | 0.5297 ± 0.0071 | 0.9509 ± 0.0014 | 0.6028 ± 0.0063 | 0.8142 ± 0.0013 |
| `new_jffn__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8796 ± 0.0028 | 0.8917 ± 0.0026 | 0.6186 ± 0.0132 | 0.9594 ± 0.0009 | 0.6643 ± 0.0068 | 0.8313 ± 0.0033 |
| `new_jffn__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.8288 ± 0.0009 | 0.8743 ± 0.0027 | 0.5308 ± 0.0164 | 0.9423 ± 0.0007 | 0.5716 ± 0.0031 | 0.8018 ± 0.0026 |
| `new_jffn__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8726 ± 0.0042 | 0.8886 ± 0.0031 | 0.6112 ± 0.0077 | 0.9586 ± 0.0016 | 0.6431 ± 0.0115 | 0.8269 ± 0.0036 |
| `new_jffn__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.8354 ± 0.0011 | 0.8764 ± 0.0018 | 0.5247 ± 0.0048 | 0.9456 ± 0.0004 | 0.5840 ± 0.0028 | 0.8039 ± 0.0020 |
| `new_jffn__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8735 ± 0.0009 | 0.8888 ± 0.0026 | 0.6036 ± 0.0133 | 0.9590 ± 0.0005 | 0.6454 ± 0.0013 | 0.8264 ± 0.0020 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__sqrt_matched_state__risk` | 0.8505 ± 0.0021 | 0.8796 ± 0.0016 | 0.5206 ± 0.0110 | 0.9518 ± 0.0009 | 0.5883 ± 0.0033 | 0.8076 ± 0.0019 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__sqrt_matched_state__risk_ev` | 0.8822 ± 0.0010 | 0.8884 ± 0.0010 | 0.6028 ± 0.0226 | 0.9634 ± 0.0004 | 0.6506 ± 0.0020 | 0.8259 ± 0.0010 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__cosine_matched_state__risk` | 0.8550 ± 0.0011 | 0.8820 ± 0.0013 | 0.5231 ± 0.0297 | 0.9530 ± 0.0003 | 0.5945 ± 0.0021 | 0.8110 ± 0.0014 |
| `new_jffn_entropy_matched__hpre_raw_logit_gauss__cosine_matched_state__risk_ev` | 0.8831 ± 0.0004 | 0.8910 ± 0.0028 | 0.6160 ± 0.0126 | 0.9634 ± 0.0001 | 0.6603 ± 0.0029 | 0.8303 ± 0.0022 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__sqrt_matched_state__risk` | 0.8419 ± 0.0017 | 0.8765 ± 0.0032 | 0.4796 ± 0.0076 | 0.9497 ± 0.0003 | 0.5517 ± 0.0052 | 0.8004 ± 0.0046 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__sqrt_matched_state__risk_ev` | 0.8811 ± 0.0030 | 0.8882 ± 0.0019 | 0.6076 ± 0.0053 | 0.9630 ± 0.0016 | 0.6464 ± 0.0023 | 0.8260 ± 0.0018 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__cosine_matched_state__risk` | 0.8402 ± 0.0018 | 0.8762 ± 0.0034 | 0.4871 ± 0.0120 | 0.9491 ± 0.0006 | 0.5473 ± 0.0014 | 0.8005 ± 0.0053 |
| `new_jffn_entropy_matched__hpre_softmax_prob_gauss__cosine_matched_state__risk_ev` | 0.8816 ± 0.0034 | 0.8900 ± 0.0010 | 0.6184 ± 0.0074 | 0.9632 ± 0.0011 | 0.6455 ± 0.0085 | 0.8292 ± 0.0013 |

### Preregistered per-seed AUROC deltas

| Comparison | seed 43 | seed 44 | seed 45 |
|---|---:|---:|---:|
| new_jffn vs old_hpre_cos | -0.0056 | -0.0187 | -0.0100 |
| new_jffn_entropy_matched vs old_hpre_cos | -0.0036 | -0.0092 | -0.0049 |

机器可读的逐 seed 全指标见 `training_results.json`。

## P 本身的验证

- 空间验证：11418 个 REAL mention 有同类别 COCO box；558 个无匹配 box；0 个网格不匹配。
- 分布曲线：`p_distribution_entropy.png`。
- 同图不同目标特异性：`p_target_specificity_js.png`。
- COCO 空间富集：`p_spatial_enrichment.png`。
- 全指标面板：`p_distribution_all_metrics.png`、`p_target_specificity_all_metrics.png`、`p_spatial_all_metrics.png`。
