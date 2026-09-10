# shikra_7b — Jacobian path

分别计算seeds 43/44/45的指标，再报告均值±总体标准差（ddof=0）；主表不使用ensemble。完整双阈值见results.json。F1阈值由各seed训练集REAL-F1选择。

| 模型 | 方法/训练器 | AUROC | HALL AUPR | HALL F1（train-F1阈值） |
|---|---|---:|---:|---:|
| shikra_7b | AE / jacobian_path | 0.8141 ± 0.0007 | 0.5283 ± 0.0054 | 0.3532 ± 0.0229 |
| shikra_7b | AE+R_cos+S / jacobian_path | 0.8597 ± 0.0009 | 0.6010 ± 0.0048 | 0.5324 ± 0.0197 |
| shikra_7b | AE+a / jacobian_path | 0.8638 ± 0.0004 | 0.6067 ± 0.0037 | 0.5568 ± 0.0050 |
| shikra_7b | AE+g / jacobian_path | 0.8612 ± 0.0015 | 0.6008 ± 0.0056 | 0.5500 ± 0.0195 |
| shikra_7b | AE+i / jacobian_path | 0.8593 ± 0.0012 | 0.5933 ± 0.0051 | 0.5328 ± 0.0082 |
| shikra_7b | AE+i+s / jacobian_path | 0.8675 ± 0.0012 | 0.6206 ± 0.0063 | 0.5541 ± 0.0222 |
| shikra_7b | AE+log1p(S) / jacobian_path | 0.8626 ± 0.0031 | 0.6048 ± 0.0079 | 0.5599 ± 0.0106 |
| shikra_7b | AE+n / jacobian_path | 0.8561 ± 0.0009 | 0.5883 ± 0.0039 | 0.5232 ± 0.0111 |
| shikra_7b | AE+s / jacobian_path | 0.8624 ± 0.0025 | 0.6018 ± 0.0067 | 0.5554 ± 0.0269 |
| shikra_7b | AE+s+L(N_end) / jacobian_path | 0.8585 ± 0.0047 | 0.5993 ± 0.0030 | 0.5768 ± 0.0172 |
| shikra_7b | AE+s+kappa_end / jacobian_path | 0.8601 ± 0.0015 | 0.6100 ± 0.0024 | 0.5783 ± 0.0174 |
| shikra_7b | AE+s+kappa_vec / jacobian_path | 0.8607 ± 0.0009 | 0.6124 ± 0.0055 | 0.5699 ± 0.0023 |
| shikra_7b | AE+s+n / jacobian_path | 0.8612 ± 0.0053 | 0.6099 ± 0.0087 | 0.5623 ± 0.0320 |
| shikra_7b | B_Q / jacobian_path | 0.7941 ± 0.0015 | 0.5026 ± 0.0035 | 0.3824 ± 0.0544 |
| shikra_7b | F / jacobian_path | 0.8626 ± 0.0031 | 0.6048 ± 0.0079 | 0.5599 ± 0.0106 |
| shikra_7b | F_B_Q / jacobian_path | 0.8537 ± 0.0045 | 0.5952 ± 0.0076 | 0.5047 ± 0.0234 |
| shikra_7b | F_J_QT / jacobian_path | 0.8650 ± 0.0015 | 0.6262 ± 0.0027 | 0.5641 ± 0.0112 |
| shikra_7b | F_K / jacobian_path | 0.8629 ± 0.0011 | 0.6155 ± 0.0029 | 0.5718 ± 0.0105 |
| shikra_7b | F_K_B_Q / jacobian_path | 0.8547 ± 0.0045 | 0.5981 ± 0.0062 | 0.5795 ± 0.0055 |
| shikra_7b | F_K_J_QT / jacobian_path | 0.8584 ± 0.0046 | 0.6214 ± 0.0031 | 0.5667 ± 0.0180 |
| shikra_7b | F_K_Q / jacobian_path | 0.8518 ± 0.0029 | 0.5906 ± 0.0061 | 0.5422 ± 0.0286 |
| shikra_7b | F_Q / jacobian_path | 0.8534 ± 0.0035 | 0.5906 ± 0.0047 | 0.5529 ± 0.0050 |
| shikra_7b | H_SK / jacobian_path | 0.8600 ± 0.0038 | 0.6169 ± 0.0085 | 0.5806 ± 0.0115 |
| shikra_7b | H_SN / jacobian_path | 0.8538 ± 0.0032 | 0.6039 ± 0.0062 | 0.5568 ± 0.0084 |
| shikra_7b | J_E / jacobian_path | 0.8265 ± 0.0023 | 0.5617 ± 0.0022 | 0.4772 ± 0.0174 |
| shikra_7b | J_QT / jacobian_path | 0.7921 ± 0.0032 | 0.5255 ± 0.0070 | 0.4206 ± 0.0109 |
| shikra_7b | J_T / jacobian_path | 0.8077 ± 0.0009 | 0.5361 ± 0.0016 | 0.4205 ± 0.0330 |
| shikra_7b | J_T+J_E / jacobian_path | 0.8398 ± 0.0017 | 0.5837 ± 0.0018 | 0.5179 ± 0.0173 |
| shikra_7b | J_cosT / jacobian_path | 0.7999 ± 0.0022 | 0.5338 ± 0.0032 | 0.4022 ± 0.0174 |
| shikra_7b | Q / jacobian_path | 0.8489 ± 0.0017 | 0.5852 ± 0.0050 | 0.5680 ± 0.0096 |
| shikra_7b | U_SK / jacobian_path | 0.8577 ± 0.0040 | 0.5983 ± 0.0108 | 0.5618 ± 0.0104 |
| shikra_7b | U_SN / jacobian_path | 0.8600 ± 0.0018 | 0.6033 ± 0.0100 | 0.5820 ± 0.0078 |
| shikra_7b | endpoint_cosine_js / jacobian_path | 0.7999 ± 0.0022 | 0.5338 ± 0.0032 | 0.4022 ± 0.0174 |
| shikra_7b | endpoint_dot_js / jacobian_path | 0.7688 ± 0.0013 | 0.4731 ± 0.0023 | 0.3733 ± 0.0037 |
| shikra_7b | endpoint_projection_js / jacobian_path | 0.7833 ± 0.0028 | 0.4955 ± 0.0036 | 0.3903 ± 0.0261 |
| shikra_7b | raw_AE+S+N_vec / jacobian_path | 0.8594 ± 0.0047 | 0.6083 ± 0.0038 | 0.5537 ± 0.0167 |
| shikra_7b | F+JS(Q,tau=0.2) / jacobian_path | 0.8614 ± 0.0042 | 0.6136 ± 0.0082 | 0.5583 ± 0.0154 |
| shikra_7b | F+JS(endpoint_cos,tau=0.2) / jacobian_path | 0.8564 ± 0.0053 | 0.5992 ± 0.0006 | 0.5301 ± 0.0161 |
| shikra_7b | R_raw / jacobian_path | 0.8337 ± 0.0071 | 0.5478 ± 0.0120 | 0.4643 ± 0.0505 |
| shikra_7b | log1p(S) / jacobian_path | 0.8584 ± 0.0010 | 0.5992 ± 0.0084 | 0.5219 ± 0.0063 |
| shikra_7b | R_raw+log1p(S) / jacobian_path | 0.8677 ± 0.0009 | 0.6183 ± 0.0018 | 0.5542 ± 0.0344 |
| shikra_7b | JS(endpoint_cos,raw_attention,tau=0.2) / jacobian_path | 0.8146 ± 0.0007 | 0.5350 ± 0.0051 | 0.4383 ± 0.0060 |
| shikra_7b | F+JS(endpoint_cos,raw_attention,tau=0.2) / jacobian_path | 0.8582 ± 0.0016 | 0.5949 ± 0.0048 | 0.5227 ± 0.0230 |
