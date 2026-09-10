# minigpt4_7b — Jacobian path

分别计算seeds 43/44/45的指标，再报告均值±总体标准差（ddof=0）；主表不使用ensemble。完整双阈值见results.json。F1阈值由各seed训练集REAL-F1选择。

| 模型 | 方法/训练器 | AUROC | HALL AUPR | HALL F1（train-F1阈值） |
|---|---|---:|---:|---:|
| minigpt4_7b | AE / jacobian_path | 0.8754 ± 0.0014 | 0.6274 ± 0.0111 | 0.5151 ± 0.0393 |
| minigpt4_7b | AE+R_cos+S / jacobian_path | 0.9039 ± 0.0023 | 0.6289 ± 0.0040 | 0.5671 ± 0.0234 |
| minigpt4_7b | AE+a / jacobian_path | 0.9106 ± 0.0002 | 0.6686 ± 0.0116 | 0.5969 ± 0.0138 |
| minigpt4_7b | AE+g / jacobian_path | 0.9107 ± 0.0024 | 0.6645 ± 0.0067 | 0.5859 ± 0.0057 |
| minigpt4_7b | AE+i / jacobian_path | 0.9083 ± 0.0042 | 0.6666 ± 0.0097 | 0.5919 ± 0.0084 |
| minigpt4_7b | AE+i+s / jacobian_path | 0.9142 ± 0.0027 | 0.6836 ± 0.0066 | 0.6283 ± 0.0082 |
| minigpt4_7b | AE+log1p(S) / jacobian_path | 0.9091 ± 0.0015 | 0.6549 ± 0.0104 | 0.5935 ± 0.0177 |
| minigpt4_7b | AE+n / jacobian_path | 0.9100 ± 0.0013 | 0.6682 ± 0.0071 | 0.6023 ± 0.0139 |
| minigpt4_7b | AE+s / jacobian_path | 0.9086 ± 0.0011 | 0.6682 ± 0.0148 | 0.5890 ± 0.0112 |
| minigpt4_7b | AE+s+L(N_end) / jacobian_path | 0.9110 ± 0.0028 | 0.6725 ± 0.0014 | 0.5995 ± 0.0201 |
| minigpt4_7b | AE+s+kappa_end / jacobian_path | 0.9138 ± 0.0038 | 0.6765 ± 0.0156 | 0.6116 ± 0.0246 |
| minigpt4_7b | AE+s+kappa_vec / jacobian_path | 0.9079 ± 0.0068 | 0.6693 ± 0.0177 | 0.6115 ± 0.0114 |
| minigpt4_7b | AE+s+n / jacobian_path | 0.9131 ± 0.0013 | 0.6869 ± 0.0027 | 0.6151 ± 0.0122 |
| minigpt4_7b | B_Q / jacobian_path | 0.7748 ± 0.0030 | 0.3899 ± 0.0065 | 0.1034 ± 0.0292 |
| minigpt4_7b | F / jacobian_path | 0.9091 ± 0.0015 | 0.6549 ± 0.0104 | 0.5935 ± 0.0177 |
| minigpt4_7b | F_B_Q / jacobian_path | 0.9010 ± 0.0070 | 0.6422 ± 0.0176 | 0.5728 ± 0.0216 |
| minigpt4_7b | F_J_QT / jacobian_path | 0.9096 ± 0.0043 | 0.6605 ± 0.0160 | 0.5796 ± 0.0406 |
| minigpt4_7b | F_K / jacobian_path | 0.9103 ± 0.0003 | 0.6671 ± 0.0085 | 0.6053 ± 0.0011 |
| minigpt4_7b | F_K_B_Q / jacobian_path | 0.9055 ± 0.0042 | 0.6512 ± 0.0151 | 0.6068 ± 0.0214 |
| minigpt4_7b | F_K_J_QT / jacobian_path | 0.9111 ± 0.0043 | 0.6827 ± 0.0083 | 0.6133 ± 0.0025 |
| minigpt4_7b | F_K_Q / jacobian_path | 0.8952 ± 0.0127 | 0.6329 ± 0.0276 | 0.5626 ± 0.0559 |
| minigpt4_7b | F_Q / jacobian_path | 0.9081 ± 0.0024 | 0.6616 ± 0.0100 | 0.6003 ± 0.0409 |
| minigpt4_7b | H_SK / jacobian_path | 0.9095 ± 0.0051 | 0.6664 ± 0.0046 | 0.6060 ± 0.0109 |
| minigpt4_7b | H_SN / jacobian_path | 0.9081 ± 0.0035 | 0.6677 ± 0.0017 | 0.5991 ± 0.0113 |
| minigpt4_7b | J_E / jacobian_path | 0.8761 ± 0.0008 | 0.5877 ± 0.0089 | 0.5137 ± 0.0190 |
| minigpt4_7b | J_QT / jacobian_path | 0.8678 ± 0.0050 | 0.5572 ± 0.0214 | 0.4808 ± 0.0415 |
| minigpt4_7b | J_T / jacobian_path | 0.8804 ± 0.0040 | 0.5799 ± 0.0088 | 0.4944 ± 0.0228 |
| minigpt4_7b | J_T+J_E / jacobian_path | 0.9017 ± 0.0017 | 0.6465 ± 0.0054 | 0.6086 ± 0.0109 |
| minigpt4_7b | J_cosT / jacobian_path | 0.8719 ± 0.0045 | 0.5540 ± 0.0138 | 0.4119 ± 0.0347 |
| minigpt4_7b | Q / jacobian_path | 0.8874 ± 0.0017 | 0.5925 ± 0.0036 | 0.5302 ± 0.0304 |
| minigpt4_7b | U_SK / jacobian_path | 0.9090 ± 0.0006 | 0.6670 ± 0.0109 | 0.6076 ± 0.0233 |
| minigpt4_7b | U_SN / jacobian_path | 0.9134 ± 0.0004 | 0.6867 ± 0.0045 | 0.6091 ± 0.0038 |
| minigpt4_7b | endpoint_cosine_js / jacobian_path | 0.8719 ± 0.0045 | 0.5540 ± 0.0138 | 0.4119 ± 0.0347 |
| minigpt4_7b | endpoint_dot_js / jacobian_path | 0.8479 ± 0.0015 | 0.5194 ± 0.0054 | 0.3548 ± 0.0713 |
| minigpt4_7b | endpoint_projection_js / jacobian_path | 0.8600 ± 0.0021 | 0.5539 ± 0.0113 | 0.4677 ± 0.0389 |
| minigpt4_7b | raw_AE+S+N_vec / jacobian_path | 0.9032 ± 0.0025 | 0.6292 ± 0.0090 | 0.5985 ± 0.0158 |
| minigpt4_7b | F+JS(Q,tau=0.2) / jacobian_path | 0.8977 ± 0.0013 | 0.6380 ± 0.0060 | 0.6006 ± 0.0218 |
| minigpt4_7b | F+JS(endpoint_cos,tau=0.2) / jacobian_path | 0.9019 ± 0.0033 | 0.6519 ± 0.0045 | 0.5912 ± 0.0320 |
| minigpt4_7b | R_raw / jacobian_path | 0.8843 ± 0.0045 | 0.5735 ± 0.0086 | 0.4211 ± 0.0358 |
| minigpt4_7b | log1p(S) / jacobian_path | 0.8947 ± 0.0012 | 0.5927 ± 0.0038 | 0.5238 ± 0.0306 |
| minigpt4_7b | R_raw+log1p(S) / jacobian_path | 0.8966 ± 0.0012 | 0.6017 ± 0.0048 | 0.5470 ± 0.0178 |
| minigpt4_7b | JS(endpoint_cos,raw_attention,tau=0.2) / jacobian_path | 0.8425 ± 0.0028 | 0.5156 ± 0.0117 | 0.4616 ± 0.0235 |
| minigpt4_7b | F+JS(endpoint_cos,raw_attention,tau=0.2) / jacobian_path | 0.9047 ± 0.0107 | 0.6542 ± 0.0166 | 0.5994 ± 0.0250 |
