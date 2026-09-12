# 原始attention＋log1p(S)：XGB / RF网格搜索

四模型×8组相同输入，原4000图/3200-800/全部mentions与原MLP完全一致。X保持原attention或attention×gate，仅S_g取log1p。
每组分别：visual、V+P相加、generation、完整VP组与G组拼接。
XGB：depth4/6/8×lr.1/.05×trees100/200/500，共18候选；RF：depth无限/10/20×trees200/400/600，共9候选。
每模型/组/分类器独立在同一2560/640图片划分上用seed43验证AUROC选参，平局AP/候选顺序；选定后全3200图重训43/44/45，原800图测试。没有根据测试结果重选。
使用项目原分类器构建器，CPU每估计器2线程；不增加标准化、类别权重或树模型早停。
共864候选拟合、192最终树模型。两种MLP只引用已完成结果，不重训。以下为三seed均值±总体std（%），F1按训练REAL-F1阈值。

八组平均AUROC总览（%，原三层MLP / XGB / RF）：

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| attention_visual | 84.624 / 81.881 / 80.004 | 89.379 / 88.425 / 88.320 | 87.516 / 86.811 / 84.002 | 85.244 / 83.435 / 83.184 |
| attention_visual_prompt_sum | 84.078 / 83.752 / 82.153 | 89.720 / 89.416 / 89.068 | 87.725 / 87.865 / 84.767 | 84.880 / 85.704 / 85.021 |
| attention_generation | 82.536 / 81.706 / 81.340 | 88.856 / 88.235 / 88.188 | 86.449 / 86.855 / 85.315 | 84.467 / 85.311 / 83.879 |
| attention_vp_generation | 84.675 / 85.400 / 82.418 | 90.021 / 89.599 / 89.186 | 89.289 / 89.056 / 85.856 | 86.695 / 87.090 / 85.275 |
| gated_visual | 84.973 / 82.763 / 80.439 | 89.501 / 88.880 / 88.452 | 87.744 / 87.138 / 84.405 | 85.419 / 84.483 / 83.944 |
| gated_visual_prompt_sum | 85.528 / 84.844 / 82.502 | 89.699 / 89.275 / 89.315 | 88.350 / 89.257 / 86.359 | 85.402 / 85.665 / 85.374 |
| gated_generation | 83.868 / 84.843 / 81.832 | 88.228 / 88.499 / 88.147 | 86.598 / 87.017 / 84.710 | 84.718 / 84.812 / 83.594 |
| gated_vp_generation | 86.635 / 86.584 / 83.416 | 90.470 / 90.519 / 89.792 | 89.665 / 90.996 / 87.607 | 86.599 / 87.338 / 85.552 |

xgb相对原三层：15/32组平均AUROC提高；8/32组平均HALL-AUPR提高。

rf相对原三层：1/32组平均AUROC提高；2/32组平均HALL-AUPR提高。

XGB相对原三层MLP，15/32组平均AUROC提高、8/32组HALL-AUPR提高；RF只有1/32组AUROC和2/32组AP提高。相对上一轮搜参单层MLP，XGB有28/32组AUROC、24/32组AP提高，RF为20/32、22/32。树模型不能跨特征统一替换原三层MLP。

XGB三seed的AUROC/AP在本设置下相同或仅有浮点舍入差，std≈0不代表跨样本不确定性为零。

完整指标与单层MLP对照：

| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR | HALL-F1 |
|---|---|---|---:|---:|---:|
| Qwen2.5-VL | attention_visual | three_hidden | 84.624 ± 0.471 | 38.360 ± 1.206 | 31.513 ± 4.856 |
| Qwen2.5-VL | attention_visual | one_hidden | 82.914 ± 0.402 | 35.082 ± 0.546 | 17.021 ± 2.504 |
| Qwen2.5-VL | attention_visual | xgb | 81.881 ± 0.000 | 33.854 ± 0.000 | 41.943 ± 0.000 |
| Qwen2.5-VL | attention_visual | rf | 80.004 ± 0.178 | 32.551 ± 0.261 | 36.451 ± 1.627 |
| Qwen2.5-VL | attention_visual_prompt_sum | three_hidden | 84.078 ± 1.756 | 38.792 ± 2.939 | 30.187 ± 2.959 |
| Qwen2.5-VL | attention_visual_prompt_sum | one_hidden | 82.240 ± 0.038 | 33.216 ± 0.220 | 5.460 ± 0.915 |
| Qwen2.5-VL | attention_visual_prompt_sum | xgb | 83.752 ± 0.000 | 38.227 ± 0.000 | 35.172 ± 0.000 |
| Qwen2.5-VL | attention_visual_prompt_sum | rf | 82.153 ± 0.164 | 37.579 ± 0.584 | 39.647 ± 1.440 |
| Qwen2.5-VL | attention_generation | three_hidden | 82.536 ± 0.630 | 36.973 ± 1.685 | 32.899 ± 2.246 |
| Qwen2.5-VL | attention_generation | one_hidden | 78.698 ± 0.093 | 28.350 ± 0.557 | 0.355 ± 0.501 |
| Qwen2.5-VL | attention_generation | xgb | 81.706 ± 0.000 | 33.167 ± 0.000 | 28.364 ± 0.000 |
| Qwen2.5-VL | attention_generation | rf | 81.340 ± 0.144 | 33.330 ± 0.615 | 40.549 ± 0.577 |
| Qwen2.5-VL | attention_vp_generation | three_hidden | 84.675 ± 0.271 | 41.737 ± 2.507 | 38.632 ± 6.507 |
| Qwen2.5-VL | attention_vp_generation | one_hidden | 82.207 ± 0.110 | 34.546 ± 0.391 | 5.439 ± 0.440 |
| Qwen2.5-VL | attention_vp_generation | xgb | 85.400 ± 0.000 | 41.002 ± 0.000 | 44.685 ± 0.000 |
| Qwen2.5-VL | attention_vp_generation | rf | 82.418 ± 0.212 | 38.214 ± 0.596 | 40.985 ± 0.591 |
| Qwen2.5-VL | gated_visual | three_hidden | 84.973 ± 0.435 | 40.489 ± 1.078 | 38.378 ± 1.949 |
| Qwen2.5-VL | gated_visual | one_hidden | 82.719 ± 0.649 | 35.085 ± 1.417 | 17.262 ± 2.561 |
| Qwen2.5-VL | gated_visual | xgb | 82.763 ± 0.000 | 35.382 ± 0.000 | 40.553 ± 0.000 |
| Qwen2.5-VL | gated_visual | rf | 80.439 ± 0.170 | 33.008 ± 0.805 | 36.789 ± 0.308 |
| Qwen2.5-VL | gated_visual_prompt_sum | three_hidden | 85.528 ± 1.063 | 42.766 ± 1.914 | 35.867 ± 1.960 |
| Qwen2.5-VL | gated_visual_prompt_sum | one_hidden | 82.765 ± 0.412 | 35.770 ± 0.282 | 1.767 ± 1.321 |
| Qwen2.5-VL | gated_visual_prompt_sum | xgb | 84.844 ± 0.000 | 40.106 ± 0.000 | 42.336 ± 0.000 |
| Qwen2.5-VL | gated_visual_prompt_sum | rf | 82.502 ± 0.149 | 40.574 ± 0.516 | 41.781 ± 0.618 |
| Qwen2.5-VL | gated_generation | three_hidden | 83.868 ± 0.507 | 40.379 ± 1.033 | 31.322 ± 3.163 |
| Qwen2.5-VL | gated_generation | one_hidden | 79.251 ± 0.225 | 29.379 ± 0.502 | 2.409 ± 1.750 |
| Qwen2.5-VL | gated_generation | xgb | 84.843 ± 0.000 | 38.817 ± 0.000 | 35.526 ± 0.000 |
| Qwen2.5-VL | gated_generation | rf | 81.832 ± 0.140 | 34.994 ± 0.495 | 40.703 ± 1.723 |
| Qwen2.5-VL | gated_vp_generation | three_hidden | 86.635 ± 0.660 | 46.780 ± 2.475 | 44.953 ± 1.068 |
| Qwen2.5-VL | gated_vp_generation | one_hidden | 83.111 ± 0.237 | 39.007 ± 0.630 | 20.903 ± 2.099 |
| Qwen2.5-VL | gated_vp_generation | xgb | 86.584 ± 0.000 | 45.783 ± 0.000 | 49.780 ± 0.000 |
| Qwen2.5-VL | gated_vp_generation | rf | 83.416 ± 0.164 | 41.413 ± 0.393 | 44.196 ± 0.877 |
| LLaVA-1.5 | attention_visual | three_hidden | 89.379 ± 0.151 | 68.622 ± 0.582 | 64.261 ± 0.499 |
| LLaVA-1.5 | attention_visual | one_hidden | 89.009 ± 0.054 | 67.331 ± 0.795 | 63.204 ± 1.087 |
| LLaVA-1.5 | attention_visual | xgb | 88.425 ± 0.000 | 64.914 ± 0.000 | 62.148 ± 0.000 |
| LLaVA-1.5 | attention_visual | rf | 88.320 ± 0.023 | 66.986 ± 0.201 | 62.262 ± 0.443 |
| LLaVA-1.5 | attention_visual_prompt_sum | three_hidden | 89.720 ± 0.286 | 69.640 ± 1.049 | 63.861 ± 0.130 |
| LLaVA-1.5 | attention_visual_prompt_sum | one_hidden | 87.816 ± 0.275 | 68.509 ± 0.528 | 58.013 ± 0.748 |
| LLaVA-1.5 | attention_visual_prompt_sum | xgb | 89.416 ± 0.000 | 68.551 ± 0.000 | 62.713 ± 0.000 |
| LLaVA-1.5 | attention_visual_prompt_sum | rf | 89.068 ± 0.029 | 69.029 ± 0.084 | 65.882 ± 0.661 |
| LLaVA-1.5 | attention_generation | three_hidden | 88.856 ± 0.149 | 66.265 ± 0.421 | 63.619 ± 1.563 |
| LLaVA-1.5 | attention_generation | one_hidden | 87.579 ± 0.294 | 65.512 ± 0.700 | 53.859 ± 0.844 |
| LLaVA-1.5 | attention_generation | xgb | 88.235 ± 0.000 | 64.316 ± 0.000 | 60.914 ± 0.000 |
| LLaVA-1.5 | attention_generation | rf | 88.188 ± 0.064 | 65.583 ± 0.342 | 63.552 ± 0.513 |
| LLaVA-1.5 | attention_vp_generation | three_hidden | 90.021 ± 0.438 | 70.067 ± 1.501 | 65.635 ± 0.878 |
| LLaVA-1.5 | attention_vp_generation | one_hidden | 89.433 ± 0.318 | 70.426 ± 0.246 | 59.392 ± 1.581 |
| LLaVA-1.5 | attention_vp_generation | xgb | 89.599 ± 0.000 | 68.186 ± 0.000 | 63.623 ± 0.000 |
| LLaVA-1.5 | attention_vp_generation | rf | 89.186 ± 0.016 | 69.069 ± 0.149 | 63.860 ± 1.549 |
| LLaVA-1.5 | gated_visual | three_hidden | 89.501 ± 0.102 | 68.828 ± 0.192 | 64.855 ± 0.716 |
| LLaVA-1.5 | gated_visual | one_hidden | 88.828 ± 0.229 | 66.320 ± 1.058 | 61.035 ± 1.253 |
| LLaVA-1.5 | gated_visual | xgb | 88.880 ± 0.000 | 66.577 ± 0.000 | 63.596 ± 0.000 |
| LLaVA-1.5 | gated_visual | rf | 88.452 ± 0.048 | 67.354 ± 0.084 | 64.612 ± 0.513 |
| LLaVA-1.5 | gated_visual_prompt_sum | three_hidden | 89.699 ± 0.043 | 70.276 ± 0.081 | 63.837 ± 1.677 |
| LLaVA-1.5 | gated_visual_prompt_sum | one_hidden | 88.415 ± 0.475 | 68.397 ± 0.825 | 59.852 ± 1.394 |
| LLaVA-1.5 | gated_visual_prompt_sum | xgb | 89.275 ± 0.000 | 67.610 ± 0.000 | 64.920 ± 0.000 |
| LLaVA-1.5 | gated_visual_prompt_sum | rf | 89.315 ± 0.078 | 69.176 ± 0.319 | 63.639 ± 0.904 |
| LLaVA-1.5 | gated_generation | three_hidden | 88.228 ± 0.504 | 66.171 ± 1.070 | 60.500 ± 1.423 |
| LLaVA-1.5 | gated_generation | one_hidden | 87.212 ± 0.132 | 64.777 ± 0.510 | 56.344 ± 1.044 |
| LLaVA-1.5 | gated_generation | xgb | 88.499 ± 0.000 | 64.588 ± 0.000 | 59.259 ± 0.000 |
| LLaVA-1.5 | gated_generation | rf | 88.147 ± 0.039 | 64.618 ± 0.348 | 59.679 ± 1.120 |
| LLaVA-1.5 | gated_vp_generation | three_hidden | 90.470 ± 0.306 | 71.947 ± 0.602 | 66.103 ± 1.734 |
| LLaVA-1.5 | gated_vp_generation | one_hidden | 89.601 ± 0.416 | 71.187 ± 0.711 | 61.910 ± 1.996 |
| LLaVA-1.5 | gated_vp_generation | xgb | 90.519 ± 0.000 | 71.134 ± 0.000 | 63.741 ± 0.000 |
| LLaVA-1.5 | gated_vp_generation | rf | 89.792 ± 0.035 | 70.443 ± 0.211 | 62.724 ± 0.550 |
| Qwen3-VL | attention_visual | three_hidden | 87.516 ± 0.195 | 59.573 ± 0.977 | 52.975 ± 1.007 |
| Qwen3-VL | attention_visual | one_hidden | 86.883 ± 0.396 | 57.888 ± 0.238 | 49.455 ± 1.195 |
| Qwen3-VL | attention_visual | xgb | 86.811 ± 0.000 | 58.984 ± 0.000 | 45.070 ± 0.000 |
| Qwen3-VL | attention_visual | rf | 84.002 ± 0.087 | 53.960 ± 0.103 | 40.077 ± 1.099 |
| Qwen3-VL | attention_visual_prompt_sum | three_hidden | 87.725 ± 1.091 | 61.527 ± 1.843 | 57.147 ± 2.739 |
| Qwen3-VL | attention_visual_prompt_sum | one_hidden | 82.591 ± 0.203 | 51.650 ± 0.394 | 39.903 ± 0.662 |
| Qwen3-VL | attention_visual_prompt_sum | xgb | 87.865 ± 0.000 | 59.933 ± 0.000 | 50.856 ± 0.000 |
| Qwen3-VL | attention_visual_prompt_sum | rf | 84.767 ± 0.055 | 54.558 ± 0.083 | 52.909 ± 0.120 |
| Qwen3-VL | attention_generation | three_hidden | 86.449 ± 0.717 | 57.586 ± 1.801 | 51.120 ± 2.864 |
| Qwen3-VL | attention_generation | one_hidden | 83.309 ± 0.308 | 50.417 ± 0.784 | 34.246 ± 1.909 |
| Qwen3-VL | attention_generation | xgb | 86.855 ± 0.000 | 57.157 ± 0.000 | 48.649 ± 0.000 |
| Qwen3-VL | attention_generation | rf | 85.315 ± 0.090 | 54.525 ± 0.121 | 39.227 ± 2.517 |
| Qwen3-VL | attention_vp_generation | three_hidden | 89.289 ± 0.110 | 64.135 ± 0.966 | 58.473 ± 0.636 |
| Qwen3-VL | attention_vp_generation | one_hidden | 85.745 ± 0.066 | 56.044 ± 0.036 | 50.600 ± 0.720 |
| Qwen3-VL | attention_vp_generation | xgb | 89.056 ± 0.000 | 62.956 ± 0.000 | 56.140 ± 0.000 |
| Qwen3-VL | attention_vp_generation | rf | 85.856 ± 0.081 | 56.569 ± 0.103 | 41.072 ± 1.358 |
| Qwen3-VL | gated_visual | three_hidden | 87.744 ± 0.305 | 59.638 ± 0.441 | 53.514 ± 0.659 |
| Qwen3-VL | gated_visual | one_hidden | 87.126 ± 0.063 | 58.328 ± 0.168 | 49.765 ± 0.987 |
| Qwen3-VL | gated_visual | xgb | 87.138 ± 0.000 | 59.336 ± 0.000 | 46.310 ± 0.000 |
| Qwen3-VL | gated_visual | rf | 84.405 ± 0.094 | 54.878 ± 0.418 | 49.870 ± 4.300 |
| Qwen3-VL | gated_visual_prompt_sum | three_hidden | 88.350 ± 0.228 | 62.379 ± 0.935 | 53.595 ± 1.226 |
| Qwen3-VL | gated_visual_prompt_sum | one_hidden | 85.916 ± 0.719 | 58.811 ± 1.111 | 45.567 ± 2.385 |
| Qwen3-VL | gated_visual_prompt_sum | xgb | 89.257 ± 0.000 | 64.507 ± 0.000 | 54.217 ± 0.000 |
| Qwen3-VL | gated_visual_prompt_sum | rf | 86.359 ± 0.078 | 61.325 ± 0.081 | 46.431 ± 0.595 |
| Qwen3-VL | gated_generation | three_hidden | 86.598 ± 0.860 | 58.313 ± 2.412 | 51.527 ± 2.700 |
| Qwen3-VL | gated_generation | one_hidden | 83.302 ± 0.091 | 51.289 ± 0.377 | 33.908 ± 1.323 |
| Qwen3-VL | gated_generation | xgb | 87.017 ± 0.000 | 58.465 ± 0.000 | 50.301 ± 0.000 |
| Qwen3-VL | gated_generation | rf | 84.710 ± 0.072 | 54.407 ± 0.359 | 42.054 ± 0.392 |
| Qwen3-VL | gated_vp_generation | three_hidden | 89.665 ± 0.073 | 66.411 ± 0.973 | 60.678 ± 2.367 |
| Qwen3-VL | gated_vp_generation | one_hidden | 88.189 ± 0.314 | 63.007 ± 0.454 | 52.162 ± 3.350 |
| Qwen3-VL | gated_vp_generation | xgb | 90.996 ± 0.000 | 69.646 ± 0.000 | 59.070 ± 0.000 |
| Qwen3-VL | gated_vp_generation | rf | 87.607 ± 0.038 | 63.676 ± 0.118 | 49.228 ± 0.322 |
| InternVL-2.5 | attention_visual | three_hidden | 85.244 ± 0.199 | 52.349 ± 0.492 | 44.158 ± 1.600 |
| InternVL-2.5 | attention_visual | one_hidden | 84.534 ± 0.330 | 52.371 ± 0.266 | 39.052 ± 5.537 |
| InternVL-2.5 | attention_visual | xgb | 83.435 ± 0.000 | 50.118 ± 0.000 | 32.335 ± 0.000 |
| InternVL-2.5 | attention_visual | rf | 83.184 ± 0.153 | 48.056 ± 0.186 | 25.187 ± 1.085 |
| InternVL-2.5 | attention_visual_prompt_sum | three_hidden | 84.880 ± 0.686 | 51.543 ± 1.375 | 41.340 ± 6.602 |
| InternVL-2.5 | attention_visual_prompt_sum | one_hidden | 81.879 ± 0.714 | 46.562 ± 1.883 | 20.556 ± 4.645 |
| InternVL-2.5 | attention_visual_prompt_sum | xgb | 85.704 ± 0.000 | 53.461 ± 0.000 | 41.016 ± 0.000 |
| InternVL-2.5 | attention_visual_prompt_sum | rf | 85.021 ± 0.125 | 51.738 ± 0.118 | 29.749 ± 1.572 |
| InternVL-2.5 | attention_generation | three_hidden | 84.467 ± 0.503 | 49.978 ± 1.162 | 39.876 ± 4.198 |
| InternVL-2.5 | attention_generation | one_hidden | 83.152 ± 0.258 | 48.594 ± 0.999 | 24.527 ± 2.268 |
| InternVL-2.5 | attention_generation | xgb | 85.311 ± 0.000 | 51.019 ± 0.000 | 36.047 ± 0.000 |
| InternVL-2.5 | attention_generation | rf | 83.879 ± 0.107 | 49.242 ± 0.405 | 19.812 ± 4.262 |
| InternVL-2.5 | attention_vp_generation | three_hidden | 86.695 ± 0.218 | 54.417 ± 0.469 | 49.615 ± 2.474 |
| InternVL-2.5 | attention_vp_generation | one_hidden | 84.626 ± 0.078 | 51.495 ± 0.173 | 37.402 ± 3.889 |
| InternVL-2.5 | attention_vp_generation | xgb | 87.090 ± 0.000 | 56.294 ± 0.000 | 42.066 ± 0.000 |
| InternVL-2.5 | attention_vp_generation | rf | 85.275 ± 0.044 | 52.336 ± 0.107 | 30.305 ± 3.013 |
| InternVL-2.5 | gated_visual | three_hidden | 85.419 ± 0.192 | 52.891 ± 1.059 | 44.311 ± 5.061 |
| InternVL-2.5 | gated_visual | one_hidden | 84.475 ± 0.341 | 51.955 ± 0.361 | 44.964 ± 2.521 |
| InternVL-2.5 | gated_visual | xgb | 84.483 ± 0.000 | 52.226 ± 0.000 | 36.502 ± 0.000 |
| InternVL-2.5 | gated_visual | rf | 83.944 ± 0.076 | 50.766 ± 0.223 | 36.768 ± 0.356 |
| InternVL-2.5 | gated_visual_prompt_sum | three_hidden | 85.402 ± 0.327 | 52.491 ± 2.117 | 46.161 ± 1.507 |
| InternVL-2.5 | gated_visual_prompt_sum | one_hidden | 82.425 ± 0.351 | 47.806 ± 0.829 | 18.076 ± 5.051 |
| InternVL-2.5 | gated_visual_prompt_sum | xgb | 85.665 ± 0.000 | 52.423 ± 0.000 | 38.346 ± 0.000 |
| InternVL-2.5 | gated_visual_prompt_sum | rf | 85.374 ± 0.095 | 53.021 ± 0.188 | 30.923 ± 2.406 |
| InternVL-2.5 | gated_generation | three_hidden | 84.718 ± 0.239 | 50.902 ± 0.361 | 43.379 ± 1.433 |
| InternVL-2.5 | gated_generation | one_hidden | 81.749 ± 0.424 | 43.269 ± 1.830 | 9.137 ± 8.316 |
| InternVL-2.5 | gated_generation | xgb | 84.812 ± 0.000 | 51.268 ± 0.000 | 35.363 ± 0.000 |
| InternVL-2.5 | gated_generation | rf | 83.594 ± 0.051 | 48.351 ± 0.169 | 14.963 ± 1.395 |
| InternVL-2.5 | gated_vp_generation | three_hidden | 86.599 ± 0.558 | 56.160 ± 0.570 | 53.479 ± 0.798 |
| InternVL-2.5 | gated_vp_generation | one_hidden | 84.825 ± 0.281 | 52.237 ± 0.690 | 36.957 ± 4.959 |
| InternVL-2.5 | gated_vp_generation | xgb | 87.338 ± 0.000 | 56.765 ± 0.000 | 44.123 ± 0.000 |
| InternVL-2.5 | gated_vp_generation | rf | 85.552 ± 0.064 | 53.286 ± 0.590 | 41.874 ± 13.700 |

trees_selected_params.csv/json包含每组验证选中参数。每候选的验证指标/耗时/警告、selection、三seed模型和概率位于各模型xgb/rf子目录。
原始数据与划分复用，不做SHA、bootstrap置信区间或全量checkpoint独立审计。RF自身按默认bootstrap=True训练。重复使用原holdout，报告为探索性点估计，不作显著性或因果声明。
运行：`python scripts/train_attention_log_strength_trees.py --stage search --workers 16`；完成后`--stage summarize`。
