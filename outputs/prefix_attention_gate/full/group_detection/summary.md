# 分组attention与attention×gate检测

六模型原4000图/3200-800/全部mentions，按原训练mention顺序；prompt含BOS、模板及特殊token，generation仅为已生成前缀。
attention为跨头均值的原始注意力，gated为其乘完整因果前缀MAD gate；按组直接求和，不再归一化、不除token数。
各自raw/log1p两套，每套prompt/generation/visual单独、三组concat、visual+prompt相加五组。先相加再log1p；未跨attention与gated混合拼接。
原MLP/seeds43-45，共360头。以下三seed均值±总体std（%），HALL-F1采用每seed训练REAL-F1阈值；原双阈值见CSV。
原始attention三组总量约1，V+P约等于1-G，属于近似冗余信号；不据分类器小差异声称新增信息。gate的总量则可变化，且本轮gate统计范围不同于原视觉AE。

三组拼接总览，AUROC均值±总体std（%）：

| 模型 | attention 原值 | attention×gate 原值 | attention log1p | attention×gate log1p |
|---|---:|---:|---:|---:|
| MiniGPT-4 | 90.605 ± 0.138 | 90.879 ± 0.059 | 90.393 ± 0.244 | 90.878 ± 0.063 |
| Shikra | 85.039 ± 0.321 | 86.081 ± 0.162 | 84.941 ± 0.313 | 86.001 ± 0.226 |
| Qwen2.5-VL | 84.533 ± 0.346 | 87.475 ± 0.489 | 84.567 ± 0.250 | 88.032 ± 0.351 |
| LLaVA-1.5 | 88.954 ± 0.388 | 89.541 ± 0.199 | 89.105 ± 0.355 | 89.611 ± 0.124 |
| Qwen3-VL | 88.091 ± 0.247 | 89.904 ± 0.245 | 88.203 ± 0.483 | 90.165 ± 0.282 |
| InternVL-2.5 | 86.007 ± 0.107 | 87.412 ± 0.171 | 85.503 ± 0.531 | 87.385 ± 0.248 |

原值和log1p两套中，gate拼接六模型的平均AUROC/AP均高于对应attention拼接，也均高于各自最佳单组。纯attention的原值拼接在Shikra略低于最佳单组（AUROC -0.052点，AP -0.433点），其他模型提高；log1p下Shikra仅小幅变化。log1p没有跨模型一致收益；以上均为三seed点估计，未作显著性检验。

以下为全部单组、相加及拼接结果：

| 模型 | 输入 | AUROC | HALL-AUPR | HALL-F1 |
|---|---|---:|---:|---:|
| MiniGPT-4 | attention_raw_prompt | 89.209 ± 0.196 | 61.475 ± 0.712 | 52.915 ± 0.569 |
| MiniGPT-4 | attention_raw_generation | 87.920 ± 0.327 | 56.004 ± 1.002 | 48.345 ± 3.587 |
| MiniGPT-4 | attention_raw_visual | 88.433 ± 0.441 | 57.335 ± 0.959 | 47.162 ± 6.622 |
| MiniGPT-4 | attention_raw_concat | 90.605 ± 0.138 | 66.028 ± 0.459 | 60.952 ± 1.601 |
| MiniGPT-4 | attention_raw_visual_prompt_sum | 87.636 ± 0.167 | 55.392 ± 0.802 | 44.874 ± 1.841 |
| MiniGPT-4 | attention_log1p_prompt | 88.945 ± 0.384 | 61.373 ± 0.666 | 49.273 ± 3.699 |
| MiniGPT-4 | attention_log1p_generation | 87.875 ± 0.118 | 55.776 ± 0.609 | 45.271 ± 1.508 |
| MiniGPT-4 | attention_log1p_visual | 88.671 ± 0.288 | 57.611 ± 0.792 | 45.324 ± 3.169 |
| MiniGPT-4 | attention_log1p_concat | 90.393 ± 0.244 | 65.635 ± 1.157 | 60.983 ± 1.211 |
| MiniGPT-4 | attention_log1p_visual_prompt_sum | 87.823 ± 0.087 | 55.934 ± 1.040 | 44.050 ± 0.705 |
| MiniGPT-4 | gated_raw_prompt | 87.406 ± 0.118 | 58.894 ± 0.662 | 41.086 ± 0.610 |
| MiniGPT-4 | gated_raw_generation | 88.373 ± 0.110 | 59.306 ± 0.800 | 43.717 ± 4.172 |
| MiniGPT-4 | gated_raw_visual | 88.771 ± 0.251 | 61.868 ± 0.315 | 48.373 ± 4.406 |
| MiniGPT-4 | gated_raw_concat | 90.879 ± 0.059 | 67.161 ± 0.111 | 59.956 ± 1.565 |
| MiniGPT-4 | gated_raw_visual_prompt_sum | 87.833 ± 0.389 | 57.631 ± 0.656 | 44.884 ± 0.566 |
| MiniGPT-4 | gated_log1p_prompt | 87.100 ± 0.074 | 57.421 ± 0.178 | 38.945 ± 2.601 |
| MiniGPT-4 | gated_log1p_generation | 88.532 ± 0.104 | 59.617 ± 0.748 | 48.367 ± 4.543 |
| MiniGPT-4 | gated_log1p_visual | 88.751 ± 0.224 | 61.774 ± 0.201 | 40.032 ± 4.390 |
| MiniGPT-4 | gated_log1p_concat | 90.878 ± 0.063 | 67.303 ± 0.217 | 61.964 ± 1.394 |
| MiniGPT-4 | gated_log1p_visual_prompt_sum | 87.891 ± 0.319 | 58.182 ± 0.679 | 44.069 ± 2.636 |
| MiniGPT-4 | F_E_reference | 90.907 ± 0.148 | 65.486 ± 1.035 | 59.354 ± 1.769 |
| Shikra | attention_raw_prompt | 84.437 ± 0.138 | 58.610 ± 0.591 | 54.163 ± 0.212 |
| Shikra | attention_raw_generation | 85.091 ± 0.030 | 59.070 ± 0.412 | 52.584 ± 0.970 |
| Shikra | attention_raw_visual | 83.422 ± 0.486 | 55.227 ± 0.133 | 46.449 ± 2.138 |
| Shikra | attention_raw_concat | 85.039 ± 0.321 | 58.637 ± 0.678 | 54.586 ± 1.520 |
| Shikra | attention_raw_visual_prompt_sum | 85.259 ± 0.118 | 58.499 ± 0.305 | 52.172 ± 3.962 |
| Shikra | attention_log1p_prompt | 84.383 ± 0.125 | 58.633 ± 0.896 | 51.088 ± 1.327 |
| Shikra | attention_log1p_generation | 84.896 ± 0.047 | 58.633 ± 0.076 | 54.666 ± 1.579 |
| Shikra | attention_log1p_visual | 83.677 ± 0.071 | 55.226 ± 0.445 | 47.990 ± 1.664 |
| Shikra | attention_log1p_concat | 84.941 ± 0.313 | 58.666 ± 0.297 | 53.084 ± 0.634 |
| Shikra | attention_log1p_visual_prompt_sum | 85.098 ± 0.110 | 58.129 ± 0.331 | 51.573 ± 0.983 |
| Shikra | gated_raw_prompt | 83.759 ± 0.132 | 55.172 ± 0.498 | 40.341 ± 2.969 |
| Shikra | gated_raw_generation | 84.986 ± 0.260 | 58.839 ± 0.988 | 53.685 ± 1.948 |
| Shikra | gated_raw_visual | 82.678 ± 0.277 | 53.986 ± 0.186 | 40.430 ± 1.040 |
| Shikra | gated_raw_concat | 86.081 ± 0.162 | 60.918 ± 0.046 | 53.035 ± 1.562 |
| Shikra | gated_raw_visual_prompt_sum | 83.484 ± 0.077 | 56.068 ± 0.084 | 42.513 ± 3.095 |
| Shikra | gated_log1p_prompt | 83.782 ± 0.173 | 55.435 ± 0.542 | 38.142 ± 3.249 |
| Shikra | gated_log1p_generation | 84.870 ± 0.218 | 58.444 ± 0.910 | 52.349 ± 1.451 |
| Shikra | gated_log1p_visual | 82.687 ± 0.272 | 54.087 ± 0.642 | 43.033 ± 2.270 |
| Shikra | gated_log1p_concat | 86.001 ± 0.226 | 61.100 ± 0.513 | 52.056 ± 3.862 |
| Shikra | gated_log1p_visual_prompt_sum | 83.723 ± 0.131 | 56.349 ± 0.095 | 43.640 ± 1.571 |
| Shikra | F_E_reference | 86.265 ± 0.308 | 60.481 ± 0.794 | 55.992 ± 1.056 |
| Qwen2.5-VL | attention_raw_prompt | 83.437 ± 0.391 | 38.491 ± 1.754 | 33.048 ± 1.937 |
| Qwen2.5-VL | attention_raw_generation | 81.297 ± 0.130 | 32.817 ± 1.252 | 15.694 ± 2.640 |
| Qwen2.5-VL | attention_raw_visual | 81.787 ± 0.663 | 33.479 ± 1.388 | 14.007 ± 4.805 |
| Qwen2.5-VL | attention_raw_concat | 84.533 ± 0.346 | 40.933 ± 1.321 | 39.195 ± 0.253 |
| Qwen2.5-VL | attention_raw_visual_prompt_sum | 80.911 ± 0.651 | 32.420 ± 2.613 | 20.513 ± 2.832 |
| Qwen2.5-VL | attention_log1p_prompt | 83.387 ± 0.332 | 38.610 ± 1.306 | 33.467 ± 1.864 |
| Qwen2.5-VL | attention_log1p_generation | 80.815 ± 0.272 | 31.974 ± 1.589 | 9.499 ± 9.045 |
| Qwen2.5-VL | attention_log1p_visual | 82.313 ± 0.256 | 34.337 ± 0.850 | 20.192 ± 5.454 |
| Qwen2.5-VL | attention_log1p_concat | 84.567 ± 0.250 | 40.948 ± 1.373 | 37.574 ± 0.083 |
| Qwen2.5-VL | attention_log1p_visual_prompt_sum | 80.975 ± 0.216 | 32.978 ± 1.641 | 22.304 ± 6.695 |
| Qwen2.5-VL | gated_raw_prompt | 84.378 ± 0.425 | 41.836 ± 0.224 | 32.216 ± 2.401 |
| Qwen2.5-VL | gated_raw_generation | 81.789 ± 0.087 | 35.890 ± 0.327 | 26.251 ± 2.719 |
| Qwen2.5-VL | gated_raw_visual | 82.829 ± 0.452 | 36.132 ± 0.529 | 17.367 ± 4.449 |
| Qwen2.5-VL | gated_raw_concat | 87.475 ± 0.489 | 48.195 ± 1.564 | 45.643 ± 2.551 |
| Qwen2.5-VL | gated_raw_visual_prompt_sum | 83.298 ± 0.363 | 42.372 ± 0.879 | 32.395 ± 1.920 |
| Qwen2.5-VL | gated_log1p_prompt | 84.233 ± 0.270 | 41.641 ± 0.260 | 29.975 ± 4.545 |
| Qwen2.5-VL | gated_log1p_generation | 82.423 ± 0.553 | 36.908 ± 1.241 | 30.790 ± 4.336 |
| Qwen2.5-VL | gated_log1p_visual | 82.867 ± 0.660 | 36.350 ± 0.517 | 15.725 ± 0.983 |
| Qwen2.5-VL | gated_log1p_concat | 88.032 ± 0.351 | 49.342 ± 0.838 | 45.628 ± 1.931 |
| Qwen2.5-VL | gated_log1p_visual_prompt_sum | 83.171 ± 0.353 | 41.752 ± 1.656 | 31.238 ± 3.272 |
| Qwen2.5-VL | F_E_reference | 87.449 ± 0.085 | 43.982 ± 1.678 | 38.401 ± 3.469 |
| LLaVA-1.5 | attention_raw_prompt | 87.872 ± 0.102 | 64.717 ± 0.405 | 58.795 ± 0.789 |
| LLaVA-1.5 | attention_raw_generation | 88.057 ± 0.087 | 64.977 ± 0.639 | 57.756 ± 3.517 |
| LLaVA-1.5 | attention_raw_visual | 87.960 ± 0.215 | 65.996 ± 0.096 | 62.046 ± 1.425 |
| LLaVA-1.5 | attention_raw_concat | 88.954 ± 0.388 | 66.515 ± 0.702 | 62.311 ± 2.393 |
| LLaVA-1.5 | attention_raw_visual_prompt_sum | 88.013 ± 0.060 | 64.806 ± 0.311 | 57.404 ± 0.950 |
| LLaVA-1.5 | attention_log1p_prompt | 87.869 ± 0.062 | 64.380 ± 0.300 | 58.363 ± 1.845 |
| LLaVA-1.5 | attention_log1p_generation | 88.027 ± 0.126 | 64.846 ± 0.731 | 56.666 ± 1.945 |
| LLaVA-1.5 | attention_log1p_visual | 88.059 ± 0.213 | 65.908 ± 0.246 | 63.514 ± 0.526 |
| LLaVA-1.5 | attention_log1p_concat | 89.105 ± 0.355 | 67.420 ± 0.524 | 64.195 ± 0.837 |
| LLaVA-1.5 | attention_log1p_visual_prompt_sum | 87.148 ± 0.398 | 63.152 ± 0.856 | 56.762 ± 1.005 |
| LLaVA-1.5 | gated_raw_prompt | 84.818 ± 0.179 | 56.538 ± 0.719 | 44.909 ± 3.567 |
| LLaVA-1.5 | gated_raw_generation | 87.865 ± 0.204 | 65.406 ± 0.281 | 58.083 ± 2.219 |
| LLaVA-1.5 | gated_raw_visual | 87.871 ± 0.141 | 66.323 ± 0.396 | 60.285 ± 0.066 |
| LLaVA-1.5 | gated_raw_concat | 89.541 ± 0.199 | 69.361 ± 0.172 | 62.967 ± 0.503 |
| LLaVA-1.5 | gated_raw_visual_prompt_sum | 87.213 ± 0.075 | 62.204 ± 0.168 | 56.666 ± 1.233 |
| LLaVA-1.5 | gated_log1p_prompt | 85.121 ± 0.128 | 57.135 ± 0.624 | 45.688 ± 2.513 |
| LLaVA-1.5 | gated_log1p_generation | 87.745 ± 0.088 | 65.053 ± 0.179 | 58.527 ± 0.830 |
| LLaVA-1.5 | gated_log1p_visual | 87.898 ± 0.095 | 66.280 ± 0.194 | 60.770 ± 0.447 |
| LLaVA-1.5 | gated_log1p_concat | 89.611 ± 0.124 | 69.473 ± 0.151 | 60.911 ± 1.630 |
| LLaVA-1.5 | gated_log1p_visual_prompt_sum | 87.132 ± 0.159 | 62.958 ± 0.501 | 55.418 ± 0.745 |
| LLaVA-1.5 | F_E_reference | 90.015 ± 0.272 | 70.815 ± 0.619 | 66.788 ± 0.073 |
| Qwen3-VL | attention_raw_prompt | 85.917 ± 0.246 | 58.886 ± 0.394 | 52.217 ± 0.864 |
| Qwen3-VL | attention_raw_generation | 85.228 ± 0.141 | 54.978 ± 0.807 | 47.300 ± 1.597 |
| Qwen3-VL | attention_raw_visual | 85.535 ± 0.116 | 56.905 ± 0.464 | 46.151 ± 0.441 |
| Qwen3-VL | attention_raw_concat | 88.091 ± 0.247 | 62.715 ± 0.934 | 57.336 ± 1.775 |
| Qwen3-VL | attention_raw_visual_prompt_sum | 85.141 ± 0.210 | 54.666 ± 0.283 | 44.709 ± 1.547 |
| Qwen3-VL | attention_log1p_prompt | 85.783 ± 0.271 | 58.482 ± 0.634 | 49.720 ± 3.434 |
| Qwen3-VL | attention_log1p_generation | 84.538 ± 0.267 | 53.741 ± 0.487 | 44.483 ± 2.757 |
| Qwen3-VL | attention_log1p_visual | 85.732 ± 0.172 | 57.198 ± 0.476 | 47.329 ± 1.660 |
| Qwen3-VL | attention_log1p_concat | 88.203 ± 0.483 | 62.668 ± 0.895 | 58.329 ± 1.024 |
| Qwen3-VL | attention_log1p_visual_prompt_sum | 84.380 ± 0.518 | 53.872 ± 0.827 | 46.866 ± 1.653 |
| Qwen3-VL | gated_raw_prompt | 82.857 ± 0.385 | 55.898 ± 0.101 | 44.965 ± 1.037 |
| Qwen3-VL | gated_raw_generation | 85.275 ± 0.363 | 56.903 ± 0.582 | 50.523 ± 1.220 |
| Qwen3-VL | gated_raw_visual | 85.915 ± 0.170 | 56.334 ± 0.499 | 45.957 ± 2.650 |
| Qwen3-VL | gated_raw_concat | 89.904 ± 0.245 | 66.928 ± 0.221 | 61.218 ± 0.704 |
| Qwen3-VL | gated_raw_visual_prompt_sum | 85.510 ± 0.253 | 58.233 ± 0.456 | 46.763 ± 1.805 |
| Qwen3-VL | gated_log1p_prompt | 83.168 ± 0.313 | 56.423 ± 0.602 | 44.976 ± 1.277 |
| Qwen3-VL | gated_log1p_generation | 85.310 ± 0.435 | 56.271 ± 0.671 | 49.142 ± 1.267 |
| Qwen3-VL | gated_log1p_visual | 86.004 ± 0.129 | 56.327 ± 0.313 | 49.704 ± 1.810 |
| Qwen3-VL | gated_log1p_concat | 90.165 ± 0.282 | 67.436 ± 0.673 | 60.912 ± 0.358 |
| Qwen3-VL | gated_log1p_visual_prompt_sum | 86.185 ± 0.150 | 59.334 ± 0.029 | 46.259 ± 0.918 |
| Qwen3-VL | F_E_reference | 88.655 ± 0.305 | 62.832 ± 0.353 | 58.839 ± 1.088 |
| InternVL-2.5 | attention_raw_prompt | 83.976 ± 0.477 | 47.214 ± 0.864 | 40.235 ± 3.529 |
| InternVL-2.5 | attention_raw_generation | 84.055 ± 0.112 | 46.330 ± 0.864 | 34.383 ± 3.658 |
| InternVL-2.5 | attention_raw_visual | 83.417 ± 0.371 | 46.925 ± 0.838 | 33.851 ± 1.065 |
| InternVL-2.5 | attention_raw_concat | 86.007 ± 0.107 | 51.775 ± 0.289 | 48.914 ± 2.222 |
| InternVL-2.5 | attention_raw_visual_prompt_sum | 82.199 ± 1.504 | 44.144 ± 1.719 | 27.503 ± 2.828 |
| InternVL-2.5 | attention_log1p_prompt | 83.244 ± 1.193 | 45.870 ± 1.032 | 32.477 ± 3.917 |
| InternVL-2.5 | attention_log1p_generation | 84.024 ± 0.304 | 46.118 ± 1.296 | 33.672 ± 3.245 |
| InternVL-2.5 | attention_log1p_visual | 83.322 ± 0.330 | 46.302 ± 0.745 | 31.545 ± 1.140 |
| InternVL-2.5 | attention_log1p_concat | 85.503 ± 0.531 | 50.948 ± 1.327 | 48.703 ± 1.254 |
| InternVL-2.5 | attention_log1p_visual_prompt_sum | 77.912 ± 3.186 | 40.063 ± 2.464 | 19.351 ± 2.035 |
| InternVL-2.5 | gated_raw_prompt | 83.872 ± 0.171 | 49.219 ± 0.604 | 39.999 ± 1.911 |
| InternVL-2.5 | gated_raw_generation | 83.355 ± 0.126 | 45.038 ± 0.666 | 35.263 ± 4.210 |
| InternVL-2.5 | gated_raw_visual | 83.542 ± 0.265 | 47.612 ± 0.224 | 33.987 ± 2.705 |
| InternVL-2.5 | gated_raw_concat | 87.412 ± 0.171 | 56.414 ± 0.507 | 55.130 ± 0.852 |
| InternVL-2.5 | gated_raw_visual_prompt_sum | 83.760 ± 0.439 | 46.799 ± 1.032 | 37.561 ± 6.634 |
| InternVL-2.5 | gated_log1p_prompt | 83.001 ± 0.689 | 47.273 ± 0.645 | 32.943 ± 3.677 |
| InternVL-2.5 | gated_log1p_generation | 83.501 ± 0.204 | 45.421 ± 0.882 | 38.061 ± 0.434 |
| InternVL-2.5 | gated_log1p_visual | 83.713 ± 0.312 | 47.329 ± 0.418 | 37.464 ± 3.430 |
| InternVL-2.5 | gated_log1p_concat | 87.385 ± 0.248 | 55.987 ± 1.203 | 54.342 ± 0.613 |
| InternVL-2.5 | gated_log1p_visual_prompt_sum | 80.977 ± 3.333 | 42.090 ± 5.032 | 29.054 ± 9.722 |
| InternVL-2.5 | F_E_reference | 85.825 ± 0.587 | 53.034 ± 1.201 | 51.378 ± 0.362 |

F_E_reference为已有AE+log1p(S_E)，仅作参照，未重新训练；本次输入不含AE或FFN强度。
所有曲线为4000图全部mentions等权，先变换后汇总，阴影为IQR而非置信区间；训练评估仍只使用原800图holdout。
[attention原值](all_attention_raw.png) · [attention log1p](all_attention_log1p.png) · [attention×gate原值](all_gated_raw.png) · [attention×gate log1p](all_gated_log1p.png)
逐层曲线数据curves.csv；逐seed双阈值seed_metrics.csv；均值/std detection.csv；描述性配对comparisons.csv。原训练汇总JSON兼容保留ensemble字段，主表不用ensemble。
运行：`python scripts/train_prefix_attention_groups.py --stage prepare`，然后`--stage train --models <列表> --device cuda:0`；完成后`--stage summarize`。
保留前缀长度/位置混杂及原生attention舍入限制，不作显著性或因果声明，不调参或bootstrap。
