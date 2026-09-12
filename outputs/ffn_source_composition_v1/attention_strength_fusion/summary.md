# 对应attention与完整S_g配对拼接检测

四模型原4000图/3200-800/全部mentions，原MLP/seeds43-45；只训练20融合组×4模型×3seed=240新头，原单独结果复用。
attention/gated各raw/log1p两套，每套prompt/generation/visual、三组全拼接、V+P相加五组。
同尺度配对：[A_g,S_g]、[U_g,S_g]、[log1p(A_g),log1p(S_g)]、[log1p(U_g),log1p(S_g)]；没有跨尺度组合。
单组/相加为2L维，全拼接6L维，按[attention的P/G/V全层,S的P/G/V全层]顺序。V+P先各自求和再log1p。
S_g是K50完整来源贡献范数之和，视觉为S_C；BOS并入prompt；gate沿用完整前缀MAD。MiniGPT/Shikra尚无完整S_g缓存，本轮未混用旧e_m强度。
以下为三seed均值±总体std（%），HALL-F1用各seed训练REAL-F1阈值；不使用ensemble，不调参/bootstrap。

三组全拼接总览，AUROC（%）：

| 模型 | A+S 原值 | U+S 原值 | log1p(A)+log1p(S) | log1p(U)+log1p(S) |
|---|---:|---:|---:|---:|
| Qwen2.5-VL | 85.486 ± 0.528 | 86.309 ± 0.398 | 85.405 ± 0.422 | 86.796 ± 0.661 |
| LLaVA-1.5 | 90.087 ± 0.039 | 90.482 ± 0.221 | 90.125 ± 0.166 | 90.296 ± 0.073 |
| Qwen3-VL | 90.099 ± 0.516 | 90.525 ± 0.121 | 90.083 ± 0.136 | 90.878 ± 0.366 |
| InternVL-2.5 | 87.333 ± 0.275 | 87.280 ± 0.172 | 86.939 ± 0.206 | 86.818 ± 0.452 |

各模型预设融合组中的最高平均AUROC：

| 模型 | 融合组 | AUROC | HALL-AUPR |
|---|---|---:|---:|
| Qwen2.5-VL | gated_log1p_concat | 86.796 ± 0.661 | 44.449 ± 0.928 |
| LLaVA-1.5 | gated_raw_concat | 90.482 ± 0.221 | 72.194 ± 0.252 |
| Qwen3-VL | gated_log1p_concat | 90.878 ± 0.366 | 68.972 ± 0.180 |
| InternVL-2.5 | gated_raw_prompt | 88.282 ± 0.184 | 59.672 ± 0.880 |

结果：LLaVA原值gate全拼接、Qwen3双方log1p的gate全拼接、InternVL原值gate prompt配对，平均AUROC/AP均超过各自两个对应单独基线，三个seed的AUROC差也均为正。Qwen2本轮最高AUROC融合组仍低于对应gate单独使用（86.796 vs 88.032），三个seed均下降；融合没有跨模型统一增益。最高均值只作本轮预设组的描述性对比，未据此修改训练配置或宣称显著性。

A为attention区域和，U为attention×gate区域和；每一列都包含对应S_g。完整单组及相加组如下。

| 模型 | 融合组 | AUROC | HALL-AUPR | HALL-F1 |
|---|---|---:|---:|---:|
| Qwen2.5-VL | attention_raw_prompt | 82.912 ± 0.437 | 36.352 ± 1.169 | 21.007 ± 10.313 |
| Qwen2.5-VL | attention_raw_generation | 80.449 ± 0.401 | 35.174 ± 1.618 | 19.561 ± 7.810 |
| Qwen2.5-VL | attention_raw_visual | 85.063 ± 0.432 | 38.085 ± 0.518 | 25.610 ± 6.982 |
| Qwen2.5-VL | attention_raw_concat | 85.486 ± 0.528 | 43.824 ± 0.954 | 42.374 ± 0.770 |
| Qwen2.5-VL | attention_raw_visual_prompt_sum | 83.874 ± 0.068 | 36.773 ± 1.203 | 15.702 ± 0.309 |
| Qwen2.5-VL | attention_log1p_prompt | 84.367 ± 0.115 | 38.825 ± 0.710 | 29.042 ± 2.057 |
| Qwen2.5-VL | attention_log1p_generation | 82.127 ± 1.045 | 37.789 ± 2.002 | 31.791 ± 3.084 |
| Qwen2.5-VL | attention_log1p_visual | 84.597 ± 0.305 | 38.984 ± 0.828 | 33.019 ± 1.736 |
| Qwen2.5-VL | attention_log1p_concat | 85.405 ± 0.422 | 40.901 ± 1.578 | 41.766 ± 4.515 |
| Qwen2.5-VL | attention_log1p_visual_prompt_sum | 84.438 ± 1.069 | 39.822 ± 1.203 | 31.602 ± 3.152 |
| Qwen2.5-VL | gated_raw_prompt | 83.981 ± 0.432 | 38.163 ± 1.153 | 22.400 ± 7.763 |
| Qwen2.5-VL | gated_raw_generation | 80.495 ± 0.421 | 35.421 ± 1.383 | 18.305 ± 2.492 |
| Qwen2.5-VL | gated_raw_visual | 85.398 ± 0.168 | 38.311 ± 0.346 | 19.437 ± 2.071 |
| Qwen2.5-VL | gated_raw_concat | 86.309 ± 0.398 | 45.278 ± 0.528 | 43.565 ± 0.376 |
| Qwen2.5-VL | gated_raw_visual_prompt_sum | 84.686 ± 0.152 | 38.573 ± 0.671 | 23.739 ± 6.771 |
| Qwen2.5-VL | gated_log1p_prompt | 84.847 ± 0.472 | 39.568 ± 1.228 | 29.170 ± 7.254 |
| Qwen2.5-VL | gated_log1p_generation | 83.583 ± 0.234 | 39.842 ± 0.504 | 34.595 ± 1.027 |
| Qwen2.5-VL | gated_log1p_visual | 84.844 ± 0.568 | 40.023 ± 0.717 | 38.466 ± 2.992 |
| Qwen2.5-VL | gated_log1p_concat | 86.796 ± 0.661 | 44.449 ± 0.928 | 43.957 ± 1.989 |
| Qwen2.5-VL | gated_log1p_visual_prompt_sum | 85.910 ± 0.652 | 42.552 ± 1.186 | 27.550 ± 5.537 |
| LLaVA-1.5 | attention_raw_prompt | 89.965 ± 0.026 | 71.203 ± 0.081 | 64.250 ± 0.473 |
| LLaVA-1.5 | attention_raw_generation | 88.365 ± 0.122 | 65.287 ± 0.370 | 60.854 ± 1.961 |
| LLaVA-1.5 | attention_raw_visual | 88.948 ± 0.071 | 68.798 ± 0.145 | 62.476 ± 1.192 |
| LLaVA-1.5 | attention_raw_concat | 90.087 ± 0.039 | 70.669 ± 0.822 | 65.510 ± 0.708 |
| LLaVA-1.5 | attention_raw_visual_prompt_sum | 89.681 ± 0.086 | 70.887 ± 0.399 | 63.584 ± 0.059 |
| LLaVA-1.5 | attention_log1p_prompt | 89.766 ± 0.190 | 70.614 ± 0.526 | 64.038 ± 1.528 |
| LLaVA-1.5 | attention_log1p_generation | 88.768 ± 0.088 | 66.264 ± 0.049 | 62.753 ± 0.506 |
| LLaVA-1.5 | attention_log1p_visual | 89.316 ± 0.247 | 68.527 ± 1.039 | 64.619 ± 0.632 |
| LLaVA-1.5 | attention_log1p_concat | 90.125 ± 0.166 | 70.227 ± 0.999 | 64.826 ± 0.171 |
| LLaVA-1.5 | attention_log1p_visual_prompt_sum | 89.738 ± 0.102 | 70.325 ± 0.251 | 64.363 ± 1.541 |
| LLaVA-1.5 | gated_raw_prompt | 89.164 ± 0.206 | 70.562 ± 0.504 | 61.135 ± 1.197 |
| LLaVA-1.5 | gated_raw_generation | 88.062 ± 0.092 | 64.878 ± 0.199 | 60.220 ± 1.573 |
| LLaVA-1.5 | gated_raw_visual | 89.120 ± 0.061 | 69.492 ± 0.243 | 63.714 ± 0.630 |
| LLaVA-1.5 | gated_raw_concat | 90.482 ± 0.221 | 72.194 ± 0.252 | 66.982 ± 1.360 |
| LLaVA-1.5 | gated_raw_visual_prompt_sum | 89.128 ± 0.175 | 69.396 ± 0.534 | 63.249 ± 0.620 |
| LLaVA-1.5 | gated_log1p_prompt | 89.318 ± 0.186 | 70.617 ± 0.413 | 61.147 ± 1.140 |
| LLaVA-1.5 | gated_log1p_generation | 88.525 ± 0.125 | 66.814 ± 0.415 | 60.497 ± 0.452 |
| LLaVA-1.5 | gated_log1p_visual | 89.378 ± 0.019 | 68.862 ± 0.366 | 64.113 ± 0.599 |
| LLaVA-1.5 | gated_log1p_concat | 90.296 ± 0.073 | 71.258 ± 0.727 | 65.723 ± 1.644 |
| LLaVA-1.5 | gated_log1p_visual_prompt_sum | 89.673 ± 0.080 | 70.394 ± 0.534 | 63.922 ± 0.522 |
| Qwen3-VL | attention_raw_prompt | 86.418 ± 0.625 | 57.675 ± 1.435 | 46.926 ± 3.430 |
| Qwen3-VL | attention_raw_generation | 86.329 ± 0.265 | 57.210 ± 0.930 | 53.112 ± 1.917 |
| Qwen3-VL | attention_raw_visual | 86.563 ± 0.224 | 56.254 ± 0.347 | 51.623 ± 1.053 |
| Qwen3-VL | attention_raw_concat | 90.099 ± 0.516 | 65.673 ± 1.379 | 61.112 ± 1.292 |
| Qwen3-VL | attention_raw_visual_prompt_sum | 87.005 ± 0.162 | 59.248 ± 1.003 | 47.679 ± 0.885 |
| Qwen3-VL | attention_log1p_prompt | 87.992 ± 0.879 | 61.605 ± 1.915 | 58.147 ± 1.749 |
| Qwen3-VL | attention_log1p_generation | 87.521 ± 0.117 | 59.911 ± 1.814 | 53.832 ± 1.202 |
| Qwen3-VL | attention_log1p_visual | 87.309 ± 0.464 | 59.198 ± 0.300 | 53.490 ± 0.341 |
| Qwen3-VL | attention_log1p_concat | 90.083 ± 0.136 | 66.941 ± 0.753 | 61.773 ± 1.280 |
| Qwen3-VL | attention_log1p_visual_prompt_sum | 85.149 ± 1.977 | 57.753 ± 3.064 | 50.607 ± 5.999 |
| Qwen3-VL | gated_raw_prompt | 85.503 ± 1.743 | 57.518 ± 2.058 | 49.441 ± 3.871 |
| Qwen3-VL | gated_raw_generation | 86.470 ± 0.427 | 58.002 ± 1.349 | 54.758 ± 1.544 |
| Qwen3-VL | gated_raw_visual | 86.729 ± 0.150 | 56.954 ± 0.194 | 51.683 ± 0.865 |
| Qwen3-VL | gated_raw_concat | 90.525 ± 0.121 | 66.631 ± 0.447 | 62.075 ± 0.922 |
| Qwen3-VL | gated_raw_visual_prompt_sum | 86.991 ± 0.089 | 59.666 ± 0.361 | 50.991 ± 1.596 |
| Qwen3-VL | gated_log1p_prompt | 87.398 ± 0.749 | 62.444 ± 1.041 | 56.917 ± 2.417 |
| Qwen3-VL | gated_log1p_generation | 86.322 ± 1.237 | 57.730 ± 3.799 | 49.885 ± 5.416 |
| Qwen3-VL | gated_log1p_visual | 87.679 ± 0.313 | 59.425 ± 0.392 | 54.519 ± 1.249 |
| Qwen3-VL | gated_log1p_concat | 90.878 ± 0.366 | 68.972 ± 0.180 | 64.171 ± 0.467 |
| Qwen3-VL | gated_log1p_visual_prompt_sum | 87.989 ± 0.194 | 62.627 ± 0.416 | 56.273 ± 0.837 |
| InternVL-2.5 | attention_raw_prompt | 87.656 ± 0.285 | 56.295 ± 1.271 | 46.127 ± 1.552 |
| InternVL-2.5 | attention_raw_generation | 83.581 ± 0.297 | 47.868 ± 1.071 | 30.255 ± 4.809 |
| InternVL-2.5 | attention_raw_visual | 83.886 ± 0.234 | 49.982 ± 0.866 | 35.163 ± 9.001 |
| InternVL-2.5 | attention_raw_concat | 87.333 ± 0.275 | 58.109 ± 0.562 | 53.471 ± 0.976 |
| InternVL-2.5 | attention_raw_visual_prompt_sum | 85.514 ± 0.255 | 52.603 ± 0.500 | 39.484 ± 2.512 |
| InternVL-2.5 | attention_log1p_prompt | 85.194 ± 1.683 | 51.936 ± 4.119 | 41.242 ± 4.477 |
| InternVL-2.5 | attention_log1p_generation | 85.174 ± 0.406 | 51.531 ± 1.081 | 43.694 ± 0.348 |
| InternVL-2.5 | attention_log1p_visual | 84.947 ± 0.542 | 51.677 ± 0.812 | 40.275 ± 0.704 |
| InternVL-2.5 | attention_log1p_concat | 86.939 ± 0.206 | 54.592 ± 1.325 | 49.226 ± 2.080 |
| InternVL-2.5 | attention_log1p_visual_prompt_sum | 84.587 ± 0.756 | 51.371 ± 1.251 | 42.421 ± 5.545 |
| InternVL-2.5 | gated_raw_prompt | 88.282 ± 0.184 | 59.672 ± 0.880 | 50.218 ± 1.509 |
| InternVL-2.5 | gated_raw_generation | 83.664 ± 0.509 | 48.826 ± 1.350 | 36.635 ± 4.916 |
| InternVL-2.5 | gated_raw_visual | 84.311 ± 0.060 | 50.404 ± 0.611 | 42.168 ± 2.861 |
| InternVL-2.5 | gated_raw_concat | 87.280 ± 0.172 | 58.627 ± 0.640 | 51.075 ± 0.544 |
| InternVL-2.5 | gated_raw_visual_prompt_sum | 85.698 ± 0.200 | 53.132 ± 0.792 | 43.271 ± 3.263 |
| InternVL-2.5 | gated_log1p_prompt | 86.677 ± 1.199 | 55.643 ± 2.362 | 46.762 ± 5.144 |
| InternVL-2.5 | gated_log1p_generation | 84.748 ± 0.158 | 51.051 ± 1.006 | 41.312 ± 3.162 |
| InternVL-2.5 | gated_log1p_visual | 84.994 ± 0.226 | 51.942 ± 0.233 | 46.883 ± 0.934 |
| InternVL-2.5 | gated_log1p_concat | 86.818 ± 0.452 | 56.468 ± 2.340 | 52.145 ± 2.588 |
| InternVL-2.5 | gated_log1p_visual_prompt_sum | 85.353 ± 1.419 | 52.853 ± 2.824 | 45.512 ± 3.459 |

comparisons.csv逐组报告融合相对对应attention/gated单独、S_g单独的平均指标差和逐seed AUROC差；增量为原始比例单位。
输入来自同一批目标但不同提取缓存，原mention顺序、目标ID/标签/划分已核对。两路源各自既有数值与gate口径限制保留；维度增加，不据提升宣称显著性或因果。
运行：`python scripts/train_attention_strength_fusion.py --stage prepare`，再`--stage train --models <模型> --device cuda:0`；完成后`--stage summarize`。
各模型保存matrices.pt/protocol.json/heads/detection.json，根目录detection.csv和seed_metrics.csv为均值/std与逐seed双阈值。
