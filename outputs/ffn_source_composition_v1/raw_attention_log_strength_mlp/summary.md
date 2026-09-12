# 原始attention + log1p(S)：固定MLP与单隐藏层网格搜索

四模型原4000图/3200-800/全部mentions。X为原始attention或原始attention×gate，只有S_g取log1p。
每种X四组：visual、visual_prompt_sum、generation，以及vp_generation=[完整VP组,完整G组]。前三组2L维，最后4L维。
固定对照为原Torch三隐藏层MLP；单隐藏层用项目sklearn.MLPClassifier，宽度64/128/256×学习率.01/.001×adam/sgd，max_iter500，共12候选。
每个模型/特征组独立选参：原训练3200图中2560/640按图片划分，seed43验证AUROC优先，平局HALL-AUPR；选定后3200图重训43/44/45，原800图只用于最终报告。
未额外标准化。sklearn与Torch还有BN/dropout、正则、batch和停止规则差异，结果不能只归因为隐藏层数。
三seed均值±总体std（%），非ensemble；HALL-F1采用各seed训练REAL-F1阈值，原双阈值详见CSV。

八组AUROC均值总览（%，原三隐藏层 → 搜参单隐藏层；标准差见下方完整表）：

| 特征 | Qwen2.5 | LLaVA | Qwen3 | InternVL |
|---|---:|---:|---:|---:|
| attention_visual | 84.624 → 82.914 | 89.379 → 89.009 | 87.516 → 86.883 | 85.244 → 84.534 |
| attention_visual_prompt_sum | 84.078 → 82.240 | 89.720 → 87.816 | 87.725 → 82.591 | 84.880 → 81.879 |
| attention_generation | 82.536 → 78.698 | 88.856 → 87.579 | 86.449 → 83.309 | 84.467 → 83.152 |
| attention_vp_generation | 84.675 → 82.207 | 90.021 → 89.433 | 89.289 → 85.745 | 86.695 → 84.626 |
| gated_visual | 84.973 → 82.719 | 89.501 → 88.828 | 87.744 → 87.126 | 85.419 → 84.475 |
| gated_visual_prompt_sum | 85.528 → 82.765 | 89.699 → 88.415 | 88.350 → 85.916 | 85.402 → 82.425 |
| gated_generation | 83.868 → 79.251 | 88.228 → 87.212 | 86.598 → 83.302 | 84.718 → 81.749 |
| gated_vp_generation | 86.635 → 83.111 | 90.470 → 89.601 | 89.665 → 88.189 | 86.599 → 84.825 |

单隐藏层相对原对照：0/32组平均AUROC提高，2/32组平均HALL-AUPR提高。只解释当前输入与实现，不将差异归因于纯隐藏层数。

本表记录单隐藏层MLP阶段；后续XGB/RF已完成，见[四种分类器完整对照](trees_summary.md)。

| 模型 | 特征 | 分类器 | AUROC | HALL-AUPR | HALL-F1 |
|---|---|---|---:|---:|---:|
| Qwen2.5-VL | attention_visual | fixed_three_hidden | 84.624 ± 0.471 | 38.360 ± 1.206 | 31.513 ± 4.856 |
| Qwen2.5-VL | attention_visual | searched_one_hidden | 82.914 ± 0.402 | 35.082 ± 0.546 | 17.021 ± 2.504 |
| Qwen2.5-VL | attention_visual_prompt_sum | fixed_three_hidden | 84.078 ± 1.756 | 38.792 ± 2.939 | 30.187 ± 2.959 |
| Qwen2.5-VL | attention_visual_prompt_sum | searched_one_hidden | 82.240 ± 0.038 | 33.216 ± 0.220 | 5.460 ± 0.915 |
| Qwen2.5-VL | attention_generation | fixed_three_hidden | 82.536 ± 0.630 | 36.973 ± 1.685 | 32.899 ± 2.246 |
| Qwen2.5-VL | attention_generation | searched_one_hidden | 78.698 ± 0.093 | 28.350 ± 0.557 | 0.355 ± 0.501 |
| Qwen2.5-VL | attention_vp_generation | fixed_three_hidden | 84.675 ± 0.271 | 41.737 ± 2.507 | 38.632 ± 6.507 |
| Qwen2.5-VL | attention_vp_generation | searched_one_hidden | 82.207 ± 0.110 | 34.546 ± 0.391 | 5.439 ± 0.440 |
| Qwen2.5-VL | gated_visual | fixed_three_hidden | 84.973 ± 0.435 | 40.489 ± 1.078 | 38.378 ± 1.949 |
| Qwen2.5-VL | gated_visual | searched_one_hidden | 82.719 ± 0.649 | 35.085 ± 1.417 | 17.262 ± 2.561 |
| Qwen2.5-VL | gated_visual_prompt_sum | fixed_three_hidden | 85.528 ± 1.063 | 42.766 ± 1.914 | 35.867 ± 1.960 |
| Qwen2.5-VL | gated_visual_prompt_sum | searched_one_hidden | 82.765 ± 0.412 | 35.770 ± 0.282 | 1.767 ± 1.321 |
| Qwen2.5-VL | gated_generation | fixed_three_hidden | 83.868 ± 0.507 | 40.379 ± 1.033 | 31.322 ± 3.163 |
| Qwen2.5-VL | gated_generation | searched_one_hidden | 79.251 ± 0.225 | 29.379 ± 0.502 | 2.409 ± 1.750 |
| Qwen2.5-VL | gated_vp_generation | fixed_three_hidden | 86.635 ± 0.660 | 46.780 ± 2.475 | 44.953 ± 1.068 |
| Qwen2.5-VL | gated_vp_generation | searched_one_hidden | 83.111 ± 0.237 | 39.007 ± 0.630 | 20.903 ± 2.099 |
| LLaVA-1.5 | attention_visual | fixed_three_hidden | 89.379 ± 0.151 | 68.622 ± 0.582 | 64.261 ± 0.499 |
| LLaVA-1.5 | attention_visual | searched_one_hidden | 89.009 ± 0.054 | 67.331 ± 0.795 | 63.204 ± 1.087 |
| LLaVA-1.5 | attention_visual_prompt_sum | fixed_three_hidden | 89.720 ± 0.286 | 69.640 ± 1.049 | 63.861 ± 0.130 |
| LLaVA-1.5 | attention_visual_prompt_sum | searched_one_hidden | 87.816 ± 0.275 | 68.509 ± 0.528 | 58.013 ± 0.748 |
| LLaVA-1.5 | attention_generation | fixed_three_hidden | 88.856 ± 0.149 | 66.265 ± 0.421 | 63.619 ± 1.563 |
| LLaVA-1.5 | attention_generation | searched_one_hidden | 87.579 ± 0.294 | 65.512 ± 0.700 | 53.859 ± 0.844 |
| LLaVA-1.5 | attention_vp_generation | fixed_three_hidden | 90.021 ± 0.438 | 70.067 ± 1.501 | 65.635 ± 0.878 |
| LLaVA-1.5 | attention_vp_generation | searched_one_hidden | 89.433 ± 0.318 | 70.426 ± 0.246 | 59.392 ± 1.581 |
| LLaVA-1.5 | gated_visual | fixed_three_hidden | 89.501 ± 0.102 | 68.828 ± 0.192 | 64.855 ± 0.716 |
| LLaVA-1.5 | gated_visual | searched_one_hidden | 88.828 ± 0.229 | 66.320 ± 1.058 | 61.035 ± 1.253 |
| LLaVA-1.5 | gated_visual_prompt_sum | fixed_three_hidden | 89.699 ± 0.043 | 70.276 ± 0.081 | 63.837 ± 1.677 |
| LLaVA-1.5 | gated_visual_prompt_sum | searched_one_hidden | 88.415 ± 0.475 | 68.397 ± 0.825 | 59.852 ± 1.394 |
| LLaVA-1.5 | gated_generation | fixed_three_hidden | 88.228 ± 0.504 | 66.171 ± 1.070 | 60.500 ± 1.423 |
| LLaVA-1.5 | gated_generation | searched_one_hidden | 87.212 ± 0.132 | 64.777 ± 0.510 | 56.344 ± 1.044 |
| LLaVA-1.5 | gated_vp_generation | fixed_three_hidden | 90.470 ± 0.306 | 71.947 ± 0.602 | 66.103 ± 1.734 |
| LLaVA-1.5 | gated_vp_generation | searched_one_hidden | 89.601 ± 0.416 | 71.187 ± 0.711 | 61.910 ± 1.996 |
| Qwen3-VL | attention_visual | fixed_three_hidden | 87.516 ± 0.195 | 59.573 ± 0.977 | 52.975 ± 1.007 |
| Qwen3-VL | attention_visual | searched_one_hidden | 86.883 ± 0.396 | 57.888 ± 0.238 | 49.455 ± 1.195 |
| Qwen3-VL | attention_visual_prompt_sum | fixed_three_hidden | 87.725 ± 1.091 | 61.527 ± 1.843 | 57.147 ± 2.739 |
| Qwen3-VL | attention_visual_prompt_sum | searched_one_hidden | 82.591 ± 0.203 | 51.650 ± 0.394 | 39.903 ± 0.662 |
| Qwen3-VL | attention_generation | fixed_three_hidden | 86.449 ± 0.717 | 57.586 ± 1.801 | 51.120 ± 2.864 |
| Qwen3-VL | attention_generation | searched_one_hidden | 83.309 ± 0.308 | 50.417 ± 0.784 | 34.246 ± 1.909 |
| Qwen3-VL | attention_vp_generation | fixed_three_hidden | 89.289 ± 0.110 | 64.135 ± 0.966 | 58.473 ± 0.636 |
| Qwen3-VL | attention_vp_generation | searched_one_hidden | 85.745 ± 0.066 | 56.044 ± 0.036 | 50.600 ± 0.720 |
| Qwen3-VL | gated_visual | fixed_three_hidden | 87.744 ± 0.305 | 59.638 ± 0.441 | 53.514 ± 0.659 |
| Qwen3-VL | gated_visual | searched_one_hidden | 87.126 ± 0.063 | 58.328 ± 0.168 | 49.765 ± 0.987 |
| Qwen3-VL | gated_visual_prompt_sum | fixed_three_hidden | 88.350 ± 0.228 | 62.379 ± 0.935 | 53.595 ± 1.226 |
| Qwen3-VL | gated_visual_prompt_sum | searched_one_hidden | 85.916 ± 0.719 | 58.811 ± 1.111 | 45.567 ± 2.385 |
| Qwen3-VL | gated_generation | fixed_three_hidden | 86.598 ± 0.860 | 58.313 ± 2.412 | 51.527 ± 2.700 |
| Qwen3-VL | gated_generation | searched_one_hidden | 83.302 ± 0.091 | 51.289 ± 0.377 | 33.908 ± 1.323 |
| Qwen3-VL | gated_vp_generation | fixed_three_hidden | 89.665 ± 0.073 | 66.411 ± 0.973 | 60.678 ± 2.367 |
| Qwen3-VL | gated_vp_generation | searched_one_hidden | 88.189 ± 0.314 | 63.007 ± 0.454 | 52.162 ± 3.350 |
| InternVL-2.5 | attention_visual | fixed_three_hidden | 85.244 ± 0.199 | 52.349 ± 0.492 | 44.158 ± 1.600 |
| InternVL-2.5 | attention_visual | searched_one_hidden | 84.534 ± 0.330 | 52.371 ± 0.266 | 39.052 ± 5.537 |
| InternVL-2.5 | attention_visual_prompt_sum | fixed_three_hidden | 84.880 ± 0.686 | 51.543 ± 1.375 | 41.340 ± 6.602 |
| InternVL-2.5 | attention_visual_prompt_sum | searched_one_hidden | 81.879 ± 0.714 | 46.562 ± 1.883 | 20.556 ± 4.645 |
| InternVL-2.5 | attention_generation | fixed_three_hidden | 84.467 ± 0.503 | 49.978 ± 1.162 | 39.876 ± 4.198 |
| InternVL-2.5 | attention_generation | searched_one_hidden | 83.152 ± 0.258 | 48.594 ± 0.999 | 24.527 ± 2.268 |
| InternVL-2.5 | attention_vp_generation | fixed_three_hidden | 86.695 ± 0.218 | 54.417 ± 0.469 | 49.615 ± 2.474 |
| InternVL-2.5 | attention_vp_generation | searched_one_hidden | 84.626 ± 0.078 | 51.495 ± 0.173 | 37.402 ± 3.889 |
| InternVL-2.5 | gated_visual | fixed_three_hidden | 85.419 ± 0.192 | 52.891 ± 1.059 | 44.311 ± 5.061 |
| InternVL-2.5 | gated_visual | searched_one_hidden | 84.475 ± 0.341 | 51.955 ± 0.361 | 44.964 ± 2.521 |
| InternVL-2.5 | gated_visual_prompt_sum | fixed_three_hidden | 85.402 ± 0.327 | 52.491 ± 2.117 | 46.161 ± 1.507 |
| InternVL-2.5 | gated_visual_prompt_sum | searched_one_hidden | 82.425 ± 0.351 | 47.806 ± 0.829 | 18.076 ± 5.051 |
| InternVL-2.5 | gated_generation | fixed_three_hidden | 84.718 ± 0.239 | 50.902 ± 0.361 | 43.379 ± 1.433 |
| InternVL-2.5 | gated_generation | searched_one_hidden | 81.749 ± 0.424 | 43.269 ± 1.830 | 9.137 ± 8.316 |
| InternVL-2.5 | gated_vp_generation | fixed_three_hidden | 86.599 ± 0.558 | 56.160 ± 0.570 | 53.479 ± 0.798 |
| InternVL-2.5 | gated_vp_generation | searched_one_hidden | 84.825 ± 0.281 | 52.237 ± 0.690 | 36.957 ± 4.959 |

selected_params.json及selected_params.csv保存每组验证选择；one_hidden/<group>/search逐候选记录验证指标、迭代次数、loss和收敛警告。达到max_iter的警告保留，不自动延长预算。
每组最终三seed模型及概率已保存，固定与搜索方法各96个正式头；MLP内层共384次候选拟合。
运行：`python scripts/train_attention_log_strength_mlp.py --stage prepare`；`--stage fixed --models <模型> --device cuda:0`；`--stage search --workers 12`；完成后`--stage summarize`。
完整来源S_g目前只有四模型，不混用MiniGPT/Shikra旧S_E。保留原提取数值、gate口径及前缀长度/位置限制；没有bootstrap或独立图像验证。
