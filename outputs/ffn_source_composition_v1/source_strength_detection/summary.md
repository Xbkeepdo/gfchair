# 完整来源强度：原始值与log1p检测

S_g=sum_group ||c_j||，不除全部来源总量或token数。分组prompt含模板和特殊token，generation仅含已生成前缀，visual为S_C。
每套5组：prompt / generation / visual 单独；concat三组全层向量；visual_prompt_sum=S_V+S_P。log版本先相加再log1p。
四模型原4000图、3200/800、全部mentions，MLP/seeds43/44/45保持，共120头。每格为三seed均值±总体标准差，单位%。
HALL-F1使用每seed训练REAL-F1阈值；原双阈值详见CSV。无超参数搜索或bootstrap。F_E/F_C及份额拼接仅引用已有基线。

| 模型 | 信号 | AUROC | HALL-AUPR | HALL-F1 |
|---|---|---:|---:|---:|
| Qwen2.5-VL | raw_prompt | 81.539 ± 0.190 | 33.831 ± 1.033 | 10.326 ± 1.914 |
| Qwen2.5-VL | raw_generation | 80.050 ± 0.314 | 32.304 ± 0.972 | 11.000 ± 3.558 |
| Qwen2.5-VL | raw_visual | 84.391 ± 0.301 | 37.307 ± 1.573 | 15.634 ± 5.268 |
| Qwen2.5-VL | raw_concat | 85.536 ± 0.146 | 42.039 ± 1.280 | 40.913 ± 1.765 |
| Qwen2.5-VL | raw_visual_prompt_sum | 83.165 ± 0.282 | 35.174 ± 0.638 | 16.581 ± 3.775 |
| Qwen2.5-VL | log1p_prompt | 82.457 ± 0.084 | 35.145 ± 1.039 | 26.433 ± 3.481 |
| Qwen2.5-VL | log1p_generation | 80.975 ± 0.249 | 33.713 ± 0.983 | 11.215 ± 4.757 |
| Qwen2.5-VL | log1p_visual | 84.774 ± 0.414 | 38.485 ± 0.936 | 29.485 ± 6.283 |
| Qwen2.5-VL | log1p_concat | 85.617 ± 0.103 | 42.018 ± 0.309 | 41.328 ± 2.739 |
| Qwen2.5-VL | log1p_visual_prompt_sum | 83.525 ± 0.365 | 35.152 ± 0.553 | 22.377 ± 3.841 |
| Qwen2.5-VL | F_E | 87.449 ± 0.085 | 43.982 ± 1.678 | 38.401 ± 3.469 |
| Qwen2.5-VL | F_C | 87.182 ± 0.306 | 45.568 ± 0.204 | 42.087 ± 4.250 |
| Qwen2.5-VL | share_concat | 85.773 ± 0.110 | 42.661 ± 1.120 | 42.002 ± 3.260 |
| LLaVA-1.5 | raw_prompt | 88.888 ± 0.035 | 70.422 ± 0.293 | 61.708 ± 1.383 |
| LLaVA-1.5 | raw_generation | 87.785 ± 0.069 | 64.061 ± 0.048 | 59.130 ± 1.016 |
| LLaVA-1.5 | raw_visual | 88.421 ± 0.075 | 68.218 ± 0.247 | 61.704 ± 0.324 |
| LLaVA-1.5 | raw_concat | 90.052 ± 0.112 | 70.899 ± 0.559 | 65.733 ± 0.800 |
| LLaVA-1.5 | raw_visual_prompt_sum | 88.917 ± 0.160 | 69.741 ± 0.314 | 63.115 ± 1.442 |
| LLaVA-1.5 | log1p_prompt | 89.055 ± 0.141 | 70.617 ± 0.390 | 62.416 ± 1.750 |
| LLaVA-1.5 | log1p_generation | 87.895 ± 0.183 | 63.604 ± 0.261 | 58.780 ± 1.669 |
| LLaVA-1.5 | log1p_visual | 88.944 ± 0.095 | 68.824 ± 0.123 | 62.909 ± 0.986 |
| LLaVA-1.5 | log1p_concat | 89.974 ± 0.308 | 70.339 ± 0.626 | 65.318 ± 1.450 |
| LLaVA-1.5 | log1p_visual_prompt_sum | 89.353 ± 0.158 | 70.097 ± 0.176 | 64.143 ± 1.082 |
| LLaVA-1.5 | F_E | 90.015 ± 0.272 | 70.815 ± 0.619 | 66.788 ± 0.073 |
| LLaVA-1.5 | F_C | 90.101 ± 0.081 | 71.224 ± 0.251 | 66.731 ± 0.435 |
| LLaVA-1.5 | share_concat | 89.607 ± 0.084 | 68.778 ± 0.271 | 64.751 ± 0.236 |
| Qwen3-VL | raw_prompt | 84.085 ± 0.184 | 53.125 ± 0.895 | 44.818 ± 2.057 |
| Qwen3-VL | raw_generation | 85.771 ± 0.241 | 55.224 ± 0.527 | 45.851 ± 0.096 |
| Qwen3-VL | raw_visual | 86.399 ± 0.275 | 55.704 ± 0.640 | 48.628 ± 3.944 |
| Qwen3-VL | raw_concat | 90.202 ± 0.111 | 66.351 ± 0.625 | 62.252 ± 0.674 |
| Qwen3-VL | raw_visual_prompt_sum | 86.031 ± 0.382 | 56.857 ± 0.874 | 47.340 ± 2.063 |
| Qwen3-VL | log1p_prompt | 85.896 ± 0.248 | 56.117 ± 1.144 | 51.122 ± 0.653 |
| Qwen3-VL | log1p_generation | 85.681 ± 0.801 | 55.594 ± 2.444 | 48.669 ± 2.690 |
| Qwen3-VL | log1p_visual | 86.906 ± 0.664 | 56.957 ± 1.625 | 49.892 ± 1.056 |
| Qwen3-VL | log1p_concat | 90.141 ± 0.323 | 65.740 ± 0.752 | 61.515 ± 0.127 |
| Qwen3-VL | log1p_visual_prompt_sum | 86.312 ± 0.704 | 58.570 ± 0.838 | 52.999 ± 1.753 |
| Qwen3-VL | F_E | 88.655 ± 0.305 | 62.832 ± 0.353 | 58.839 ± 1.088 |
| Qwen3-VL | F_C | 88.607 ± 0.323 | 62.627 ± 0.770 | 56.987 ± 1.453 |
| Qwen3-VL | share_concat | 89.404 ± 0.405 | 64.711 ± 1.060 | 60.105 ± 0.517 |
| InternVL-2.5 | raw_prompt | 85.129 ± 0.193 | 51.070 ± 0.286 | 35.300 ± 2.302 |
| InternVL-2.5 | raw_generation | 83.304 ± 0.358 | 46.676 ± 1.371 | 32.676 ± 1.057 |
| InternVL-2.5 | raw_visual | 83.847 ± 0.119 | 50.505 ± 0.374 | 31.327 ± 0.304 |
| InternVL-2.5 | raw_concat | 87.040 ± 0.122 | 57.071 ± 0.630 | 51.755 ± 0.984 |
| InternVL-2.5 | raw_visual_prompt_sum | 84.462 ± 0.253 | 50.090 ± 0.191 | 34.665 ± 7.385 |
| InternVL-2.5 | log1p_prompt | 85.438 ± 0.279 | 51.488 ± 1.130 | 42.697 ± 3.068 |
| InternVL-2.5 | log1p_generation | 83.746 ± 0.263 | 47.964 ± 0.271 | 35.963 ± 2.314 |
| InternVL-2.5 | log1p_visual | 84.723 ± 0.269 | 51.843 ± 0.255 | 37.781 ± 6.428 |
| InternVL-2.5 | log1p_concat | 86.787 ± 0.263 | 55.029 ± 0.971 | 52.315 ± 0.633 |
| InternVL-2.5 | log1p_visual_prompt_sum | 84.303 ± 0.585 | 49.607 ± 1.407 | 30.083 ± 6.626 |
| InternVL-2.5 | F_E | 85.825 ± 0.587 | 53.034 ± 1.201 | 51.378 ± 0.362 |
| InternVL-2.5 | F_C | 84.895 ± 0.600 | 50.890 ± 0.702 | 48.848 ± 1.210 |
| InternVL-2.5 | share_concat | 86.474 ± 0.132 | 51.972 ± 1.778 | 47.776 ± 2.468 |

结果解读：四模型在raw/log1p两套中，concat平均AUROC均高于其余四组。log1p使单独prompt和visual的AUROC/AP在四模型都提高，但concat仅Qwen2的AUROC略升，另外三模型下降；concat的AP均略降或下降。V+P相加没有跨模型一致超过单独visual：raw在LLaVA/InternVL更高、两个Qwen更低；log1p仅LLaVA更高（按AUROC）。
原始concat对照AE+S：Qwen3/InternVL的AUROC/AP均超过F_E/F_C；LLaVA与F_E接近且略低于F_C；Qwen2仍低于两基线。相对份额concat，原始concat在LLaVA/Qwen3/InternVL双指标提高，Qwen2下降。以上均为点估计，不宣称显著性或因果，concat输入维度为单组3倍。

曲线为每模型4000图全部mentions等权：先对每条mention计算S或log1p(S)，再按REAL/HALL统计。
![原始强度](all_raw.png)

![log1p强度](all_log1p.png)

模型的matrices.pt、protocol.json、heads、detection.json保存在各子目录；detection.csv为均值/std，seed_metrics.csv为逐seed双阈值，comparisons.csv为配对均值差。
原汇总器JSON含ensemble字段，本报告不使用ensemble。长度/位置混杂及原K50来源重建误差限制仍适用。
训练：`python scripts/train_ffn_source_strengths.py --models <模型列表> --device cuda:0`；完成后`--summarize`生成表格和曲线。
