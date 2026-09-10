# 三类来源份额的单组与拼接检测效果

四模型原4000图、3200/800划分和全部mentions；三组分别单独训练，并新增prompt_generation_visual三组拼接。均未拼接AE或强度。
拼接按[prompt全层, generation全层, visual全层]顺序，四模型维度84/96/108/96；单组L维、拼接3L维。
份额=sum_group ||c_j|| / gross_all；分母含residual、attention输出偏置和FFN(0)范数。prompt包括模板，generation仅含已生成前缀。
原MLP (128/64/32、dropout .3、Adam、batch 256)、原调度/早停/最低训练loss checkpoint，seeds43/44/45。
每格为三seed均值±总体标准差（ddof=0），单位%；HALL-F1使用各seed训练集REAL-F1阈值。没有调参或bootstrap。
F_E=AE+log1p(S_E)、F_C=AE+log1p(S_C)直接引用同划分已有结果，仅作基线，输入均为2L维。

| 模型 | 特征 | AUROC | HALL-AUPR | HALL-F1 |
|---|---|---:|---:|---:|
| Qwen2.5-VL | prompt | 81.395 ± 0.444 | 35.102 ± 0.890 | 23.051 ± 1.574 |
| Qwen2.5-VL | generation | 80.054 ± 0.474 | 29.805 ± 1.029 | 15.107 ± 3.044 |
| Qwen2.5-VL | visual | 83.846 ± 0.299 | 38.037 ± 1.483 | 24.288 ± 4.380 |
| Qwen2.5-VL | prompt_generation_visual | 85.773 ± 0.110 | 42.661 ± 1.120 | 42.002 ± 3.260 |
| Qwen2.5-VL | F_E | 87.449 ± 0.085 | 43.982 ± 1.678 | 38.401 ± 3.469 |
| Qwen2.5-VL | F_C | 87.182 ± 0.306 | 45.568 ± 0.204 | 42.087 ± 4.250 |
| LLaVA-1.5 | prompt | 88.273 ± 0.086 | 66.457 ± 0.054 | 59.197 ± 1.986 |
| LLaVA-1.5 | generation | 88.054 ± 0.088 | 65.186 ± 0.749 | 58.323 ± 2.859 |
| LLaVA-1.5 | visual | 88.544 ± 0.028 | 68.785 ± 0.156 | 62.970 ± 0.265 |
| LLaVA-1.5 | prompt_generation_visual | 89.607 ± 0.084 | 68.778 ± 0.271 | 64.751 ± 0.236 |
| LLaVA-1.5 | F_E | 90.015 ± 0.272 | 70.815 ± 0.619 | 66.788 ± 0.073 |
| LLaVA-1.5 | F_C | 90.101 ± 0.081 | 71.224 ± 0.251 | 66.731 ± 0.435 |
| Qwen3-VL | prompt | 85.310 ± 0.271 | 54.436 ± 0.320 | 46.601 ± 1.413 |
| Qwen3-VL | generation | 85.906 ± 0.304 | 54.447 ± 1.360 | 45.837 ± 3.601 |
| Qwen3-VL | visual | 86.190 ± 0.164 | 56.497 ± 0.714 | 49.680 ± 2.230 |
| Qwen3-VL | prompt_generation_visual | 89.404 ± 0.405 | 64.711 ± 1.060 | 60.105 ± 0.517 |
| Qwen3-VL | F_E | 88.655 ± 0.305 | 62.832 ± 0.353 | 58.839 ± 1.088 |
| Qwen3-VL | F_C | 88.607 ± 0.323 | 62.627 ± 0.770 | 56.987 ± 1.453 |
| InternVL-2.5 | prompt | 84.434 ± 0.137 | 49.463 ± 0.580 | 35.788 ± 3.395 |
| InternVL-2.5 | generation | 83.912 ± 0.270 | 45.992 ± 0.196 | 34.900 ± 3.455 |
| InternVL-2.5 | visual | 84.106 ± 0.117 | 48.846 ± 0.207 | 32.084 ± 4.349 |
| InternVL-2.5 | prompt_generation_visual | 86.474 ± 0.132 | 51.972 ± 1.778 | 47.776 ± 2.468 |
| InternVL-2.5 | F_E | 85.825 ± 0.587 | 53.034 ± 1.201 | 51.378 ± 0.362 |
| InternVL-2.5 | F_C | 84.895 ± 0.600 | 50.890 ± 0.702 | 48.848 ± 1.210 |

拼接相对本次最佳单组（按三seed平均AUROC）的增量，单位百分点；仅描述性比较。

| 模型 | 最佳单组 | 拼接ΔAUROC | 拼接ΔHALL-AUPR |
|---|---|---:|---:|
| qwen2_5_vl_7b | visual | +1.927 | +4.624 |
| llava_1_5_7b | visual | +1.062 | -0.007 |
| qwen3_vl_8b | visual | +3.214 | +8.214 |
| internvl_2_5_8b | prompt | +2.041 | +2.509 |

拼接结果：四模型平均AUROC均高于最佳单组，三seed的AUROC增量也均为正。HALL-AUPR在Qwen2.5/Qwen3/InternVL提高，LLaVA基本持平（-0.007点）。对照AE+S：Qwen3的AUROC/AP均高于F_E和F_C；InternVL的AUROC高于两者，AP低于F_E但高于F_C；Qwen2.5/LLaVA两项仍低于两种基线。本次不作显著性声明。

此前单组结果：视觉份额在Qwen2.5/LLaVA/Qwen3的三组中平均AUROC与HALL-AUPR最高；InternVL两项均以prompt份额最高。四模型三组单独输入的平均AUROC与HALL-AUPR都低于各自F_E和F_C。单组对照不支持单独替代AE+S；三组份额自身拼接结果见上。尚未检验与AE+S拼接后的增益，不声称显著性。

结果位于本目录各模型matrices.pt、protocol.json、heads和detection.json。detection.json由原汇总函数生成，含历史兼容的ensemble字段；本报告只使用逐seed统计。
训练：`python scripts/train_ffn_source_shares.py --models <模型列表> --device cuda:0`；汇总：`python scripts/train_ffn_source_shares.py --summarize`。
这是来源位置的范数份额检测，不是attention检测；生成长度/位置混杂和原K50重建误差限制仍适用。
