# Prompt / generation 完整来源贡献曲线

复用四模型原4000图K50缓存和原3200/800划分，按全部mentions统计；不重跑模型或训练检测器。

主图合并全部4000图，每条mention等权（非每图等权）；先逐mention计算比值，再按REAL/HALL统计。另保留train/test分图。

- prompt包含全部非视觉提示位置（含system、聊天模板及特殊token），不等同于纯instruction。
- generation仅含预测目标之前已生成的前缀；包含当前query token，不含待预测目标与未来。
- gross = 组内逐来源 ||c_j|| 相加；per_token = gross / 该组token数；fraction = gross / gross_all。
- gross_all包含视觉、全部文本、residual、attention输出偏置和FFN(0)范数；三条来源份额不要求相加为1。
- 空generation的gross和fraction为0，per_token未定义并排除；数量见counts.json。
- 实线均值、虚线中位数、阴影IQR（非置信区间）；是来源位置归因，不是纯模态信息或因果效应。
- 本次只有FFN贡献曲线；非视觉raw attention未保存，不能由范数恢复。原K50重建误差限制保留。

| 模型 | 来源 | 全4000图gross：REAL均值>HALL层数 | per_token：REAL均值>HALL层数 |
|---|---|---:|---:|
| Qwen2.5-VL | prompt | 28/28 | 28/28 |
| Qwen2.5-VL | generation | 0/28 | 28/28 |
| Qwen2.5-VL | visual | 28/28 | 28/28 |
| LLaVA-1.5 | prompt | 32/32 | 32/32 |
| LLaVA-1.5 | generation | 1/32 | 32/32 |
| LLaVA-1.5 | visual | 26/32 | 26/32 |
| Qwen3-VL | prompt | 36/36 | 36/36 |
| Qwen3-VL | generation | 2/36 | 36/36 |
| Qwen3-VL | visual | 36/36 | 36/36 |
| InternVL-2.5 | prompt | 32/32 | 32/32 |
| InternVL-2.5 | generation | 0/32 | 32/32 |
| InternVL-2.5 | visual | 32/32 | 32/32 |

四模型prompt在train/test全部层均REAL均值更高。generation的gross几乎各层HALL更高，但逐样本除以前缀token数后，四模型全部层均REAL更高。HALL前缀明显更长，总强度不能直接解释成更依赖生成文本；除以长度也不等于完成位置/长度匹配或证明因果。

| 模型 | 全量REAL平均生成前缀token数 | 全量HALL平均生成前缀token数 |
|---|---:|---:|
| Qwen2.5-VL | 33.00 | 58.67 |
| LLaVA-1.5 | 31.97 | 70.74 |
| Qwen3-VL | 101.39 | 154.09 |
| InternVL-2.5 | 70.13 | 116.65 |

![4000图贡献总强度](all_gross.png)

![4000图每token强度](all_per_token.png)

![4000图来源范数份额](all_fraction.png)

训练/测试集对应train_*.png/pdf和test_*.png/pdf；完整逐层统计curves.csv；PNG与PDF均已保存。

运行：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /opt/conda/private/envs/vicr/bin/python scripts/plot_ffn_text_sources.py`。
检查：合成边界例子（含空generation）、全量未来位置范数为0、分组加residual/bias/FFN(0)范数重建gross_all。
