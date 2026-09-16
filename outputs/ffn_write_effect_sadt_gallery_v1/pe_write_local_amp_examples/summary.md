# 未求和的 PE–WRITE local amp：10个案例

每个视觉token绘制 $d_m=|P_{E,m}-P_{W,m}|$，不乘 $1/2$、不求和、不做TopK。
图中 `sum` 是该层 $\sum_m d_m=2TV(P_E,P_W)$，`max` 是最大单token差值。
10图统一使用候选局部差值的99.5%分位数 `0.024823` 作为显示上限；颜色可跨图比较，超过上限的极少数位置仅在显示时饱和，保存的统计不截断。
选择为每模型各一个高差异HALL和REAL，再补全局最高的未重复HALL和REAL，共HALL/REAL各5。

| # | 模型 | 标签 | 目标 | 图片 | 最大层sum | 最大单token差 |
|---:|---|---|---|---|---:|---:|
| 1 | Qwen2.5-VL-7B | HALL | keyboard (keyboard) | [332570](01_qwen2_5_vl_7b_hall_332570_keyboard.jpg) | 0.594 | 0.0746 |
| 2 | Qwen2.5-VL-7B | REAL | player (person) | [390435](02_qwen2_5_vl_7b_real_390435_person.jpg) | 0.535 | 0.0467 |
| 3 | LLaVA-1.5-7B | HALL | table (dining table) | [483135](03_llava_1_5_7b_hall_483135_dining-table.jpg) | 0.884 | 0.0989 |
| 4 | LLaVA-1.5-7B | REAL | dog (dog) | [548331](04_llava_1_5_7b_real_548331_dog.jpg) | 1.054 | 0.0892 |
| 5 | Qwen3-VL-8B | HALL | tv (tv) | [70471](05_qwen3_vl_8b_hall_70471_tv.jpg) | 0.847 | 0.3484 |
| 6 | Qwen3-VL-8B | REAL | player (person) | [427117](06_qwen3_vl_8b_real_427117_person.jpg) | 0.900 | 0.2149 |
| 7 | InternVL2.5-8B | HALL | calf (cow) | [196295](07_internvl_2_5_8b_hall_196295_cow.jpg) | 0.788 | 0.2344 |
| 8 | InternVL2.5-8B | REAL | skis (skis) | [233233](08_internvl_2_5_8b_real_233233_skis.jpg) | 0.564 | 0.0649 |
| 9 | LLaVA-1.5-7B | HALL | apples (apple) | [472828](09_llava_1_5_7b_hall_472828_apple.jpg) | 0.630 | 0.1543 |
| 10 | LLaVA-1.5-7B | REAL | baseball bat (baseball bat) | [35049](10_llava_1_5_7b_real_35049_baseball-bat.jpg) | 0.959 | 0.0577 |

运行：

```bash
/opt/conda/private/envs/vicr/bin/python scripts/plot_pe_write_local_amp_examples.py
```

只读取已有K50 attention缓存和all-attention K32路径结果，不运行模型或路径积分。
