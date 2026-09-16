# LLaVA-1.5-7B SADT风格逐层案例

每图顶部为模型实际输入视图与完整生成原文，红色突出本次检测词；下方依次为四种信号的Top16与Top32。
热图采用SADT作者代码的视觉原则：未保留位置置零，JET以0.45权重覆盖0.55原图，不画patch框。
main共10例（HALL/REAL各5）；divergence_hall另含3个Attention与P_E差异大的HALL案例。

| 组 | # | 标签 | 目标 | 图片 | mean JS | max JS@层 | min overlap@16 | min overlap@32 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| main | 1 | HALL | mouse (mouse) | [287741](main_01_hall_287741_mouse.jpg) | 0.065 | 0.104@L32 | 0.375 | 0.375 |
| main | 2 | HALL | person (person) | [183204](main_02_hall_183204_person.jpg) | 0.073 | 0.104@L10 | 0.500 | 0.594 |
| main | 3 | HALL | skis (skis) | [8771](main_03_hall_8771_skis.jpg) | 0.077 | 0.119@L10 | 0.375 | 0.531 |
| main | 4 | HALL | forks (fork) | [266951](main_04_hall_266951_fork.jpg) | 0.066 | 0.127@L10 | 0.438 | 0.375 |
| main | 5 | REAL | frisbee (frisbee) | [481413](main_05_real_481413_frisbee.jpg) | 0.036 | 0.087@L32 | 0.312 | 0.438 |
| main | 6 | REAL | umbrella (umbrella) | [431364](main_06_real_431364_umbrella.jpg) | 0.034 | 0.065@L32 | 0.375 | 0.531 |
| main | 7 | REAL | cake (cake) | [417023](main_07_real_417023_cake.jpg) | 0.057 | 0.087@L10 | 0.438 | 0.438 |
| main | 8 | REAL | kite (kite) | [246124](main_08_real_246124_kite.jpg) | 0.038 | 0.061@L10 | 0.688 | 0.469 |
| main | 9 | HALL | table (dining table) | [483135](main_09_hall_483135_dining-table.jpg) | 0.081 | 0.125@L10 | 0.125 | 0.375 |
| main | 10 | REAL | television (tv) | [546444](main_10_real_546444_tv.jpg) | 0.052 | 0.104@L10 | 0.312 | 0.438 |
| divergence_hall | 1 | HALL | apples (apple) | [472828](divergence_hall_01_hall_472828_apple.jpg) | 0.042 | 0.078@L10 | 0.125 | 0.188 |
| divergence_hall | 2 | HALL | broccoli (broccoli) | [157789](divergence_hall_02_hall_157789_broccoli.jpg) | 0.057 | 0.133@L10 | 0.312 | 0.312 |
| divergence_hall | 3 | HALL | handbag (handbag) | [148662](divergence_hall_03_hall_148662_handbag.jpg) | 0.059 | 0.113@L10 | 0.125 | 0.531 |
