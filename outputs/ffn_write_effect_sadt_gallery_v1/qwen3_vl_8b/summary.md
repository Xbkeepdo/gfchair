# Qwen3-VL-8B SADT风格逐层案例

每图顶部为模型实际输入视图与完整生成原文，红色突出本次检测词；下方依次为四种信号的Top16与Top32。
热图采用SADT作者代码的视觉原则：未保留位置置零，JET以0.45权重覆盖0.55原图，不画patch框。
main共10例（HALL/REAL各5）；divergence_hall另含3个Attention与P_E差异大的HALL案例。

| 组 | # | 标签 | 目标 | 图片 | mean JS | max JS@层 | min overlap@16 | min overlap@32 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| main | 1 | HALL | ball (sports ball) | [500084](main_01_hall_500084_sports-ball.jpg) | 0.077 | 0.165@L30 | 0.625 | 0.750 |
| main | 2 | HALL | bowl (bowl) | [465552](main_02_hall_465552_bowl.jpg) | 0.062 | 0.115@L36 | 0.500 | 0.375 |
| main | 3 | HALL | book (book) | [531707](main_03_hall_531707_book.jpg) | 0.035 | 0.087@L36 | 0.375 | 0.406 |
| main | 4 | HALL | bow (tie) | [314557](main_04_hall_314557_tie.jpg) | 0.062 | 0.125@L30 | 0.438 | 0.562 |
| main | 5 | REAL | bottle (bottle) | [489344](main_05_real_489344_bottle.jpg) | 0.025 | 0.066@L36 | 0.438 | 0.625 |
| main | 6 | REAL | suitcase (suitcase) | [429063](main_06_real_429063_suitcase.jpg) | 0.019 | 0.048@L36 | 0.688 | 0.750 |
| main | 7 | REAL | mobile phone (cell phone) | [414236](main_07_real_414236_cell-phone.jpg) | 0.025 | 0.082@L36 | 0.500 | 0.531 |
| main | 8 | REAL | fork (fork) | [157789](main_08_real_157789_fork.jpg) | 0.031 | 0.099@L36 | 0.625 | 0.531 |
| main | 9 | HALL | refrigerator (refrigerator) | [369082](main_09_hall_369082_refrigerator.jpg) | 0.061 | 0.122@L36 | 0.312 | 0.562 |
| main | 10 | REAL | table (dining table) | [120853](main_10_real_120853_dining-table.jpg) | 0.021 | 0.061@L36 | 0.688 | 0.531 |
| divergence_hall | 1 | HALL | keyboard (keyboard) | [455975](divergence_hall_01_hall_455975_keyboard.jpg) | 0.043 | 0.142@L36 | 0.438 | 0.531 |
| divergence_hall | 2 | HALL | chair (chair) | [552947](divergence_hall_02_hall_552947_chair.jpg) | 0.033 | 0.132@L36 | 0.438 | 0.500 |
| divergence_hall | 3 | HALL | remote (remote) | [556025](divergence_hall_03_hall_556025_remote.jpg) | 0.030 | 0.102@L36 | 0.375 | 0.438 |
