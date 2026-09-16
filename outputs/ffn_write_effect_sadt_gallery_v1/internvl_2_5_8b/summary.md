# InternVL2.5-8B SADT风格逐层案例

每图顶部为模型实际输入视图与完整生成原文，红色突出本次检测词；下方依次为四种信号的Top16与Top32。
热图采用SADT作者代码的视觉原则：未保留位置置零，JET以0.45权重覆盖0.55原图，不画patch框。
main共10例（HALL/REAL各5）；divergence_hall另含3个Attention与P_E差异大的HALL案例。

| 组 | # | 标签 | 目标 | 图片 | mean JS | max JS@层 | min overlap@16 | min overlap@32 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| main | 1 | HALL | bowl (bowl) | [452121](main_01_hall_452121_bowl.jpg) | 0.016 | 0.026@L30 | 0.625 | 0.688 |
| main | 2 | HALL | car (car) | [410101](main_02_hall_410101_car.jpg) | 0.026 | 0.038@L25 | 0.625 | 0.719 |
| main | 3 | HALL | toaster (toaster) | [88562](main_03_hall_88562_toaster.jpg) | 0.016 | 0.020@L10 | 0.625 | 0.750 |
| main | 4 | HALL | ski (skis) | [74101](main_04_hall_74101_skis.jpg) | 0.024 | 0.060@L30 | 0.625 | 0.750 |
| main | 5 | REAL | tie (tie) | [447854](main_05_real_447854_tie.jpg) | 0.011 | 0.016@L32 | 0.812 | 0.781 |
| main | 6 | REAL | phone (cell phone) | [414236](main_06_real_414236_cell-phone.jpg) | 0.016 | 0.021@L30 | 0.750 | 0.812 |
| main | 7 | REAL | kayak (boat) | [578522](main_07_real_578522_boat.jpg) | 0.015 | 0.026@L25 | 0.812 | 0.812 |
| main | 8 | REAL | scooter (motorcycle) | [142324](main_08_real_142324_motorcycle.jpg) | 0.020 | 0.041@L30 | 0.750 | 0.750 |
| main | 9 | HALL | oven (oven) | [566920](main_09_hall_566920_oven.jpg) | 0.025 | 0.045@L15 | 0.625 | 0.750 |
| main | 10 | REAL | tabby (cat) | [542487](main_10_real_542487_cat.jpg) | 0.013 | 0.020@L20 | 0.750 | 0.844 |
| divergence_hall | 1 | HALL | calf (cow) | [196295](divergence_hall_01_hall_196295_cow.jpg) | 0.033 | 0.090@L25 | 0.625 | 0.719 |
| divergence_hall | 2 | HALL | remote (remote) | [407146](divergence_hall_02_hall_407146_remote.jpg) | 0.026 | 0.059@L15 | 0.625 | 0.656 |
| divergence_hall | 3 | HALL | foal (horse) | [1818](divergence_hall_03_hall_1818_horse.jpg) | 0.025 | 0.068@L25 | 0.625 | 0.750 |
