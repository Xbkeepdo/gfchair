# Qwen2.5-VL-7B SADT风格逐层案例

每图顶部为模型实际输入视图与完整生成原文，红色突出本次检测词；下方依次为四种信号的Top16与Top32。
热图采用SADT作者代码的视觉原则：未保留位置置零，JET以0.45权重覆盖0.55原图，不画patch框。
main共10例（HALL/REAL各5）；divergence_hall另含3个Attention与P_E差异大的HALL案例。

| 组 | # | 标签 | 目标 | 图片 | mean JS | max JS@层 | min overlap@16 | min overlap@32 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| main | 1 | HALL | streetlights (traffic light) | [385078](main_01_hall_385078_traffic-light.jpg) | 0.022 | 0.034@L20 | 0.562 | 0.750 |
| main | 2 | HALL | bike (bicycle) | [290231](main_02_hall_290231_bicycle.jpg) | 0.022 | 0.031@L25 | 0.688 | 0.656 |
| main | 3 | HALL | bottle (bottle) | [183204](main_03_hall_183204_bottle.jpg) | 0.017 | 0.025@L15 | 0.625 | 0.688 |
| main | 4 | HALL | ski (skis) | [151589](main_04_hall_151589_skis.jpg) | 0.025 | 0.048@L28 | 0.688 | 0.750 |
| main | 5 | REAL | kayak (boat) | [578522](main_05_real_578522_boat.jpg) | 0.009 | 0.011@L10 | 0.875 | 0.875 |
| main | 6 | REAL | kitten (cat) | [334062](main_06_real_334062_cat.jpg) | 0.013 | 0.021@L10 | 0.812 | 0.875 |
| main | 7 | REAL | baseball bat (baseball bat) | [214742](main_07_real_214742_baseball-bat.jpg) | 0.013 | 0.018@L15 | 0.812 | 0.844 |
| main | 8 | REAL | motorbike (motorcycle) | [195267](main_08_real_195267_motorcycle.jpg) | 0.019 | 0.047@L10 | 0.812 | 0.906 |
| main | 9 | HALL | train (train) | [82327](main_09_hall_82327_train.jpg) | 0.017 | 0.042@L28 | 0.562 | 0.656 |
| main | 10 | REAL | chair (chair) | [156372](main_10_real_156372_chair.jpg) | 0.016 | 0.027@L10 | 0.875 | 0.812 |
| divergence_hall | 1 | HALL | keyboard (keyboard) | [332570](divergence_hall_01_hall_332570_keyboard.jpg) | 0.028 | 0.061@L28 | 0.625 | 0.500 |
| divergence_hall | 2 | HALL | cars (car) | [567315](divergence_hall_02_hall_567315_car.jpg) | 0.026 | 0.053@L28 | 0.625 | 0.562 |
| divergence_hall | 3 | HALL | calf (cow) | [196295](divergence_hall_03_hall_196295_cow.jpg) | 0.024 | 0.061@L28 | 0.562 | 0.812 |
