# 三层：学习率调度与checkpoint选择

固定全AE+log1p(S)、[128,64,32]、StandardScaler、BN-off、AdamW、dropout.1、lr.001、wd1e-5、batch128。
内层同轨比较：2560优化/640验证；最高验证AUROC与最低训练损失；共同100epochs上限、train-loss patience10。
调度开/关对照；开启时精确复用旧ReduceLROnPlateau(train_loss,factor.5,patience5)。
完整3200重训：迁移内层max-AUC epoch中位数、固定末轮；对比直接3200最小训练损失。不是在800图上选AUC。
所有新结果为旧800图的探索性评价；数值FAIL保留，无bootstrap、新2000图或batch网格。

## 三seed概率ensemble AUROC / HALL-AUPR（%）

| 组 | llava_1_5_7b | internvl_2_5_8b | qwen2_5_vl_7b | qwen3_vl_8b |
|---|---:|---:|---:|---:|
| full/schedule_off/min_train_loss | 89.242 / 69.953 | 84.747 / 50.756 | 88.070 / 46.445 | 89.498 / 65.602 |
| full/schedule_off/val_epoch_transfer | 90.405 / 71.210 | 85.625 / 51.868 | 87.652 / 44.453 | 89.872 / 67.117 |
| full/schedule_on/min_train_loss | 89.232 / 69.658 | 84.809 / 49.841 | 87.999 / 47.361 | 89.562 / 65.737 |
| full/schedule_on/val_epoch_transfer | 90.405 / 71.210 | 85.625 / 51.868 | 87.652 / 44.453 | 89.872 / 67.117 |
| inner/schedule_off/max_val_auc | 90.158 / 70.972 | 85.685 / 50.669 | 87.182 / 43.070 | 89.434 / 66.034 |
| inner/schedule_off/min_train_loss | 89.184 / 69.730 | 84.448 / 47.214 | 86.805 / 44.910 | 88.904 / 65.537 |
| inner/schedule_on/max_val_auc | 90.158 / 70.972 | 85.685 / 50.669 | 87.182 / 43.070 | 89.434 / 66.034 |
| inner/schedule_on/min_train_loss | 89.184 / 69.730 | 84.322 / 47.575 | 87.190 / 45.623 | 88.818 / 65.217 |
| legacy_3200_three | 90.496 / 71.928 | 86.381 / 54.615 | 88.214 / 45.774 | 89.651 / 64.934 |
| previous_2560_three | 90.158 / 70.972 | 85.685 / 50.669 | 87.182 / 43.070 | 89.434 / 66.034 |
| previous_3200_three | 90.405 / 71.210 | 85.625 / 51.868 | 87.652 / 44.453 | 89.872 / 67.117 |

## 完整3200重训的固定AUC迁移轮数

- llava_1_5_7b: {"schedule_off": 20, "schedule_on": 20}
- internvl_2_5_8b: {"schedule_off": 22, "schedule_on": 22}
- qwen2_5_vl_7b: {"schedule_off": 13, "schedule_on": 13}
- qwen3_vl_8b: {"schedule_off": 23, "schedule_on": 23}

72条训练轨迹、96个checkpoint评价；完整逐seed/均值标准差、两阈值及实际epoch/LR在JSON/CSV和各模型history中。
