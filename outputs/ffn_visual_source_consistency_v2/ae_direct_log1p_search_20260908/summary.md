# AE + log1p(原始 S)：单独扩大调参

四模型原3200/800图片划分；2560/640内部训练/验证。无tau、无额外特征标准化。
每模型/分类器48候选(seed43)，内部前三各补seed44/45，以三seed平均验证AUROC选参。
选参封存后才评估800图；最终每头seeds43/44/45。旧800图已用于研究探索，不是独立确认。
同参数scaled对照仅改变输入变换，不是分别对两种变换调到最优。原三隐藏层参数的direct对照单列。
保留K4数值FAIL例外及旧InternVL严格CPU复核状态；无bootstrap或新2000图实验。

## 文献依据与差异

- [Gorishniy等，NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf)：验证集选超参数，测试集最终评估，选定配置多seed复跑。
- [官方MLP配置](https://raw.githubusercontent.com/yandex-research/rtdl-revisiting-models/main/output/california_housing/mlp/tuning/0.toml)：100次搜索；lr 1e-5至1e-2，dropout 0至.5，weight_decay允许0或1e-6至1e-3。
- [Grinsztajn等，NeurIPS2022](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)：约400次随机搜索，比较不同搜索预算；默认配置纳入搜索。
- [Bergstra与Bengio，JMLR2012](https://www.jmlr.org/papers/v13/bergstra12a.html)：随机搜索是固定预算下的基本对照。
- 本轮48组而非100/400组；MLP保留现有BatchNorm、Adam、batch256、最多100epoch和train-loss checkpoint，不照搬论文的AdamW/验证早停/quantile变换。
- XGBoost扩展深度、树数、学习率、行/列采样与正则；固定树数，无验证早停。参数预算相同不代表耗时相同。

## 选定参数

| 模型 | 分类器 | 配置 | 内部验证 AUROC三seed均值 |
|---|---|---|---:|
| llava_1_5_7b | one_hidden | `{"dropout": 0.3, "hidden_sizes": [128], "learning_rate": 0.001, "weight_decay": 1e-05}` | 0.909754 |
| llava_1_5_7b | three_hidden | `{"dropout": 0.3, "hidden_sizes": [256, 128, 64], "learning_rate": 0.0003, "weight_decay": 1e-05}` | 0.907866 |
| llava_1_5_7b | xgboost | `{"colsample_bytree": 0.7168183799123804, "learning_rate": 0.06460620369374312, "max_depth": 6, "min_child_weight": 2.0394318683988217, "n_estimators": 1000, "reg_alpha": 0.07473674186184502, "reg_lambda": 0.029540945510080602, "subsample": 0.8744738238433856}` | 0.901931 |
| internvl_2_5_8b | one_hidden | `{"dropout": 0.2743046698735786, "hidden_sizes": [32], "learning_rate": 0.0019533187834888588, "weight_decay": 3.7294875898043635e-06}` | 0.872938 |
| internvl_2_5_8b | three_hidden | `{"dropout": 0.3, "hidden_sizes": [128, 64, 32], "learning_rate": 0.001, "weight_decay": 1e-05}` | 0.879891 |
| internvl_2_5_8b | xgboost | `{"colsample_bytree": 0.7168183799123804, "learning_rate": 0.06460620369374312, "max_depth": 6, "min_child_weight": 2.0394318683988217, "n_estimators": 1000, "reg_alpha": 0.07473674186184502, "reg_lambda": 0.029540945510080602, "subsample": 0.8744738238433856}` | 0.867371 |
| qwen2_5_vl_7b | one_hidden | `{"dropout": 0.2743046698735786, "hidden_sizes": [32], "learning_rate": 0.0019533187834888588, "weight_decay": 3.7294875898043635e-06}` | 0.858997 |
| qwen2_5_vl_7b | three_hidden | `{"dropout": 0.3, "hidden_sizes": [512, 256, 128], "learning_rate": 0.0003, "weight_decay": 1e-05}` | 0.858322 |
| qwen2_5_vl_7b | xgboost | `{"colsample_bytree": 0.7168183799123804, "learning_rate": 0.06460620369374312, "max_depth": 6, "min_child_weight": 2.0394318683988217, "n_estimators": 1000, "reg_alpha": 0.07473674186184502, "reg_lambda": 0.029540945510080602, "subsample": 0.8744738238433856}` | 0.848862 |
| qwen3_vl_8b | one_hidden | `{"dropout": 0.4491483401659788, "hidden_sizes": [128], "learning_rate": 0.002123706936354634, "weight_decay": 3.219073501797785e-05}` | 0.885076 |
| qwen3_vl_8b | three_hidden | `{"dropout": 0.3, "hidden_sizes": [512, 256, 128], "learning_rate": 0.0003, "weight_decay": 1e-05}` | 0.884760 |
| qwen3_vl_8b | xgboost | `{"colsample_bytree": 0.7168183799123804, "learning_rate": 0.06460620369374312, "max_depth": 6, "min_child_weight": 2.0394318683988217, "n_estimators": 1000, "reg_alpha": 0.07473674186184502, "reg_lambda": 0.029540945510080602, "subsample": 0.8744738238433856}` | 0.879580 |

## 800图结果：三seed概率ensemble AUROC / HALL-AUPR（%）

| 特征/分类器 | llava_1_5_7b | internvl_2_5_8b | qwen2_5_vl_7b | qwen3_vl_8b |
|---|---:|---:|---:|---:|
| old_three_hidden_scaled | 90.646 / 72.298 | 86.388 / 55.221 | 88.198 / 47.011 | 89.845 / 66.074 |
| one_hidden/AE+log1p(S) | 89.813 / 70.325 | 85.230 / 52.926 | 87.867 / 44.951 | 88.927 / 65.089 |
| one_hidden/AE+log1p(S/tau)_matched_config | 89.887 / 70.907 | 85.378 / 52.698 | 88.083 / 45.027 | 89.472 / 66.399 |
| three_hidden/AE+log1p(S) | 90.258 / 71.644 | 86.381 / 54.615 | 87.347 / 45.566 | 89.755 / 64.433 |
| three_hidden/AE+log1p(S/tau)_matched_config | 90.270 / 70.996 | 86.388 / 55.221 | 87.172 / 46.024 | 90.214 / 66.214 |
| three_hidden_fixed_direct | 90.496 / 71.928 | 86.381 / 54.615 | 88.214 / 45.774 | 89.651 / 64.934 |
| xgboost/AE+log1p(S) | 89.583 / 69.409 | 85.267 / 51.026 | 85.613 / 43.320 | 89.025 / 62.975 |
| xgboost/AE+log1p(S/tau)_matched_config | 89.532 / 68.874 | 85.227 / 51.097 | 85.652 / 43.735 | 88.902 / 62.391 |

逐seed、mean±std、REAL/HALL AUPR、固定0.5与训练REAL-F1阈值的P/R/F1见summary.json。
