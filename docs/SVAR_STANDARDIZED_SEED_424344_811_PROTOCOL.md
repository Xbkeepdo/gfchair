# 比例层SVAR标准化单因素对照协议（固定811）

## 目的

测量逐特征训练集标准化对比例层原生SVAR检测结果的影响。

## 固定项

- 图片级固定811：3200 train / 400 validation / 400 test，split seed 20260912，保留全部mentions。
- 训练种子42/43/44。
- LLaVA使用零基层5–18，即`[5,19)`；其余模型用`start=round(L*5/32)`、`end=round(L*19/32)`同比例映射。
- 输入仍为所选层、各attention head的视觉attention mass展平向量。
- 分类器仍为`Linear(D,248)-ReLU-Linear(248,2)`；Adam lr0.001、batch32、最多50 epochs、最低validation loss checkpoint、patience5；无BN、dropout、weighted sampler或weight decay。

## 唯一变化

每个输入列仅用train mentions拟合均值和总体标准差；train/validation/test统一应用`(x-mean)/std`。小于`1e-6`的标准差置为1。每个训练结果保存mean/scale，重载验证必须核对它们等于train-only重算值。

本实验在固定test已访问后提出，属于探索性消融；不据此宣称独立测试集推广效果。

入口：`scripts/evaluate_svar_standardized_seed_424344_811.py`。
