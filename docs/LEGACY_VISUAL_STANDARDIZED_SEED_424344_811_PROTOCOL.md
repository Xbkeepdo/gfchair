# Visual-only冻结单层MLP统一标准化协议（固定811）

## 目的

测量对`legacy_visual=[AE_V,log1p(S_E)]`冻结单层MLP统一使用train-only逐特征z-score后的效果。

## 设置

- 固定图片级3200 train / 400 validation / 400 test、全部mentions及seeds42/43/44。
- 保留各模型上一轮由validation选择的网络宽度、BN、dropout、激活、学习率、weight decay、batch、monitor和早停设置，不重新选参。
- 仅强制`standardize=True`；每列mean/std只在train mentions拟合，并原样用于validation/test。
- Qwen2.5和LLaVA原配置为false，需要训练标准化对照；Qwen3和InternVL原配置已为true，直接复用已有同种子权重及概率。
- 固定test已在前序实验访问，本实验属于探索性消融。

入口：`scripts/evaluate_legacy_visual_standardized_seed_424344_811.py`。
