# Visual-only统一标准化后去BN协议（固定811）

## 目的

测量统一train-only标准化的`legacy_visual=[AE_V,log1p(S_E)]`冻结单层MLP去掉BatchNorm后的检测效果。

## 固定项与唯一变化

- 固定图片级3200 train / 400 validation / 400 test、全部mentions及seeds42/43/44。
- 每列mean/std仅用train mentions拟合，并用于validation/test。
- 保留各模型冻结配置的width、dropout、activation、learning rate、weight decay、batch size、monitor、scheduler和early stopping。
- 唯一变化为强制`batch_norm=false`。Qwen2.5、LLaVA、Qwen3原为true，需重新训练；InternVL原为false，直接复用。
- 不重新选参；固定test已在前序实验访问，本实验属于探索性消融。

入口：`scripts/evaluate_legacy_visual_standardized_no_bn_seed_424344_811.py`。
