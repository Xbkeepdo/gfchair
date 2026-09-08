# qwen2_5_vl_7b Baseline（三层共享 MLP） 结果汇总

## 实验协议

- 随机种子：`44`。
- headline 正类：real。
- 标签协议：`shared_inslen_unique_detected_word_exact_response_offsets`。
- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。
- test set 只用于最终评估，不用于调参、早停或选阈值。
- image split：train/val/test = 3200/0/800。
- token 样本：train/val/test = 6985/0/1732。
- 表中数值来自单次随机种子运行，不是多 seed 平均。
- 原始结果：`/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/baseline/results/shared_torch_mlp/seed44/qwen2_5_vl_7b_baselines_shared_torch_mlp.json`。

## Test 结果（Train Real-F1 搜索阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8955 | 0.9286 | 0.9567 | 0.9424 | 0.8330 | 0.9712 | 0.5074 | 0.3770 | 0.4326 | 0.8330 | 0.3902 |
| MetaToken-MLP | 0.8943 | 0.8943 | 1.0000 | 0.9442 | 0.7577 | 0.9603 | 0.0000 | 0.0000 | 0.0000 | 0.7577 | 0.2422 |

## Test 结果（固定阈值 0.5）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.8990 | 0.9204 | 0.9709 | 0.9450 | 0.8330 | 0.9712 | 0.5408 | 0.2896 | 0.3772 | 0.8330 | 0.3902 |
| MetaToken-MLP | 0.8943 | 0.8943 | 1.0000 | 0.9442 | 0.7577 | 0.9603 | 0.0000 | 0.0000 | 0.0000 | 0.7577 | 0.2422 |

## Train 结果（train-F1 模式在此选择阈值）

| 方法 | Accuracy | Real P | Real R | Real F1 | Real AUROC | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVAR | 0.9933 | 0.9976 | 0.9949 | 0.9962 | 0.9993 | 0.9999 | 0.9573 | 0.9795 | 0.9683 | 0.9993 | 0.9937 |
| MetaToken-MLP | 0.8951 | 0.8951 | 1.0000 | 0.9446 | 0.7870 | 0.9683 | 0.0000 | 0.0000 | 0.0000 | 0.7870 | 0.2668 |

## 简要结论

- Test Real F1 最高的方法是 **MetaToken-MLP**：0.9442。
