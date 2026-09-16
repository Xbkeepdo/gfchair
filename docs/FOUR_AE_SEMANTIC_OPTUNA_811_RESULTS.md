# 四模型 AE 与语义注意力：Optuna 单层 MLP 811 结果

复用 ENDAC exact controlled mentions、固定 3200/400/400 图片划分与已提取特征。每模型五组输入：`[AE_V,log1p(S_E)]`，以及 raw/norm × Top16/32 的 `[Σ_{v∈T_k}a_v p_v(y),log1p(S_E)]`。所有头均用训练集 z-score、无 BN；Optuna 5.0.0 TPE 每组 24 trial（seed43），验证 AUROC/HALL-AUPR 筛前三后补 seeds44/45，按三种子验证均值冻结配置，再评估 test。测试集此前已被多次查看，因此只作探索性比较。完整 20 组测试均值/标准差、全部选中参数和逐种子明细分别见 [summary.md](../outputs/coco4000_512_endac_four_ae_semantic_optuna_811/summary.md)、[selected_configs.csv](../outputs/coco4000_512_endac_four_ae_semantic_optuna_811/selected_configs.csv)、[summary.csv](../outputs/coco4000_512_endac_four_ae_semantic_optuna_811/summary.csv)。

## 验证集选出的模型内配置

下列 AUROC/HALL-AUPR 是 seeds43/44/45 测试集均值，单位%；不是按测试集挑选的最好值。

| 模型 | 验证选中 | Optuna AUROC / AP | 同轮 AE 对照 | 既有原生 SVAR | 选中头参数（宽度、dropout、lr、wd、batch、monitor） |
|---|---|---:|---:|---:|---|
| Qwen2.5 | AE | 87.18 / 54.77 | 87.18 / 54.77 | 86.19 / 43.67 | 256, .5, .000547, 1e-6, 128, val AUROC |
| LLaVA | norm Top32 语义 | 92.41 / 76.99 | 90.39 / 72.63 | 90.68 / 72.37 | 512, .5, .00112, 1e-6, 64, val AUROC |
| Qwen3 | AE | 89.85 / 71.75 | 89.85 / 71.75 | 88.95 / 68.25 | 256, .5, .00128, 1e-4, 128, val AUROC |
| InternVL | norm Top32 语义 | 88.23 / 62.41 | 87.81 / 62.36 | 88.47 / 66.46 | 128, .5, .00415, 0, 128, val AUROC |

全部四个模型内 champion 都选到 ReLU、dropout .5、val AUROC checkpoint；这只是本次候选与验证数据的结果，不证明其他激活/正则单独无效。LLaVA 语义相对同轮 AE 高 +2.02 AUROC / +4.36 AP 百分点；InternVL 语义仅高 +0.42 / +0.05。Qwen2.5、Qwen3 的模型内 champion 都是 AE。InternVL 的 raw Top32 在测试上是 89.61/66.88，高于验证选中的 norm Top32，但不能据此事后更改选择。

与先前固定 128/noBN/标准化的同一 `semantic_attention+logS` 对照相比，验证选中的 LLaVA norm Top32 为 92.41/76.99（旧 92.47/77.40），InternVL norm Top32 为 88.23/62.41（旧 88.35/62.73）；扩大搜参预算没有保证测试分数提高。此前单特征对照见 [single_feature_summary.md](../outputs/coco4000_512_endac_semantic_attention_logs_fusion_811/single_feature_summary.md)。

## 核验

- 四模型各 5×24=120 个 TPE trial 加前三候选的额外 30 次拟合，共 600 次；四卡各一模型。本地 Qwen2.5/LLaVA，远端 Qwen3/InternVL。
- 四模型各自 `selection.json` 均在测试前写入。五组×三种子×四模型共 60 个最终头、预测文件齐备。
- [validation.json](../outputs/coco4000_512_endac_four_ae_semantic_optuna_811/validation.json)：保存预测与保存指标重算误差 0；train-only scaler 与 CPU checkpoint 重载通过，最大概率差 3.58e-7。Qwen2.5 一组有近并列概率，CPU 与 GPU 重载后的 AUROC 最大差 7.56e-6，不改变保存的测试预测或选型。
