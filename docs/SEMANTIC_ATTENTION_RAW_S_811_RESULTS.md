# 全视觉语义注意力：原始 S_E 与 log1p(S_E) 的 811 对照

## 协议和口径

复用 ENDAC exact 同一批 mention 的全视觉 token 六项特征，只把拼接的 Visual-only 条件路径强度 `log1p(S_E)` 换为原始 `S_E`。原始 S 由已保存的 FP32 `log1p(S_E)` 做 `expm1` 恢复，不重新运行 VLM；四模型均有限、非负，反变换再取 log 与原缓存最大差 `1.2e-7`。

所有变体仍先拼接、再只用训练集拟合逐列 z-score；固定图片级 3200/400/400、种子 43/44/45、128/ReLU/dropout0.3/无BN单隐藏层 MLP，最低 validation BCE loss checkpoint。线性标准化不会消除 log 的非线性压缩。228 个头验证完成后，才按三种子 validation AUROC、HALL-AUPR 冻结 raw/norm/特征组，随后评估 test。下面为 test 三种子均值，单位百分比；此项目 test 已在其他实验中查看，结论仅作探索性对照。

**更正前序 AE 比较**：此前引用的旧 `legacy_visual` AE 表来自不同 mention cohort：旧四模型分别为 8717/15463/14873/11759 条，本轮 ENDAC exact 为 7924/12686/11751/10781 条。旧表不能与本轮语义注意力结果直接相减。本报告的 `AE_only`、`AE+S`、`AE+logS` 都在**当前相同样本**上用上述同配置重新训练。

## 六项拼接与 AE 对照

六项拼接的 raw/norm 仅凭 validation 选择：Qwen3 为 raw，其他三模型为 norm。`Δ` 是原始 S 减 logS 的同组 test 百分点差。

| 模型 | 六项+S AUROC/AP | 六项+logS AUROC/AP | Δ | AE+S AUROC/AP | AE+logS AUROC/AP | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 | 85.84/52.01 | 85.99/52.51 | −0.15/−0.50 | 86.57/52.38 | 86.98/53.06 | −0.41/−0.68 |
| LLaVA | 93.29/79.88 | 93.24/79.47 | +0.05/+0.41 | 90.21/72.33 | 90.32/72.37 | −0.11/−0.04 |
| Qwen3 | 92.01/76.99 | 92.09/76.67 | −0.08/+0.32 | 89.76/71.87 | 89.83/71.73 | −0.07/+0.14 |
| InternVL | 89.86/66.66 | 89.96/66.55 | −0.10/+0.11 | 87.10/60.42 | 86.98/60.23 | +0.12/+0.19 |

不取 log 没有一致优势；六项拼接在 LLaVA 双指标小幅上升、Qwen2.5 双指标小幅下降，Qwen3/InternVL 则 AUROC 下降而 AP 上升。Qwen2.5 上同样本 `AE+S` 和 `AE+logS` 均高于对应的语义六项；另外三个模型则相反。差值多与三种子波动同量级，不能宣称显著改进。

## S 与 AE 的单独参考

| 模型 | S-only | logS-only | AE-only |
|---|---:|---:|---:|
| Qwen2.5 | 82.38/40.52 | 81.93/38.69 | 78.07/37.37 |
| LLaVA | 88.64/68.81 | 89.00/69.89 | 85.62/61.35 |
| Qwen3 | 88.19/69.03 | 88.45/68.36 | 83.17/57.71 |
| InternVL | 85.97/57.68 | 86.45/60.22 | 82.75/53.21 |

六个单项按原 `+logS` 验证集选出的 raw/norm **固定同一变体**后，其 `+S`、`+logS` 与差值见 [`single_feature_summary.md`](../outputs/coco4000_512_endac_semantic_attention_raw_s_811/single_feature_summary.md)；这样差值只改变 S 变换。全部 raw/norm 组合及 AE 参考见 [`summary.md`](../outputs/coco4000_512_endac_semantic_attention_raw_s_811/summary.md)。测试明细、标准差、checkpoint 和预测保存在该目录的各模型子目录。

入口：`scripts/train_semantic_attention_raw_s_811.py`。228/228 头的指标与预测文件复算误差为零；四个冠军及 AE 参考头 CPU 重载最大概率误差 `8.94e-7`，见输出目录 `validation.json`。
