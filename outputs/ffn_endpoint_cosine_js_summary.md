# Endpoint-cosine distribution vs T：逐层 JS 幻觉检测

每格为三 seed 概率 ensemble 的 AUROC / HALL-AUPR；F1 是各 seed 在训练集 REAL-F1 阈值下的测试 HALL-F1 均值。

| Model | AUROC | HALL-AUPR | Mean HALL-F1 |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 0.788201 | 0.302792 | 0.184791 |
| LLaVA-1.5-7B | 0.869978 | 0.634024 | 0.554503 |
| Qwen3-VL-8B | 0.852969 | 0.569851 | 0.485376 |
| InternVL2.5-8B | 0.844669 | 0.508138 | 0.405045 |

输入仅从已保存的 `path_signed_q`、`ffn_path_gross` 和 `attention_evidence` 恢复；未运行VLM。无bootstrap或独立cohort，结果为探索性。
