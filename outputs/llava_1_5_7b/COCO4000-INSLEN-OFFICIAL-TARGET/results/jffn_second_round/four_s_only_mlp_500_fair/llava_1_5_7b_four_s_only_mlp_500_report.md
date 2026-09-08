# LLaVA：四种 Jacobian S 单独训练（500 图公平 cohort）

四个分类头均只输入 32 层 S，不包含 risk 或 EV；MLP、图片级 split、seeds 43/44/45 和训练协议完全一致。

| S-only feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| All-token aggregate S | 0.826055 ± 0.004179 | 0.558968 | 0.559724 | 0.831725 | 0.564988 |
| All-token tokenwise S | 0.851866 ± 0.010224 | 0.624512 | 0.633475 | 0.859181 | 0.642747 |
| Union-TopK tokenwise S | 0.849838 ± 0.006390 | 0.577861 | 0.620931 | 0.854977 | 0.627291 |
| Union-TopK aggregate S | 0.830376 ± 0.003443 | 0.562651 | 0.549222 | 0.835961 | 0.554423 |

## 两两 seed-ensemble 图片级 bootstrap

- All-token aggregate S − All-token tokenwise S：AUROC Δ=-0.027457，95% CI [-0.054067,-0.001905]；Hall-AUPR Δ=-0.077759，95% CI [-0.146401,-0.015498]。
- All-token aggregate S − Union-TopK tokenwise S：AUROC Δ=-0.023252，95% CI [-0.051927,+0.003532]；Hall-AUPR Δ=-0.062303，95% CI [-0.137442,-0.000017]。
- All-token aggregate S − Union-TopK aggregate S：AUROC Δ=-0.004236，95% CI [-0.031420,+0.021974]；Hall-AUPR Δ=+0.010565，95% CI [-0.056804,+0.061405]。
- All-token tokenwise S − Union-TopK tokenwise S：AUROC Δ=+0.004204，95% CI [-0.011050,+0.018550]；Hall-AUPR Δ=+0.015456，95% CI [-0.023996,+0.051935]。
- All-token tokenwise S − Union-TopK aggregate S：AUROC Δ=+0.023220，95% CI [-0.004707,+0.050957]；Hall-AUPR Δ=+0.088324，95% CI [+0.013162,+0.157038]。
- Union-TopK tokenwise S − Union-TopK aggregate S：AUROC Δ=+0.019016，95% CI [-0.002263,+0.040435]；Hall-AUPR Δ=+0.072868，95% CI [+0.013322,+0.133260]。

## 协议核验

- train/test images：395/105。
- train/test mentions：1513/404。
- 每个输入均为 32 维；无特征标准化。
- checkpoint 按 minimum train loss，阈值只用 train F1 选择。
