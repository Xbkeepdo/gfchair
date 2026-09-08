# LLaVA：四种 Jacobian S 单独训练（full 3971-image official-target cohort）

四个分类头均只输入 32 层 S，不包含 risk 或 EV；MLP、图片级 split、seeds 43/44/45 和训练协议完全一致。

| S-only feature | Mean AUROC | Mean Hall F1 | Mean Hall AUPR | Ensemble AUROC | Ensemble Hall AUPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| All-token aggregate S | 0.864417 ± 0.000909 | 0.582569 | 0.623901 | 0.868466 | 0.631508 |
| All-token tokenwise S | 0.879812 ± 0.001614 | 0.591302 | 0.651029 | 0.882407 | 0.655472 |
| Union-TopK tokenwise S | 0.874654 ± 0.001957 | 0.596298 | 0.625348 | 0.877570 | 0.630148 |
| Union-TopK aggregate S | 0.862309 ± 0.001108 | 0.576299 | 0.602870 | 0.865925 | 0.609410 |

## 两两 seed-ensemble 图片级 bootstrap

- All-token aggregate S − All-token tokenwise S：AUROC Δ=-0.013941，95% CI [-0.021859,-0.006120]；Hall-AUPR Δ=-0.023964，95% CI [-0.047159,-0.000116]。
- All-token aggregate S − Union-TopK tokenwise S：AUROC Δ=-0.009104，95% CI [-0.017917,-0.000573]；Hall-AUPR Δ=+0.001359，95% CI [-0.025148,+0.027097]。
- All-token aggregate S − Union-TopK aggregate S：AUROC Δ=+0.002541，95% CI [-0.004288,+0.009726]；Hall-AUPR Δ=+0.022097，95% CI [+0.000314,+0.042337]。
- All-token tokenwise S − Union-TopK tokenwise S：AUROC Δ=+0.004837，95% CI [-0.000063,+0.009667]；Hall-AUPR Δ=+0.025323，95% CI [+0.011462,+0.038318]。
- All-token tokenwise S − Union-TopK aggregate S：AUROC Δ=+0.016482，95% CI [+0.009003,+0.024296]；Hall-AUPR Δ=+0.046061，95% CI [+0.020336,+0.069602]。
- Union-TopK tokenwise S − Union-TopK aggregate S：AUROC Δ=+0.011645，95% CI [+0.004931,+0.018454]；Hall-AUPR Δ=+0.020738，95% CI [-0.001706,+0.042699]。

## 协议核验

- train/test images：3174/797。
- train/test mentions：12317/3146。
- 每个输入均为 32 维；无特征标准化。
- checkpoint 按 minimum train loss，阈值只用 train F1 选择。
