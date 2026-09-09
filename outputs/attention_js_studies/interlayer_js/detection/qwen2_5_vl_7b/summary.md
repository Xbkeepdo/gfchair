# Qwen2.5-VL-7B 层间 JS 幻觉检测

严格图片级 3200/800 split；唯一无冲突 target token；三层 MLP [128,64,32]；seeds 43/44/45。主排序看 seed-ensemble AUROC。

最佳层间 JS 特征为 `adj_A_all`：AUROC `0.7841`，Hall AUPR `0.2780`。

| Feature set | Dim | AUROC mean±std | Ensemble AUROC | Ensemble Hall AUPR | Hall F1 mean |
|---|---:|---:|---:|---:|---:|
| `adj_A_all` | 27 | 0.7787±0.0048 | 0.7841 | 0.2780 | 0.1932 |
| `adj_E_all` | 27 | 0.7765±0.0052 | 0.7835 | 0.2774 | 0.1379 |
| `adj_E_top32` | 27 | 0.7611±0.0028 | 0.7685 | 0.2527 | 0.1381 |
| `adj_A_top32` | 27 | 0.7571±0.0070 | 0.7636 | 0.2617 | 0.1034 |

四组特征均独立训练，只使用相邻层 JS；A/E 为 plain attention/evidence。没有融合，且全层两两 JS 不进入检测器。

## 不训练：相邻层 JS 均值

这里直接把每个 token 的全部相邻层 JS 取均值作为 hall score；用于判断信号来自总体幅度还是层位模式。

| Block | Dim | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|
| `adj_A_all` | 27 | 0.6314 | 0.1536 |
| `adj_A_top32` | 27 | 0.6306 | 0.1545 |
| `adj_E_all` | 27 | 0.6271 | 0.1552 |
| `adj_E_top32` | 27 | 0.6291 | 0.1567 |

测试集 HALL 比例为 `0.1063`；因此 Hall AUPR 应同时与该先验比较。均值分数明显弱于层向量 MLP，说明主要信息来自哪几个层转换发生变化，而非总体 JS 单调升高。
