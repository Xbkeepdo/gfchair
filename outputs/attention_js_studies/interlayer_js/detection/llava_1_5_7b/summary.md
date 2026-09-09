# LLaVA-1.5-7B 层间 JS 幻觉检测

严格图片级 3200/800 split；唯一无冲突 target token；三层 MLP [128,64,32]；seeds 43/44/45。主排序看 seed-ensemble AUROC。

最佳层间 JS 特征为 `adj_A_all`：AUROC `0.8575`，Hall AUPR `0.6244`。

| Feature set | Dim | AUROC mean±std | Ensemble AUROC | Ensemble Hall AUPR | Hall F1 mean |
|---|---:|---:|---:|---:|---:|
| `adj_A_all` | 31 | 0.8540±0.0008 | 0.8575 | 0.6244 | 0.5598 |
| `adj_E_all` | 31 | 0.8536±0.0020 | 0.8574 | 0.6239 | 0.5563 |
| `adj_E_top32` | 31 | 0.8511±0.0016 | 0.8550 | 0.6128 | 0.5555 |
| `adj_A_top32` | 31 | 0.8473±0.0016 | 0.8506 | 0.6165 | 0.5455 |

四组特征均独立训练，只使用相邻层 JS；A/E 为 plain attention/evidence。没有融合，且全层两两 JS 不进入检测器。

## 不训练：相邻层 JS 均值

这里直接把每个 token 的全部相邻层 JS 取均值作为 hall score；用于判断信号来自总体幅度还是层位模式。

| Block | Dim | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|
| `adj_A_all` | 31 | 0.7162 | 0.4428 |
| `adj_A_top32` | 31 | 0.7145 | 0.4502 |
| `adj_E_all` | 31 | 0.7172 | 0.4215 |
| `adj_E_top32` | 31 | 0.7031 | 0.4001 |

测试集 HALL 比例为 `0.2282`；因此 Hall AUPR 应同时与该先验比较。均值分数明显弱于层向量 MLP，说明主要信息来自哪几个层转换发生变化，而非总体 JS 单调升高。
