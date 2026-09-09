# Qwen3-VL-8B 层间 JS 幻觉检测

严格图片级 3200/800 split；唯一无冲突 target token；三层 MLP [128,64,32]；seeds 43/44/45。主排序看 seed-ensemble AUROC。

最佳层间 JS 特征为 `adj_E_all`：AUROC `0.7962`，Hall AUPR `0.4727`。

| Feature set | Dim | AUROC mean±std | Ensemble AUROC | Ensemble Hall AUPR | Hall F1 mean |
|---|---:|---:|---:|---:|---:|
| `adj_E_all` | 35 | 0.7908±0.0043 | 0.7962 | 0.4727 | 0.3640 |
| `adj_A_all` | 35 | 0.7899±0.0022 | 0.7951 | 0.4672 | 0.3542 |
| `adj_E_top32` | 35 | 0.7840±0.0054 | 0.7911 | 0.4545 | 0.3538 |
| `adj_A_top32` | 35 | 0.7827±0.0018 | 0.7877 | 0.4537 | 0.3557 |

四组特征均独立训练，只使用相邻层 JS；A/E 为 plain attention/evidence。没有融合，且全层两两 JS 不进入检测器。

## 不训练：相邻层 JS 均值

这里直接把每个 token 的全部相邻层 JS 取均值作为 hall score；用于判断信号来自总体幅度还是层位模式。

| Block | Dim | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|
| `adj_A_all` | 35 | 0.5925 | 0.2119 |
| `adj_A_top32` | 35 | 0.6028 | 0.2194 |
| `adj_E_all` | 35 | 0.5993 | 0.2176 |
| `adj_E_top32` | 35 | 0.6060 | 0.2244 |

测试集 HALL 比例为 `0.1725`；因此 Hall AUPR 应同时与该先验比较。均值分数明显弱于层向量 MLP，说明主要信息来自哪几个层转换发生变化，而非总体 JS 单调升高。
