# InternVL2.5-8B 层间 JS 幻觉检测

严格图片级 3200/800 split；唯一无冲突 target token；三层 MLP [128,64,32]；seeds 43/44/45。主排序看 seed-ensemble AUROC。

最佳层间 JS 特征为 `adj_A_all`：AUROC `0.8002`，Hall AUPR `0.3927`。

| Feature set | Dim | AUROC mean±std | Ensemble AUROC | Ensemble Hall AUPR | Hall F1 mean |
|---|---:|---:|---:|---:|---:|
| `adj_A_all` | 31 | 0.7945±0.0011 | 0.8002 | 0.3927 | 0.2540 |
| `adj_E_all` | 31 | 0.7849±0.0017 | 0.7894 | 0.3886 | 0.2645 |
| `adj_A_top32` | 31 | 0.7825±0.0019 | 0.7886 | 0.3857 | 0.1885 |
| `adj_E_top32` | 31 | 0.7687±0.0029 | 0.7738 | 0.3718 | 0.1882 |

四组特征均独立训练，只使用相邻层 JS；A/E 为 plain attention/evidence。没有融合，且全层两两 JS 不进入检测器。

## 不训练：相邻层 JS 均值

这里直接把每个 token 的全部相邻层 JS 取均值作为 hall score；用于判断信号来自总体幅度还是层位模式。

| Block | Dim | Hall AUROC | Hall AUPR |
|---|---:|---:|---:|
| `adj_A_all` | 31 | 0.5300 | 0.1617 |
| `adj_A_top32` | 31 | 0.5166 | 0.1592 |
| `adj_E_all` | 31 | 0.5580 | 0.1759 |
| `adj_E_top32` | 31 | 0.5446 | 0.1728 |

测试集 HALL 比例为 `0.1580`；因此 Hall AUPR 应同时与该先验比较。均值分数明显弱于层向量 MLP，说明主要信息来自哪几个层转换发生变化，而非总体 JS 单调升高。
