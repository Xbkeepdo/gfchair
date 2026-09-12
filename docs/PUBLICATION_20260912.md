# 2026-09-12 代码、实验结果与绘图数据发布

基于上次发布 `2a96b3d`，本次发布当前已完成实验的代码、测试、中文报告、结果表、
协议及绘图数据。图只发布数值数据，不新增PNG/PDF/SVG或照片；模型权重、
逐图PT特征、逐目标谱和mention明细、训练矩阵、预测缓存、训练头及运行日志保留本地。

部分报告的图片链接指向本地生成的文件。GitHub上使用下表的CSV；旧有已跟踪
图片不在此次变更中。旧报告中的“未上传”描述属于当时的运行记录，本次发布以
此索引及Git提交为准。

## 数据入口

| 实验 | 结果与方法 | 已发布的绘图数据 |
|---|---|---|
| FFN scalar identity：四模型500图、全部目标/层 | [报告](FFN_SCALAR_IDENTITY.md)、[逐token汇总](../outputs/ffn_scalar_identity_20260912/tokens500/summary.csv)、[子空间汇总](../outputs/ffn_scalar_identity_20260912/subspace500/summary.csv) | [C/g/R曲线](../outputs/ffn_scalar_identity_20260912/tokens500/curves.csv)、[Q/Y/B曲线](../outputs/ffn_scalar_identity_20260912/subspace500/curves.csv)；各模型曲线、配对结果及audit.json |
| FFN amplification/rotation/cancellation | [报告](FFN_INPUT_GEOMETRY.md)、[4000图汇总](../outputs/ffn_input_geometry_20260912/summary.csv)、[500图汇总](../outputs/ffn_input_geometry_20260912/cohort500/summary.csv) | [4000图曲线](../outputs/ffn_input_geometry_20260912/curves.csv)、[500图三指标及输入/输出抵消曲线](../outputs/ffn_input_geometry_20260912/cohort500/curves.csv)、paired_images.csv |
| attention及attention×gate同分组检测 | [报告](PREFIX_ATTENTION_GROUP_DETECTION.md)、[六模型总表](../outputs/prefix_attention_gate/full/group_detection/summary.md) | [曲线](../outputs/prefix_attention_gate/full/group_detection/curves.csv)、detection/seed_metrics/comparisons.csv、原始总量舍入检查 |
| visual/generation比值 | [绘图协议](../outputs/prefix_attention_gate/full/region_plots/visual_generation_ratio/README.md)、[检测结果](../outputs/prefix_attention_gate/full/region_plots/visual_generation_ratio/detection/summary.md) | [曲线](../outputs/prefix_attention_gate/full/region_plots/visual_generation_ratio/curves.csv)、protocol.json、检测CSV与input_check.json |
| attention与对应S_g融合 | [报告](ATTENTION_STRENGTH_FUSION.md)、[四模型80组结果](../outputs/ffn_source_composition_v1/attention_strength_fusion/summary.md) | detection.csv、seed_metrics.csv、comparisons.csv及各模型detection/protocol.json |
| 原始attention/gate + log1p(S)：MLP与树模型 | [MLP报告](RAW_ATTENTION_LOG_STRENGTH_MLP.md)、[树模型报告](RAW_ATTENTION_LOG_STRENGTH_TREES.md)、[MLP总表](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/summary.md)、[四种分类器总表](../outputs/ffn_source_composition_v1/raw_attention_log_strength_mlp/trees_summary.md) | detection/seed_metrics/comparisons/selected_params及trees前缀CSV/JSON；各组selection.json保留全部候选验证结果 |

保留4000图与500图、逐mention与唯一target、REAL/HALL和原train/test划分的区别。
scalar identity统计为描述性研究；K4与数值秩阈值等限制见原报告。分类器表保留
逐seed均值/标准差及原有阈值口径，不将不同协议的数据合并成一个指标。

## 仅用发布数据复画

以下示例只读CSV，调用已有绘图函数，不加载模型、PT特征或训练矩阵：

```python
import csv
from pathlib import Path
from scripts import analyze_ffn_input_geometry as geometry
from scripts.run_ffn_scalar_subspace import TITLES

geometry.TITLES.update(TITLES)
path = Path('outputs/ffn_scalar_identity_20260912/subspace500/curves.csv')
with path.open() as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    row['layer'] = int(row['layer'])
    for key in ('mean', 'median', 'q25', 'q75'):
        row[key] = float(row[key])
geometry.plot(
    rows, 'all',
    ('scalar_error_full', 'leakage_ratio', 'sigma_Y_p90_p10'),
    Path('outputs/reproduced_subspace'),
)
```

其他曲线CSV提供相应的model、scope/split、metric/信号、label、layer和
mean/median/q25/q75；按同样字段分组即可复画。原始提取、训练或重新计算全部
统计的脚本仍需要本地模型与缓存，发布CSV用于直接绘图和核对已完成结果。

## 发布检查

- 10项现有定向unittest通过；prefix分组、原始attention+log-strength分组及树网格检查通过。
- 上述CSV-only复画示例已在禁用CUDA的环境实际生成PNG/PDF，通过；生成图片未加入发布。
- 发布JSON全部可解析；新Python文件编译通过；暂存差异及文件类型检查通过。
- 没有新增图片、PDF、模型/特征/训练二进制文件、日志、环境密钥或对话交接文档。
- 数值结果使用已有完成产物，没有重新训练或运行VLM。历史CURRENT_TASK归档保留，
  本次未纳入旧的CODEX_CONVERSATION_HANDOFF对话文件。
