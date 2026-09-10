# 2026-09-10 实验代码与结果发布

本次发布包含上次提交 `cecfb37` 之后的实验代码、说明、结果表、逐 seed 指标和绘图数据。
按用户要求，本次不新增 PNG/PDF/SVG、COCO照片、模型权重、逐目标完整特征张量或训练缓存。
历史已跟踪图片保持不变。文档中原有 PNG/PDF 链接指向本地生成产物；GitHub 上请使用以下数值数据。

## 结果与绘图数据索引

| 实验 | 结果/说明 | 绘图数据 |
|---|---|---|
| 六模型 raw attention + log1p(S) | [结果表](../outputs/raw_attention_strength/summary.md) | 各模型 `curves.csv`，train/test、HALL/REAL、mean/median/q25/q75 |
| 六模型 cosine-Q 与 raw attention 的 JS | [结果表](../outputs/cosine_raw_attention_js02/summary.md) | 各模型 `curves.csv` |
| 完整因果前缀四区域 | [协议](PREFIX_ATTENTION_GATE.md) | [数据根](../outputs/prefix_attention_gate/full/region_plots)，各模型 `curves.csv` 和 `metadata.json` |
| MiniGPT-4 / Shikra 主方法与baseline | [总表](../outputs/minigpt4_shikra_path_summary/summary.md) | 两模型结果根的 JSON；原前缀/路径大张量不发布 |
| MiniGPT/Shikra F+raw-Q JS(.2) | [结果表](../outputs/ae_s_qjs02/summary.md) | 各模型结果 JSON 内逐 seed 指标 |
| MiniGPT/Shikra F+cosine JS(.2) | [结果表](../outputs/ae_s_cosinejs02/summary.md) | 各模型结果 JSON 内逐 seed 指标 |
| 四模型完整FFN来源分解 | [实验说明](FFN_SOURCE_COMPOSITION_EXPERIMENT.md) | [数据根](../outputs/ffn_source_composition_v1)：各模型 `curves.csv`、fixed-QK CSV；热图示例 `example_label*.npz/.json` |
| 来源份额、原始/对数强度 | [份额表](../outputs/ffn_source_composition_v1/source_share_detection/summary.md)、[强度表](../outputs/ffn_source_composition_v1/source_strength_detection/summary.md) | `text_source_curves/curves.csv`、`source_strength_detection/curves.csv`、逐seed CSV |
| QE/QC 温度.2融合 | [结果表](../outputs/ffn_source_composition_v1/q_softmax_tau02/summary.md) | 各模型 `curves.csv`、`seed_metrics.csv`、`paired.csv` |
| Endpoint温度与融合 | [温度表](../outputs/endpoint_temperatures4000/summary.md)、[融合根](../outputs/endpoint_tau02_fusion4000) | 各模型曲线 CSV 与结果 JSON |
| Endpoint内积、标量投影与JS分解 | [结果索引](EXPERIMENT_RESULTS_INDEX.md) | 四模型 `results/ffn_visual_source_attribution_v1/tables/endpoint*csv` |
| Q-softmax及温度.07 | [数据根](../outputs/ffn_q_softmax_js_20260909) | 模型结果 JSON、逐层曲线 CSV |
| FFN-output cosine及温度 | [数据根](../outputs/ffn_output_cosine_20260909/cohort500) | 各模型/温度 `curves.csv`、结果 JSON；明确是500图实验 |
| C/Q/B_Q 500图子集 | [数据根](../outputs/ffn_target_consequence_cqb_v1/subset500_20260909) | 结果 JSON/CSV和子集协议；不上传原始 C 张量 |

## 复画与统计口径

- CSV 提供已完成图的实际统计量，不用推理模型即可复画。对含 `layer/mean/q25/q75` 的数据，
  按模型、split、label以及signal/metric/region分组，画mean曲线和q25–q75阴影；阴影是IQR，不是置信区间。
- 六模型四区域总览可直接执行 `python scripts/plot_prefix_attention_regions.py --overview`，只读发布的CSV/metadata。
- Raw-attention总览可运行 `python scripts/train_raw_attention_strength.py --summarize`；
  cosine/raw-attention JS总览可运行 `python scripts/train_cosine_raw_attention_js.py --summarize`。
- 热图NPZ包含每个命名map的 `[layer,visual_token]` 数组，使用 `numpy.load(..., allow_pickle=False)`；
  相邻JSON包含COCO image ID、目标信息和visual_grid。按相应层reshape到网格即可重画，不包含COCO原照片。
- 新六模型主表按seeds43/44/45独立指标的均值±总体标准差（ddof=0），不以ensemble作为主表。
  部分历史结果仍保留明确标注的ensemble，统计口径不能混用。
- 原有数值失败/探索性限制、500图与4000图区别、前缀BOS与模板标记的区别均保留在说明和审计结果中。
  上传不代表重新验证大模型或重跑实验。

## 本次验证

- 45项相关标准库unittest通过，覆盖模型wrapper、位置/gate/区域划分、来源分解、JS特征及InsLen协议。
- 对发布的JSON进行解析检查，并检查暂存内容、文件大小和凭据特征；不新增图片或权重文件。
- CSV保留标准CRLF行尾；vendor EVA代码保留上游尾空白，不为发布改写冻结产物。
- 具体发布文件及SHA256列于 [publication_20260910_manifest.csv](publication_20260910_manifest.csv)。
