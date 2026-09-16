# Visual-only 条件路径单层 MLP 搜参协议（811）

- 特征固定为 `legacy_visual=[AE_V, log1p(S_E)]`。`AE_V` 是视觉 attention mass；`S_E` 是沿真实 RMS Visual-only 条件路径 `z-A_V→z` 得到的逐视觉 token FFN 响应范数之和。
- 直接复用 `outputs/all_attention_ae_811_v1/<model>/matrices.pt`，不重新抽取模型特征。
- 数据固定为图片级 3200 train / 400 validation / 400 test，保留全部 mentions；split seed 20260912。
- 复用 `SINGLE_MLP_SEARCH_811` 的同一组24个候选：严格单隐藏层，宽度、StandardScaler、BatchNorm、激活、dropout、学习率、weight decay、batch size及验证监控项可变；最多150 epochs，patience20，训练 seeds 43/44/45。
- 每模型先用 seed43 跑24候选并按 validation AUROC、HALL-AUPR、候选序号排序；前三名补 seeds44/45，再按三seed validation均值冻结最终参数。
- 四模型参数全部冻结后才运行 test；不合并 train/validation 重训。报告三seed测试AUROC和HALL-AUPR均值±总体标准差。
- 同表复用早期 `legacy_visual` 三层MLP、sklearn单层12候选、XGB 18候选及原生SVAR已有结果。各分类器搜参预算不同。
- 固定400图测试集已经在此前实验中访问，本轮属于探索性比较，不据test继续扩参或选择分类器。

运行入口：`scripts/search_legacy_visual_single_mlp_811.py`；输出：`outputs/legacy_visual_single_mlp_search_811_v1/`。
