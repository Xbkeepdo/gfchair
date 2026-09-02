# TC-FVPA 当前实验结果汇总 — 2026-09-02

本文汇总截至 2026-09-02 已持久化并核验的 TC-FVPA 实验。正式完成、部分失败和未运行严格分开；本文不是“四模型全部完成”的最终结论。

## 1. 执行状态

| 模型 | Local Riesz | Path/LOO | 真 FP32 | Shapley | Analyze/Reports | 总体状态 |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 500 图、3496 cases、0 failure | 200 图、1316 cases、0 failure | 216384 rows、全预注册层、0 failure | 300 rows、128 permutations、0 failure | 已完成 | **正式可审阅，研究总体仍 PARTIAL** |
| LLaVA-1.5-7B | 500 图、3656 cases、0 failure | 200 图、1420 cases、0 failure | 60144 measured rows；低层 6 条 BLOCKED | 300 rows、128 permutations、0 failure | 已完成 | **正式可审阅，研究总体仍 PARTIAL** |
| Qwen3-VL-8B | 3743 成功 + 1 OOM | 1446 成功 + 10 OOM | 未运行 | 未运行 | 未运行 | **FAIL / partial，不可作正式结论** |
| InternVL2.5-8B | 未启动 | 未启动 | 未启动 | 未启动 | 未启动 | **NOT RUN** |

Qwen3 两个 Path rank 都扫描到 100/100 images，但分别以 5 个 OOM 结束为 FAIL。后续控制器因缺少 checksum-verified Qwen3 结果而停止，明确没有启动 InternVL。Qwen3 成功行可以用于失败诊断，不能与 Qwen2/LLaVA 的封存结果并列为正式结果。

VQA、正式幻觉检测器、10,000 次配对 bootstrap、fixed-QK、完整 activation patching、pixel counterfactual 和真实 neuron intervention 均未在本轮执行。

## 2. 正式样本与可复核规模

| 项目 | Qwen2.5-VL-7B | LLaVA-1.5-7B |
|---|---:|---:|
| 预注册层 | 7/14/21/28 | 8/16/24/32 |
| Local images | 500 | 500 |
| Local target-layer cases | 3496 | 3656 |
| REAL / HALL case rows | 2348 / 1148 | 2528 / 1128 |
| Path images | 200 | 200 |
| Path target-layer cases | 1316 | 1420 |
| Path convergence rows | 39480 | 42600 |
| Frozen-write measured rows | 43428 | 46815 |
| Spatial metric rows | 53200 | 57464 |
| Token-score sample rows | 30888 | 76032 |

每个 Path case 完整包含 `3 scalars × K(1/4/8/16/32) × 2 quadratures = 30` 条 convergence 记录。两模型的正式 Path 均固定 `path_batch_size=1`。

## 3. 数值分解与 Local Riesz 一致性

### Qwen2.5-VL-7B

- Attention 重建相对误差：中位数 `8.48e-5`，最大值 `0.00620`。
- FFN component-sum 相对误差：中位数 `0.00279`，最大值 `0.00669`。
- Local Riesz duality 最大绝对误差上界：log-probability `0.0078125`，margin/logit `0.015625`。

### LLaVA-1.5-7B

- Attention 重建相对误差：中位数 `2.69e-4`，最大值 `7.21e-4`。
- FFN component-sum 相对误差：中位数 `4.31e-4`，最大值 `7.70e-4`。
- Local Riesz duality 最大绝对误差上界：log-probability `1.53e-4`，margin `1.22e-4`，logit `2.14e-4`。

这些结果支持实现层面的分解与 JVP/VJP 对偶一致性，但不等价于完整图像删除因果效应。

## 4. Path 相对点 Jacobian 的完整性改进

下表比较 trapezoid K=1 与 K=32 的 median absolute completeness error：

| 模型 / scalar | K=1 | K=32 | 相对下降 |
|---|---:|---:|---:|
| Qwen2 log-probability | 0.022123 | 0.017965 | 18.79% |
| Qwen2 margin | 0.095032 | 0.082031 | 13.68% |
| Qwen2 logit | 0.142151 | 0.126892 | 10.73% |
| LLaVA log-probability | 0.002481 | 0.002008 | 19.08% |
| LLaVA margin | 0.008080 | 0.006592 | 18.42% |
| LLaVA logit | 0.008190 | 0.005005 | 38.89% |

因此，多点 Path integration 在两个已完成模型上都比 K=1 点近似更接近当前块 zero-write finite effect；残差仍非零，不能声称完美恢复。

## 5. Frozen-write 有限效应关联

aggregate visual WRITE 与 frozen-write observed finite effect 的 Pearson 相关：

| 模型 | log-probability | margin | logit |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 0.875 | 0.906 | 0.732 |
| LLaVA-1.5-7B | 0.992 | 0.991 | 0.997 |

对应 sign agreement：

| 模型 | log-probability | margin | logit |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 67.48% | 55.62% | 56.69% |
| LLaVA-1.5-7B | 86.27% | 74.93% | 74.51% |

这是在 clean attention decomposition 条件下的 current-block frozen-write estimand，不是去除原始 image patch 的完整因果效应。

## 6. 真 FP32 局部因果验证

在 `eta=0.025`、strategy=`aggregate_visual_response` 时：

| 模型 / scalar | 样本行 | Sign agreement | Median relative error |
|---|---:|---:|---:|
| Qwen2 log-probability | 616 | 97.56% | 1.06% |
| Qwen2 logit | 616 | 99.35% | 0.64% |
| Qwen2 margin | 616 | 99.35% | 0.99% |
| LLaVA log-probability | 172 | 99.42% | 0.20% |
| LLaVA logit | 172 | 100.00% | 0.09% |
| LLaVA margin | 172 | 100.00% | 0.19% |

Qwen2 的真 FP32 覆盖全部四个预注册层。LLaVA 只在最终层 32 完成真 FP32，layers 8/16/24 各 rank 各保留一条明确 BLOCKED 记录，所以 LLaVA 的 FP32 family 仍为 PARTIAL/BLOCKED。

## 7. Shapley

两个已完成模型都持久化：

- 最终层 50 images。
- 8/16 regions。
- 128 permutations。
- 300 scalar case rows。
- 0 failures。
- Shapley completeness absolute error 的中位数和最大值均为 `0`。

非最终层不在当前已实现 Shapley scope 中，不能外推为全层验证。

## 8. 描述性空间定位

以下是具有有效 REAL bounding box 的 case-method 平均值；尚未执行预注册的 image-cluster paired bootstrap，因此不能把点估计差异解释为统计显著提升。

### Qwen2.5-VL-7B

| 方法 | Top-1 pointing | Patch AUPRC | BBox mass |
|---|---:|---:|---:|
| WRITE | 0.54330 | 0.46846 | 0.39058 |
| JFFN | 0.52098 | 0.46356 | 0.38731 |
| signed Q | 0.51295 | 0.47000 | 0.40933 |
| Path log-probability GL-K32 | 0.50687 | 0.42877 | 0.38625 |

Qwen2 中 WRITE 的 Top-1 最高；signed Q 的 AUPRC 与 BBox mass 略高，但 Top-1 更低，属于混合结果。

### LLaVA-1.5-7B

| 方法 | Top-1 pointing | Patch AUPRC | BBox mass |
|---|---:|---:|---:|
| WRITE | 0.54457 | 0.52466 | 0.45718 |
| JFFN | 0.55306 | 0.51657 | 0.45405 |
| signed Q | 0.59635 | 0.53745 | 0.50181 |
| Path log-probability GL-K32 | 0.53309 | 0.44073 | 0.44565 |

LLaVA 的 signed Q 在三个描述性指标上均为这四项中最高，但在完成配对 bootstrap 前只能报告为候选趋势，不能宣布稳定优于 WRITE。

## 9. Qwen3 失败审计与 InternVL 状态

Qwen3 Local 原始 shard 保存 `3743` 个成功 case 和 `1` 个 OOM；旧 resume 分支随后错误地把已有 shard 标成 PASS，因此权威解释必须同时读取 shard failure，而不能只读 `run_status.json`。

Qwen3 Path 保存：

- rank0：100 images、743 成功、5 OOM、stage FAIL。
- rank1：100 images、703 成功、5 OOM、stage FAIL。
- 合计：1446 成功、10 OOM；峰值显存约 24.6/24.4 GB（十进制）。

失败集中在 6 个长前缀 target 上，共 10 个 layer-case，均为 batch size 已降到 1 后仍 OOM 或 JVP 额外分配约 1.8 GiB 失败。Qwen3 没有生成 analyze、FP32、Shapley、reports 或完整 output checksum。

队列控制器于 2026-08-30 23:03 明确记录 Qwen3 checksum 验证失败，并以 `InternVL not started` 停止。InternVL 没有本轮 formal output root。

## 10. 当前可支持与不可支持的结论

### 可以支持

1. Qwen2 与 LLaVA 的 source-write decomposition、Local Riesz duality 和正式 Path 网格已经形成可审阅证据。
2. 两模型 K=32 Path 的 median completeness error 均低于 K=1。
3. 真 FP32 小扰动支持 aggregate visual-response 的局部线性预测；Qwen2 为全预注册层，LLaVA 限最终层。
4. 当前 frozen-write conditional effect 与 aggregate visual WRITE 高度相关，LLaVA 尤其强。
5. 空间结果是混合的：Qwen2 不支持 Path/JFFN 稳定优于 WRITE；LLaVA signed Q 有积极描述性趋势，但尚无正式显著性检验。

### 不能支持

1. 不能声称四模型均完成；Qwen3 是 FAIL，InternVL 是 NOT RUN。
2. 不能声称 signed Q 或 Path 已显著改善空间定位，因为正式 paired bootstrap 未运行。
3. 不能声称提高幻觉检测，因为本轮 detector 与 10,000 次 detection bootstrap 未运行。
4. 不能把 frozen-write 当作 pixel deletion、fixed-QK 或完整 activation patching。
5. 不能把 LLaVA 最终层 FP32/Shapley 外推到所有层。
6. 不能把未执行 VQA benchmark 解释为外部泛化通过。

## 11. GitHub 精简快照边界

本地 LLaVA 正式根约 486 MiB，Qwen3 partial 根约 199 MiB。GitHub 精简快照只纳入：

- 本汇总、执行说明和报告。
- LLaVA 的 run status、hardware、final audit、output checksums、metrics、schemas。
- LLaVA 的 compact CSV.GZ 表与小型 `shapley_cases.pt`。
- Qwen3 的小型 run status 和失败日志。

以下大型文件保留在本地、不上传：完整 token maps、local/path/FP32/Shapley rank shards、audit vectors、Shapley running estimates、handoff tarball。GitHub 快照足以复核本文统计，但不是完整的张量级实验归档。

## 12. 审阅入口

- Qwen2 正式根：`outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260829/`
- LLaVA 正式根：`outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260830/`
- Qwen3 partial 根：`outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260830/`
- 阶段记录：`docs/CURRENT_TASK.md`
- 运行与解释规则：`docs/TC_FVPA_RUNBOOK.md`
- 旧 JFFN 完整讨论：`docs/JFFN_EXPERIMENT_SUMMARY_FOR_DISCUSSION.md`
