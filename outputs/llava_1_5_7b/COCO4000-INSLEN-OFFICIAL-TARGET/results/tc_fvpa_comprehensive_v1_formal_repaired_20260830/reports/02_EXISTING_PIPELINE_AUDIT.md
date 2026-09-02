# 既有流水线预检审计（编辑前基线）

生成日期：2026-08-29（UTC）
仓库提交：`dd3c1ecec02c37bac23687959a0508b0b53a452f`
工作树：预检时干净，分支 `main` 与 `origin/main` 对齐。

## 已经实现的内容

- 已按 InsLen official first-token/first-occurrence 协议固定目标首 subtoken、第一次出现及目标进入 prefix 前的因果预测行；已有记录保存 `target_token_id`、`response_index`，标签方向为 `0=HALL, 1=REAL`。
- `features/visual_ffn_jacobian.py` 已实现 source-wise attention residual write：覆盖 LLaVA/Qwen separate QKV、GQA、InternLM2 packed QKV，output projection bias 只在完整 attention 重建中加入一次，不分摊给视觉 token。
- 已实现 `FFN(Norm(z))` 的 batched `vmap(jvp)`、自适应 chunk、CUDA OOM 回退、aggregate WRITE/JFFN/gain/direction，以及 WRITE、signed-Q、cancellation 的二轮 payload。
- 已有原子 `.pt` shard、双 rank 稳定分片、resume、有限值拒绝、正式 cohort 完整性和唯一 position/mention 检查；已有 probe 采用图片级 split、train-only calibration 和 10,000 次图片级 paired bootstrap。
- 已实现 target logit 与 fixed-clean-competitor margin 的 FFN-output gradient、local attribution additivity 和低精度 FFN-output intervention；尚未实现 log-probability、显式 Riesz pullback接口、finite-path attribution和真正 FP32 downstream。

## 已完成模型和实验

- LLaVA-1.5-7B：正式 JFFN/WRITE 全 cohort 完成，3971 图、14951 唯一目标位置、15463 mentions；target-logit/margin 子集完成 25 图、200 target-layer cases。
- InternVL2.5-8B：正式 JFFN/WRITE 全 cohort 完成，3927 图、11630 唯一目标位置、11759 mentions；target-logit/margin 子集完成 25 图、200 target-layer cases。
- Qwen2.5-VL-7B 与 Qwen3-VL-8B：只完成 JFFN 五图 smoke、熵校准及若干旧特征/QA 实验；Qwen2.5 仅有不完整的早期正式 shard，Qwen3 没有正式 JFFN shard。二者不得计为正式 TC-FVPA 结果。
- POPE、CLEVR、AMBER 已有旧 `risk+EV` QA 结果，但没有本次 local-Riesz/path 变体，因此不能作为本研究的外部泛化结果。

## 当前有证据支持的科学结论

- Attention write 分解和局部 `J_G a_m` 在合成测试、真实 smoke、LLaVA FP32 FFN 副本有限差分中数值可信；它描述的是在观测 clean attention pattern 下的 value-path write，不是删除图像 patch 的完整因果效应。
- `P_JFFN` 比旧 hidden-cosine source 更尖锐、目标特异且空间定位更好，但 WRITE magnitude 已解释几乎全部 token 排名：`corr(I_m,E_m)` 为 LLaVA 0.9806、InternVL 0.9901。
- 固定 16--32 层上，JFFN 相对 WRITE 的 patch AUPRC/bbox mass 在两模型均显著下降；因此现有证据属于 Outcome C，而不是 Outcome A。
- 多层 aggregate/tokenwise gain `S` 作为旧 `risk+EV` 的附加块在 LLaVA 与 InternVL 提高 AUROC；但 `S` 单独不能替代 `risk+EV`，且最优聚合方式依赖模型。

## 已失败或仍未验证的主张

- “JFFN token map 优于 WRITE”已在两个正式模型上失败。
- “简单 gain `S` 在 WRITE/Input magnitude 条件下具有稳定独立价值”在线性条件分析中失败；非线性 `[risk,EV,S]` 增益不能直接归因于独立 FFN token reranking。
- target-margin attribution 的 REAL/HALL 分离没有跨模型复现；正 attribution map 的空间定位弱于 WRITE。
- 现有 FP16/BF16 intervention 的 observed change 大量量化为零；把低精度 logits cast 到 FP32 不构成 FP32 因果验证。当前 causal magnitude 为 FAIL，direction 仅 PARTIAL。
- Qwen 正式复现、log-probability Riesz、current-block path integration、finite LOO/fixed-QK/full activation/pixel counterfactual、Shapley、SwiGLU neuron intervention、完整 geometry controls 和 TC-FVPA hallucination probes均未执行。

## 最重要的理论、数值和实验缺口

1. 缺少将 downstream scalar differential 明确表示为 FFN-output Riesz vector `g` 及 pre-FFN pullback `J_G^T g` 的统一接口和 VJP/JVP/finite-difference验证。
2. current-point Jacobian 只是一阶局部量；完整移除 current-block visual write 的既有 median relative error 约 0.084，需要有限路径分解和 completeness/convergence 审计。
3. 因果验证必须让 intervention vector 在插入低精度张量前保持 FP32，并以 FP32 执行下游 decoder/final norm/LM head；当前实现不满足。
4. 既有 shard 没有保存完整 `a_m`、`J_Ga_m` 和 downstream gradient，不能离线推导 path/Riesz；正式新实验需要重新 forward，并需要更严格的 manifest/checksum/failure persistence。
5. 尚无四模型 preregistered minimum cohort、region counterfactual、Shapley、SwiGLU neuron 或跨数据集 TC-FVPA 结果；任何报告必须标为 `NOT RUN`/`BLOCKED`，不能由旧结果外推。

## 预检环境

- `python --version`：Python 3.9.18（当前 shell 默认解释器；正式模型脚本通常使用 `/opt/conda/private/envs/vicr/bin/python`）。
- GPU：2 × NVIDIA GeForce RTX 3090，24 GiB；预检时均 0 MiB、无运行进程；driver 570.153.02，CUDA capability report 12.8。
- `pip freeze`、`nvidia-smi`、Git 命令的原始输出将在本实验的 environment/hardware manifest 中持久化；默认解释器仅含轻量环境，不代表 vicr 正式模型环境。
