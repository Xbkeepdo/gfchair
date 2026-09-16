# Codex 对话交接：TC-FVPA 四模型实验与双 4090 LLaVA 续跑

更新时间：2026-08-30 22:06 CST（UTC+8）

> 这是给新 Codex 对话读取的自包含交接文档。它不是客户端导出的逐 token 原始聊天记录，而是按时间整理的完整需求脉络、已经执行的工作、实验结论、后台队列状态、文件路径和下一步操作。不要仅凭旧聊天摘要猜测当前状态；开始工作前必须重新核验进程、日志和权威状态文件。

## 1. 用户当前最终安排

用户最后明确要求把计算拆到两台机器：

1. 原双 RTX 3090 对话/机器继续完成 `qwen3_vl_8b`，通过完整性门禁后再运行 `internvl_2_5_8b`。
2. 新对话所在的双 RTX 4090 机器只负责 `llava_1_5_7b` 正式实验。
3. 4090 不要运行 Qwen3 或 InternVL，也不要修改/覆盖旧 3090 正在使用的 Qwen3/InternVL 输出根。
4. 当前暂不运行 VQA benchmark。
5. 正式 Path 固定使用安全的 `path_batch_size=1`；不能为了加速静默改协议。
6. LLaVA 正式阶段应包括 `local`、`path`、真 FP32、Shapley、analyze、counterfactuals、reports；未实现或未执行项必须诚实写成 `BLOCKED/NOT RUN`，不能伪造 PASS。
7. 后台任务需脱离 VS Code/终端运行，并汇报 PID、日志、GPU 显存、进度和 ETA。真实失败必须保留，零失败门禁不得绕过。

## 2. 新对话应先读取的文件

工作目录：

```text
/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
```

依次读取：

```text
AGENTS.md
docs/CURRENT_TASK.md
docs/TC_FVPA_RUNBOOK.md
docs/TC_FVPA_CURRENT_RESULTS_20260830.md
docs/CODEX_CONVERSATION_HANDOFF_20260830.md
```

用户最初提供的 2226 行综合实验协议仍保存在：

```text
/home/apulis-dev/userdata/CODEX/codex-home/attachments/8320765b-11ff-42c8-9143-4921220a6ef6/pasted-text.txt
```

该原始协议标题为：

```text
Comprehensive Codex Prompt: FFN Visual Update Components, Riesz Functionals,
Path Attribution, and Causal Validation
```

新对话必须读取原协议，不能只依赖本交接摘要。

## 3. GitHub 与代码状态

仓库：

```text
https://github.com/Xbkeepdo/gfchair
```

分支：`main`

交接时已推送的最新提交：

```text
6e92f8f Record split GPU execution plan
e2cdd33 Add TC-FVPA framework and Qwen2 formal snapshot
```

2026-08-30 22:06 CST 核验时工作树为：

```text
## main...origin/main
```

即创建本交接文档之前没有未提交改动。本文件创建后会成为新的本地改动，除非之后另行提交和推送。

`e2cdd33` 已上传当前 TC-FVPA 代码、测试、报告、schema、Qwen2 权威状态和压缩结果表。Qwen2 本地正式结果根约 438 MiB；GitHub 只上传了约 19 MiB 的可审阅快照，没有上传大型 token-map、FP32 shard 和 audit vector。不能把 GitHub 子集描述成完整本地归档。

发布前已完成的验证：

- TC-FVPA 定向测试 `42/42` PASS。
- 相关 Python `py_compile` PASS。
- `bash -n run_tc_fvpa_comprehensive.sh` PASS。
- `git diff --check` PASS。
- 敏感信息扫描没有发现凭据值。

## 4. 当前机器与旧 3090 队列的关系

本交接生成时，当前 shell 主机为：

```text
dev-75b45280-7d25-4953-b8d3-bea2c35d319b-5b2kb
```

当前 `nvidia-smi` 显示：

```text
GPU 0: NVIDIA GeForce RTX 4090, 24564 MiB, idle
GPU 1: NVIDIA GeForce RTX 4090, 24564 MiB, idle
```

因此，这个 shell 已是新的双 4090 主机。旧 3090 机器上的 PID 不会出现在这里的本地进程表中。**不得因为在 4090 上 `pgrep` 不到旧 PID，就断言旧 3090 队列已经终止。** 应通过共享日志是否持续更新、旧机器会话或平台实例状态来判断。

旧 3090 队列此前的 PID：

- Qwen3 coordinator：`558763`，此前已脱离终端，`PPID=1`。
- Qwen3 Path workers：`559546`、`559547`。
- 等待 Qwen3 后启动 InternVL 的控制器：`593002`，此前为 `PPID=1/SID=593002`。

控制器日志：

```text
outputs/tc_fvpa_qwen3_then_intern_20260830.log
```

控制器逻辑：

1. 等待 Qwen3 coordinator `558763` 退出。
2. 使用 checksum-verified loader 检查 Qwen3 L36 的 `PATH_LOG_PROBABILITY_GAUSS_LEGENDRE_K32`。
3. 只有 Qwen3 完整封存且验证通过，才启动 InternVL formal resume。
4. 如果 Qwen3 失败或未封存，记录 `BLOCKED` 并以非零状态退出，不启动 InternVL。
5. LLaVA 明确不在该控制器命令中。

关闭用户本地电脑或 VS Code 通常不会终止已脱离终端、`PPID=1` 的远端进程；但停止/释放平台实例、服务器重启、容器/Pod 回收、手工 kill、致命错误或磁盘写满会终止任务。

## 5. Qwen3 最新共享日志快照

Qwen3 正式结果根：

```text
outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET/results/
tc_fvpa_comprehensive_v1_formal_repaired_20260830
```

Local Riesz 阶段已完成，两 rank 均 PASS/复用成功。

截至 2026-08-30 22:03 CST，共享 Path 日志显示：

- rank 0：`81/100` images，`591` measured cases，`5` failures。
- rank 1：`85/100` images，`599` measured cases，`5` failures。
- 合计已看到 `10` 个失败 case。
- rank 0 日志 mtime：`22:03:19`。
- rank 1 日志 mtime：`22:03:15`。
- `manifests/run_status.json` 仍是过期的 `RUNNING`，不能单独当成当前事实。

已知失败发生在：

- rank 0：image `486991` 增加 2 个失败；`24693`、`240250`、`259342` 各增加 1 个失败。
- rank 1：image `314026` 增加 1 个失败；`321516` 增加 4 个失败。

Path runner 在扫描完成后才把完整 failure traceback 写入 shard，所以运行中日志只有累计失败数，尚不能从日志断言根因。正式协议要求 0 failures；若这些失败保留到结束，Qwen3 应为 FAIL，控制器不得启动 InternVL。

重要 resume 风险：当前 Path runner 的早期 resume 分支可能仅看到 case/token shard 文件存在就误标 PASS，而没有检查持久化 failures。若 Qwen3 最终写出含失败的 shard，**禁止直接无脑执行 `--resume`**。必须先读取 PT payload 中的 `failures`，修复 resume 门禁，并针对失败 case 诊断/重算，最后重新做完整性与 checksum 验证。

在 4090 新对话中不要修复或接管该 Qwen3 根；这属于原 3090 对话的责任范围。4090 只运行 LLaVA。

## 6. 双 4090 上的 LLaVA 正式任务

模型：

```text
llava_1_5_7b
```

正式输出根：

```text
outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/
tc_fvpa_comprehensive_v1_formal_repaired_20260830
```

正式预注册层：

```text
8,16,24,32
```

这些是按网络深度冻结的四分位代表层，不是为了得到更好结果后选的层。Path 是最耗时阶段；全层运行约会把主要计算量按层数成比例放大，所以当前 formal protocol 使用四个预注册深度探针。不得在结果出来后改变层位。

推荐在执行前先运行：

```bash
cd /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
git status --short --branch
git rev-parse HEAD
nvidia-smi
```

然后确认：

- 两张 4090 均空闲。
- `/opt/conda/private/envs/vicr/bin/python` 可用。
- LLaVA 模型权重可见。
- COCO4000、CHAIR、InsLen official target 输入完整可见。
- 目标输出根没有被另一进程写入。
- 新旧主机共享同一输出盘时，不要对 Qwen3/InternVL 根执行任何写操作。

现有综合入口格式：

```bash
/opt/conda/private/envs/vicr/bin/python scripts/run_tc_fvpa_comprehensive.py \
  --model llava_1_5_7b \
  --device cuda:0 \
  --devices cuda:0,cuda:1 \
  --num-shards 2 \
  --output-dir /home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/llava_1_5_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/tc_fvpa_comprehensive_v1_formal_repaired_20260830 \
  --formal \
  --resume
```

正式启动前必须核对 `--help`、runbook 和当前实现，确认该入口确实使用 `layers=8,16,24,32`、三个 target scalar、K=`1,4,8,16,32`、两种 quadrature，且 Path 的默认 `path_batch_size=1`。不要仅复制命令后盲跑。

如果双 4090 分成两 shard，应一张 GPU 对应一个 shard。任务必须用项目既有的 detached/setsid 方式启动，记录 PID 与日志。启动后至少核验：

```text
PID / PPID / SID
两张 GPU 的显存与利用率
rank 0 / rank 1 日志是否都推进
run_status 是否更新
输出根是否正确
是否意外出现 qwen3 或 internvl 命令
```

## 7. Path 加速工作的结论

Path 的主要瓶颈是每个 case 的 96 个唯一积分内点：K=`1/4/8/16/32`、trapezoid/Gauss-Legendre 复用端点和重复节点后，仍需反复执行下游 decoder。另一个浪费是 11 个 frozen-write LOO 原本只需要 scalar score，却构造了随后丢弃的梯度。

已经实现：

- exact-suffix Path 节点 microbatch。
- LOO score-only forward。
- 三 scalar 梯度统一入口。
- OOM 时释放失败图并自动回退 batch 1。
- FP32/FP64 可使用 batched VJP。
- 原生 BF16/FP16 保持三 scalar 串行 VJP，避免改变低精度累加顺序。

四模型统一安全 A/B（2 images、双 shard、四层、三 scalar、完整 K 网格、正式 `path_batch_size=1`）：

| 模型 | 旧版合计 | 优化版合计 | 加速 | 耗时下降 |
|---|---:|---:|---:|---:|
| Qwen3 | 313.393 s | 293.604 s | 1.067x | 6.31% |
| Qwen2.5 | 288.991 s | 274.670 s | 1.052x | 4.96% |
| InternVL | 422.307 s | 405.685 s | 1.041x | 3.94% |
| LLaVA | 579.272 s | 563.347 s | 1.028x | 2.75% |

四模型均为 `12/12 MEASURED`、0 failures；每模型 528/528 方法 token maps 与旧结果逐元素一致，frozen-write LOO observed effects 也完全一致。

Qwen3 曾测试显式 `path_batch_size=2`，吞吐提高约 10.26%，但 BF16 GEMM 的 M 维变化导致 Path 图出现小数值差异：最大绝对差约 `0.0040283`、最大相对差约 `0.12177`、最低 cosine `0.994753`。虽然正式 GL-K32 的最低 Spearman 为 `0.998967` 且 Top-1 一致，个别 Top-4 overlap 只有 `0.75`。因此四模型正式协议冻结为 batch 1，batch 2 不用于本轮正式结果。

换双 4090 相比双 3090，曾给出的保守估计是总任务约 `1.3--1.6x` 加速；两者单卡都只有约 24 GB 显存，所以 4090 不意味着可以安全提高 Path batch。单张 4090 通常不如双 3090 的两 shard 总吞吐。

## 8. Qwen2.5 已完成正式结果

Qwen2 正式结果根：

```text
outputs/qwen2_5_vl_7b/COCO4000-INSLEN-OFFICIAL-TARGET/results/
tc_fvpa_comprehensive_v1_formal_repaired_20260829
```

完成情况：

- local：500 images，3496 target-layer cases，0 failures。
- path：200 images，1316 cases，0 failures。
- 真 FP32 intervention：216,384 rows，0 failures。
- frozen-write：43,428 rows。
- Shapley：300 scalar rows，128 permutations。
- analyze、counterfactuals、reports 已完成并封存。
- 正式 hallucination detector 与 VQA 没有运行。

数值结果：

- Path K32 相对 K1 的 median absolute completeness error reduction：
  - log probability：18.79%
  - margin：13.68%
  - logit：10.73%
- 真 FP32 aggregate，`eta=0.025`：
  - sign agreement：log probability 97.56%、logit 99.35%、margin 99.35%
  - median relative error：log probability 1.06%、logit 0.64%、margin 0.99%
- Qwen2 spatial：
  - WRITE：Top-1 0.54330，AUPRC 0.46846，BBox 0.39058
  - JFFN：Top-1 0.52098，AUPRC 0.46356，BBox 0.38731
  - signed Q：Top-1 0.51295，AUPRC 0.47000，BBox 0.40933
  - Path log-probability GL-K32：Top-1 0.50687，AUPRC 0.42877，BBox 0.38625

这些 spatial 数字尚未做 Qwen2 正式 bootstrap，不能宣称显著优于其他方法。

## 9. 之前 JFFN 第二轮实验已经支持的结论

主要报告：

```text
jffn_second_round_incremental_validation_report.md
```

当前结论是 Outcome C：JFFN 没有在空间指标上稳定优于 WRITE；S 也没有提供稳定的幻觉检测增量。

- LLaVA `[I,S]-I` AUROC 增量：`+0.02317`，CI `[-0.00300,+0.04917]`。
- InternVL `[I,S]-I` AUROC 增量：`+0.00150`，CI `[-0.00043,+0.00330]`。

置信区间跨 0，不能宣称稳定显著提升。必须保留这个负结果，不能只报告有利方向的点估计。

## 10. 用户问过的问题与已给出的答复脉络

按对话顺序，用户主要问过：

1. “什么意思？任务未完成吗”——回答应明确区分已实现代码、已运行 smoke、已完成 formal 与仍 BLOCKED/NOT RUN 的实验；当时四模型并未全部完成。
2. “为什么会这么久？”——主要因为 Path 对每图、每 target、每层、三个 scalar、两种 quadrature 和多个 K 执行大量下游模型计算，且正式协议要保存逐 token/case 证据并做门禁。
3. “可以，你修复吧”——随后修复了 Qwen 真 FP32 microbatch、resume 门禁等问题，并恢复正式队列。
4. “qwen 怎么没修复？修复后就执行正式实验吧”——完成 Qwen 低层/长前缀相关路由与 OOM 修复，并启动正式队列；不允许把失败 shard 静默当 PASS。
5. “4 个模型的所有实验都跑完了吗”——没有；Qwen2 后来完成，Qwen3 正在 Path，LLaVA/InternVL 尚未完成，VQA/fixed-QK/pixel 等部分仍 NOT RUN/BLOCKED。
6. “现在进度如何？”——持续通过 PID、rank 日志、case 数、failure 数和 GPU 状态汇报，而不是只看一个状态 JSON。
7. “为什么会保存这么多行结果，现在的显卡显存有多少？”——大量行来自 `方向 × eta × 正负号 × scalar × layer × target × image` 的笛卡尔积；这是逐干预证据，不等于重复保存无用结果。3090 正式运行时常用约 20--22 GB/卡。
8. “可以切成小批次，然后跑完 4 个模型估计要多久？”——实现了 FP32 query microbatch；当时从剩余阶段估算双 3090 约 30--36 小时，但这是运行阶段估算，不是保证全部科学协议 PASS。
9. “为什么只跑 4 层？”——四层是实验前冻结的四分位深度探针，控制 Path 计算量并防止结果后挑层；不是因为只实现了四层。
10. “最耗费时间的是哪个实验？”——Path attribution 是主要瓶颈，LLaVA Path 在旧校准中尤其慢。
11. “Path 能优化计算来加速吗？”——可以，已做 exact-suffix microbatch、LOO score-only 和复用；正式 batch 1 的收益约 2.75%--6.31%，不会夸大成 batch 2 的 10.26%。
12. “其他模型加速多少？”——见上面的四模型 A/B 表。
13. “之前跑的实验可以总结出实验结果吗？”——可以，但只能总结已完成的 Qwen2 formal 与旧 JFFN 结果，不能把 Qwen3 partial 或 smoke 升格为完整结论。
14. “10,000 bootstrap 是什么意思？如果先不跑 VQA benchmark，整个实验大概要多久？”——bootstrap 是对评估单位有放回重采样 10,000 次，用来估计指标差异的置信区间；不是重新做 10,000 次模型推理。跳过 VQA 后，当时双 3090 当前已实现阶段估计约 30--36 小时，实际取决于 Path、共享盘冷启动和失败重跑。
15. “先把当前代码和实验结果上传到 GitHub”——已完成 `e2cdd33` 和后续 `6e92f8f` 推送；大型本地张量没有上传。
16. “换 4090 会快些吗？”——双 4090 相比双 3090保守估计约 1.3--1.6x；同为 24 GB，不能因此扩大 formal batch。单 4090 不一定胜过双 3090。
17. “Qwen3 还剩多久？”——当时基于 Path 进度估计仍需约 1.5--2 小时扫描，然后还取决于失败门禁；现在应以第 5 节最新日志为准，不沿用旧 ETA。
18. “当前对话跑完 Qwen3 后跑 InternVL，新开对话在 4090 继续 LLaVA”——已按此拆分队列并写入 `docs/CURRENT_TASK.md`。
19. “关电脑或关闭 VS Code，进程会终止吗？”——已脱离终端的远端进程不会因本地客户端关闭而自动终止，但释放服务器实例会终止。
20. “把当前对话发送给新分支 / 为什么看不到 Branch in new chat”——Codex/IDE 当前界面没有可由模型调用的创建分支工具，所以改用本 Markdown 文档交接。

## 11. 新对话的第一条建议指令

用户可以在新双 4090 对话直接发送：

```text
请完整读取：
/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/docs/CODEX_CONVERSATION_HANDOFF_20260830.md
以及其中指向的 AGENTS.md、CURRENT_TASK.md、TC_FVPA runbook、当前结果摘要和原始综合实验协议。

你现在只负责双 RTX 4090 上的 llava_1_5_7b 正式 TC-FVPA 实验。先做只读 preflight，核对 Git、环境、两张 GPU、模型与数据、输出根和现有进程；确认不会碰旧 3090 的 Qwen3/InternVL 输出后，按冻结协议以两个 shard、Path batch_size=1、detached 方式启动或安全 resume LLaVA。启动后汇报 PID、日志、显存、阶段、进度与 ETA。遇到失败必须保留并停止门禁，不得伪造 PASS。暂不运行 VQA benchmark。
```

## 12. 交接原则

- 先核验，再声明进度。
- partial、smoke、BLOCKED、FAIL 与 formal PASS 必须严格区分。
- 不覆盖旧结果，不删除失败证据，不修改预注册样本/层/阈值来改善结果。
- 不隐藏负贡献或负结果。
- 不把 BF16/FP16 数值强转 float32 冒充真 FP32 验证。
- 不因运行昂贵而静默缩小范围。
- 不在仓库中写入 token、密码或学校账号信息。
- 未经用户新授权，不提交或推送本交接文档及后续改动。
