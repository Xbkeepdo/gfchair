# GFChair 四模型 ENDAC-811 实验进度

最后更新：2026-09-15 18:56 UTC

## 当前状态

- 状态：四模型特征、训练、预测及总汇总全部完成并校验通过。
- generation：全部复用既有结果，没有重新生成。
- 标注：四模型 exact ENDAC 标注与 token 定位已完成，0 个对齐失败。
- 划分：共享 image-level `3200 train / 400 validation / 400 test`，held-out 划分种子 `20260912`。
- 保存：沿用 token-detector 格式，每图结果追加到 `features.partN.pkl`，结束后合并为 `features.pkl`。
- 并行：本地 2 张 RTX 4090、远端 2 张 RTX 4090；空闲卡通过独立 part 分担剩余提取。

## 四模型任务

| 机器 / GPU | 模型 | controlled samples | 正式提取进度 | 状态 |
|---|---|---:|---:|---|
| 远端 GPU 0 | qwen2_5_vl_7b | 7,924 | 3,679 / 3,679 images（100%） | training complete |
| 本地 GPU 0 | llava_1_5_7b | 12,686 | 3,968 / 3,968 images（100%） | training complete |
| 本地 GPU 1 | qwen3_vl_8b | 11,751 | 3,948 / 3,948 images（100%） | training complete |
| 远端 GPU 1 | internvl_2_5_8b | 10,781 | 3,939 / 3,939 images（100%） | training complete |

进度分母是“至少含一个 controlled mention 的图片数”，不是全部 4,000 张；无目标图片不产生特征行。

这份 Markdown 是阶段性人工快照，不会由四个提取进程并发改写；实时计数以各模型目录下的
`feature_progress.json` 或 `feature_progress.partN.json` 为准。到达重要里程碑或进入训练阶段时会同步刷新本文档。

Qwen2.5 合并产物已验证：`features.pkl` 共 7,924 行，sample ID 无重复、顺序正确，
SVAR/MetaToken/Visual-only/All-attention 特征均无 NaN/Inf，维度与 smoke 一致。

InternVL 合并产物也已验证：`features.pkl` 共 10,781 行，sample ID 无重复、顺序正确，
全部特征无 NaN/Inf，维度为 `448 / 42 / 64 / 64 / 128`。

LLaVA 两个分片已完成并合并为 12,686 行；Qwen3 合并为 11,751 行。两者均已验证
sample ID 唯一、顺序正确，全部特征无 NaN/Inf，维度与 smoke 一致。

## 训练进度

- Qwen2.5：native 9/9、固定三层 MLP 12/12、四方法单层搜索及最终三种子均已完成，汇总已生成。
- Qwen2.5 首次进入单层搜索时因远端缺少 `CUBLAS_WORKSPACE_CONFIG` 退出；补入 `:4096:8` 后已从现有结果续跑，未丢失结果。
- InternVL：native 9/9、固定三层 MLP 12/12、四方法单层搜索及最终三种子均已完成，汇总已生成。
- LLaVA：native 9/9、固定三层 MLP 12/12、四方法单层搜索及最终三种子均已完成。
- Qwen3：native 9/9、固定三层 MLP 12/12、四方法单层搜索及最终三种子均已完成。
- 已完成结果均保存对应 `result.pt` 与 `predictions.npz`。

最新调度：Qwen2.5 训练完成后，空出的远端 GPU 0 分担了 LLaVA shard 1；两个分片
各自完成 1,984 / 1,984 后已单进程合并。当前本地 GPU 0/1 分别训练 LLaVA/Qwen3。

## 标注产物

| 模型 | all mentions | controlled | SVAR official |
|---|---:|---:|---:|
| qwen2_5_vl_7b | 15,686 | 7,924 | 8,382 |
| llava_1_5_7b | 25,504 | 12,686 | 13,208 |
| qwen3_vl_8b | 32,631 | 11,751 | 12,982 |
| internvl_2_5_8b | 40,086 | 10,781 | 11,867 |

Qwen3 新产出的 `labeling.json` 已与协议来源
`token-detector/outputs/qwen3_vl_8b/COCO4000-512-ENDAC/labeling.json`
逐字段比较，4,000 条记录完全相同。

## Smoke

四种模型架构均完成 1 图真实模型 smoke，特征均为有限值，pickle 追加、合并和按 image ID 续跑正常。

| 模型 | image | samples | SVAR | MetaToken | Visual-only / All-V / All-VP+G |
|---|---:|---:|---:|---:|---:|
| qwen2_5_vl_7b | 283 | 2 | 364 | 38 | 56 / 56 / 112 |
| llava_1_5_7b | 283 | 4 | 448 | 42 | 64 / 64 / 128 |
| qwen3_vl_8b | 283 | 5 | 480 | 42 | 72 / 72 / 144 |
| internvl_2_5_8b | 283 | 4 | 448 | 42 | 64 / 64 / 128 |

## 最终检查

1. 每模型 33 份 `predictions.npz` 均与对应特征行数一致，概率无 NaN/Inf。
2. 每模型 native 9、固定三层 12、单层最终 12 个结果齐全。
3. 四模型总汇总已生成，共 44 行方法结果。

输出根目录：`outputs/coco4000_512_endac_811/`
