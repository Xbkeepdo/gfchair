# MiniGPT-4 / Shikra ENDAC exact + Top-K 特征进度

## 特征提取完成状态（2026-09-16 19:03）

- 两模型特征提取均完成并合并：MiniGPT-4 3929 张有 controlled mention 的图片、9140 条样本；Shikra 3976 张、13067 条样本。完整的 4000 张 generation/ENDAC exact 标注及共享 3200/400/400 split 均保留。
- 合并产物：`outputs/coco4000_512_endac_811/<model>/features.pkl`、`feature_summary.json`。每条样本包含 SVAR、MetaToken、Visual-only、All-attention、raw/norm Top16/32 六项及可复原全视觉求和的四种 `[L,N_V]` 矩阵；另存原始 S_E。
- 全量核对 PASS：两模型各自样本数等于 controlled 标注数；sample ID 唯一、合并顺序一致，逐条标签、首 token ID、split 与源数据相符；全部特征及矩阵数值有限，形状分别为 MiniGPT-4 `[32,32]`、Shikra `[32,256]`。四张提取 GPU 均已释放。

## 811 训练（2026-09-16 19:11 UTC 启动）

- 本地 GPU0/1 分别训练 MiniGPT-4/Shikra：原生 SVAR MLP、MetaToken LR/GB，四类统一三层 MLP，以及四类方法的冻结 24 候选单层 MLP 搜索；seeds 43/44/45。结果写回 `outputs/coco4000_512_endac_811/<model>/results/`，日志为同目录 `train_full.log`。
- 远端 GPU0/1 分别训练 MiniGPT-4/Shikra 语义 Top-K：raw/norm×Top16/32，每组六个单项及六项拼接，各自 standalone、`+log1p(S)`、`+S`；固定单隐藏 128、train-only 标准化、三种子，验证选择冻结后才评估 test。独立输出 `outputs/coco4000_512_endac_prefix_semantic_attention_topk_detection_811/<model>/`。
- 11:22 UTC：两模型四方法的原生与共享 MLP 训练已完成，见各自 `results/summary.md`；语义 Top-K 验证候选仍在远端两卡训练，最终选型、测试和验证尚未完成。
- 11:34 UTC：两模型全部 504 个语义 Top-K 训练头完成，先按三种子 validation AUROC、再 HALL-AUPR 冻结每模型设置，之后才评估 test。MiniGPT-4 选 `norm_top16/all_six+S`，test AUROC/HALL-AUPR 为 `93.67±0.04/78.16±0.08%`；Shikra 选 `raw_top32/all_six+S`，为 `87.79±0.24/69.91±0.37%`。完整 raw/norm × Top16/32 × 六单项、六项拼接及各自 `+logS`、`+S` 见 `outputs/coco4000_512_endac_prefix_semantic_attention_topk_detection_811/<model>/summary.md`。
- 原四方法三种子结果：MiniGPT-4 的 SVAR 单层 `94.26/80.20%`，Shikra 的原生 SVAR `88.23/68.96%`、SVAR 单层 `88.08/69.65%`（AUROC/HALL-AUPR）；其余方法见各自 `outputs/coco4000_512_endac_811/<model>/results/summary.md`。语义结果没有在两个模型上统一胜过 SVAR。
- 504/504 个语义头的 checkpoint、预测、指标重算通过，`validation.json` 状态 `pass`，最大 CPU 重载概率差 `2.98e-7`；原方法每模型 33/33 个 `result.pt` 及预测文件齐备。四个训练/测试 GPU 进程均已结束。

- 目标模型：`minigpt4_7b`、`shikra_7b`；复用各自 `COCO4000-JACOBIAN-PATH/generations.json`，不重新生成。
- 协议：Qwen3 ENDAC exact response offsets、first canonical mention、首个目标 token；保留全部 mention 与 SVAR official 样本；共享既定 3200/400/400 图片划分。
- 提取：每模型每图一次前向，同步保存 SVAR、MetaToken、Visual-only K4、All-attention K32，以及 raw/norm Top16/32 六项特征；另存 `[L,N_V]` attention、目标词概率和 cosine 矩阵，之后可重构全视觉求和。
- MiniGPT-4 的 `N_V=32` 为 Q-Former query（Top32 等于全部 query）；Shikra 的 `N_V=256` 为图像 patch。两者不能把 query 当成同一空间分辨率的 patch 比较。
- 2026-09-16：两模型各 4000 张 ENDAC exact 标注完成，controlled 样本 MiniGPT-4 9140、Shikra 13067；图片划分 3200/400/400，无 token 对齐失败。
- 两模型各 1 图烟测通过：MiniGPT-4 保存 `[32,32]`、Shikra 保存 `[32,256]` 的四种语义注意力矩阵，raw/norm Top16/32 六项特征及四类原特征均有限值。前缀首 token 因果位置单测通过。
- 全量提取已在持久 screen 会话 `endac811_minigpt4`（GPU0）和 `endac811_shikra`（GPU1）启动；日志和逐图续跑进度分别在 `outputs/coco4000_512_endac_811/<model>/extract_full.log` 与 `feature_progress.json`。训练尚未开始。此前一次普通 nohup 后台尝试被终端回收，未写入全量特征。
- 16:52 UTC 快照：MiniGPT-4 `104/3929`、Shikra `57/3976` 张有 controlled mention 的图片完成，两个 screen 进程仍在运行；动态进度以各模型 `feature_progress.json` 为准。烟测行与原标签的 sample ID、首 token ID、标签逐一匹配。
- 18:20 UTC：MiniGPT-4 全量完成，3929 张、9140 条 controlled 样本，已合并到 `minigpt4_7b/features.pkl`；其中 raw/norm Top16/32 六项和四种 `[32,32]` 矩阵均已核对。Shikra 继续提取。
- 18:33 UTC：Shikra 单卡进程在 `2224/3976` 张处暂停；已有 `features.part0.pkl` 的 7199 行、2224 个 image ID 完整可读。剩余任务改为四份续跑：本地 GPU1/0 分别运行 part0/1，远端两张 4090 运行 part2/3。各进程写自己的 `features.part<i>.pkl`、`feature_progress.part<i>.json` 和 `extract_part<i>.log`；旧 `feature_progress.json` 停留在切换点，不再代表总进度。四份结束后需执行 `--merge-only` 并核对 3976 张、13067 条样本。
- 18:38 UTC：四个 part 均已开始新图片提取，进度合计约 `2555/3976`（64%）；本地/远端各两张 4090 均在计算，远端两卡约 15 GB 显存、79%/81% 利用率。远端共享存储首次加载权重用了约五分钟，之后正常写盘。
