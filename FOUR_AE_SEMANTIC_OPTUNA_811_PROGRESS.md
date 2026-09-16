# 四模型 AE / 语义注意力 Optuna 单层 MLP

## 协议

- 模型：Qwen2.5-VL-7B、LLaVA-1.5-7B、Qwen3-VL-8B、InternVL2.5-8B。复用现有 ENDAC exact 811 controlled mentions、3200/400/400 图片划分与已提取特征，不重新运行 VLM。
- 每模型五组：`ae_logs=[AE_V,log1p(S_E)]`，以及 raw/norm × Top16/32 的 `[semantic_attention,log1p(S_E)]`。`semantic_attention` 是语义概率 Top-K 区域内逐层 `attention × 目标词概率` 的和。四模型的 AE 与语义缓存已核对 sample ID、标签、行数完全相同。
- Torch 单隐藏层，全部 trial 强制训练集 z-score、无 BatchNorm。Optuna 5.0.0 TPE 每组 24 trial（包含一个已知固定 128 配置），seed43 验证 AUROC 优先、HALL-AUPR 次优选前 3；补 seeds44/45，再按三种子验证均值冻结每组配置及模型内 champion。四模型选择均冻结后才评估 test。
- 每模型预计 150 次拟合，四模型共 600 次；四张 GPU 各自运行一个模型、单进程写入，Optuna 每组独立 SQLite study 供中断续跑。不做 train+val 重训。固定测试图片在此前实验中已查看，结果只作探索性比较。
- 入口：`scripts/optuna_four_ae_semantic_single_mlp_811.py`。输出：`outputs/coco4000_512_endac_four_ae_semantic_optuna_811/`。

## 进度

- 2026-09-16 12:25 UTC：`vicr` 安装 Optuna 5.0.0；本地 GPU0/1 分别训练 Qwen2.5/LLaVA，远端 GPU0/1 分别训练 Qwen3/InternVL。四进程都已产生 Optuna trial；动态进度见各模型 `progress.json`，日志在 `outputs/optuna_four_<model>_train.log`。测试尚未运行。
- 12:41 UTC：四模型各 150/150 次拟合全部完成，四个 validation selection 均先于 test 冻结；20 组×三种子测试完成。验证选中：Qwen2.5 AE 87.18/54.77，LLaVA 语义 norm Top32 92.41/76.99，Qwen3 AE 89.85/71.75，InternVL 语义 norm Top32 88.23/62.41（测试 AUROC/HALL-AUPR，%）。完整选中参数与解读见 `docs/FOUR_AE_SEMANTIC_OPTUNA_811_RESULTS.md`，20 组表见输出 `summary.md`。
- 60/60 最终头保存预测与指标重算完全一致；train-only scaler 和 CPU 重载通过，概率最大差 `3.58e-7`，近并列排序引起的 CPU 指标最大差 `7.56e-6`，核验状态 `pass`。四个训练进程已结束，本地与远端 GPU 释放。
