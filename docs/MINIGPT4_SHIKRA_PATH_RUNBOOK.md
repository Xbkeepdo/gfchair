# MiniGPT-4 / Shikra COCO4000 路径积分实验

本轮主方法为 local-FP32 K4 Gauss–Legendre 向量雅可比路径积分。保留
consistency-v2 的 16 组、AE+log1p(S)、AE+R_cos+S、Q/B_Q 八组、
Q-softmax JS 六组、endpoint cosine/内积/有符号标量投影 JS 及 J_T/J_E
分解；共 36 个命名组合，相同输入矩阵共用训练。没有新增 C 提取或搜索超参数。

Baseline 为 SVAR、ProjectAway、MetaToken 的 native_paper/shared_torch_mlp，
以及 ADS+CGC MLP。ProjectAway 只做检测。正式 cohort 为现有 seed42
COCO4000、3200/800 图像划分，所有方法使用相同标注目标和 seeds43/44/45。

## 入口

在项目根目录、目标服务器运行：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NLTK_DATA=/home/apulis-dev/userdata/nltk_data
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN=/opt/conda/private/envs/vicr/bin/python
$PYTHON_BIN scripts/run_minigpt4_shikra_path.py --model minigpt4_7b --device cuda:0 --smoke
$PYTHON_BIN scripts/run_minigpt4_shikra_path.py --model shikra_7b --device cuda:1 --smoke
# 对应模型 smoke 数值门槛通过后：
$PYTHON_BIN scripts/run_minigpt4_shikra_path.py --model minigpt4_7b --device cuda:0
$PYTHON_BIN scripts/run_minigpt4_shikra_path.py --model shikra_7b --device cuda:1
```

`--stage generate|baselines|extract|train` 可单独恢复阶段；默认 pipeline。
本轮输出为 `outputs/<model>/COCO4000-JACOBIAN-PATH`，smoke 独立放在
`smoke8/`。检查实际进度和阶段状态，不以启动当成完成。

## 原始注意力与空间适配

`path/shards/image_*.pt` 按唯一目标存储 `raw_attention_mean:[L,M]`，FP32、
softmax 后跨头均值，不在视觉子集上重新归一化，不保存逐头矩阵或 PNG。
同时保存目标 token、响应位置、视觉位置、图片 SHA、标注 mention 表和各层路径信号。

MiniGPT-4 原始支持是 32 个查询，不具有网格。另存
`mapped_attention_mean:[L,256]` 和 `mapped_grid:[16,16]`：最后一个
Q-Former 跨注意力层、头均值、去除 CLS、逐查询归一化后映射。ADS 使用此图，
CGC 和其他方法仍在 decoder 的查询支持上计算。此映射是空间适配，不是精确因果归因。
Shikra 遵循官方白色 Expand2square + CLIP 预处理，在 16×16 原生 patch 网格计算。

两模型使用 Vicuna/Llama tokenizer 的 InsLen 默认 surface 第一子词/首次出现
规则；标注兼容说明明确这是 tokenizer 适配，不声称上游发布了这两个模型的 wrapper。

## 数值与训练验收

smoke 固定 8 图，对全部有效目标/层比较 K4/K64，使用既有门槛：闭合误差
p90≤1e-3、max≤1e-2，P_FFN Spearman 中位数≥.99、JS≤.005、Top32 overlap≥.95，
S/N_vec 相对差 p90≤.01、κ_vec 绝对差 p90≤.01。任何失败均保留诊断，不能使用旧四模型豁免。

训练冻结原 MLP/三 seed/双阈值，尺度仅从训练图片 mentions 拟合。逐头保存并
重载 checkpoint 复核训练和测试概率。正式 baseline 与主方法共享图片划分。
`training_smoke/` 只验证接口：一 seed、一 epoch、小 batch；若原 8 图测试集只有
一个类别，先在独立技术 smoke 目录尝试图像划分；若幻觉仅出现在一张图片，增加固定 seed43 的8张技术样本，保存原/新 split，
其结果不得作为正式检测性能。正式图片划分不变。

## 当前开发验证

- 新增 tiny-Llama 测试覆盖因果预测行、未来 token 屏蔽、批量/单前缀一致性、
  FP32 未重归一化均值注意力；映射测试覆盖质量守恒、查询排列不变性和非法输入。
- 初始 EVA meta 初始化失败已通过显式 CPU stochastic-depth 标量修正；checkpoint
  的第40块/classifier 是官方39块特征模型的预期额外键，显式过滤且仍严格校验其余键。
- Q-Former 官方 encoder 的非 cross 层返回 KV tuple，取最后一个真实 tensor
  cross-attention；没有将 KV cache 当作空间权重。
- 初次 InsLen 回归测试未设置 NLTK_DATA，4 项因找不到 punkt 失败；使用共享
  NLTK 路径重新验证，不安装重复数据。

## 2026-09-10 主表统计口径更新

主表统一采用seeds43/44/45各自指标的均值±总体标准差（ddof=0），不使用ensemble。F1按各seed的训练REAL-F1阈值评价。历史ensemble仅保留在原始结果JSON中。总表已加入温度.2的Q-JS融合组合，无需重训。
