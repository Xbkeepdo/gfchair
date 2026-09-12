# 完整因果前缀 attention 与 attention × gate

六模型原 COCO4000、原描述、原 InsLen 目标；本页记录原始提取流程。后续360头分组检测已完成，见[分组检测说明](PREFIX_ATTENTION_GROUP_DETECTION.md)。

## 定义

对目标响应 token x_t，q 是预测该 token 的 decoder 位置。每层保存实际输入
位置 0..q（包含 q）的向量，长度为 prompt_length+t。支持含完整模板/提示、视觉
嵌入和已生成文本，不含目标 x_t 或未来位置。没有独立 BOS 的模板从实际位置0开始，
不人为插入 BOS。多目标共享一次最长目标前缀前向，逐目标裁剪，实际检查未来注意力为零。

- raw_attention：原生 softmax 后的 attention，跨头 FP32 均值，不再归一化。
- attention_x_gate：上述 raw_attention 与 gate 的逐位置原值乘积，不归一化。
- gate：hpre 原始隐藏状态直接乘目标 unembedding 行，无 final norm 或 bias；
  sigmoid((logit-median)/(1.4826*MAD+epsilon))，epsilon沿用配置（通常1e-6）。
- 本轮默认 full：每个目标、每层独立在其完整因果前缀计算 median/MAD，调用项目原
  hpre_raw_logit_gauss 公式。median/MAD遵循torch.median的偶数长度约定。
  --gate-reference visual 可单独使用视觉统计量扩展到全前缀，两者输出目录隔离。

## 产物

`outputs/prefix_attention_gate/full/<model>/shards/image_*.pt`：

- positions：每个唯一目标一条；raw_attention、attention_x_gate 为 FP32 [L,N]，
  N=该目标prefix_length。另存target_token_id、response_index、prediction_position、
  gate_median/gate_mad [L] 和 target_key。
- layout：最长输入的token_ids/token_pieces/position_types/is_special、prompt_length、
  visual_range/visual_grid、实际BOS标记。对一个目标读取这些列表的前N项即可。
  视觉位置token_id用None表示连续视觉嵌入，token_pieces使用明确的visual占位标记。
- sample_table：原始全部mentions及标签，对应positions里的target_key；无目标图片也保存完成分片。
- 原生attention精度和raw_mass_tolerance记在layout。原生FP16/BF16概率已发生舍入，
  raw和1之间容差采用0.5*finfo(native_dtype).eps+1e-5；不调整保存的raw数值。

```python
import torch
record = torch.load(path, map_location="cpu", weights_only=False)
target = record["positions"][0]
a = target["raw_attention"]
ag = target["attention_x_gate"]
ids = record["layout"]["token_ids"][:target["prefix_length"]]
```

## 执行与验收

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/opt/conda/private/envs/vicr/bin/python scripts/extract_prefix_attention_gate.py \
  --model minigpt4_7b --device cuda:0 --smoke
# 六模型各smoke完成，独立核对保存文件：
/opt/conda/private/envs/vicr/bin/python scripts/audit_prefix_attention_gate.py --smoke
# 正式运行去掉--smoke；每模型使用各自固定GPU。
# 查看进度并在所有模型完成后自动进行独立文件核验：
/opt/conda/private/envs/vicr/bin/python scripts/audit_prefix_attention_gate.py --watch
```

每模型manifest记录配置、输入文件摘要及实现摘要；断点恢复检查分片归属，拒绝混协议。
status.json只在4000图写完后标记COMPLETE；全局audit.json另记录独立检查的结果，
包括原目标/mentions对齐、响应prefix IDs、位置长度、层数、FP32格式、raw质量与乘积范围。
未来位置为零的检查在前向提取阶段进行。smoke8_initial_fixed_tolerance保留最初固定容差版本，
正式版本保存原生精度容差元数据。
