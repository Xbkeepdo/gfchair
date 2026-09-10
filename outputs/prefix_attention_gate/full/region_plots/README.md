# 完整前缀四区域注意力曲线

区域互斥：真实起点BOS、视觉、其余prompt、目标之前的生成文本。无BOS模型不伪造BOS；其起点模板token计入prompt。各区域直接求和，不除以token数、不再归一化。全部mentions等权，空生成文本区域记0；实线均值、阴影IQR，非置信区间。

[测试集 raw attention](test_raw_attention.png) · [测试集 attention×gate](test_attention_x_gate.png)
[训练集 raw attention](train_raw_attention.png) · [训练集 attention×gate](train_attention_x_gate.png)

- minigpt4_7b：[测试集](minigpt4_7b/test.png) · [训练集](minigpt4_7b/train.png) · [逐层数据](minigpt4_7b/curves.csv)
- shikra_7b：[测试集](shikra_7b/test.png) · [训练集](shikra_7b/train.png) · [逐层数据](shikra_7b/curves.csv)
- qwen2_5_vl_7b：[测试集](qwen2_5_vl_7b/test.png) · [训练集](qwen2_5_vl_7b/train.png) · [逐层数据](qwen2_5_vl_7b/curves.csv)
- llava_1_5_7b：[测试集](llava_1_5_7b/test.png) · [训练集](llava_1_5_7b/train.png) · [逐层数据](llava_1_5_7b/curves.csv)
- qwen3_vl_8b：[测试集](qwen3_vl_8b/test.png) · [训练集](qwen3_vl_8b/train.png) · [逐层数据](qwen3_vl_8b/curves.csv)
- internvl_2_5_8b：[测试集](internvl_2_5_8b/test.png) · [训练集](internvl_2_5_8b/train.png) · [逐层数据](internvl_2_5_8b/curves.csv)
