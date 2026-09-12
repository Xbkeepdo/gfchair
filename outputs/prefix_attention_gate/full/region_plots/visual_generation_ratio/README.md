# 视觉注意力 / 生成文本注意力

先逐目标、逐层求比值，再按HALL/REAL统计。使用原全部mentions，分别提供all/train/test。分母为0的比值未定义，仅在对应层统计中排除并在CSV计数，不加epsilon、不截断极值。
实线均值、虚线中位数、阴影IQR（非置信区间）。纵轴为对数刻度；统计的是原始比值，不是log比值。各面板纵轴独立。合法的零比值保留在CSV及统计中，但不在对数图显示。

- all：[raw attention](all_raw_attention.png) · [attention×gate](all_attention_x_gate.png)
- train：[raw attention](train_raw_attention.png) · [attention×gate](train_attention_x_gate.png)
- test：[raw attention](test_raw_attention.png) · [attention×gate](test_attention_x_gate.png)

[逐层数据与无效计数](curves.csv) · [协议](protocol.json)
