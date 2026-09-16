# Prompt 组内 κ_P：REAL / HALL 逐层曲线

复用固定811检测所用四模型4000图缓存（测试集400图），不重跑VLM、不训练检测器。横轴decoder layer；蓝色REAL，红色HALL；实线为mention加权均值，浅色带为各类mention的IQR；下排为HALL−REAL均值。缓存中κ_P均在(0,1]且无零值。

| 模型 | all HALL−REAL层均值 | all正差层数 | test HALL−REAL层均值 | test正差层数 |
|---|---:|---:|---:|---:|
| Qwen2.5-VL | +0.0591 | 26/28 | +0.0632 | 26/28 |
| LLaVA-1.5 | +0.0706 | 32/32 | +0.0669 | 32/32 |
| Qwen3-VL | +0.0446 | 33/36 | +0.0443 | 34/36 |
| InternVL2.5 | +0.0708 | 31/32 | +0.0763 | 31/32 |

κ_P越大，prompt来源内的token响应方向越一致、相互抵消越少。图显示HALL总体更高，但这是相关性描述，不证明prompt导致幻觉；两类mention数量、图片/文本长度等可能混杂。test集此前已查看，此图仅作探索性展示。

[全部mention曲线](prompt_kappa_all.png) · [811测试曲线](prompt_kappa_test.png) · [逐层数据](curves.csv) · [样本数与汇总](summary.csv)
