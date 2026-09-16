# All-attention WRITE / SS / gain：固定协议（2026-09-15）

仅复用四模型各4000图的 `outputs/ffn_all_source_paths_v1` 正式缓存，真实Norm+FFN、局部FP32、GL K32，路径z−A_all→z。不重新运行VLM、不使用B1/B2逐组净响应替代token响应。

1. P/V/G/ALL四区域分别计算I=Σ||a_m||、S=Σ||e_m||、G=S/I。ALL先对全部attention token求和，不包含residual/bias。曲线all/train/validation/test，mentions等权；均值、中位数和跨mentions IQR分别保存。
2. 每target-layer-region内先计算lambda_m=||e_m||/||a_m||，保存Q25/median/Q75/IQR、CV、>1比例；再跨mentions分REAL/HALL汇总。区域内IQR/CV用于来源方向离散度，跨mentions IQR不能代替它。仅a=0时lambda未定义；正的微小a不按阈值排除，空区域I=S=0、G和lambda统计NaN。
3. 每层Spearman(I,S)、log-Pearson；使用共同正值样本检查logS=logI+logG并分解HALL−REAL平均log差。另做同图配对，排除标签冲突target，图片等权。相关不代表因果；小的总体类别差不等于没有方向依赖增益。
4. 固定现有811图片划分3200/400/400（seed20260912），保留全部mentions。检测输入两套：PVG按来源拼接、ALL按token合并；分别I-only、S-only、I+S，四模型×6组×3seed=72头。
5. 输入逐层log1p强度，然后train-only逐列Z-score。统一单隐藏128/ReLU/dropout.3，无BN；Adam lr.001/wd1e-5/batch128，最多150epoch，最低validation BCE checkpoint，早停20、LR plateau patience6/factor.5/min1e-6。所有模型/特征同一配置，不搜索、不加入AE/gain/缺失指示，不train+val重训。
6. seeds43/44/45先分别计算AUROC、HALL-AUPR，再报告均值±总体std。配对比较S−I、(I+S)−I、(I+S)−S；测试图片簇bootstrap2000次，对三seed指标差取均值，固定seed20260915，名义95%区间未多重校正。已有测试集已查看，属于探索性对照。
7. 核验4000shards、协议签名/K、mentions/target/层/811划分、逐token S对旧gross、真实路径闭合≤1%，保存文件manifest与派生特征指纹；72头均CPU重载复算概率、train-only scaler、最低val-loss epoch和指标。原始缓存不改写。

输出：`outputs/all_attention_write_gain_811_v1/`。入口：`scripts/analyze_all_attention_write_gain.py`。本次不做第四项frozen-RMS直接对照。
