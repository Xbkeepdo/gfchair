# S_G / position 检测协议

用户确认比较S_G-only、S_G/position-only、position-only。四模型真RMS All-attention z−A_all→z K32，复用all_attention_write_gain_811_v1/statistics.pt，S_G为仅生成前缀token的gross响应；position=response_index=N_G，不是包含prompt/视觉的绝对位置。

- 固定原811图片3200/400/400、保留全部mentions，seeds43/44/45；四模型×三特征×三seed=36头。
- S_G与比值保留全部decoder层；position-only为单列，不跨层重复。先除position，再log1p与train-only逐列Z-score。position=0时raw存NaN、检测值置0，保留样本。
- 全部组使用前轮统一128/ReLU/dropout.3/noBN单隐藏MLP，Adam lr.001/wd1e-5/batch128，max150、早停20、LR plateau patience6，最低val BCE checkpoint；无调参，不输入AE，不重训VLM。
- AUROC、HALL-AUPR先逐seed再均值±总体std。每模型2000次测试图片簇配对bootstrap，seed20260915；比较ratio−S_G、ratio−position、S_G−position。包括空图片，名义95%区间未多重校正；旧test已查看，探索性结果。
- 原统计SHA256、mention/标签/划分、生成S矩阵一致性、36头CPU重载/scaler/最佳epoch/概率和指标复算。

入口scripts/evaluate_generation_per_position_811.py，输出outputs/generation_per_position_811_v1/。
