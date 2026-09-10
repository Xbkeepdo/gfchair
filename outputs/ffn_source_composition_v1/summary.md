# FFN 来源分解验证结果

原4000图3200/800划分；10组×3seeds（原9组加F_C=AE+log1p(S_C)）；探索性结果，无bootstrap。

| 模型 | 组 | AUROC | HALL-AUPR |
|---|---|---:|---:|
| qwen2_5_vl_7b | AE_I | 0.870320 | 0.444710 |
| qwen2_5_vl_7b | B | 0.881034 | 0.460093 |
| qwen2_5_vl_7b | AE_I_SC | 0.874516 | 0.453304 |
| qwen2_5_vl_7b | B_SC | 0.881013 | 0.450900 |
| qwen2_5_vl_7b | B_JS_TE | 0.875093 | 0.449509 |
| qwen2_5_vl_7b | B_JS_TC | 0.882667 | 0.460138 |
| qwen2_5_vl_7b | B_OT_TE | 0.879743 | 0.447930 |
| qwen2_5_vl_7b | B_OT_TC | 0.872666 | 0.447456 |
| qwen2_5_vl_7b | F | 0.882138 | 0.457740 |
| qwen2_5_vl_7b | F_C | 0.881510 | 0.471622 |
| llava_1_5_7b | AE_I | 0.902676 | 0.718400 |
| llava_1_5_7b | B | 0.906132 | 0.717329 |
| llava_1_5_7b | AE_I_SC | 0.907223 | 0.720366 |
| llava_1_5_7b | B_SC | 0.904614 | 0.718221 |
| llava_1_5_7b | B_JS_TE | 0.904857 | 0.718454 |
| llava_1_5_7b | B_JS_TC | 0.906856 | 0.726737 |
| llava_1_5_7b | B_OT_TE | 0.905254 | 0.723580 |
| llava_1_5_7b | B_OT_TC | 0.907956 | 0.723416 |
| llava_1_5_7b | F | 0.904960 | 0.719276 |
| llava_1_5_7b | F_C | 0.904821 | 0.720085 |
| qwen3_vl_8b | AE_I | 0.886481 | 0.632260 |
| qwen3_vl_8b | B | 0.896747 | 0.664920 |
| qwen3_vl_8b | AE_I_SC | 0.897139 | 0.673012 |
| qwen3_vl_8b | B_SC | 0.900019 | 0.669176 |
| qwen3_vl_8b | B_JS_TE | 0.904018 | 0.685049 |
| qwen3_vl_8b | B_JS_TC | 0.890310 | 0.645600 |
| qwen3_vl_8b | B_OT_TE | 0.899828 | 0.674214 |
| qwen3_vl_8b | B_OT_TC | 0.881986 | 0.630541 |
| qwen3_vl_8b | F | 0.896510 | 0.649342 |
| qwen3_vl_8b | F_C | 0.894223 | 0.649203 |
| internvl_2_5_8b | AE_I | 0.863149 | 0.549897 |
| internvl_2_5_8b | B | 0.870476 | 0.537155 |
| internvl_2_5_8b | AE_I_SC | 0.867782 | 0.543936 |
| internvl_2_5_8b | B_SC | 0.869596 | 0.555812 |
| internvl_2_5_8b | B_JS_TE | 0.877491 | 0.586110 |
| internvl_2_5_8b | B_JS_TC | 0.873851 | 0.550485 |
| internvl_2_5_8b | B_OT_TE | 0.870790 | 0.550492 |
| internvl_2_5_8b | B_OT_TC | 0.867541 | 0.557991 |
| internvl_2_5_8b | F | 0.863812 | 0.546154 |
| internvl_2_5_8b | F_C | 0.855203 | 0.526212 |

## 配对检测增量

| 模型 | 比较 | AUROC增量 | HALL-AUPR增量 | 逐seed AUROC差 |
|---|---|---:|---:|---|
| qwen2_5_vl_7b | B − AE_I | +0.010714 | +0.015382 | +0.007606/+0.010657/+0.010019 |
| qwen2_5_vl_7b | B − F | -0.001104 | +0.002353 | -0.002395/+0.002374/-0.011649 |
| qwen2_5_vl_7b | F_C − F | -0.000628 | +0.013882 | +0.001291/-0.004791/-0.004519 |
| qwen2_5_vl_7b | AE_I_SC − B | -0.006518 | -0.006789 | +0.002104/-0.014136/-0.006216 |
| qwen2_5_vl_7b | B_SC − B | -0.000021 | -0.009193 | +0.000014/-0.008632/+0.012957 |
| qwen2_5_vl_7b | B_JS_TC − B_JS_TE | +0.007574 | +0.010628 | +0.005786/+0.020053/+0.006809 |
| qwen2_5_vl_7b | B_OT_TC − B_OT_TE | -0.007077 | -0.000474 | +0.004863/-0.015961/-0.003069 |
| qwen2_5_vl_7b | B_JS_TC − B | +0.001633 | +0.000045 | +0.000141/+0.009648/+0.011264 |
| qwen2_5_vl_7b | B_OT_TC − B | -0.008368 | -0.012636 | -0.001057/-0.013030/+0.001598 |
| llava_1_5_7b | B − AE_I | +0.003456 | -0.001071 | +0.006160/+0.003441/-0.004759 |
| llava_1_5_7b | B − F | +0.001173 | -0.001947 | +0.007312/+0.001491/-0.007823 |
| llava_1_5_7b | F_C − F | -0.000138 | +0.000808 | +0.003958/-0.000718/-0.000662 |
| llava_1_5_7b | AE_I_SC − B | +0.001091 | +0.003037 | -0.002383/+0.000500/+0.009651 |
| llava_1_5_7b | B_SC − B | -0.001518 | +0.000891 | -0.008716/-0.001727/+0.006560 |
| llava_1_5_7b | B_JS_TC − B_JS_TE | +0.001999 | +0.008283 | +0.007599/-0.003537/+0.002997 |
| llava_1_5_7b | B_OT_TC − B_OT_TE | +0.002702 | -0.000164 | +0.011931/+0.002151/+0.000288 |
| llava_1_5_7b | B_JS_TC − B | +0.000724 | +0.009408 | +0.001015/-0.002426/+0.001872 |
| llava_1_5_7b | B_OT_TC − B | +0.001824 | +0.006087 | +0.002183/-0.000488/+0.006554 |
| qwen3_vl_8b | B − AE_I | +0.010266 | +0.032660 | -0.021328/+0.033560/-0.006981 |
| qwen3_vl_8b | B − F | +0.000238 | +0.015579 | -0.021905/+0.011880/-0.022951 |
| qwen3_vl_8b | F_C − F | -0.002287 | -0.000139 | -0.000782/+0.000636/-0.001290 |
| qwen3_vl_8b | AE_I_SC − B | +0.000392 | +0.008092 | +0.010513/+0.003771/+0.011803 |
| qwen3_vl_8b | B_SC − B | +0.003272 | +0.004256 | +0.014080/-0.003245/+0.023846 |
| qwen3_vl_8b | B_JS_TC − B_JS_TE | -0.013708 | -0.039448 | -0.010577/-0.013788/-0.070452 |
| qwen3_vl_8b | B_OT_TC − B_OT_TE | -0.017842 | -0.043673 | -0.019712/-0.004838/-0.036647 |
| qwen3_vl_8b | B_JS_TC − B | -0.006437 | -0.019320 | +0.005929/-0.014696/-0.048416 |
| qwen3_vl_8b | B_OT_TC − B | -0.014762 | -0.034380 | -0.004934/-0.016230/-0.010897 |
| internvl_2_5_8b | B − AE_I | +0.007327 | -0.012742 | +0.016560/+0.001451/-0.003572 |
| internvl_2_5_8b | B − F | +0.006664 | -0.008999 | +0.015767/+0.000835/-0.010666 |
| internvl_2_5_8b | F_C − F | -0.008609 | -0.019942 | -0.013202/-0.003196/-0.011514 |
| internvl_2_5_8b | AE_I_SC − B | -0.002694 | +0.006781 | -0.002085/+0.013431/-0.011283 |
| internvl_2_5_8b | B_SC − B | -0.000880 | +0.018657 | -0.002706/+0.005013/-0.003084 |
| internvl_2_5_8b | B_JS_TC − B_JS_TE | -0.003640 | -0.035625 | -0.002636/-0.029847/-0.005030 |
| internvl_2_5_8b | B_OT_TC − B_OT_TE | -0.003248 | +0.007499 | -0.000780/+0.000060/-0.005983 |
| internvl_2_5_8b | B_JS_TC − B | +0.003375 | +0.013330 | +0.000959/-0.020617/+0.015246 |
| internvl_2_5_8b | B_OT_TC − B | -0.002935 | +0.020836 | -0.007526/-0.000682/-0.007239 |

## 数值闭合

| 模型 | 中位数 | P90 | 最大值 | 超过1% / target-layer |
|---|---:|---:|---:|---:|
| qwen2_5_vl_7b | 0.00135913 | 0.00540263 | 0.0951321 | 7847/242312 |
| llava_1_5_7b | 0.0001276 | 0.000165371 | 0.000544653 | 0/478432 |
| qwen3_vl_8b | 0.00120894 | 0.00236552 | 0.116194 | 21295/529344 |
| internvl_2_5_8b | 0.000967912 | 0.00126066 | 0.0033839 | 0/372160 |

## fixed-QK配对差

图片内先平均四层，再计算策略之间的差；无bootstrap。

| 模型 | 比较 | 指标 | 均值差 | 中位数差 | 胜率 |
|---|---|---|---:|---:|---:|
| qwen2_5_vl_7b | composition − WRITE | absolute_log_probability_delta | -0.000033 | +0.000000 | 0.080 |
| qwen2_5_vl_7b | composition − WRITE | log_probability_delta | -0.001323 | +0.000000 | 0.100 |
| qwen2_5_vl_7b | composition − conditional | absolute_log_probability_delta | -0.000285 | +0.000000 | 0.050 |
| qwen2_5_vl_7b | composition − conditional | log_probability_delta | -0.000044 | +0.000000 | 0.060 |
| qwen2_5_vl_7b | composition − random | absolute_log_probability_delta | +0.026851 | +0.008657 | 0.780 |
| qwen2_5_vl_7b | composition − random | log_probability_delta | -0.015680 | -0.002724 | 0.410 |
| qwen2_5_vl_7b | conditional − WRITE | absolute_log_probability_delta | +0.000252 | +0.000000 | 0.100 |
| qwen2_5_vl_7b | conditional − WRITE | log_probability_delta | -0.001279 | +0.000000 | 0.110 |
| llava_1_5_7b | composition − WRITE | absolute_log_probability_delta | +0.000372 | +0.000000 | 0.420 |
| llava_1_5_7b | composition − WRITE | log_probability_delta | -0.000116 | +0.000000 | 0.300 |
| llava_1_5_7b | composition − conditional | absolute_log_probability_delta | +0.000043 | +0.000000 | 0.070 |
| llava_1_5_7b | composition − conditional | log_probability_delta | +0.000022 | +0.000000 | 0.050 |
| llava_1_5_7b | composition − random | absolute_log_probability_delta | +0.010313 | +0.003301 | 0.850 |
| llava_1_5_7b | composition − random | log_probability_delta | -0.002162 | +0.000310 | 0.570 |
| llava_1_5_7b | conditional − WRITE | absolute_log_probability_delta | +0.000329 | +0.000000 | 0.380 |
| llava_1_5_7b | conditional − WRITE | log_probability_delta | -0.000138 | +0.000000 | 0.280 |
| qwen3_vl_8b | composition − WRITE | absolute_log_probability_delta | +0.000304 | +0.000000 | 0.330 |
| qwen3_vl_8b | composition − WRITE | log_probability_delta | -0.000265 | +0.000000 | 0.280 |
| qwen3_vl_8b | composition − conditional | absolute_log_probability_delta | -0.000131 | +0.000000 | 0.030 |
| qwen3_vl_8b | composition − conditional | log_probability_delta | -0.000137 | +0.000000 | 0.010 |
| qwen3_vl_8b | composition − random | absolute_log_probability_delta | +0.009877 | +0.000690 | 0.670 |
| qwen3_vl_8b | composition − random | log_probability_delta | -0.000948 | +0.000000 | 0.480 |
| qwen3_vl_8b | conditional − WRITE | absolute_log_probability_delta | +0.000435 | +0.000000 | 0.320 |
| qwen3_vl_8b | conditional − WRITE | log_probability_delta | -0.000128 | +0.000000 | 0.290 |
| internvl_2_5_8b | composition − WRITE | absolute_log_probability_delta | -0.000034 | +0.000000 | 0.110 |
| internvl_2_5_8b | composition − WRITE | log_probability_delta | +0.000412 | +0.000000 | 0.110 |
| internvl_2_5_8b | composition − conditional | absolute_log_probability_delta | -0.000032 | +0.000000 | 0.000 |
| internvl_2_5_8b | composition − conditional | log_probability_delta | +0.000032 | +0.000000 | 0.010 |
| internvl_2_5_8b | composition − random | absolute_log_probability_delta | +0.011099 | +0.003997 | 0.650 |
| internvl_2_5_8b | composition − random | log_probability_delta | -0.006152 | -0.000324 | 0.440 |
| internvl_2_5_8b | conditional − WRITE | absolute_log_probability_delta | -0.000002 | +0.000000 | 0.110 |
| internvl_2_5_8b | conditional − WRITE | log_probability_delta | +0.000380 | +0.000000 | 0.100 |

## 相同区域比例

| 模型 | composition=conditional | composition=WRITE |
|---|---:|---:|
| qwen2_5_vl_7b | 97.25% | 94.00% |
| llava_1_5_7b | 97.50% | 82.00% |
| qwen3_vl_8b | 98.75% | 82.00% |
| internvl_2_5_8b | 99.75% | 94.50% |
