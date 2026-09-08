# qwen2_5_vl_7b union-topK JS + mass x cosine 3-seed Torch MLP summary

Seeds: `43, 44, 45`. Values are population mean+/-std.
Real is the headline positive class; hallucination-positive metrics are reported alongside it.
Strict 8:2 has no validation set: train-loss early stopping restores the minimum-train-loss checkpoint.
Every checkpoint is reported twice: fixed threshold 0.5 and a Real-F1 threshold selected on train only.
Real/Hall AUC values are equal under score inversion, while AUPR differs; hallucination metrics use the complementary prediction at the same fixed boundary.

## Best by model

| Model | Threshold mode | Best feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2_5_vl_7b | fixed_0.5 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.886+/-0.008 | 0.916+/-0.005 | 0.960+/-0.016 | 0.938+/-0.005 | 0.830+/-0.007 | 0.976+/-0.001 | 0.446+/-0.042 | 0.257+/-0.062 | 0.318+/-0.036 | 0.830+/-0.007 | 0.377+/-0.004 |
| qwen2_5_vl_7b | train_f1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.547+/-0.056 | 0.881+/-0.005 | 0.922+/-0.001 | 0.946+/-0.007 | 0.934+/-0.003 | 0.830+/-0.007 | 0.976+/-0.001 | 0.419+/-0.024 | 0.324+/-0.014 | 0.365+/-0.008 | 0.830+/-0.007 | 0.377+/-0.004 |

## qwen2_5_vl_7b

| Threshold mode | Rank | Feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.5 | 1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.886+/-0.008 | 0.916+/-0.005 | 0.960+/-0.016 | 0.938+/-0.005 | 0.830+/-0.007 | 0.976+/-0.001 | 0.446+/-0.042 | 0.257+/-0.062 | 0.318+/-0.036 | 0.830+/-0.007 | 0.377+/-0.004 |
| fixed_0.5 | 2 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.882+/-0.004 | 0.916+/-0.007 | 0.957+/-0.013 | 0.936+/-0.003 | 0.826+/-0.009 | 0.975+/-0.002 | 0.408+/-0.021 | 0.251+/-0.075 | 0.305+/-0.055 | 0.826+/-0.009 | 0.354+/-0.009 |
| fixed_0.5 | 3 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.893+/-0.002 | 0.903+/-0.001 | 0.986+/-0.004 | 0.943+/-0.001 | 0.795+/-0.008 | 0.969+/-0.002 | 0.475+/-0.036 | 0.106+/-0.017 | 0.172+/-0.022 | 0.795+/-0.008 | 0.327+/-0.002 |
| fixed_0.5 | 4 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.893+/-0.003 | 0.903+/-0.000 | 0.986+/-0.003 | 0.943+/-0.001 | 0.791+/-0.008 | 0.969+/-0.002 | 0.477+/-0.058 | 0.104+/-0.004 | 0.170+/-0.007 | 0.791+/-0.008 | 0.317+/-0.002 |
| fixed_0.5 | 5 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.892+/-0.001 | 0.898+/-0.002 | 0.991+/-0.003 | 0.942+/-0.001 | 0.785+/-0.005 | 0.968+/-0.001 | 0.402+/-0.036 | 0.051+/-0.022 | 0.089+/-0.034 | 0.785+/-0.005 | 0.295+/-0.017 |
| fixed_0.5 | 6 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.891+/-0.002 | 0.898+/-0.001 | 0.990+/-0.004 | 0.942+/-0.001 | 0.784+/-0.002 | 0.968+/-0.000 | 0.395+/-0.034 | 0.053+/-0.016 | 0.092+/-0.024 | 0.784+/-0.002 | 0.284+/-0.001 |
| train_f1 | 1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.547+/-0.056 | 0.881+/-0.005 | 0.922+/-0.001 | 0.946+/-0.007 | 0.934+/-0.003 | 0.830+/-0.007 | 0.976+/-0.001 | 0.419+/-0.024 | 0.324+/-0.014 | 0.365+/-0.008 | 0.830+/-0.007 | 0.377+/-0.004 |
| train_f1 | 2 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.585+/-0.054 | 0.874+/-0.001 | 0.922+/-0.002 | 0.939+/-0.002 | 0.930+/-0.001 | 0.826+/-0.009 | 0.975+/-0.002 | 0.388+/-0.010 | 0.328+/-0.019 | 0.355+/-0.015 | 0.826+/-0.009 | 0.354+/-0.009 |
| train_f1 | 3 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.576+/-0.018 | 0.882+/-0.003 | 0.911+/-0.001 | 0.961+/-0.005 | 0.936+/-0.002 | 0.795+/-0.008 | 0.969+/-0.002 | 0.389+/-0.018 | 0.208+/-0.016 | 0.270+/-0.013 | 0.795+/-0.008 | 0.327+/-0.002 |
| train_f1 | 4 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.591+/-0.028 | 0.880+/-0.004 | 0.912+/-0.001 | 0.959+/-0.006 | 0.935+/-0.002 | 0.791+/-0.008 | 0.969+/-0.002 | 0.385+/-0.023 | 0.217+/-0.013 | 0.277+/-0.009 | 0.791+/-0.008 | 0.317+/-0.002 |
| train_f1 | 5 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.540+/-0.011 | 0.890+/-0.002 | 0.901+/-0.002 | 0.984+/-0.002 | 0.941+/-0.001 | 0.785+/-0.005 | 0.968+/-0.001 | 0.397+/-0.039 | 0.089+/-0.019 | 0.145+/-0.026 | 0.785+/-0.005 | 0.295+/-0.017 |
| train_f1 | 6 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.537+/-0.003 | 0.888+/-0.002 | 0.901+/-0.001 | 0.983+/-0.004 | 0.940+/-0.001 | 0.784+/-0.002 | 0.968+/-0.000 | 0.372+/-0.023 | 0.082+/-0.012 | 0.134+/-0.015 | 0.784+/-0.002 | 0.284+/-0.001 |
