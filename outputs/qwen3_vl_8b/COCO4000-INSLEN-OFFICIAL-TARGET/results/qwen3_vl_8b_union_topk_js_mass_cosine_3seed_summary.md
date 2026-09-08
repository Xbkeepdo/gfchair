# qwen3_vl_8b union-topK JS + mass x cosine 3-seed Torch MLP summary

Seeds: `43, 44, 45`. Values are population mean+/-std.
Real is the headline positive class; hallucination-positive metrics are reported alongside it.
Strict 8:2 has no validation set: train-loss early stopping restores the minimum-train-loss checkpoint.
Every checkpoint is reported twice: fixed threshold 0.5 and a Real-F1 threshold selected on train only.
Real/Hall AUC values are equal under score inversion, while AUPR differs; hallucination metrics use the complementary prediction at the same fixed boundary.

## Best by model

| Model | Threshold mode | Best feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_vl_8b | fixed_0.5 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.854+/-0.006 | 0.909+/-0.008 | 0.916+/-0.017 | 0.912+/-0.005 | 0.865+/-0.002 | 0.967+/-0.001 | 0.577+/-0.027 | 0.552+/-0.049 | 0.562+/-0.013 | 0.865+/-0.002 | 0.615+/-0.007 |
| qwen3_vl_8b | train_f1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.466+/-0.040 | 0.857+/-0.003 | 0.902+/-0.002 | 0.929+/-0.006 | 0.915+/-0.002 | 0.865+/-0.002 | 0.967+/-0.001 | 0.596+/-0.013 | 0.508+/-0.014 | 0.549+/-0.005 | 0.865+/-0.002 | 0.615+/-0.007 |

## qwen3_vl_8b

| Threshold mode | Rank | Feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.5 | 1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.854+/-0.006 | 0.909+/-0.008 | 0.916+/-0.017 | 0.912+/-0.005 | 0.865+/-0.002 | 0.967+/-0.001 | 0.577+/-0.027 | 0.552+/-0.049 | 0.562+/-0.013 | 0.865+/-0.002 | 0.615+/-0.007 |
| fixed_0.5 | 2 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.852+/-0.008 | 0.904+/-0.013 | 0.920+/-0.028 | 0.911+/-0.007 | 0.863+/-0.002 | 0.966+/-0.000 | 0.586+/-0.057 | 0.519+/-0.088 | 0.541+/-0.032 | 0.863+/-0.002 | 0.594+/-0.007 |
| fixed_0.5 | 3 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.854+/-0.002 | 0.875+/-0.002 | 0.961+/-0.006 | 0.916+/-0.002 | 0.842+/-0.001 | 0.959+/-0.001 | 0.638+/-0.022 | 0.331+/-0.018 | 0.435+/-0.011 | 0.842+/-0.001 | 0.538+/-0.008 |
| fixed_0.5 | 4 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.851+/-0.001 | 0.867+/-0.006 | 0.969+/-0.007 | 0.915+/-0.000 | 0.827+/-0.006 | 0.956+/-0.002 | 0.653+/-0.018 | 0.276+/-0.041 | 0.386+/-0.036 | 0.827+/-0.006 | 0.525+/-0.006 |
| fixed_0.5 | 5 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.844+/-0.001 | 0.871+/-0.007 | 0.953+/-0.009 | 0.910+/-0.000 | 0.825+/-0.002 | 0.958+/-0.000 | 0.578+/-0.007 | 0.309+/-0.051 | 0.400+/-0.041 | 0.825+/-0.002 | 0.518+/-0.008 |
| fixed_0.5 | 6 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.842+/-0.004 | 0.871+/-0.007 | 0.951+/-0.015 | 0.909+/-0.003 | 0.824+/-0.002 | 0.957+/-0.000 | 0.577+/-0.037 | 0.313+/-0.051 | 0.401+/-0.034 | 0.824+/-0.002 | 0.507+/-0.006 |
| train_f1 | 1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.466+/-0.040 | 0.857+/-0.003 | 0.902+/-0.002 | 0.929+/-0.006 | 0.915+/-0.002 | 0.865+/-0.002 | 0.967+/-0.001 | 0.596+/-0.013 | 0.508+/-0.014 | 0.549+/-0.005 | 0.865+/-0.002 | 0.615+/-0.007 |
| train_f1 | 2 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.523+/-0.117 | 0.852+/-0.001 | 0.903+/-0.002 | 0.921+/-0.003 | 0.912+/-0.001 | 0.863+/-0.002 | 0.966+/-0.000 | 0.573+/-0.006 | 0.519+/-0.014 | 0.544+/-0.007 | 0.863+/-0.002 | 0.594+/-0.007 |
| train_f1 | 3 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.506+/-0.024 | 0.853+/-0.002 | 0.876+/-0.005 | 0.958+/-0.008 | 0.915+/-0.001 | 0.842+/-0.001 | 0.959+/-0.001 | 0.629+/-0.027 | 0.340+/-0.033 | 0.440+/-0.023 | 0.842+/-0.001 | 0.538+/-0.008 |
| train_f1 | 4 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.556+/-0.038 | 0.852+/-0.001 | 0.878+/-0.000 | 0.955+/-0.002 | 0.915+/-0.001 | 0.827+/-0.006 | 0.956+/-0.002 | 0.616+/-0.009 | 0.353+/-0.004 | 0.449+/-0.001 | 0.827+/-0.006 | 0.525+/-0.006 |
| train_f1 | 5 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.531+/-0.016 | 0.838+/-0.004 | 0.876+/-0.009 | 0.938+/-0.017 | 0.906+/-0.003 | 0.825+/-0.002 | 0.958+/-0.000 | 0.544+/-0.023 | 0.355+/-0.065 | 0.425+/-0.038 | 0.825+/-0.002 | 0.518+/-0.008 |
| train_f1 | 6 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.551+/-0.052 | 0.836+/-0.006 | 0.881+/-0.003 | 0.928+/-0.013 | 0.904+/-0.005 | 0.824+/-0.002 | 0.957+/-0.000 | 0.528+/-0.029 | 0.388+/-0.026 | 0.446+/-0.009 | 0.824+/-0.002 | 0.507+/-0.006 |
