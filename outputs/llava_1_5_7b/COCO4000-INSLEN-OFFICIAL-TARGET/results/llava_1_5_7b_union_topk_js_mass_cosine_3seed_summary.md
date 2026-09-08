# llava_1_5_7b union-topK JS + mass x cosine 3-seed Torch MLP summary

Seeds: `43, 44, 45`. Values are population mean+/-std.
Real is the headline positive class; hallucination-positive metrics are reported alongside it.
Strict 8:2 has no validation set: train-loss early stopping restores the minimum-train-loss checkpoint.
Every checkpoint is reported twice: fixed threshold 0.5 and a Real-F1 threshold selected on train only.
Real/Hall AUC values are equal under score inversion, while AUPR differs; hallucination metrics use the complementary prediction at the same fixed boundary.

## Best by model

| Model | Threshold mode | Best feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llava_1_5_7b | fixed_0.5 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.835+/-0.001 | 0.904+/-0.004 | 0.880+/-0.006 | 0.892+/-0.001 | 0.889+/-0.002 | 0.965+/-0.001 | 0.622+/-0.006 | 0.680+/-0.018 | 0.649+/-0.005 | 0.889+/-0.002 | 0.675+/-0.004 |
| llava_1_5_7b | train_f1 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.469+/-0.013 | 0.836+/-0.000 | 0.895+/-0.002 | 0.892+/-0.003 | 0.894+/-0.000 | 0.889+/-0.002 | 0.965+/-0.001 | 0.634+/-0.003 | 0.642+/-0.009 | 0.638+/-0.003 | 0.889+/-0.002 | 0.675+/-0.004 |

## llava_1_5_7b

| Threshold mode | Rank | Feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.5 | 1 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.835+/-0.001 | 0.904+/-0.004 | 0.880+/-0.006 | 0.892+/-0.001 | 0.889+/-0.002 | 0.965+/-0.001 | 0.622+/-0.006 | 0.680+/-0.018 | 0.649+/-0.005 | 0.889+/-0.002 | 0.675+/-0.004 |
| fixed_0.5 | 2 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.838+/-0.004 | 0.893+/-0.004 | 0.898+/-0.006 | 0.896+/-0.003 | 0.888+/-0.001 | 0.965+/-0.001 | 0.643+/-0.011 | 0.631+/-0.016 | 0.637+/-0.009 | 0.888+/-0.001 | 0.666+/-0.006 |
| fixed_0.5 | 3 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.830+/-0.001 | 0.881+/-0.002 | 0.902+/-0.002 | 0.892+/-0.000 | 0.875+/-0.001 | 0.959+/-0.000 | 0.634+/-0.002 | 0.580+/-0.009 | 0.606+/-0.004 | 0.875+/-0.001 | 0.641+/-0.004 |
| fixed_0.5 | 4 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.824+/-0.002 | 0.888+/-0.004 | 0.884+/-0.007 | 0.886+/-0.002 | 0.866+/-0.002 | 0.957+/-0.001 | 0.607+/-0.008 | 0.617+/-0.018 | 0.612+/-0.006 | 0.866+/-0.002 | 0.632+/-0.007 |
| fixed_0.5 | 5 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.825+/-0.002 | 0.880+/-0.006 | 0.898+/-0.009 | 0.888+/-0.001 | 0.865+/-0.003 | 0.955+/-0.001 | 0.622+/-0.009 | 0.578+/-0.028 | 0.598+/-0.012 | 0.865+/-0.003 | 0.636+/-0.004 |
| fixed_0.5 | 6 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.825+/-0.002 | 0.879+/-0.009 | 0.897+/-0.016 | 0.888+/-0.003 | 0.865+/-0.002 | 0.955+/-0.001 | 0.622+/-0.019 | 0.575+/-0.044 | 0.595+/-0.016 | 0.865+/-0.002 | 0.645+/-0.002 |
| train_f1 | 1 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.469+/-0.013 | 0.836+/-0.000 | 0.895+/-0.002 | 0.892+/-0.003 | 0.894+/-0.000 | 0.889+/-0.002 | 0.965+/-0.001 | 0.634+/-0.003 | 0.642+/-0.009 | 0.638+/-0.003 | 0.889+/-0.002 | 0.675+/-0.004 |
| train_f1 | 2 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.505+/-0.009 | 0.838+/-0.005 | 0.894+/-0.006 | 0.897+/-0.005 | 0.896+/-0.003 | 0.888+/-0.001 | 0.965+/-0.001 | 0.642+/-0.009 | 0.635+/-0.025 | 0.638+/-0.015 | 0.888+/-0.001 | 0.666+/-0.006 |
| train_f1 | 3 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.454+/-0.017 | 0.829+/-0.001 | 0.861+/-0.007 | 0.929+/-0.008 | 0.894+/-0.000 | 0.875+/-0.001 | 0.959+/-0.000 | 0.666+/-0.010 | 0.485+/-0.032 | 0.560+/-0.018 | 0.875+/-0.001 | 0.641+/-0.004 |
| train_f1 | 4 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.441+/-0.003 | 0.824+/-0.001 | 0.869+/-0.007 | 0.909+/-0.008 | 0.889+/-0.000 | 0.866+/-0.002 | 0.957+/-0.001 | 0.630+/-0.007 | 0.530+/-0.033 | 0.575+/-0.017 | 0.866+/-0.002 | 0.632+/-0.007 |
| train_f1 | 5 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.472+/-0.025 | 0.826+/-0.003 | 0.870+/-0.004 | 0.912+/-0.005 | 0.890+/-0.002 | 0.865+/-0.003 | 0.955+/-0.001 | 0.637+/-0.009 | 0.529+/-0.018 | 0.578+/-0.011 | 0.865+/-0.003 | 0.636+/-0.004 |
| train_f1 | 6 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.465+/-0.023 | 0.826+/-0.001 | 0.869+/-0.002 | 0.913+/-0.004 | 0.890+/-0.001 | 0.865+/-0.002 | 0.955+/-0.001 | 0.638+/-0.007 | 0.527+/-0.012 | 0.577+/-0.005 | 0.865+/-0.002 | 0.645+/-0.002 |
