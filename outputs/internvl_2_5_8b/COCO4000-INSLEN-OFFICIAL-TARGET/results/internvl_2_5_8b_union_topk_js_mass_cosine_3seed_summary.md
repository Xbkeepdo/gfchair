# internvl_2_5_8b union-topK JS + mass x cosine 3-seed Torch MLP summary

Seeds: `43, 44, 45`. Values are population mean+/-std.
Real is the headline positive class; hallucination-positive metrics are reported alongside it.
Strict 8:2 has no validation set: train-loss early stopping restores the minimum-train-loss checkpoint.
Every checkpoint is reported twice: fixed threshold 0.5 and a Real-F1 threshold selected on train only.
Real/Hall AUC values are equal under score inversion, while AUPR differs; hallucination metrics use the complementary prediction at the same fixed boundary.

## Best by model

| Model | Threshold mode | Best feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| internvl_2_5_8b | fixed_0.5 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.854+/-0.003 | 0.885+/-0.014 | 0.951+/-0.023 | 0.916+/-0.003 | 0.852+/-0.005 | 0.965+/-0.003 | 0.569+/-0.041 | 0.332+/-0.106 | 0.405+/-0.071 | 0.852+/-0.005 | 0.513+/-0.018 |
| internvl_2_5_8b | train_f1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.555+/-0.054 | 0.855+/-0.003 | 0.896+/-0.005 | 0.936+/-0.004 | 0.916+/-0.001 | 0.852+/-0.005 | 0.965+/-0.003 | 0.547+/-0.010 | 0.414+/-0.032 | 0.471+/-0.023 | 0.852+/-0.005 | 0.513+/-0.018 |

## internvl_2_5_8b

| Threshold mode | Rank | Feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.5 | 1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.854+/-0.003 | 0.885+/-0.014 | 0.951+/-0.023 | 0.916+/-0.003 | 0.852+/-0.005 | 0.965+/-0.003 | 0.569+/-0.041 | 0.332+/-0.106 | 0.405+/-0.071 | 0.852+/-0.005 | 0.513+/-0.018 |
| fixed_0.5 | 2 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.850+/-0.002 | 0.894+/-0.004 | 0.933+/-0.004 | 0.913+/-0.001 | 0.847+/-0.001 | 0.963+/-0.001 | 0.529+/-0.008 | 0.408+/-0.030 | 0.460+/-0.021 | 0.847+/-0.001 | 0.497+/-0.004 |
| fixed_0.5 | 3 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.852+/-0.005 | 0.874+/-0.004 | 0.963+/-0.011 | 0.916+/-0.003 | 0.839+/-0.003 | 0.964+/-0.001 | 0.566+/-0.038 | 0.250+/-0.037 | 0.344+/-0.031 | 0.839+/-0.003 | 0.480+/-0.009 |
| fixed_0.5 | 4 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.500+/-0.000 | 0.847+/-0.001 | 0.858+/-0.007 | 0.980+/-0.010 | 0.915+/-0.000 | 0.830+/-0.003 | 0.961+/-0.000 | 0.549+/-0.018 | 0.129+/-0.057 | 0.202+/-0.071 | 0.830+/-0.003 | 0.451+/-0.010 |
| fixed_0.5 | 5 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.844+/-0.003 | 0.865+/-0.005 | 0.966+/-0.012 | 0.912+/-0.002 | 0.796+/-0.001 | 0.953+/-0.001 | 0.511+/-0.029 | 0.186+/-0.045 | 0.268+/-0.043 | 0.796+/-0.001 | 0.412+/-0.001 |
| fixed_0.5 | 6 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.500+/-0.000 | 0.844+/-0.002 | 0.864+/-0.001 | 0.968+/-0.004 | 0.913+/-0.001 | 0.784+/-0.000 | 0.949+/-0.001 | 0.511+/-0.020 | 0.177+/-0.012 | 0.262+/-0.011 | 0.784+/-0.000 | 0.406+/-0.003 |
| train_f1 | 1 | `hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.555+/-0.054 | 0.855+/-0.003 | 0.896+/-0.005 | 0.936+/-0.004 | 0.916+/-0.001 | 0.852+/-0.005 | 0.965+/-0.003 | 0.547+/-0.010 | 0.414+/-0.032 | 0.471+/-0.023 | 0.852+/-0.005 | 0.513+/-0.018 |
| train_f1 | 2 | `hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.514+/-0.009 | 0.848+/-0.001 | 0.896+/-0.005 | 0.928+/-0.005 | 0.912+/-0.000 | 0.847+/-0.001 | 0.963+/-0.001 | 0.520+/-0.002 | 0.419+/-0.035 | 0.463+/-0.023 | 0.847+/-0.001 | 0.497+/-0.004 |
| train_f1 | 3 | `hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine` | 0.514+/-0.023 | 0.849+/-0.002 | 0.876+/-0.006 | 0.956+/-0.009 | 0.914+/-0.002 | 0.839+/-0.003 | 0.964+/-0.001 | 0.539+/-0.021 | 0.272+/-0.043 | 0.358+/-0.035 | 0.839+/-0.003 | 0.480+/-0.009 |
| train_f1 | 4 | `hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine` | 0.530+/-0.009 | 0.848+/-0.001 | 0.871+/-0.003 | 0.962+/-0.006 | 0.914+/-0.001 | 0.830+/-0.003 | 0.961+/-0.000 | 0.534+/-0.015 | 0.231+/-0.027 | 0.321+/-0.024 | 0.830+/-0.003 | 0.451+/-0.010 |
| train_f1 | 5 | `hpre_softmax_prob_gauss_source_target_union_topk_js` | 0.547+/-0.031 | 0.840+/-0.007 | 0.875+/-0.008 | 0.947+/-0.021 | 0.909+/-0.005 | 0.796+/-0.001 | 0.953+/-0.001 | 0.497+/-0.043 | 0.269+/-0.072 | 0.339+/-0.057 | 0.796+/-0.001 | 0.412+/-0.001 |
| train_f1 | 6 | `hpre_raw_logit_gauss_source_target_union_topk_js` | 0.518+/-0.029 | 0.842+/-0.002 | 0.867+/-0.004 | 0.960+/-0.009 | 0.911+/-0.002 | 0.784+/-0.000 | 0.949+/-0.001 | 0.491+/-0.019 | 0.205+/-0.035 | 0.286+/-0.033 | 0.784+/-0.000 | 0.406+/-0.003 |
