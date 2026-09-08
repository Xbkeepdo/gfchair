| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.895 | 0.901 | 0.991 | 0.944 | 0.806 | 0.972 | 0.517 | 0.082 | 0.142 | 0.806 | 0.324 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.878 | 0.911 | 0.956 | 0.933 | 0.806 | 0.972 | 0.364 | 0.213 | 0.269 | 0.806 | 0.324 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.893 | 0.897 | 0.994 | 0.943 | 0.787 | 0.968 | 0.400 | 0.033 | 0.061 | 0.787 | 0.285 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.890 | 0.899 | 0.988 | 0.942 | 0.787 | 0.968 | 0.387 | 0.066 | 0.112 | 0.787 | 0.285 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.894 | 0.913 | 0.974 | 0.943 | 0.835 | 0.977 | 0.500 | 0.219 | 0.304 | 0.835 | 0.374 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.887 | 0.922 | 0.955 | 0.938 | 0.835 | 0.977 | 0.452 | 0.311 | 0.369 | 0.835 | 0.374 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.897 | 0.903 | 0.990 | 0.945 | 0.802 | 0.971 | 0.559 | 0.104 | 0.175 | 0.802 | 0.320 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.879 | 0.911 | 0.958 | 0.934 | 0.802 | 0.971 | 0.369 | 0.208 | 0.266 | 0.802 | 0.320 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.893 | 0.897 | 0.994 | 0.943 | 0.781 | 0.968 | 0.438 | 0.038 | 0.070 | 0.781 | 0.277 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.891 | 0.901 | 0.987 | 0.942 | 0.781 | 0.968 | 0.429 | 0.082 | 0.138 | 0.781 | 0.277 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.883 | 0.909 | 0.966 | 0.936 | 0.838 | 0.977 | 0.384 | 0.180 | 0.245 | 0.838 | 0.358 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.875 | 0.921 | 0.941 | 0.931 | 0.838 | 0.977 | 0.389 | 0.317 | 0.349 | 0.838 | 0.358 |
