| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.890 | 0.904 | 0.982 | 0.941 | 0.790 | 0.968 | 0.429 | 0.115 | 0.181 | 0.790 | 0.330 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.882 | 0.913 | 0.960 | 0.936 | 0.790 | 0.968 | 0.398 | 0.224 | 0.287 | 0.790 | 0.330 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.888 | 0.900 | 0.985 | 0.940 | 0.784 | 0.968 | 0.351 | 0.071 | 0.118 | 0.784 | 0.283 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.885 | 0.901 | 0.979 | 0.938 | 0.784 | 0.968 | 0.340 | 0.093 | 0.146 | 0.784 | 0.283 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.889 | 0.912 | 0.969 | 0.940 | 0.835 | 0.977 | 0.442 | 0.208 | 0.283 | 0.835 | 0.373 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.877 | 0.924 | 0.940 | 0.932 | 0.835 | 0.977 | 0.404 | 0.344 | 0.372 | 0.835 | 0.373 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.891 | 0.903 | 0.984 | 0.942 | 0.785 | 0.967 | 0.444 | 0.109 | 0.175 | 0.785 | 0.316 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.886 | 0.912 | 0.966 | 0.938 | 0.785 | 0.967 | 0.418 | 0.208 | 0.277 | 0.785 | 0.316 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.891 | 0.901 | 0.986 | 0.942 | 0.792 | 0.969 | 0.417 | 0.082 | 0.137 | 0.792 | 0.317 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.890 | 0.904 | 0.981 | 0.941 | 0.792 | 0.969 | 0.420 | 0.115 | 0.180 | 0.792 | 0.317 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.887 | 0.913 | 0.966 | 0.939 | 0.821 | 0.974 | 0.435 | 0.219 | 0.291 | 0.821 | 0.342 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.872 | 0.920 | 0.939 | 0.929 | 0.821 | 0.974 | 0.375 | 0.311 | 0.340 | 0.821 | 0.342 |
