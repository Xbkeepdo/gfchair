| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.846 | 0.865 | 0.969 | 0.914 | 0.784 | 0.950 | 0.523 | 0.182 | 0.270 | 0.784 | 0.408 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.845 | 0.861 | 0.973 | 0.914 | 0.784 | 0.950 | 0.518 | 0.158 | 0.242 | 0.784 | 0.408 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.845 | 0.878 | 0.949 | 0.912 | 0.835 | 0.963 | 0.512 | 0.290 | 0.370 | 0.835 | 0.469 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.846 | 0.874 | 0.955 | 0.913 | 0.835 | 0.963 | 0.519 | 0.260 | 0.346 | 0.835 | 0.469 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.850 | 0.904 | 0.920 | 0.912 | 0.854 | 0.967 | 0.525 | 0.475 | 0.499 | 0.854 | 0.513 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.855 | 0.900 | 0.931 | 0.915 | 0.854 | 0.967 | 0.544 | 0.445 | 0.490 | 0.854 | 0.513 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.840 | 0.872 | 0.950 | 0.909 | 0.798 | 0.953 | 0.482 | 0.249 | 0.329 | 0.798 | 0.410 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.831 | 0.882 | 0.923 | 0.902 | 0.798 | 0.953 | 0.448 | 0.335 | 0.383 | 0.798 | 0.410 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.845 | 0.856 | 0.983 | 0.915 | 0.827 | 0.961 | 0.533 | 0.107 | 0.179 | 0.827 | 0.441 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.849 | 0.866 | 0.971 | 0.916 | 0.827 | 0.961 | 0.554 | 0.193 | 0.286 | 0.827 | 0.441 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.853 | 0.898 | 0.932 | 0.915 | 0.849 | 0.964 | 0.541 | 0.429 | 0.478 | 0.849 | 0.501 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.849 | 0.899 | 0.925 | 0.912 | 0.849 | 0.964 | 0.521 | 0.440 | 0.477 | 0.849 | 0.501 |
