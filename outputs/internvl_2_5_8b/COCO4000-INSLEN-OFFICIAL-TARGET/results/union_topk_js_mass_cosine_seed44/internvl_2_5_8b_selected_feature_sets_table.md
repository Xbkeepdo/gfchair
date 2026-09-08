| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.841 | 0.864 | 0.963 | 0.911 | 0.784 | 0.949 | 0.483 | 0.188 | 0.270 | 0.784 | 0.401 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.841 | 0.867 | 0.958 | 0.910 | 0.784 | 0.949 | 0.482 | 0.212 | 0.294 | 0.784 | 0.401 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.856 | 0.876 | 0.967 | 0.919 | 0.839 | 0.963 | 0.595 | 0.260 | 0.362 | 0.839 | 0.491 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.849 | 0.884 | 0.946 | 0.914 | 0.839 | 0.963 | 0.530 | 0.330 | 0.407 | 0.839 | 0.491 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.857 | 0.871 | 0.975 | 0.920 | 0.857 | 0.967 | 0.624 | 0.223 | 0.328 | 0.857 | 0.535 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.858 | 0.898 | 0.938 | 0.917 | 0.857 | 0.967 | 0.560 | 0.426 | 0.484 | 0.857 | 0.535 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.848 | 0.861 | 0.978 | 0.915 | 0.796 | 0.953 | 0.550 | 0.147 | 0.233 | 0.796 | 0.413 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.848 | 0.863 | 0.975 | 0.916 | 0.796 | 0.953 | 0.553 | 0.169 | 0.259 | 0.796 | 0.413 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.848 | 0.868 | 0.967 | 0.915 | 0.833 | 0.961 | 0.538 | 0.206 | 0.298 | 0.833 | 0.466 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.848 | 0.873 | 0.959 | 0.914 | 0.833 | 0.961 | 0.531 | 0.249 | 0.339 | 0.833 | 0.466 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.849 | 0.897 | 0.927 | 0.912 | 0.846 | 0.963 | 0.523 | 0.429 | 0.471 | 0.846 | 0.499 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.849 | 0.900 | 0.924 | 0.912 | 0.846 | 0.963 | 0.522 | 0.448 | 0.482 | 0.846 | 0.499 |
