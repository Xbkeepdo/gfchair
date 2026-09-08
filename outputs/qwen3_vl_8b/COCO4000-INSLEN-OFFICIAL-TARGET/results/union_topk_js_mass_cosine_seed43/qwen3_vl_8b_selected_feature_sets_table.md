| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.844 | 0.869 | 0.956 | 0.910 | 0.825 | 0.958 | 0.583 | 0.300 | 0.396 | 0.825 | 0.522 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.843 | 0.871 | 0.951 | 0.909 | 0.825 | 0.958 | 0.570 | 0.314 | 0.405 | 0.825 | 0.522 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.850 | 0.861 | 0.977 | 0.915 | 0.820 | 0.953 | 0.676 | 0.234 | 0.348 | 0.820 | 0.521 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.851 | 0.878 | 0.952 | 0.914 | 0.820 | 0.953 | 0.607 | 0.359 | 0.451 | 0.820 | 0.521 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.855 | 0.902 | 0.926 | 0.914 | 0.868 | 0.967 | 0.585 | 0.512 | 0.546 | 0.868 | 0.624 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.854 | 0.904 | 0.922 | 0.913 | 0.868 | 0.967 | 0.578 | 0.523 | 0.549 | 0.868 | 0.624 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.846 | 0.863 | 0.969 | 0.913 | 0.823 | 0.957 | 0.623 | 0.250 | 0.357 | 0.823 | 0.514 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.828 | 0.884 | 0.912 | 0.898 | 0.823 | 0.957 | 0.493 | 0.415 | 0.451 | 0.823 | 0.514 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.855 | 0.872 | 0.967 | 0.917 | 0.841 | 0.959 | 0.657 | 0.308 | 0.420 | 0.841 | 0.544 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.853 | 0.882 | 0.951 | 0.915 | 0.841 | 0.959 | 0.613 | 0.380 | 0.469 | 0.841 | 0.544 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.863 | 0.885 | 0.959 | 0.921 | 0.866 | 0.966 | 0.667 | 0.395 | 0.496 | 0.866 | 0.603 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.853 | 0.906 | 0.918 | 0.912 | 0.866 | 0.966 | 0.574 | 0.535 | 0.554 | 0.866 | 0.603 |
