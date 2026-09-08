| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.823 | 0.893 | 0.876 | 0.885 | 0.868 | 0.958 | 0.601 | 0.640 | 0.620 | 0.868 | 0.624 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.824 | 0.874 | 0.903 | 0.888 | 0.868 | 0.958 | 0.624 | 0.554 | 0.587 | 0.868 | 0.624 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.823 | 0.879 | 0.895 | 0.887 | 0.861 | 0.953 | 0.615 | 0.575 | 0.595 | 0.861 | 0.632 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.822 | 0.865 | 0.912 | 0.888 | 0.861 | 0.953 | 0.629 | 0.512 | 0.565 | 0.861 | 0.632 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.834 | 0.888 | 0.899 | 0.894 | 0.888 | 0.965 | 0.639 | 0.611 | 0.624 | 0.888 | 0.659 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.833 | 0.885 | 0.901 | 0.893 | 0.888 | 0.965 | 0.637 | 0.599 | 0.618 | 0.888 | 0.659 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.825 | 0.881 | 0.895 | 0.888 | 0.867 | 0.957 | 0.618 | 0.584 | 0.600 | 0.867 | 0.646 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.826 | 0.870 | 0.912 | 0.890 | 0.867 | 0.957 | 0.637 | 0.530 | 0.579 | 0.867 | 0.646 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.831 | 0.883 | 0.901 | 0.892 | 0.874 | 0.959 | 0.634 | 0.591 | 0.612 | 0.874 | 0.638 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.828 | 0.856 | 0.936 | 0.894 | 0.874 | 0.959 | 0.676 | 0.458 | 0.546 | 0.874 | 0.638 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.836 | 0.898 | 0.888 | 0.893 | 0.888 | 0.964 | 0.630 | 0.654 | 0.642 | 0.888 | 0.672 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.836 | 0.894 | 0.895 | 0.894 | 0.888 | 0.964 | 0.637 | 0.635 | 0.636 | 0.888 | 0.672 |
