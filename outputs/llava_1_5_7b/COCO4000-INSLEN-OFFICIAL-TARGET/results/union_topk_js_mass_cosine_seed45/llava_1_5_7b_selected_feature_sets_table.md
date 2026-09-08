| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.821 | 0.887 | 0.882 | 0.884 | 0.866 | 0.957 | 0.602 | 0.614 | 0.608 | 0.866 | 0.641 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.825 | 0.874 | 0.904 | 0.889 | 0.866 | 0.957 | 0.627 | 0.551 | 0.587 | 0.866 | 0.641 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.827 | 0.873 | 0.909 | 0.890 | 0.869 | 0.957 | 0.635 | 0.544 | 0.586 | 0.869 | 0.641 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.827 | 0.875 | 0.906 | 0.890 | 0.869 | 0.957 | 0.633 | 0.554 | 0.591 | 0.869 | 0.641 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.836 | 0.897 | 0.890 | 0.894 | 0.890 | 0.966 | 0.633 | 0.649 | 0.641 | 0.890 | 0.674 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.837 | 0.898 | 0.890 | 0.894 | 0.890 | 0.966 | 0.634 | 0.653 | 0.644 | 0.890 | 0.674 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.827 | 0.867 | 0.918 | 0.892 | 0.864 | 0.955 | 0.647 | 0.516 | 0.574 | 0.864 | 0.643 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.827 | 0.866 | 0.919 | 0.891 | 0.864 | 0.955 | 0.646 | 0.511 | 0.571 | 0.864 | 0.643 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.830 | 0.879 | 0.905 | 0.892 | 0.876 | 0.959 | 0.636 | 0.570 | 0.601 | 0.876 | 0.646 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.831 | 0.870 | 0.918 | 0.894 | 0.876 | 0.959 | 0.653 | 0.530 | 0.585 | 0.876 | 0.646 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.835 | 0.908 | 0.876 | 0.892 | 0.891 | 0.966 | 0.620 | 0.695 | 0.656 | 0.891 | 0.681 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.836 | 0.898 | 0.888 | 0.893 | 0.891 | 0.966 | 0.630 | 0.654 | 0.642 | 0.891 | 0.681 |
