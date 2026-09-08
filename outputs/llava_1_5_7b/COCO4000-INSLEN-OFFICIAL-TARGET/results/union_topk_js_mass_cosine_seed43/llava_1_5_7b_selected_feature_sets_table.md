| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.826 | 0.884 | 0.893 | 0.889 | 0.864 | 0.956 | 0.619 | 0.597 | 0.608 | 0.864 | 0.630 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.822 | 0.860 | 0.921 | 0.889 | 0.864 | 0.956 | 0.640 | 0.484 | 0.551 | 0.864 | 0.630 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.826 | 0.888 | 0.888 | 0.888 | 0.866 | 0.956 | 0.615 | 0.614 | 0.614 | 0.866 | 0.636 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.829 | 0.868 | 0.918 | 0.892 | 0.866 | 0.956 | 0.649 | 0.522 | 0.579 | 0.866 | 0.636 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.844 | 0.895 | 0.904 | 0.900 | 0.887 | 0.964 | 0.659 | 0.635 | 0.647 | 0.887 | 0.664 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.844 | 0.899 | 0.900 | 0.899 | 0.887 | 0.964 | 0.654 | 0.652 | 0.653 | 0.887 | 0.664 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.822 | 0.889 | 0.879 | 0.884 | 0.863 | 0.954 | 0.601 | 0.623 | 0.612 | 0.863 | 0.646 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.825 | 0.871 | 0.908 | 0.889 | 0.863 | 0.954 | 0.630 | 0.539 | 0.581 | 0.863 | 0.646 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.829 | 0.881 | 0.902 | 0.891 | 0.874 | 0.960 | 0.631 | 0.580 | 0.604 | 0.874 | 0.637 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.828 | 0.857 | 0.933 | 0.893 | 0.874 | 0.960 | 0.669 | 0.467 | 0.550 | 0.874 | 0.637 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.833 | 0.906 | 0.874 | 0.890 | 0.888 | 0.964 | 0.615 | 0.690 | 0.650 | 0.888 | 0.674 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.835 | 0.894 | 0.893 | 0.894 | 0.888 | 0.964 | 0.634 | 0.636 | 0.635 | 0.888 | 0.674 |
