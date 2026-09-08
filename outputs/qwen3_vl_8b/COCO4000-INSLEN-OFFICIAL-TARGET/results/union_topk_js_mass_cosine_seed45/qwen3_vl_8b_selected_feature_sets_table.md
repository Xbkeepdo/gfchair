| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.842 | 0.862 | 0.963 | 0.910 | 0.822 | 0.957 | 0.583 | 0.252 | 0.352 | 0.822 | 0.508 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.838 | 0.869 | 0.948 | 0.907 | 0.822 | 0.957 | 0.547 | 0.304 | 0.391 | 0.822 | 0.508 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.850 | 0.865 | 0.971 | 0.915 | 0.827 | 0.956 | 0.651 | 0.264 | 0.375 | 0.827 | 0.521 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.852 | 0.878 | 0.954 | 0.914 | 0.827 | 0.956 | 0.613 | 0.353 | 0.448 | 0.827 | 0.521 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.861 | 0.905 | 0.930 | 0.917 | 0.864 | 0.968 | 0.605 | 0.523 | 0.561 | 0.864 | 0.611 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.859 | 0.899 | 0.935 | 0.917 | 0.864 | 0.968 | 0.607 | 0.490 | 0.542 | 0.864 | 0.611 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.843 | 0.871 | 0.953 | 0.910 | 0.826 | 0.958 | 0.575 | 0.312 | 0.405 | 0.826 | 0.508 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.843 | 0.876 | 0.944 | 0.909 | 0.826 | 0.958 | 0.563 | 0.353 | 0.434 | 0.826 | 0.508 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.856 | 0.875 | 0.963 | 0.917 | 0.844 | 0.960 | 0.649 | 0.333 | 0.440 | 0.844 | 0.542 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.855 | 0.871 | 0.969 | 0.917 | 0.844 | 0.960 | 0.667 | 0.298 | 0.412 | 0.844 | 0.542 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.846 | 0.911 | 0.904 | 0.907 | 0.862 | 0.966 | 0.548 | 0.568 | 0.558 | 0.862 | 0.592 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.853 | 0.900 | 0.926 | 0.913 | 0.862 | 0.966 | 0.580 | 0.500 | 0.537 | 0.862 | 0.592 |
