| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.846 | 0.862 | 0.973 | 0.914 | 0.784 | 0.949 | 0.526 | 0.161 | 0.246 | 0.784 | 0.408 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.839 | 0.871 | 0.950 | 0.909 | 0.784 | 0.949 | 0.474 | 0.244 | 0.322 | 0.784 | 0.408 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.853 | 0.868 | 0.974 | 0.918 | 0.842 | 0.965 | 0.591 | 0.201 | 0.300 | 0.842 | 0.479 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.852 | 0.871 | 0.968 | 0.917 | 0.842 | 0.965 | 0.568 | 0.225 | 0.322 | 0.842 | 0.479 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.853 | 0.880 | 0.956 | 0.916 | 0.845 | 0.961 | 0.558 | 0.298 | 0.388 | 0.845 | 0.490 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.851 | 0.889 | 0.941 | 0.914 | 0.845 | 0.961 | 0.537 | 0.370 | 0.438 | 0.845 | 0.490 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.843 | 0.862 | 0.970 | 0.913 | 0.795 | 0.951 | 0.500 | 0.161 | 0.243 | 0.795 | 0.412 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.842 | 0.879 | 0.942 | 0.909 | 0.795 | 0.951 | 0.491 | 0.303 | 0.375 | 0.795 | 0.412 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.846 | 0.852 | 0.990 | 0.916 | 0.830 | 0.962 | 0.574 | 0.072 | 0.129 | 0.830 | 0.447 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.846 | 0.873 | 0.957 | 0.913 | 0.830 | 0.962 | 0.517 | 0.249 | 0.336 | 0.830 | 0.447 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.848 | 0.888 | 0.938 | 0.913 | 0.847 | 0.963 | 0.523 | 0.365 | 0.430 | 0.847 | 0.492 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.847 | 0.889 | 0.936 | 0.912 | 0.847 | 0.963 | 0.517 | 0.370 | 0.431 | 0.847 | 0.492 |
