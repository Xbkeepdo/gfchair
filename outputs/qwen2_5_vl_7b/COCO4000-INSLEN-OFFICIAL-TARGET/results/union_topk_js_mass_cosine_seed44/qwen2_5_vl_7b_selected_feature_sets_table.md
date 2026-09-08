| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.893 | 0.905 | 0.985 | 0.943 | 0.789 | 0.968 | 0.478 | 0.120 | 0.192 | 0.789 | 0.326 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.885 | 0.910 | 0.968 | 0.938 | 0.789 | 0.968 | 0.405 | 0.186 | 0.255 | 0.789 | 0.326 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.893 | 0.899 | 0.992 | 0.943 | 0.782 | 0.967 | 0.435 | 0.055 | 0.097 | 0.782 | 0.284 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.889 | 0.901 | 0.984 | 0.941 | 0.782 | 0.967 | 0.390 | 0.087 | 0.143 | 0.782 | 0.284 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.875 | 0.924 | 0.938 | 0.931 | 0.819 | 0.974 | 0.396 | 0.344 | 0.368 | 0.819 | 0.383 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.878 | 0.921 | 0.944 | 0.932 | 0.819 | 0.974 | 0.400 | 0.317 | 0.354 | 0.819 | 0.383 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.891 | 0.902 | 0.985 | 0.942 | 0.788 | 0.967 | 0.429 | 0.098 | 0.160 | 0.788 | 0.315 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.876 | 0.913 | 0.952 | 0.932 | 0.788 | 0.967 | 0.368 | 0.235 | 0.287 | 0.788 | 0.315 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.891 | 0.897 | 0.993 | 0.942 | 0.783 | 0.967 | 0.353 | 0.033 | 0.060 | 0.783 | 0.291 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.887 | 0.900 | 0.984 | 0.940 | 0.783 | 0.967 | 0.342 | 0.071 | 0.118 | 0.783 | 0.291 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.877 | 0.925 | 0.939 | 0.932 | 0.819 | 0.973 | 0.406 | 0.355 | 0.379 | 0.819 | 0.363 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.875 | 0.925 | 0.937 | 0.931 | 0.819 | 0.973 | 0.399 | 0.355 | 0.376 | 0.819 | 0.363 |
