| Feature Set | Classifier | Accuracy | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.845 | 0.880 | 0.941 | 0.910 | 0.828 | 0.958 | 0.567 | 0.376 | 0.452 | 0.828 | 0.526 |
| hpre_raw_logit_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.834 | 0.889 | 0.914 | 0.901 | 0.828 | 0.958 | 0.515 | 0.446 | 0.478 | 0.828 | 0.526 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.853 | 0.875 | 0.961 | 0.916 | 0.834 | 0.958 | 0.633 | 0.331 | 0.435 | 0.834 | 0.534 |
| hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.854 | 0.877 | 0.957 | 0.916 | 0.834 | 0.958 | 0.627 | 0.349 | 0.448 | 0.834 | 0.534 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.846 | 0.920 | 0.892 | 0.906 | 0.863 | 0.966 | 0.541 | 0.620 | 0.578 | 0.863 | 0.609 |
| hpre_raw_logit_gauss_source_target_union_topk_js+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.860 | 0.903 | 0.931 | 0.917 | 0.863 | 0.966 | 0.604 | 0.512 | 0.554 | 0.863 | 0.609 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[fixed_0.5] | 0.837 | 0.879 | 0.932 | 0.905 | 0.823 | 0.957 | 0.533 | 0.376 | 0.441 | 0.823 | 0.499 |
| hpre_softmax_prob_gauss_source_target_union_topk_js | torch_probe[train_f1] | 0.836 | 0.882 | 0.927 | 0.904 | 0.823 | 0.957 | 0.527 | 0.397 | 0.453 | 0.823 | 0.499 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.851 | 0.877 | 0.953 | 0.914 | 0.841 | 0.959 | 0.607 | 0.351 | 0.445 | 0.841 | 0.527 |
| hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.850 | 0.876 | 0.955 | 0.914 | 0.841 | 0.959 | 0.608 | 0.343 | 0.439 | 0.841 | 0.527 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[fixed_0.5] | 0.846 | 0.915 | 0.898 | 0.906 | 0.863 | 0.966 | 0.544 | 0.593 | 0.568 | 0.863 | 0.587 |
| hpre_softmax_prob_gauss_source_target_union_topk_js+hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine | torch_probe[train_f1] | 0.850 | 0.903 | 0.918 | 0.911 | 0.863 | 0.966 | 0.566 | 0.521 | 0.543 | 0.863 | 0.587 |
