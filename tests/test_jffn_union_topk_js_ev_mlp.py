import unittest

import numpy as np

from scripts.train_feature_sets import build_selected_matrix, parse_feature_set
from scripts.train_jffn_union_topk_js_ev_mlp import union_topk_js


class JffnUnionTopkJsTests(unittest.TestCase):
    def test_matches_existing_union_topk_js_definition(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.random((3, 80), dtype=np.float64)
        target = rng.random((3, 80), dtype=np.float64)
        record = {
            "label": 1,
            "dgst_t_transport_top_k": 64,
            "dgst_t_source_dist_per_layer": source,
            "dgst_t_attention_support_per_layer": target,
            "dgst_t_hpre_raw_logit_gauss_gate_per_layer": np.ones_like(target),
        }
        existing, labels = build_selected_matrix(
            [record], parse_feature_set("hpre_raw_logit_gauss_source_target_union_topk_js")
        )
        actual = union_topk_js(source, target, side_top_k=32)
        self.assertEqual(labels.tolist(), [1])
        np.testing.assert_allclose(actual, existing[0], atol=1.0e-7, rtol=1.0e-6)
        self.assertTrue(np.all(actual >= 0.0))
        self.assertTrue(np.all(actual <= np.log(2.0) + 1.0e-7))


if __name__ == "__main__":
    unittest.main()
