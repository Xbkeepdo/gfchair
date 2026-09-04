import unittest

import numpy as np

from scripts.train_union_topk_irs_ablation import build_feature_sets


class UnionTopKIRSFeatureTest(unittest.TestCase):
    def test_feature_dimensions_and_block_order(self):
        blocks = {
            "i": np.full((2, 3), 1.0, dtype=np.float32),
            "r": np.full((2, 3), 2.0, dtype=np.float32),
            "s": np.full((2, 3), 3.0, dtype=np.float32),
            "old_risk": np.full((2, 3), 4.0, dtype=np.float32),
            "jffn_risk": np.full((2, 3), 5.0, dtype=np.float32),
            "ev": np.full((2, 3), 6.0, dtype=np.float32),
        }
        result = build_feature_sets(blocks)
        self.assertEqual(result["union_i_only"].shape, (2, 3))
        self.assertEqual(result["union_i_r_s"].shape, (2, 9))
        self.assertEqual(result["old_risk_ev_union_i_r_s"].shape, (2, 15))
        np.testing.assert_array_equal(
            result["jffn_risk_ev_union_i_r_s"][0],
            [5, 5, 5, 6, 6, 6, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        )

    def test_mismatch_and_nonfinite_are_rejected(self):
        names = ("i", "r", "s", "old_risk", "jffn_risk", "ev")
        blocks = {name: np.ones((2, 3), dtype=np.float32) for name in names}
        blocks["r"] = np.ones((2, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_feature_sets(blocks)
        blocks["r"] = np.ones((2, 3), dtype=np.float32)
        blocks["s"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_feature_sets(blocks)


if __name__ == "__main__":
    unittest.main()
