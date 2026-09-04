import unittest

import numpy as np

from scripts.train_union_topk_risk_s_no_ev import build_feature_sets


class UnionTopKRiskSNoEVTest(unittest.TestCase):
    def test_feature_order_and_ev_is_absent(self):
        blocks = {
            "old_risk": np.full((2, 3), 1.0, dtype=np.float32),
            "jffn_risk": np.full((2, 3), 2.0, dtype=np.float32),
            "s": np.full((2, 3), 3.0, dtype=np.float32),
            "ev": np.full((2, 3), 99.0, dtype=np.float32),
        }
        result = build_feature_sets(blocks)
        self.assertEqual(result["old_risk_only"].shape, (2, 3))
        self.assertEqual(result["old_risk_union_s"].shape, (2, 6))
        np.testing.assert_array_equal(
            result["old_risk_union_s"][0], [1, 1, 1, 3, 3, 3]
        )
        np.testing.assert_array_equal(
            result["jffn_risk_union_s"][0], [2, 2, 2, 3, 3, 3]
        )
        self.assertFalse(any(np.any(value == 99.0) for value in result.values()))

    def test_bad_shape_and_nonfinite_are_rejected(self):
        blocks = {
            "old_risk": np.ones((2, 3), dtype=np.float32),
            "jffn_risk": np.ones((2, 3), dtype=np.float32),
            "s": np.ones((2, 4), dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            build_feature_sets(blocks)
        blocks["s"] = np.ones((2, 3), dtype=np.float32)
        blocks["s"][0, 0] = np.inf
        with self.assertRaises(ValueError):
            build_feature_sets(blocks)


if __name__ == "__main__":
    unittest.main()
