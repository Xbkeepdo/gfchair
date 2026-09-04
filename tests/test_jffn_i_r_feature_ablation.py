import unittest

import numpy as np

from scripts.train_jffn_i_r_feature_ablation import build_feature_sets, curve_statistics


class JFFNIRFeatureAblationTest(unittest.TestCase):
    def test_feature_blocks_are_concatenated_in_declared_order(self):
        blocks = {
            "i": np.full((3, 2), 1.0, dtype=np.float32),
            "r": np.full((3, 2), 2.0, dtype=np.float32),
            "old_risk": np.full((3, 2), 3.0, dtype=np.float32),
            "jffn_risk": np.full((3, 2), 4.0, dtype=np.float32),
            "ev": np.full((3, 2), 5.0, dtype=np.float32),
        }
        result = build_feature_sets(blocks)
        self.assertEqual(result["i_only"].shape, (3, 2))
        self.assertEqual(result["i_r_only"].shape, (3, 4))
        self.assertEqual(result["old_risk_ev_i_r"].shape, (3, 8))
        self.assertEqual(result["jffn_risk_ev_i_r"].shape, (3, 8))
        np.testing.assert_array_equal(result["old_risk_ev_i_r"][0], [3, 3, 5, 5, 1, 1, 2, 2])
        np.testing.assert_array_equal(result["jffn_risk_ev_r"][0], [4, 4, 5, 5, 2, 2])

    def test_mismatched_or_nonfinite_blocks_are_rejected(self):
        blocks = {name: np.ones((3, 2), dtype=np.float32) for name in ("i", "r", "old_risk", "jffn_risk", "ev")}
        blocks["r"] = np.ones((3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_feature_sets(blocks)
        blocks["r"] = np.ones((3, 2), dtype=np.float32)
        blocks["ev"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_feature_sets(blocks)

    def test_curve_direction_uses_hall_minus_real(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
        blocks = {
            "i": np.asarray([[3, 4], [3, 4], [1, 2], [1, 2]], dtype=np.float32),
            "r": np.asarray([[1, 2], [1, 2], [4, 5], [4, 5]], dtype=np.float32),
        }
        label_rows, effect_rows = curve_statistics(blocks, labels)
        self.assertEqual(len(label_rows), 8)
        self.assertEqual(len(effect_rows), 4)
        i_rows = [row for row in effect_rows if row["feature"] == "I"]
        r_rows = [row for row in effect_rows if row["feature"] == "R"]
        self.assertTrue(all(row["hall_minus_real"] < 0 for row in i_rows))
        self.assertTrue(all(row["hall_minus_real"] > 0 for row in r_rows))


if __name__ == "__main__":
    unittest.main()
