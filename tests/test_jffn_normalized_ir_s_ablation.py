import unittest

import numpy as np

from scripts.train_jffn_normalized_ir_s_ablation import (
    build_feature_sets,
    curve_statistics,
    l2_normalize_rows,
)


class JFFNNormalizedIRSFeatureTest(unittest.TestCase):
    def test_normalize_then_sum_and_feature_order(self):
        blocks = {
            "i": np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
            "r": np.asarray([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "s": np.full((2, 2), 6.0, dtype=np.float32),
            "old_risk": np.full((2, 2), 7.0, dtype=np.float32),
            "jffn_risk": np.full((2, 2), 8.0, dtype=np.float32),
            "ev": np.full((2, 2), 9.0, dtype=np.float32),
        }
        result = build_feature_sets(blocks)
        expected_n = l2_normalize_rows(blocks["i"]) + l2_normalize_rows(blocks["r"])
        np.testing.assert_allclose(result["normalized_ir_sum_only"], expected_n)
        np.testing.assert_array_equal(result["s_only"], blocks["s"])
        self.assertEqual(result["normalized_ir_sum_s"].shape, (2, 4))
        self.assertEqual(result["old_risk_ev_normalized_ir_sum_s"].shape, (2, 8))
        np.testing.assert_allclose(
            result["jffn_risk_ev_normalized_ir_sum_s"][0],
            np.concatenate((blocks["jffn_risk"][0], blocks["ev"][0], expected_n[0], blocks["s"][0])),
        )

    def test_each_i_r_row_has_unit_norm_before_sum(self):
        values = np.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
        normalized = l2_normalize_rows(values)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])
        with self.assertRaises(ValueError):
            l2_normalize_rows(np.zeros((1, 2), dtype=np.float32))

    def test_curve_direction_covers_n_and_s(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
        blocks = {
            "i": np.asarray([[3, 4], [3, 4], [4, 3], [4, 3]], dtype=np.float32),
            "r": np.asarray([[3, 4], [3, 4], [4, 3], [4, 3]], dtype=np.float32),
            "s": np.asarray([[3, 4], [3, 4], [1, 2], [1, 2]], dtype=np.float32),
        }
        label_rows, effect_rows = curve_statistics(blocks, labels)
        self.assertEqual({row["feature"] for row in label_rows}, {"N", "S"})
        self.assertEqual(len(effect_rows), 4)
        s_rows = [row for row in effect_rows if row["feature"] == "S"]
        self.assertTrue(all(row["hall_minus_real"] < 0 for row in s_rows))


if __name__ == "__main__":
    unittest.main()
