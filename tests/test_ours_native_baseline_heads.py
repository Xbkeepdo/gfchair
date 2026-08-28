from __future__ import annotations

import unittest

import numpy as np

from scripts.train_ours_with_native_baseline_heads import (
    HEADS,
    aggregate_seed_metrics,
    feature_variants,
)


class OursNativeBaselineHeadsTests(unittest.TestCase):
    def test_svar_width_ablation_is_a_distinct_head(self) -> None:
        self.assertIn("svar_native", HEADS)
        self.assertIn("svar_native_h128", HEADS)

    def test_feature_variants_are_exact_block_slices(self) -> None:
        matrix = np.arange(24, dtype=np.float32).reshape(2, 12)
        variants = feature_variants(matrix, layers=4)
        np.testing.assert_array_equal(variants["old_risk_ev"], matrix[:, :8])
        np.testing.assert_array_equal(variants["old_risk_ev_s"], matrix)

    def test_feature_variants_reject_wrong_width_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected"):
            feature_variants(np.zeros((2, 11), dtype=np.float32), layers=4)
        matrix = np.zeros((2, 12), dtype=np.float32)
        matrix[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            feature_variants(matrix, layers=4)

    def test_native_metric_aggregation(self) -> None:
        rows = []
        for value in (0.7, 0.8, 0.9):
            rows.append(
                {
                    "test_metrics": {
                        "accuracy": value - 0.1,
                        "real_positive": {
                            "auc": value,
                            "f1": value - 0.2,
                            "aupr": value - 0.05,
                        },
                        "hallucination_positive": {
                            "f1": value - 0.3,
                            "aupr": value - 0.15,
                        },
                    }
                }
            )
        result = aggregate_seed_metrics(rows)
        self.assertAlmostEqual(result["auroc"]["mean"], 0.8)
        self.assertAlmostEqual(result["auroc"]["std"], np.std([0.7, 0.8, 0.9]))
        self.assertAlmostEqual(result["hall_aupr"]["mean"], 0.65)


if __name__ == "__main__":
    unittest.main()
