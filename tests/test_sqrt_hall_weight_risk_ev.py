import math
import unittest

import numpy as np

from scripts.train_sqrt_hall_weight_risk_ev import (
    aggregate_fixed_threshold_metrics,
    aggregate_metrics,
    sqrt_hall_sample_weights,
)


class SqrtHallWeightRiskEVTest(unittest.TestCase):
    def test_only_hall_rows_receive_training_count_sqrt_weight(self):
        labels = np.asarray([1, 1, 1, 1, 0], dtype=np.int32)
        weights, audit = sqrt_hall_sample_weights(labels)
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0, 1.0, 2.0])
        self.assertEqual(audit["real_count"], 4)
        self.assertEqual(audit["hall_count"], 1)
        self.assertAlmostEqual(audit["hall_sample_weight"], math.sqrt(4.0))

    def test_nonbinary_or_single_class_labels_are_rejected(self):
        for labels in (
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([0, 1, 2], dtype=np.int32),
        ):
            with self.assertRaises(ValueError):
                sqrt_hall_sample_weights(labels)

    def test_aggregate_includes_hall_precision_and_recall(self):
        rows = []
        for value in (0.2, 0.4, 0.6):
            rows.append(
                {
                    "auc": 0.8,
                    "accuracy": 0.7,
                    "real_positive": {"f1": 0.8, "aupr": 0.9},
                    "hallucination_positive": {
                        "f1": 0.5,
                        "aupr": 0.6,
                        "precision": value,
                        "recall": value + 0.1,
                    },
                }
            )
        aggregate = aggregate_metrics(rows)
        self.assertAlmostEqual(aggregate["hall_precision"]["mean"], 0.4)
        self.assertAlmostEqual(aggregate["hall_recall"]["mean"], 0.5)

    def test_fixed_threshold_aggregation_uses_fixed_report(self):
        rows = []
        for value in (0.4, 0.6):
            rows.append(
                {
                    "threshold_reports": {
                        "fixed_0.5": {
                            "test_metrics": {
                                "accuracy": 0.7,
                                "real_positive": {"f1": 0.8},
                                "hallucination_positive": {
                                    "precision": value,
                                    "recall": value + 0.1,
                                    "f1": value + 0.05,
                                },
                            }
                        }
                    }
                }
            )
        result = aggregate_fixed_threshold_metrics(rows)
        self.assertAlmostEqual(result["hall_precision"]["mean"], 0.5)
        self.assertAlmostEqual(result["hall_recall"]["mean"], 0.6)
        self.assertAlmostEqual(result["hall_f1"]["mean"], 0.55)


if __name__ == "__main__":
    unittest.main()
