import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.evaluate_visual_layer_sum_training_free import build_scores, hall_metrics


class VisualLayerSumTrainingFreeTests(unittest.TestCase):
    def test_sum_then_log_and_fixed_weights(self):
        ae = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        gross = np.asarray([[2.0, 3.0], [4.0, 5.0]])
        scores = build_scores(ae, gross)
        np.testing.assert_allclose(scores["ae_sum"], [3.0, 7.0])
        np.testing.assert_allclose(scores["strength_free"], np.log1p([5.0, 9.0]))
        np.testing.assert_allclose(
            scores["evidence_w0.2"],
            0.8 * np.log1p([5.0, 9.0]) + 0.2 * np.asarray([3.0, 7.0]),
        )
        np.testing.assert_allclose(scores["evidence_w0"], scores["strength_free"])
        np.testing.assert_allclose(scores["evidence_w1"], scores["ae_sum"])

    def test_hall_is_positive_and_lower_evidence_is_riskier(self):
        y_real = np.asarray([0, 0, 1, 1])
        evidence = np.asarray([0.1, 0.2, 0.8, 0.9])
        result = hall_metrics(y_real, evidence)
        self.assertEqual(result["AUROC"], 1.0)
        self.assertEqual(result["HALL_AUPR"], 1.0)
        np.testing.assert_allclose(
            result["AUROC"], roc_auc_score(1 - y_real, -evidence)
        )
        np.testing.assert_allclose(
            result["HALL_AUPR"], average_precision_score(1 - y_real, -evidence)
        )

    def test_rejects_invalid_source_arrays(self):
        with self.assertRaises(ValueError):
            build_scores(np.ones((2, 2)), np.ones((2, 3)))
        with self.assertRaises(ValueError):
            build_scores(np.asarray([[np.nan]]), np.ones((1, 1)))
        with self.assertRaises(ValueError):
            build_scores(np.ones((1, 1)), -np.ones((1, 1)))


if __name__ == "__main__":
    unittest.main()
